"""
effect size 병합 규칙의 문턱값 민감도 분석 (ENERGY, 읽기 전용).

실행: uv run --group demo python -m rdh_data_adapter.analysis.rdh_effect_size_sensitivity
산출물: rdh_data_adapter/analysis/reports/rdh_effect_size_sensitivity.md
        rdh_data_adapter/analysis/reports/rdh_effect_size_sensitivity_raw.md
        rdh_data_adapter/analysis/reports/figures/ees_*.png

지표는 모두 repo 구현을 그대로 쓴다.
- 짝짓기/그래프: rdh_data_adapter.load.resource_graph.get_duckdb_delegate_graph_full (R1 FIFO, warm-up/horizon 403,200블록, read_only)
- s_in, s_out: rdh.models.effect.concentration.effect_concentration
- active: rdh.models.effect.coverage.effect_coverage (bucket = 1200블록)
- phi, m, mu: rdh.models.effect.menu_ratio.effect_menu_ratio (open 제외)
"""
import inspect
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import rdh_data_adapter.load.connection
import rdh_data_adapter.load.resource_graph
import rdh.graph.transform
import rdh.models.effect.concentration
import rdh.models.effect.coverage
import rdh.models.effect.menu_ratio
import rdh.pipeline.effect.pair_delegate_undelegate
from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full
from rdh.models.effect.concentration import effect_concentration
from rdh.models.effect.coverage import effect_coverage
from rdh.models.effect.menu_ratio import effect_menu_ratio

START, END = 80_000_000, 80_200_000
BUCKET = 1200
EXPECTED_PAIRS = 3_905_329
MENU_PARAMS = dict(flash_max=100, rho=0.05, k=3, min_n=50)

GRID_A = (3, 5, 10)
GRID_MU0 = (0.3, 0.4, 0.5, 0.6, 0.7, "otsu", "none")
GRID_TAU = (0.2, 0.3, 0.4, 0.5, 0.8, 1.0)
BASE = (5, 0.5, 0.5)
MU_LINES = (0.24, 0.90)  # 지시서에 주어진 시뮬레이션 기준선

OUT_DIR = Path(__file__).resolve().parent / "reports"
FIG_DIR = OUT_DIR / "figures"
NAME = "rdh_effect_size_sensitivity"


def otsu_threshold(x, bins=100, lo=0.0, hi=1.0):
    h, e = np.histogram(x, bins=bins, range=(lo, hi))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c)
    mu0 = s0 / np.maximum(w0, 1); mu1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return e[np.argmax(w0 * w1 * (mu0 - mu1) ** 2) + 1]


# ---------- 1. 지표 ----------

def pair_metrics(g) -> pd.DataFrame:
    conc = effect_concentration(g)
    cov = effect_coverage(g, START, END, bucket=BUCKET)
    N = g["g_node_count"]
    n_P = np.bincount(g["uk_src"], weights=g["uk_cnt"], minlength=N)
    n_in = np.bincount(g["uk_dst"], weights=g["uk_cnt"], minlength=N)
    df = pd.DataFrame({
        "P": g["uk_src"], "r": g["uk_dst"], "n_pair": g["uk_cnt"],
        "n_P": n_P[g["uk_src"]].astype(np.int64), "n_in": n_in[g["uk_dst"]].astype(np.int64),
        "s_out": conc["s_out"], "s_in": conc["s_in"], "active": cov["active"],
    })
    df["e"] = np.minimum(df["s_in"], df["s_out"])

    # assert
    has_out, has_in = n_P > 0, n_in > 0
    np.testing.assert_allclose(np.bincount(df["P"], weights=df["s_out"], minlength=N)[has_out], 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.bincount(df["r"], weights=df["s_in"], minlength=N)[has_in], 1.0, rtol=0, atol=1e-9)
    assert (df["active"] <= df["n_pair"]).all()
    assert len(df) == EXPECTED_PAIRS, len(df)
    return df


def provider_metrics(g) -> tuple[dict, np.ndarray]:
    r = effect_menu_ratio(g, **MENU_PARAMS)
    mu_node = np.full(g["g_node_count"], np.nan)
    mu_node[r["id"]] = r["mu"]
    return r, mu_node


# ---------- 2. 병합 규칙 / 격자 ----------

def merge_mask(df, mu_pair, A, mu0, tau, mu0_otsu):
    mask = df["active"].to_numpy() > A
    if mu0 != "none":
        th = mu0_otsu if mu0 == "otsu" else mu0
        mask &= np.isnan(mu_pair) | (mu_pair < th)
    e = df["e"].to_numpy()
    mask &= (e == 1.0) if tau == 1.0 else (e > tau)
    return mask


def component_stats(df, mask, N) -> tuple[dict, np.ndarray]:
    P, r = df["P"].to_numpy()[mask], df["r"].to_numpy()[mask]
    row = {"merged_pairs": int(mask.sum()), "merged_nodes": int(np.unique(np.concatenate((P, r))).size)}
    if row["merged_pairs"] == 0:
        return row | {"components": 0, "max_comp": 0, "median_comp": np.nan, "p99_comp": np.nan,
                      "max_out_deg": 0, "max_in_deg": 0}, np.array([], dtype=np.int64)
    adj = csr_matrix((np.ones(P.size), (P, r)), shape=(N, N))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    comps = sizes[sizes >= 2]
    row |= {
        "components": int(comps.size), "max_comp": int(comps.max()),
        "median_comp": float(np.median(comps)), "p99_comp": float(np.quantile(comps, 0.99)),
        "max_out_deg": int(np.bincount(P).max()), "max_in_deg": int(np.bincount(r).max()),
    }
    return row, comps


def run_grid(df, mu_pair, mu0_otsu, N):
    masks, rows, comps = {}, [], {}
    for A, mu0, tau in itertools.product(GRID_A, GRID_MU0, GRID_TAU):
        mask = merge_mask(df, mu_pair, A, mu0, tau, mu0_otsu)
        row, c = component_stats(df, mask, N)
        masks[(A, mu0, tau)] = mask
        comps[(A, mu0, tau)] = c
        rows.append({"A": A, "mu0": mu0, "tau": tau, **row})
    grid = pd.DataFrame(rows)
    base = masks[BASE]
    grid["jaccard_vs_base"] = [
        (m & base).sum() / max((m | base).sum(), 1) for m in masks.values()
    ]
    return grid, masks, comps


def funnel(df, mu_pair):
    A, mu0, tau = BASE
    s1 = df["active"].to_numpy() > A
    s2 = s1 & (np.isnan(mu_pair) | (mu_pair < mu0))
    s3 = s2 & (df["e"].to_numpy() > tau)
    n = len(df)
    steps = [("all pairs", n), (f"active > {A}", s1.sum()), (f"mu_P < {mu0} or NaN", s2.sum()), (f"e > {tau}", s3.sum())]
    out = pd.DataFrame(steps, columns=["step", "pairs"]).set_index("step")
    out["share_of_all"] = out["pairs"] / n
    out["kept_from_prev"] = out["pairs"] / out["pairs"].shift(1).fillna(n)
    return out


# ---------- 4. 그림 ----------

def plot_mu(r, mu0_otsu, path):
    mu = r["mu"][~np.isnan(r["mu"])]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(mu, bins=50, range=(0, 1))
    for x, label, c in ((MU_LINES[0], "0.24", "tab:gray"), (MU_LINES[1], "0.90", "tab:gray"), (mu0_otsu, f"otsu {mu0_otsu:.2f}", "tab:red")):
        ax[0].axvline(x, c=c, ls="--"); ax[0].text(x, 1, f" {label}", transform=ax[0].get_xaxis_transform(), va="top", fontsize=9)
    ax[0].set_yscale("log"); ax[0].set_xlabel("mu_P"); ax[0].set_ylabel("providers")
    ax[0].set_title(f"mu_P histogram (providers with m_P >= {MENU_PARAMS['min_n']}: {mu.size:,})")
    v = ~np.isnan(r["mu"])
    sc = ax[1].scatter(r["m"][v], r["mu"][v], c=r["phi"][v], s=6, alpha=0.6, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax[1], label="phi_P")
    ax[1].set_xscale("log"); ax[1].set_xlabel("m_P"); ax[1].set_ylabel("mu_P"); ax[1].set_title("mu_P vs m_P")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_e(df, path):
    e = df.loc[df["active"] > BASE[0], "e"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(e, bins=50, range=(0, 1))
    ax.set_yscale("log"); ax.set_xlabel("e = min(s_in, s_out)"); ax.set_ylabel("pairs")
    ax.set_title(f"e histogram, active > {BASE[0]} (pairs {len(e):,})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_heatmaps(grid, mu0_otsu, path):
    sub = grid[grid["A"] == BASE[0]]
    cols = [str(m) if m != "otsu" else f"otsu({mu0_otsu:.2f})" for m in GRID_MU0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a, metric in zip(ax, ("merged_pairs", "max_comp")):
        t = sub.pivot(index="tau", columns="mu0", values=metric).reindex(index=list(GRID_TAU), columns=list(GRID_MU0))
        im = a.imshow(t.to_numpy(dtype=float), aspect="auto", cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1))
        for i, j in itertools.product(range(t.shape[0]), range(t.shape[1])):
            a.text(j, i, f"{int(t.iat[i, j]):,}", ha="center", va="center", fontsize=8, color="w")
        a.set_xticks(range(len(cols)), cols); a.set_yticks(range(len(GRID_TAU)), GRID_TAU)
        a.set_xlabel("mu0"); a.set_ylabel("tau"); a.set_title(f"{metric} (A={BASE[0]})")
        fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_comp_sizes(comps, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    A, mu0, _ = BASE
    for tau in (BASE[2], 0.3, 0.2):
        c = comps[(A, mu0, tau)]
        if c.size == 0:
            continue
        sizes, counts = np.unique(c, return_counts=True)
        ax.plot(sizes, counts, "o-", ms=3, label=f"A={A}, mu0={mu0}, tau={tau}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("component size (nodes)"); ax.set_ylabel("components"); ax.legend()
    ax.set_title("component size distribution (size >= 2)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------- 5. 보고서 ----------

def md(df, pct=(), floatfmt=".3f"):
    df = df.copy()
    for c in pct:
        df[c] = df[c].map(lambda v: f"{v:.4%}")
    # 정수 열을 먼저 문자열로 바꾼다 (섞인 dtype이면 to_markdown이 전부 float으로 올림)
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:,}")
    return df.to_markdown(floatfmt=floatfmt, intfmt=",")


def summary_lines(fun, grid, otsu, n_mu, n_excl):
    g = grid.set_index(["A", "mu0", "tau"])
    pairs = lambda key: int(g.loc[key, "merged_pairs"])
    maxc = lambda key: int(g.loc[key, "max_comp"])
    base_pairs, base_max = pairs(BASE), maxc(BASE)
    drops = fun["kept_from_prev"].iloc[1:]
    biggest = drops.idxmin()
    A, mu0, tau = BASE
    lines = [
        f"1. 기준 설정(A={A}, mu0={mu0}, tau={tau})의 병합 쌍은 {base_pairs:,}개다. "
        + ", ".join(f"{s}: {int(fun.loc[s, 'pairs']):,} (직전 단계의 {fun.loc[s, 'kept_from_prev']:.2%})" for s in fun.index[1:])
        + f". 직전 단계 대비 남는 비율이 가장 낮은 단계는 `{biggest}`이다.",
        "2. tau를 0.5 아래로 내릴 때 (A=5, mu0=0.5): "
        + "; ".join(f"tau={t}: 병합 쌍 {pairs((A, mu0, t)):,} (x{pairs((A, mu0, t)) / base_pairs:.2f}), "
                    f"최대 컴포넌트 {maxc((A, mu0, t)):,} (x{maxc((A, mu0, t)) / base_max:.2f})" for t in (0.4, 0.3, 0.2))
        + f" (tau=0.5: 병합 쌍 {base_pairs:,}, 최대 컴포넌트 {base_max:,}).",
        "3. mu0를 0.3~0.7로 바꿀 때 (A=5, tau=0.5) 병합 쌍 수: "
        + ", ".join(f"{m}: {pairs((A, m, tau)):,}" for m in (0.3, 0.4, 0.5, 0.6, 0.7))
        + f" (mu 조건 없음: {pairs((A, 'none', tau)):,}). mu0_otsu = {otsu:.4f}이고, 그때 병합 쌍은 {pairs((A, 'otsu', tau)):,}개다."
        + f" mu가 정의된 provider는 {n_mu:,}명, mu0=0.5에서 mu로 제외되는 provider는 {n_excl:,}명이다.",
        "4. A를 바꿀 때 (mu0=0.5, tau=0.5) 병합 쌍 수: "
        + ", ".join(f"A={a}: {pairs((a, mu0, tau)):,}" for a in GRID_A) + ".",
    ]
    hi = grid[grid["tau"] >= 0.5]
    ok = ((hi["max_out_deg"] <= 1) & (hi["max_in_deg"] <= 1)).all()
    n_zero = int((hi["merged_pairs"] == 0).sum())
    lines.append(
        f"5. tau >= 0.5인 {len(hi)}개 설정에서 provider 최대 차수와 receiver 최대 차수가 모두 1 이하인가: {'예' if ok else '아니오'}"
        f" (최대 차수 최댓값: out {hi['max_out_deg'].max()}, in {hi['max_in_deg'].max()}; 병합 쌍이 0개라 차수가 0인 설정 {n_zero}개)."
    )
    return lines


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    g = get_duckdb_delegate_graph_full(START, END)
    N = g["g_node_count"]

    df = pair_metrics(g)
    r, mu_node = provider_metrics(g)
    mu_pair = mu_node[df["P"].to_numpy()]
    mu_valid = r["mu"][~np.isnan(r["mu"])]
    otsu = float(otsu_threshold(mu_valid))
    n_mu, n_excl = int(mu_valid.size), int((mu_valid >= BASE[1]).sum())

    fun = funnel(df, mu_pair)
    grid, masks, comps = run_grid(df, mu_pair, otsu, N)

    figs = {
        "mu": FIG_DIR / "ees_1_2_mu_hist_scatter.png",
        "e": FIG_DIR / "ees_3_e_hist.png",
        "heat": FIG_DIR / "ees_4_heatmaps_A5.png",
        "comp": FIG_DIR / "ees_5_component_sizes.png",
    }
    plot_mu(r, otsu, figs["mu"])
    plot_e(df, figs["e"])
    plot_heatmaps(grid, otsu, figs["heat"])
    plot_comp_sizes(comps, figs["comp"])

    hi = grid[grid["tau"] >= 0.5][["A", "mu0", "tau", "merged_pairs", "max_out_deg", "max_in_deg"]]
    rel = lambda p: p.relative_to(OUT_DIR)
    lines = [
        "# RDH effect size 문턱 민감도 (ENERGY)", "",
        f"- 창: 블록 {START:,} ~ {END:,}, ENERGY delegate. 짝짓기: repo의 `pair_fifo`(R1, warm-up/horizon 403,200블록, 같은 블록 안은 client_time, tx_id 순).",
        "  `pair_fifo`는 기존 `check_undelegate_pairing.py`의 SQL 구현과 같은 결과를 내는 것을 `tests/test_pair_fifo.py`로 확인했다.",
        f"- 지표: s_in/s_out `effect_concentration`, active `effect_coverage`(bucket {BUCKET}블록), phi/m/mu `effect_menu_ratio`({MENU_PARAMS}).",
        f"- 실행: `uv run --group demo python -m rdh_data_adapter.analysis.{NAME}`. 코드와 SQL 전체는 [`{NAME}_raw.md`]({NAME}_raw.md).",
        "- assert 통과: provider별 s_out 합 = 1, receiver별 s_in 합 = 1, active <= n_pair, 쌍 수 = 3,905,329.",
        "- 컴포넌트 크기의 중앙값·99분위는 크기 2 이상인 컴포넌트에 대한 값이다.", "",
        "## 요약", "", *summary_lines(fun, grid, otsu, n_mu, n_excl), "",
        "## (a) 단계별 감소 (A=5, mu0=0.5, tau=0.5)", md(fun, pct=["share_of_all", "kept_from_prev"]), "",
        "## (e) mu", "",
        f"- mu0_otsu = {otsu:.4f}", f"- mu가 정의된 provider (m_P >= {MENU_PARAMS['min_n']}): {n_mu:,}",
        f"- mu0=0.5에서 mu로 제외된 provider (mu_P >= 0.5): {n_excl:,}", "",
        "## (b)(d) 설정별 표", md(grid.assign(mu0=grid["mu0"].astype(str)).set_index(["A", "mu0", "tau"]), pct=["jaccard_vs_base"]), "",
        "## (c) tau >= 0.5 최대 차수", md(hi.assign(mu0=hi["mu0"].astype(str)).set_index(["A", "mu0", "tau"])), "",
        "## 그림", "",
        f"1~2. mu_P 히스토그램(세로선 0.24, 0.90: 지시서에 주어진 시뮬레이션 기준선, 빨간선: otsu)과 mu_P 대 m_P", f"![mu]({rel(figs['mu'])})", "",
        f"3. active > 5인 쌍의 e 히스토그램", f"![e]({rel(figs['e'])})", "",
        f"4. A=5에서 (tau, mu0) 격자의 병합 쌍 수와 최대 컴포넌트 크기", f"![heat]({rel(figs['heat'])})", "",
        f"5. 컴포넌트 크기 분포 (A=5, mu0=0.5, tau=0.5 / 0.3 / 0.2)", f"![comp]({rel(figs['comp'])})", "",
    ]
    (OUT_DIR / f"{NAME}.md").write_text("\n".join(lines))

    modules = [inspect.getmodule(main), rdh_data_adapter.load.resource_graph, rdh_data_adapter.load.connection,
               rdh.pipeline.effect.pair_delegate_undelegate, rdh.graph.transform,
               rdh.models.effect.concentration, rdh.models.effect.coverage, rdh.models.effect.menu_ratio]
    c_horizon = (14 * 24 * 60 * 60) // 3
    raw = ["# RDH effect size 문턱 민감도 — 실행한 SQL과 코드", "",
           "## 실행한 SQL", "",
           f"`rdh_data_adapter/load/resource_graph.py`의 `get_duckdb_delegate_graph_full` 안의 쿼리 하나만 실행했다. 파라미터: [{START - c_horizon}, {END + c_horizon}], 연결은 read_only.",
           "```sql", inspect.getsource(get_duckdb_delegate_graph_full).split('sql = """')[1].split('"""')[0].strip(), "```", "",
           "## 코드", ""]
    for mod in modules:
        raw += [f"### {Path(mod.__file__).resolve().relative_to(Path.cwd())}", "```python", inspect.getsource(mod), "```", ""]
    (OUT_DIR / f"{NAME}_raw.md").write_text("\n".join(raw))
    print("\n".join(lines[:lines.index("## (b)(d) 설정별 표")]))


if __name__ == "__main__":
    main()
