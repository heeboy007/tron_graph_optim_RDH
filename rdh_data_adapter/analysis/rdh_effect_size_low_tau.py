"""
rdh_effect_size_sensitivity의 후속: 낮은 tau에서 병합 결과와 큰 컴포넌트의 구성 (ENERGY, 읽기 전용).

실행: uv run --group demo python -m rdh_data_adapter.analysis.rdh_effect_size_low_tau
산출물: rdh_data_adapter/analysis/reports/rdh_effect_size_low_tau.md, rdh_data_adapter/analysis/reports/figures/eelt_component_sizes.png

지표와 병합 규칙은 rdh_effect_size_sensitivity의 함수를 그대로 쓴다.
k_P = 창 안에서 P가 위임한 서로 다른 receiver 수 (병합 여부와 무관).
컴포넌트의 provider = 그 컴포넌트 안에서 병합 간선의 provider 쪽 끝에 있는 노드.
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

from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full
from rdh_data_adapter.analysis.rdh_effect_size_sensitivity import (
    START, END, OUT_DIR, FIG_DIR, pair_metrics, provider_metrics, merge_mask, md,
)

A = 5
GRID_MU0 = (0.61, "none")
# 처음 4개 + 0.1 ~ 0.001 로그 간격 8개 (유효숫자 3자리)
LOG_TAU = tuple(float(f"{t:.3g}") for t in np.logspace(-1, -3, 8))
GRID_TAU = tuple(sorted(set((0.05, 0.1, 0.15, 0.2) + LOG_TAU)))
TOP_COMPONENTS = 10
APPENDIX_PROVIDERS = 20  # 부록에는 컴포넌트당 n_P 상위 provider만
NAME = "rdh_effect_size_low_tau"


def components(df, mask, N):
    P, r = df["P"].to_numpy()[mask], df["r"].to_numpy()[mask]
    adj = csr_matrix((np.ones(P.size), (P, r)), shape=(N, N))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    return P, r, labels, sizes


def config_summary(P, r, sizes):
    comps = sizes[sizes >= 2]
    return {
        "merged_pairs": int(P.size),
        "merged_nodes": int(np.unique(np.concatenate((P, r))).size),
        "components": int(comps.size),
        "max_comp": int(comps.max()) if comps.size else 0,
        "median_comp": float(np.median(comps)) if comps.size else np.nan,
        "p99_comp": float(np.quantile(comps, 0.99)) if comps.size else np.nan,
        "max_out_deg": int(np.bincount(P).max()) if P.size else 0,
        "max_in_deg": int(np.bincount(r).max()) if r.size else 0,
    }, comps


def top_component_table(P, r, labels, sizes, node_info, decompress_map):
    # 크기 내림차순, 동률이면 라벨 번호 순
    big = np.argsort(-sizes, kind="stable")[:TOP_COMPONENTS]
    big = big[sizes[big] >= 2]
    edge_label = labels[P]
    rows = []
    for rank, lab in enumerate(big, start=1):
        in_comp = edge_label == lab
        providers = np.unique(P[in_comp])
        receivers = np.unique(r[in_comp])
        both = np.intersect1d(providers, receivers).size
        for p in providers:
            info = node_info.loc[p]
            rows.append({
                "rank": rank, "nodes": int(sizes[lab]), "edges": int(in_comp.sum()),
                "providers": providers.size, "receivers": receivers.size, "both": both,
                "provider_address_id": int(decompress_map[p]),
                "out_deg": int((P[in_comp] == p).sum()),
                "k_P": int(info["k_P"]), "n_P": int(info["n_P"]),
                "phi_P": info["phi_P"], "mu_P": info["mu_P"],
            })
    return pd.DataFrame(rows)


def component_overview(t: pd.DataFrame) -> pd.DataFrame:
    # 컴포넌트당 한 줄: 크기와 그 안 provider들의 k_P, phi_P, mu_P 분포
    g = t.groupby(["rank", "nodes", "edges", "providers", "receivers", "both"])
    return pd.DataFrame({
        "k_P min": g["k_P"].min(), "k_P median": g["k_P"].median(), "k_P max": g["k_P"].max(),
        "phi_P median": g["phi_P"].median(), "phi_P max": g["phi_P"].max(),
        "mu_P median": g["mu_P"].median(), "mu_P max": g["mu_P"].max(),
        "mu_P NaN": g["mu_P"].apply(lambda v: int(v.isna().sum())),
    }).reset_index().set_index("rank")


def plot_comp_sizes(comp_sizes, path):
    fig, axs = plt.subplots(1, len(GRID_MU0), figsize=(14, 5), sharey=True)
    for ax, mu0 in zip(axs, GRID_MU0):
        colors = plt.cm.viridis(np.linspace(0, 1, len(GRID_TAU)))
        for tau, color in zip(GRID_TAU, colors):
            c = comp_sizes[(mu0, tau)]
            if c.size:
                s, n = np.unique(c, return_counts=True)
                ax.plot(s, n, "o-", ms=3, color=color, label=f"tau={tau}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("component size (nodes)"); ax.set_ylabel("components")
        ax.set_title(f"A={A}, mu0={mu0} (size >= 2)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    g = get_duckdb_delegate_graph_full(START, END)
    N = g["g_node_count"]
    df = pair_metrics(g)
    menu, mu_node = provider_metrics(g)
    mu_pair = mu_node[df["P"].to_numpy()]

    phi_node = np.full(N, np.nan)
    phi_node[menu["id"]] = menu["phi"]
    node_info = pd.DataFrame({
        "k_P": np.bincount(g["uk_src"], minlength=N),
        "n_P": np.bincount(g["uk_src"], weights=g["uk_cnt"], minlength=N),
        "phi_P": phi_node, "mu_P": mu_node,
    })

    rows, comp_sizes, tops = [], {}, {}
    for mu0, tau in itertools.product(GRID_MU0, GRID_TAU):
        mask = merge_mask(df, mu_pair, A, mu0, tau, mu0_otsu=None)
        P, r, labels, sizes = components(df, mask, N)
        summ, comps = config_summary(P, r, sizes)
        rows.append({"mu0": str(mu0), "tau": tau, **summ})
        comp_sizes[(mu0, tau)] = comps
        tops[(mu0, tau)] = top_component_table(P, r, labels, sizes, node_info, g["decompress_map"])
    grid = pd.DataFrame(rows).set_index(["mu0", "tau"])

    size_dist = pd.DataFrame({
        f"mu0={mu0}, tau={tau}": pd.Series(*np.unique(c, return_counts=True)[::-1]) if c.size else pd.Series(dtype=int)
        for (mu0, tau), c in comp_sizes.items()
    }).fillna(0).astype(int)
    size_dist.index.name = "size"

    fig = FIG_DIR / "eelt_component_sizes.png"
    plot_comp_sizes(comp_sizes, fig)

    lines = [
        "# RDH effect size: 낮은 tau 재계산 (ENERGY)", "",
        f"- `rdh_effect_size_sensitivity.md`의 후속. 창 {START:,} ~ {END:,}, A = {A}, mu0 ∈ {{0.61, none}}, tau ∈ {{{', '.join(map(str, GRID_TAU))}}}.",
        f"  tau 중 {', '.join(map(str, LOG_TAU))}는 0.1 ~ 0.001을 로그 간격으로 8등분한 값(유효숫자 3자리)이다.",
        "- 병합 조건: active > A, (mu_P < mu0 또는 mu_P가 NaN), e > tau. 지표 계산과 병합 규칙은 `rdh_effect_size_sensitivity.py`의 함수를 그대로 썼다.",
        "- k_P는 창 안에서 P가 위임한 서로 다른 receiver 수, n_P는 P의 위임 수다. 둘 다 병합 여부와 무관하다.",
        "- 컴포넌트의 provider는 그 컴포넌트 안 병합 간선의 provider 쪽 노드이고, `provider_address_id`는 `resource_tx.from_addr` 값이다. out_deg는 그 컴포넌트 안에서의 병합 간선 수다.",
        f"- 코드: [`rdh_data_adapter/analysis/{NAME}.py`](../{NAME}.py). 실행: `uv run --group demo python -m rdh_data_adapter.analysis.{NAME}`.", "",
        "## 설정별 요약", md(grid), "",
        "## 컴포넌트 크기 분포 (크기 2 이상, 개수)", md(size_dist), "",
        f"![sizes](figures/{fig.name})", "",
        f"## 가장 큰 컴포넌트 {TOP_COMPONENTS}개의 구성", "",
    ]
    lines += ["컴포넌트당 한 줄. providers/receivers는 병합 간선의 provider 쪽/receiver 쪽 노드 수, both는 양쪽에 모두 있는 노드 수다.",
              "k_P, phi_P, mu_P 열은 그 컴포넌트 안 provider들에 대한 값이다 (mu_P 통계는 NaN 제외). provider별 목록은 부록에 있다.", ""]
    for (mu0, tau), t in tops.items():
        lines += [f"### mu0={mu0}, tau={tau}", md(component_overview(t), floatfmt=".3f"), ""]
    lines += ["## 부록: 가장 큰 컴포넌트의 provider별 목록", "",
              f"컴포넌트당 n_P가 큰 순서로 최대 {APPENDIX_PROVIDERS}명까지만 적는다. 전체 provider 수는 위 표의 providers 열에 있다.", ""]
    for (mu0, tau), t in tops.items():
        cols = ["rank", "nodes", "provider_address_id", "out_deg", "k_P", "n_P", "phi_P", "mu_P"]
        shown = t.sort_values(["rank", "n_P"], ascending=[True, False]).groupby("rank").head(APPENDIX_PROVIDERS)
        lines += [f"### mu0={mu0}, tau={tau}", md(shown[cols].set_index("rank"), floatfmt=".3f"), ""]
    lines += ["## 코드", "", f"### rdh_data_adapter/analysis/{NAME}.py", "```python", inspect.getsource(inspect.getmodule(main)), "```", "",
              "SQL과 지표 계산 코드는 `rdh_effect_size_sensitivity_raw.md`와 같다.", ""]
    (OUT_DIR / f"{NAME}.md").write_text("\n".join(lines))
    print(md(grid)); print(md(size_dist))


if __name__ == "__main__":
    main()
