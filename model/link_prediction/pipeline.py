"""Link-prediction repetitions (manuscript §3.7, §6.1): query caches, hub-capped labels, G and the residual models.

Repetition k (seed s): G is trained on the global candidate pairs of window k (label = new common recipient in window k + 1 whose in-degree
is <= H; pairs inside one F3 component removed); residual models are trained on the query pools of windows <= k and evaluated on the
queries of window k + 2.
Inputs per window w (all under one Store): the pair table (rdh.pair_table) for F3, the sponsorship tables, the CPM labels; F2 sets {w: set}.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rdh.rdh import component_rep, f3_edges
from .features import TF, WG, build_queries, build_window, cpm_same, s_features, same_comp, t_features
from .graph import WSZ
from .ranking import DRDH, DS, FSETS, LGB, MAX_T, PATIENCE, SEED, TCOLS, W_LO, fit_eval, fit_g, logit, permute_D, qmet, z_pool

CONF_S = 1


def log(m):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {m}", flush=True)


@dataclass
class Store:
    """Folder layout of the intermediate files (default root: $RDH_DUMP_DIR/v1)."""
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    def pairs(self, w):
        return self.root / "windows" / f"w{w:04d}" / "pairs.parquet"

    def sponsor(self, w):
        return self.root / "sponsorship" / f"w{w}"

    @property
    def cpm(self):
        return self.root / "cpm"

    @property
    def cpm_labels(self):
        return self.root / "cpm" / "labels"

    @property
    def f2(self):
        return self.root / "f2"

    def cache(self, H=None):
        return self.root / "lp" / ("cache" if H is None else f"cache_H{H}")

    def result(self, name):
        return self.root / "lp" / "results" / name

    @property
    def tmp(self):
        return self.root / "duckdb_tmp"


def f3_rep(store: Store, w, f2_w):
    """(address, representative) of the F3 components of window w."""
    pairs = pd.read_parquet(store.pairs(w), columns=["P_addr", "r_addr", "s_in", "s_out", "active", "mu_P", "phi_P"])
    return component_rep(f3_edges(pairs, f2_w))


# ---------------- query caches ----------------
def window_cache(store: Store, w, con, rep, roles):
    """roles: list of (role, seed). Writes <cache>/w<w>_<role>_s<seed>.parquet with T, D_RDH, D_S."""
    cache = store.cache()
    jobs = [(r, s) for r, s in roles if not (cache / f"w{w}_{r}_s{s}.parquet").exists()]
    if not jobs:
        return
    t0 = time.time()
    cache.mkdir(parents=True, exist_ok=True)
    sdir = store.sponsor(w)
    G = WG(con, w)
    Qs = {}
    for role, s in jobs:
        key = [SEED, 5000 + s, w] if role == "train" else [SEED, 2000 + s, w - 2]
        Qs[(role, s)] = build_queries(G, rep, w, np.random.default_rng(key), sdir)
    allp = pd.concat([q[["a_id", "b_id"]] for q, _ in Qs.values()]).drop_duplicates().reset_index(drop=True)
    a, b = allp["a_id"].to_numpy(np.int64), allp["b_id"].to_numpy(np.int64)
    T = t_features(G.E, None, a, b, cpm_same(store.cpm_labels, w, a, b))
    Tc = t_features(G.E, rep, a, b, T["same_cpm"].to_numpy())
    sz = pd.Series(rep[:, 1]).value_counts(); rs = pd.Series(rep[:, 1], index=rep[:, 0])
    ina, inb = pd.Series(a).isin(rs.index).to_numpy(), pd.Series(b).isin(rs.index).to_numpy()
    ca = np.log1p(pd.Series(a).map(rs).map(sz).fillna(1).to_numpy(float)); cb = np.log1p(pd.Series(b).map(rs).map(sz).fillna(1).to_numpy(float))
    D = pd.DataFrame({f"{c}_d": Tc[c].to_numpy(float) - T[c].to_numpy(float) for c in TF})
    D["lcs_min"], D["lcs_max"] = np.minimum(ca, cb), np.maximum(ca, cb)
    D["in_any"], D["in_both"] = (ina | inb).astype(int), (ina & inb).astype(int)
    D.loc[~(ina | inb), DRDH] = 0.0
    S = s_features(w, G.E, a, b, sdir)[DS]
    F = pd.concat([allp, T[TCOLS].reset_index(drop=True), D, S.reset_index(drop=True)], axis=1)
    del G
    for (role, s), (Q, info) in Qs.items():
        out = Q[["q", "a_id", "b_id", "y", "bin", "src_2hop", "src_spon"]].merge(F, on=["a_id", "b_id"], how="left")
        assert len(out) == len(Q)
        out.to_parquet(cache / f"w{w}_{role}_s{s}.parquet", index=False)
        (cache / f"w{w}_{role}_s{s}.json").write_text(json.dumps(info))
    log(f"[cache w={w}] {len(jobs)} sets, {len(allp):,} pairs ({time.time()-t0:.0f}s)")


def hub_flags(a, b, klab, con, H):
    """For positive pairs (a, b): True if every common recipient in label window klab has in-degree > H (in-degree = distinct senders of
    successful transfers, from != to). Returns a boolean array (all False when H is None)."""
    if H is None or len(a) == 0:
        return np.zeros(len(a), bool)
    lo, hi = klab * WSZ, (klab + 1) * WSZ - 1
    con.register("ps", pd.DataFrame({"id": np.union1d(a, b)}))
    Sd = con.execute(f"""SELECT DISTINCT from_addr AS f, to_addr AS t FROM usdt_tx JOIN ps ON from_addr = ps.id
WHERE block_number BETWEEN {lo} AND {hi} AND success AND type = 'transfer' AND from_addr <> to_addr""").df()
    con.register("rs", pd.DataFrame({"id": Sd["t"].unique()}))
    indeg = con.execute(f"""SELECT to_addr AS t, count(DISTINCT from_addr) AS n FROM usdt_tx JOIN rs ON to_addr = rs.id
WHERE block_number BETWEEN {lo} AND {hi} AND success AND type = 'transfer' AND from_addr <> to_addr GROUP BY to_addr""").df().set_index("t")["n"]
    con.unregister("ps"); con.unregister("rs")
    outs = Sd.groupby("f")["t"].apply(lambda s: np.sort(s.to_numpy()))
    common = [np.intersect1d(outs.get(x, np.empty(0, np.int64)), outs.get(y, np.empty(0, np.int64)), assume_unique=True) for x, y in zip(a, b)]
    assert all(c.size > 0 for c in common), "positive pair without a common recipient"
    return np.array([bool((indeg.reindex(c).to_numpy() > H).all()) for c in common])


def hcache_file(store: Store, name, con, H):
    """Query cache with the hub-capped label: hub-only positives become negatives; queries left without a positive are dropped."""
    src, dst = store.cache(), store.cache(H)
    if (dst / f"{name}.parquet").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    w = int(name[1:].split("_")[0])
    Q = pd.read_parquet(src / f"{name}.parquet")
    pos = Q.index[Q["y"] == 1]
    flag = np.zeros(len(Q), bool)
    flag[np.asarray(pos)] = hub_flags(Q.loc[pos, "a_id"].to_numpy(np.int64), Q.loc[pos, "b_id"].to_numpy(np.int64), w + 1, con, H)
    yH = (Q["y"].to_numpy() == 1) & ~flag
    qpos = pd.Series(yH).groupby(Q["q"].to_numpy()).sum()
    keep = Q["q"].isin(qpos.index[qpos > 0]).to_numpy()
    QH = Q.assign(y=yH.astype(Q["y"].dtype))[keep].reset_index(drop=True)
    QH.to_parquet(dst / f"{name}.parquet", index=False)
    info = json.loads((src / f"{name}.json").read_text())
    info.update(queries_used_H_none=int(Q["q"].nunique()), queries_used=int(QH["q"].nunique()), positives_H_none=int(Q["y"].sum()),
                positives=int(QH["y"].sum()), hub_only_positives=int(flag.sum()), H=H)
    (dst / f"{name}.json").write_text(json.dumps(info))


def plan(ks, seeds=(CONF_S,), lmax=8):
    """Query sets needed by repetitions ks: {window: [(role, seed)]} (training pools of L <= lmax windows, evaluation on k + 2)."""
    ws = range(min(ks) - lmax + 1, max(ks) + 3)
    return {w: [(r, s) for s in seeds for r in ("train", "eval") if (r == "train" and w <= max(ks)) or (r == "eval" and (w - 2) in set(ks))] for w in ws}


# ---------------- one repetition ----------------
def g_training(store: Store, k, con, rng, rep_k, H):
    """Global candidate pairs of window k for G (pairs inside one F3 component removed) with the hub-capped label."""
    dtr = build_window(k, con, rng, store.sponsor(k), store.cpm_labels)
    dtr = dtr[~same_comp(rep_k, dtr["a_id"].to_numpy(), dtr["b_id"].to_numpy())].reset_index(drop=True)
    pos = np.flatnonzero(dtr["y"].to_numpy() == 1)
    hub = hub_flags(dtr["a_id"].to_numpy(np.int64)[pos], dtr["b_id"].to_numpy(np.int64)[pos], k + 1, con, H)
    y = dtr["y"].to_numpy().copy(); y[pos[hub]] = 0
    info = {"pairs": int(len(dtr)), "positives_H_none": int(len(pos)), "hub_only_positives": int(hub.sum()), "positives": int(y.sum())}
    return dtr.assign(y=y), info


def pools(store: Store, k, s, Lmax, H, G, folds):
    P = {}
    for j in range(max(W_LO, k - Lmax + 1), k + 1):
        x = pd.read_parquet(store.cache(H) / f"w{j}_train_s{s}.parquet").assign(window=j)
        x["z"] = z_pool(x, k, G, folds)
        P[j] = x
    return P


def run_rep(store: Store, k, s, con, f2, H, Ls, settings, cand_out=None):
    """One repetition: G on window k (hub-capped label), residual models for each L and feature set; evaluation on window k + 2.
    f2: {window: set of F2 providers}. Returns the result dict (metrics per query for G, each setting and its placebo)."""
    t0 = time.time()
    rep_k = f3_rep(store, k, f2[k])
    dtr, ginfo = g_training(store, k, con, np.random.default_rng([SEED, 1000 + s]), rep_k, H)
    G, folds = fit_g(dtr)
    Qe = pd.read_parquet(store.cache(H) / f"w{k + 2}_eval_s{s}.parquet")
    ze = logit(G.predict_proba(Qe[TCOLS].to_numpy(float))[:, 1])
    Qe = Qe.assign(window=k + 2)
    De_perm, _ = permute_D(Qe, DRDH + DS, np.random.default_rng([SEED, 9500 + s, k]))
    P = pools(store, k, s, max(Ls), H, G, folds)
    out = {"k": k, "seed": s, "H": H, "eval_queries": int(Qe["q"].nunique()), "g_training": ginfo, "models": {"G": qmet(Qe, ze)}, "params": {}, "pools": {}}
    for Lw in Ls:
        scores = {} if cand_out is not None else None
        res, params, info = fit_eval(P, Qe, ze, De_perm, k, s, Lw, settings, None, scores)
        out["pools"][str(Lw)] = info
        for (name, kind), v in res.items():
            out["models"][f"{name}|L{Lw}|{kind}"] = v; out["params"][f"{name}|L{Lw}|{kind}"] = params[(name, kind)]
        if cand_out is not None:
            C_ = Qe[["q", "a_id", "b_id", "y", "bin", "src_2hop", "src_spon"]].assign(G=ze)
            for (name, kind), v in scores.items():
                C_[f"{name}|L{Lw}|{kind}"] = v
            Path(cand_out).parent.mkdir(parents=True, exist_ok=True)
            C_.to_parquet(cand_out, index=False)
    out["t_s"] = time.time() - t0
    return out


def check_config(cfg):
    """A settings file (frozen_config.json format) must use this module's LightGBM parameters, tree limit and patience."""
    assert cfg["lightgbm_params"] == LGB and cfg["max_trees"] == MAX_T and cfg["early_stopping"]["patience"] == PATIENCE
    assert cfg["features"] == FSETS[cfg["feature_set"]]


def confirm(store: Store, ks, con, f2, H, cfg, out_name):
    """Repetitions ks with one settings file; writes <result>/k<k>.json and cand_scores/k<k>.parquet (existing results are kept)."""
    check_config(cfg)
    res = store.result(out_name); res.mkdir(parents=True, exist_ok=True)
    for k in ks:
        f = res / f"k{k}.json"
        if f.exists():
            continue
        r = run_rep(store, k, CONF_S, con, f2, H, [cfg["L"]], [(cfg["feature_set"], cfg["features"])], cand_out=res / "cand_scores" / f"k{k}.parquet")
        Lw, fs = cfg["L"], cfg["feature_set"]
        r["models"]["setting"] = r["models"].pop(f"{fs}|L{Lw}|setting"); r["models"]["placebo"] = r["models"].pop(f"{fs}|L{Lw}|placebo")
        r["trees"] = {"setting": r["params"][f"{fs}|L{Lw}|setting"]["trees"], "placebo": r["params"][f"{fs}|L{Lw}|placebo"]["trees"]}
        r["pool"] = r["pools"][str(Lw)]
        f.write_text(json.dumps(r)); log(f"[{out_name} k={k} H={H}] {r['t_s']:.0f}s queries {r['eval_queries']} trees {r['trees']}")
