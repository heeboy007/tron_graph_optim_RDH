"""Ranking models (manuscript §3.7): global model G on transfer-graph features T and residual boosting on RDH / sponsorship features.

G: HistGradientBoostingClassifier on T (+ same_cpm), trained on the global candidate pairs of window k with the hub-capped label.
   Its logit z_G is the base score; training pairs of window k get out-of-fold scores (5 folds by a pair hash), earlier windows the full model.
Residual model: LightGBM boosting started from z_G (init_score) on the setting's features, trained on the query pools of windows
   max(303, k − L + 1) … k; the number of trees is chosen by early stopping on the validation-query MRR (patience 50, at most 500 trees,
   zero trees allowed). Score = z_G + residual.
Placebo: the same residual model on D rows permuted between queries of the same window (derangement), repeated / truncated to each
   query's length.
Metrics per query: MRR, AP, hits@10 (rank by score descending, ties by tie_key ascending).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import S_FEATS, TF, query_metrics, tie_key

SEED = 20261026
W_LO = 303                                     # first window of every training pool
DRDH = [f"{c}_d" for c in TF] + ["lcs_min", "lcs_max", "in_any", "in_both"]
DS = list(S_FEATS)
FSETS = {"D_RDH": DRDH, "D_S": DS, "D_RDH+D_S": DRDH + DS}
TCOLS = TF + ["same_cpm"]
HGB = dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=300, early_stopping=False, random_state=0)
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=100, lambda_l2=5.0, feature_fraction=0.8, feature_fraction_seed=0,
           bagging_seed=0, seed=0, deterministic=True, verbose=-1, num_threads=6)
MAX_T, PATIENCE = 500, 50
METS = ["mrr", "ap", "hits10"]


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def pair_fold(a_id, b_id):
    lo, hi = np.minimum(a_id, b_id).astype(np.uint64), np.maximum(a_id, b_id).astype(np.uint64)
    h = (lo * np.uint64(0x9E3779B97F4A7C15) ^ hi * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(33)
    return (h % np.uint64(5)).astype(int)


def fit_g(dtr):
    """G on all training pairs and the 5 fold models (out-of-fold scores of window k)."""
    Xd, yd = dtr[TCOLS].to_numpy(float), dtr["y"].to_numpy()
    G = HistGradientBoostingClassifier(**HGB).fit(Xd, yd)
    fd = pair_fold(dtr["a_id"].to_numpy(np.int64), dtr["b_id"].to_numpy(np.int64))
    folds = [HistGradientBoostingClassifier(**HGB).fit(Xd[fd != f], yd[fd != f]) for f in range(5)]
    return G, folds


def z_pool(P, k, G, folds):
    X = P[TCOLS].to_numpy(float)
    if P["window"].iloc[0] < k:
        return logit(G.predict_proba(X)[:, 1])
    fq = pair_fold(P["a_id"].to_numpy(np.int64), P["b_id"].to_numpy(np.int64)); z = np.zeros(len(P))
    for f in range(5):
        z[fq == f] = logit(folds[f].predict_proba(X[fq == f])[:, 1])
    return z


def make_mrr(qv, yv, tb):
    """Validation-query MRR; ties broken by tb ascending (tb = tie_key of the pair)."""
    qcode = pd.factorize(qv)[0]

    def f(score):
        order = np.lexsort((tb, -score, qcode))
        qs, ys = qcode[order], yv[order]
        start = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
        rank = np.arange(len(qs)) - np.repeat(start, np.diff(np.r_[start, len(qs)])) + 1
        first = pd.Series(np.where(ys == 1, rank, np.inf)).groupby(qs).min().to_numpy()
        return float(np.mean(1.0 / first))
    return f


def train_residual(Xt, yt, zt, Xv, zv, mrr_v, params=None):
    ds = lgb.Dataset(Xt, label=yt, init_score=zt, free_raw_data=False)
    bst = lgb.Booster(params=LGB if params is None else params, train_set=ds)
    cum = np.zeros(len(zv)); best_t, best = 0, mrr_v(zv); curve = [best]
    for t in range(1, MAX_T + 1):
        bst.update()
        cum += bst.predict(Xv, raw_score=True, start_iteration=t - 1, num_iteration=1)
        m = mrr_v(zv + cum); curve.append(m)
        if m > best:
            best, best_t = m, t
        if t - best_t >= PATIENCE:
            break
    return bst, best_t, curve


def derangement(n, rng):
    """Random permutation without fixed points (n >= 2), by rejection with the given generator; n = 1 returns [0]."""
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def permute_D(df, cols, rng):
    """Placebo: within each window, query i gets the D rows of query perm[i] != i (derangement), repeated / truncated to its length.
    Returns (array, number of windows with a single query)."""
    out = df[cols].to_numpy(float).copy()
    single = 0
    for w, g in df.groupby("window", sort=False):
        qs = g["q"].unique(); perm = derangement(len(qs), rng)
        single += int(len(qs) < 2)
        idx_by_q = {q: g.index[g["q"] == q].to_numpy() for q in qs}
        for i, q in enumerate(qs):
            dst, src = idx_by_q[q], idx_by_q[qs[perm[i]]]
            out[df.index.get_indexer(dst)] = df.loc[np.resize(src, dst.size), cols].to_numpy(float)
    return out, single


def qmet(Q, score):
    m = query_metrics(Q, score)
    return {x: np.asarray(v).tolist() for x, v in m.items() if x != "q"}


def fit_eval(P, Qe, ze, De_perm, k, s, Lw, settings, lgb_params=None, scores=None):
    """Residual and placebo models for each setting (name, feature columns) on the pool of windows max(W_LO, k − Lw + 1) … k.
    Returns per-setting metrics, trees and pool info. scores: dict receiving the evaluation scores per (name, kind)."""
    P = pd.concat([P[j] for j in sorted(P) if j >= max(W_LO, k - Lw + 1)], ignore_index=True)
    P["qkey"] = P["window"].astype(str) + "_" + P["q"].astype(str)
    srng = np.random.default_rng([SEED, 8000 + s, k, Lw])
    val_keys = set()
    for j, g in P.groupby("window"):
        qk = np.array(sorted(g["qkey"].unique())); val_keys |= set(qk[srng.permutation(qk.size)[: int(round(0.2 * qk.size))]].tolist())
    isv = P["qkey"].isin(val_keys).to_numpy()
    tb = tie_key(P.loc[isv, "a_id"].to_numpy(), P.loc[isv, "b_id"].to_numpy())
    mrr_v = make_mrr(P.loc[isv, "qkey"].to_numpy(), P.loc[isv, "y"].to_numpy(), tb)
    Dperm, single = permute_D(P.assign(q=P["qkey"]), DRDH + DS, np.random.default_rng([SEED, 9000 + s, k, Lw]))
    info = {"windows": int(P["window"].nunique()), "queries": int(P["qkey"].nunique()), "pairs": int(len(P)), "positives": int(P["y"].sum()),
            "val_queries": len(val_keys), "placebo_single_query_windows": single}
    res, params = {}, {}
    for name, cols in settings:
        ci = [(DRDH + DS).index(c) for c in cols]
        for kind in ("setting", "placebo"):
            Xall = P[cols].to_numpy(float) if kind == "setting" else Dperm[:, ci]
            Xe = Qe[cols].to_numpy(float) if kind == "setting" else De_perm[:, ci]
            bst, nt, curve = train_residual(Xall[~isv], P["y"].to_numpy()[~isv], P["z"].to_numpy()[~isv], Xall[isv], P["z"].to_numpy()[isv], mrr_v, lgb_params)
            sc = ze + (bst.predict(Xe, raw_score=True, num_iteration=nt) if nt > 0 else 0.0)
            if scores is not None:
                scores[(name, kind)] = np.asarray(sc, float)
            res[(name, kind)] = qmet(Qe, sc)
            params[(name, kind)] = {"trees": nt, "val_mrr_best": float(max(curve)), "val_mrr_t0": float(curve[0])}
    return res, params, info
