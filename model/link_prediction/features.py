"""Candidate pairs, transfer-graph features T, sponsorship features S and ranking queries (manuscript §3.7).

Window k = blocks [k·200,000, (k+1)·200,000). Sponsorship tables: model.ext_features.sponsorship.build, one folder per window (``sdir``).
CPM labels: model.ext_features.cpm, one parquet per chunk (``cpm_dir``/leiden_cpm_week_<k·200,000>.parquet).

build_window (global candidate pairs of window k, the training data of G): union of three pools, each N = 500,000 draws —
  (a) 2-hop sender pairs through a connector of undirected degree ≤ 1,000 (connector drawn ∝ its number of sender pairs),
  (b) receivers sharing a sponsor P with 2 ≤ d_P ≤ 1,000, (c) random sender pairs; pairs with a common recipient in window k are dropped.
  Label y = a new common recipient in window k + 1 (hub cap applied by the caller, see model.link_prediction.pipeline.g_training).
build_queries: query = sender in window w belonging to an F3 component of size ≥ 2; candidates = ≤ 200 2-hop senders (connector degree
  ≤ 1,000), ≤ 200 senders sharing a sponsor (2 ≤ d_P ≤ 1,000), 50 random senders; pairs with a common recipient in w or in the same F3
  component are dropped; label = new common recipient in w + 1; queries with ≥ 1 positive and ≥ 20 candidates are kept.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import WSZ, csr, load_edges, pair_has, pair_inter, pair_sponsor

H_CONN, N_POOL = 1000, 500_000
DMIN, DMAX = 2, 1000
PTYPES = ["flash-type", "menu-type", "other"]
TF = ["log_out_min", "log_out_max", "log_in_min", "log_in_max", "cn", "aa_t", "jaccard", "direct"]
T_FEATS = TF + ["same_cpm"]
S_FEATS = ["n_shared", "aa_s", "log_min_dP", "top_flash", "top_menu", "top_other", "top_top1_share", "top_log_nrecip",
           "spon_ratio_min", "spon_ratio_max", "no_shared"]
Q_2HOP, Q_SPON, Q_RAND, Q_MINC, HUBDEG = 200, 200, 50, 20, 1000
BINS = [("B1", 0, 10), ("B2", 11, 100), ("B3", 101, 1000), ("B4", 1001, 10 ** 12)]


def read_sponsored(sdir, k, cols):
    lo, hi = k * WSZ, (k + 1) * WSZ - 1
    SP = pd.read_parquet(Path(sdir) / "sponsored_nonambiguous.parquet", columns=cols)
    return SP[(SP["b"] >= lo) & (SP["b"] <= hi)]


def cpm_labels(cpm_dir, k):
    p = Path(cpm_dir) / f"leiden_cpm_week_{k * WSZ}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def build_window(k, con, rng, sdir, cpm_dir):
    Ek, Ek1 = load_edges(con, k), load_edges(con, k + 1)
    ids = np.unique(np.concatenate([Ek["f"], Ek["t"], Ek1["f"], Ek1["t"]]).astype(np.int64))
    n = ids.size
    loc = lambda x: np.searchsorted(ids, np.asarray(x, np.int64))
    fk, tk = loc(Ek["f"]), loc(Ek["t"]); f1, t1 = loc(Ek1["f"]), loc(Ek1["t"])
    tcount = np.bincount(fk, minlength=n)
    del Ek, Ek1
    m = fk != tk
    o_ptr, o_idx = csr(fk[m], tk[m], n)
    outdeg = np.diff(o_ptr); indeg = np.bincount(o_idx, minlength=n)
    u_ptr, u_idx = csr(np.concatenate([fk[m], tk[m]]), np.concatenate([tk[m], fk[m]]), n)
    du = np.diff(u_ptr)
    m1 = f1 != t1
    o1_ptr, o1_idx = csr(f1[m1], t1[m1], n)
    del fk, tk, f1, t1, m, m1
    sender = outdeg > 0
    senders = np.nonzero(sender)[0]
    cpm = np.full(n, -1, np.int64)
    c = cpm_labels(cpm_dir, k)
    if c is not None:
        pos = np.searchsorted(ids, c["address_id"].to_numpy(np.int64))
        ok = (pos < n) & (ids[np.minimum(pos, n - 1)] == c["address_id"].to_numpy(np.int64))
        cpm[pos[ok]] = pd.factorize(c["cluster_id"])[0][ok]
    SP = read_sponsored(sdir, k, ["P", "r", "t", "b", "ptype"])
    anyc = pd.read_parquet(Path(sdir) / "sponsored_any_by_sender.parquet")
    spon_any = np.zeros(n); pa = loc(anyc["sender"]); okp = (pa < n) & (ids[np.minimum(pa, n - 1)] == anyc["sender"].to_numpy(np.int64))
    spon_any[pa[okp]] = anyc["sponsored_transfers"].to_numpy()[okp]
    spon_ratio = np.where(tcount > 0, np.minimum(spon_any / np.maximum(tcount, 1), 1.0), 0.0)
    pcode, puniq = pd.factorize(SP["P"], sort=True)      # sponsor code = ascending provider id (ties of smallest d_P → smaller id)
    SP = SP.assign(pc=pcode, ai=loc(SP["r"]))
    npc = len(puniq)
    dP = SP.groupby("pc")["ai"].nunique().reindex(range(npc)).to_numpy(float)
    wP = np.where((dP >= DMIN) & (dP <= DMAX), 1.0 / np.log(np.maximum(dP, 2)), 0.0)
    rc = SP.groupby(["pc", "t"]).size()
    top1 = (rc.groupby(level=0).max() / rc.groupby(level=0).sum()).reindex(range(npc)).to_numpy()
    nrec = rc.groupby(level=0).size().reindex(range(npc)).to_numpy(float)
    ptyp = SP.groupby("pc")["ptype"].first().reindex(range(npc)).to_numpy()
    sp_ptr, sp_idx = csr(SP["ai"].to_numpy(), SP["pc"].to_numpy(), n)
    # (a) 2-hop
    keep = sender[u_idx]
    s_cnt = np.bincount(np.repeat(np.arange(n), du)[keep], minlength=n).astype(np.int64)
    s_ptr = np.r_[0, np.cumsum(s_cnt)]; s_idx = u_idx[keep]
    wpair = s_cnt.astype(float) * (s_cnt - 1) / 2
    ok_c = (du <= H_CONN) & (s_cnt >= 2)
    pr = np.where(ok_c, wpair, 0.0); pr = pr / pr.sum()
    rows = rng.choice(n, size=N_POOL, p=pr)
    sc = s_cnt[rows]; i1 = (rng.random(N_POOL) * sc).astype(np.int64); i2 = (rng.random(N_POOL) * (sc - 1)).astype(np.int64); i2 += i2 >= i1
    pa_ = s_idx[s_ptr[rows] + i1]; pb_ = s_idx[s_ptr[rows] + i2]
    key_a = np.unique(np.minimum(pa_, pb_) * n + np.maximum(pa_, pb_))
    # (b) co-sponsored
    psd = SP.drop_duplicates(["pc", "ai"])
    psd = psd[(dP[psd["pc"]] >= DMIN) & (dP[psd["pc"]] <= DMAX)]
    kb = []
    for _, g in psd.groupby("pc"):
        s = np.sort(g["ai"].to_numpy(np.int64)); iu, ju = np.triu_indices(s.size, 1)
        kb.append(s[iu] * n + s[ju])
    kb = np.unique(np.concatenate(kb)) if kb else np.empty(0, np.int64)
    key_b = rng.choice(kb, size=min(N_POOL, kb.size), replace=False) if kb.size else kb
    # (c) random
    ra, rb = rng.choice(senders, N_POOL), rng.choice(senders, N_POOL)
    okr = ra != rb
    key_c = np.unique(np.minimum(ra[okr], rb[okr]) * n + np.maximum(ra[okr], rb[okr]))
    keys = np.unique(np.concatenate([key_a, key_b, key_c]))
    df = pd.DataFrame({"a": keys // n, "b": keys % n})
    df["pool_a"], df["pool_b"], df["pool_c"] = np.isin(keys, key_a), np.isin(keys, key_b), np.isin(keys, key_c)
    a, b = df["a"].to_numpy(np.int64), df["b"].to_numpy(np.int64)
    cr_k, _ = pair_inter(o_ptr, o_idx, a, b, np.zeros(n))
    df = df[cr_k == 0].reset_index(drop=True)
    a, b = df["a"].to_numpy(np.int64), df["b"].to_numpy(np.int64)
    wu = np.where(du >= 2, 1.0 / np.log(np.maximum(du, 2)), 0.0)
    cn, aa_t = pair_inter(u_ptr, u_idx, a, b, wu)
    df["cn"] = cn; df["aa_t"] = aa_t
    df["jaccard"] = cn / np.maximum(du[a] + du[b] - cn, 1)
    df["direct"] = pair_has(u_ptr, u_idx, a, b).astype(int)
    lo_o, lo_i = np.log1p(outdeg), np.log1p(indeg)
    df["log_out_min"], df["log_out_max"] = np.minimum(lo_o[a], lo_o[b]), np.maximum(lo_o[a], lo_o[b])
    df["log_in_min"], df["log_in_max"] = np.minimum(lo_i[a], lo_i[b]), np.maximum(lo_i[a], lo_i[b])
    df["same_cpm"] = ((cpm[a] == cpm[b]) & (cpm[a] >= 0)).astype(int)
    ns, aa_s, mind, top = pair_sponsor(sp_ptr, sp_idx, a, b, wP, dP)
    has = top >= 0
    df["n_shared"] = ns; df["aa_s"] = aa_s; df["log_min_dP"] = np.where(has, np.log(np.maximum(mind, 1)), 0.0)
    tt = np.where(has, ptyp[np.maximum(top, 0)] if npc else "none", "none")
    for t_, col in zip(PTYPES, ["top_flash", "top_menu", "top_other"]):
        df[col] = (tt == t_).astype(int)
    df["top_top1_share"] = np.where(has, top1[np.maximum(top, 0)] if npc else 0.0, 0.0)
    df["top_log_nrecip"] = np.where(has, np.log1p(nrec[np.maximum(top, 0)]) if npc else 0.0, 0.0)
    df["spon_ratio_min"], df["spon_ratio_max"] = np.minimum(spon_ratio[a], spon_ratio[b]), np.maximum(spon_ratio[a], spon_ratio[b])
    df["no_shared"] = (~has).astype(int)
    cr1, _ = pair_inter(o1_ptr, o1_idx, a, b, np.zeros(n))
    df["y"] = (cr1 > 0).astype(int)
    df["a_id"], df["b_id"] = ids[a], ids[b]
    return df.drop(columns=["a", "b"])


class WG:
    """Transfer graph of window w (directed, undirected) and directed graph of label window w + 1; nodes = union of both windows."""

    def __init__(self, con, w):
        self.w = w
        self.E = load_edges(con, w)
        E1 = load_edges(con, w + 1)
        self.ids = np.unique(np.concatenate([self.E["f"], self.E["t"], E1["f"], E1["t"]]).astype(np.int64))
        n = self.n = self.ids.size
        f, t = self.loc(self.E["f"]), self.loc(self.E["t"]); m = f != t
        self.o_ptr, self.o_idx = csr(f[m], t[m], n)
        self.outdeg = np.diff(self.o_ptr)
        self.u_ptr, self.u_idx = csr(np.r_[f[m], t[m]], np.r_[t[m], f[m]], n)
        self.du = np.diff(self.u_ptr)
        f1, t1 = self.loc(E1["f"]), self.loc(E1["t"]); m1 = f1 != t1
        self.o1_ptr, self.o1_idx = csr(f1[m1], t1[m1], n)
        self.indeg1 = np.bincount(self.o1_idx, minlength=n)
        self.senders = np.nonzero(self.outdeg > 0)[0]

    def loc(self, x):
        return np.searchsorted(self.ids, np.asarray(x, np.int64))

    def has(self, x):
        x = np.asarray(x, np.int64); p = np.minimum(np.searchsorted(self.ids, x), self.n - 1)
        return self.ids[p] == x

    def common_now(self, a, b):
        return pair_inter(self.o_ptr, self.o_idx, a, b, np.zeros(self.n))[0]

    def label(self, a, b):
        return pair_inter(self.o1_ptr, self.o1_idx, a, b, np.zeros(self.n))[0] > 0

    def min_common_indeg(self, a, b):
        out = np.full(len(a), -1, np.int64)
        for i, (x, y) in enumerate(zip(a, b)):
            c = np.intersect1d(self.o1_idx[self.o1_ptr[x]:self.o1_ptr[x + 1]], self.o1_idx[self.o1_ptr[y]:self.o1_ptr[y + 1]], assume_unique=True)
            if c.size:
                out[i] = int(self.indeg1[c].min())
        return out


def bin_of(v):
    return np.select([v <= 10, v <= 100, v <= 1000], ["B1", "B2", "B3"], "B4")


def same_comp(rep, a_id, b_id):
    rs = pd.Series(rep[:, 1], index=rep[:, 0])
    ra, rb = pd.Series(a_id).map(rs).to_numpy(), pd.Series(b_id).map(rs).to_numpy()
    return (~pd.isna(ra)) & (ra == rb)


def cpm_same(cpm_dir, w, a_id, b_id):
    c = cpm_labels(cpm_dir, w)
    if c is None:
        return np.zeros(len(a_id), int)
    c = c.set_index("address_id")["cluster_id"]
    ca, cb = pd.Series(a_id).map(c).to_numpy(), pd.Series(b_id).map(c).to_numpy()
    return ((ca == cb) & ~pd.isna(ca)).astype(int)


def build_queries(G, rep, w, rng, sdir):
    members = np.unique(rep[:, 0])
    q_ids = members[G.has(members)]
    q_idx = G.loc(q_ids); q_idx = q_idx[G.outdeg[q_idx] > 0]
    SP = read_sponsored(sdir, w, ["P", "r", "b"])[["P", "r"]].drop_duplicates()
    dP = SP.groupby("P")["r"].nunique(); okP = set(dP.index[(dP >= DMIN) & (dP <= DMAX)])
    SP = SP[SP["P"].isin(okP)]
    by_r = SP.groupby("r")["P"].apply(list).to_dict(); by_P = SP.groupby("P")["r"].apply(lambda s: s.to_numpy(np.int64)).to_dict()
    is_sender = G.outdeg > 0
    rows = []
    for qa in q_idx:
        nb_ = G.u_idx[G.u_ptr[qa]:G.u_ptr[qa + 1]]
        nb_ = nb_[G.du[nb_] <= HUBDEG]
        two = np.unique(np.concatenate([G.u_idx[G.u_ptr[c]:G.u_ptr[c + 1]] for c in nb_])) if nb_.size else np.empty(0, np.int64)
        two = two[(two != qa) & is_sender[two]]
        if two.size > Q_2HOP:
            two = rng.choice(two, Q_2HOP, replace=False)
        aid = int(G.ids[qa])
        cs = np.unique(np.concatenate([by_P[p] for p in by_r.get(aid, [])])) if aid in by_r else np.empty(0, np.int64)
        cs = cs[cs != aid]; cs = cs[G.has(cs)]; cs = G.loc(cs); cs = cs[is_sender[cs]]
        if cs.size > Q_SPON:
            cs = rng.choice(cs, Q_SPON, replace=False)
        rnd = rng.choice(G.senders, Q_RAND, replace=False)
        cand = np.unique(np.concatenate([two, cs, rnd])); cand = cand[cand != qa]
        rows.append(pd.DataFrame({"q": qa, "b": cand, "src_2hop": np.isin(cand, two), "src_spon": np.isin(cand, cs)}))
    Q = pd.concat(rows, ignore_index=True)
    a, b = Q["q"].to_numpy(np.int64), Q["b"].to_numpy(np.int64)
    Q["a_id"], Q["b_id"] = G.ids[a], G.ids[b]
    keep = (G.common_now(a, b) == 0) & ~same_comp(rep, Q["a_id"].to_numpy(), Q["b_id"].to_numpy())
    Q = Q[keep].reset_index(drop=True)
    Q["y"] = G.label(Q["q"].to_numpy(np.int64), Q["b"].to_numpy(np.int64)).astype(int)
    st = Q.groupby("q").agg(n=("y", "size"), pos=("y", "sum"))
    good = st[(st["pos"] >= 1) & (st["n"] >= Q_MINC)].index
    Q = Q[Q["q"].isin(good)].reset_index(drop=True)
    pos = Q["y"].to_numpy() == 1
    mi = np.full(len(Q), -1, np.int64)
    mi[pos] = G.min_common_indeg(Q.loc[pos, "q"].to_numpy(np.int64), Q.loc[pos, "b"].to_numpy(np.int64))
    Q["bin"] = np.where(pos, bin_of(mi), "")
    return Q, {"queries_raw": int(len(q_idx)), "queries_used": int(Q["q"].nunique())}


def t_features(E, rep, a, b, same_cpm):
    f, t = E["f"].to_numpy(np.int64), E["t"].to_numpy(np.int64)
    if rep is not None and len(rep):
        rs = pd.Series(rep[:, 1], index=rep[:, 0])
        mapf = lambda x: pd.Series(x).map(rs).fillna(pd.Series(x)).to_numpy(np.int64)
        f, t, a, b = mapf(f), mapf(t), mapf(a), mapf(b)
    ids = np.unique(np.concatenate([f, t, a, b]))
    n = ids.size
    loc = lambda x: np.searchsorted(ids, x)
    fk, tk, ai, bi = loc(f), loc(t), loc(a), loc(b)
    m = fk != tk
    o_ptr, o_idx = csr(fk[m], tk[m], n)
    outdeg = np.diff(o_ptr); indeg = np.bincount(o_idx, minlength=n)
    u_ptr, u_idx = csr(np.concatenate([fk[m], tk[m]]), np.concatenate([tk[m], fk[m]]), n)
    du = np.diff(u_ptr)
    wu = np.where(du >= 2, 1.0 / np.log(np.maximum(du, 2)), 0.0)
    cn, aa = pair_inter(u_ptr, u_idx, ai, bi, wu)
    lo_o, lo_i = np.log1p(outdeg), np.log1p(indeg)
    return pd.DataFrame({"log_out_min": np.minimum(lo_o[ai], lo_o[bi]), "log_out_max": np.maximum(lo_o[ai], lo_o[bi]),
                         "log_in_min": np.minimum(lo_i[ai], lo_i[bi]), "log_in_max": np.maximum(lo_i[ai], lo_i[bi]),
                         "cn": cn, "aa_t": aa, "jaccard": cn / np.maximum(du[ai] + du[bi] - cn, 1),
                         "direct": pair_has(u_ptr, u_idx, ai, bi).astype(int), "same_cpm": same_cpm})


def s_features(k, E, a, b, sdir, drop_P=None):
    """S of pairs (a, b) in window k. drop_P: sponsors to leave out (the sponsored ratio is then computed from the remaining
    non-overlapping sponsored transfers)."""
    SP = read_sponsored(sdir, k, ["P", "r", "t", "b", "ptype"])
    if drop_P is not None:
        SP = SP[~SP["P"].isin(drop_P)]
    ids = np.unique(np.concatenate([SP["r"].to_numpy(np.int64), a, b]))
    n = ids.size
    loc = lambda x: np.searchsorted(ids, np.asarray(x, np.int64))
    ai, bi = loc(a), loc(b)
    tcount = E["f"].value_counts().reindex(ids).fillna(0).to_numpy()
    if drop_P is None:
        anyc = pd.read_parquet(Path(sdir) / "sponsored_any_by_sender.parquet").set_index("sender")["sponsored_transfers"]
        sp_n = anyc.reindex(ids).fillna(0).to_numpy()
    else:
        sp_n = SP.groupby("r").size().reindex(ids).fillna(0).to_numpy()
    ratio = np.where(tcount > 0, np.minimum(sp_n / np.maximum(tcount, 1), 1.0), 0.0)
    pcode, puniq = pd.factorize(SP["P"], sort=True)
    SP = SP.assign(pc=pcode, ai=loc(SP["r"]))
    npc = len(puniq)
    dP = SP.groupby("pc")["ai"].nunique().reindex(range(npc)).to_numpy(float)
    wP = np.where((dP >= DMIN) & (dP <= DMAX), 1.0 / np.log(np.maximum(dP, 2)), 0.0)
    rc = SP.groupby(["pc", "t"]).size()
    top1 = (rc.groupby(level=0).max() / rc.groupby(level=0).sum()).reindex(range(npc)).to_numpy()
    nrec = rc.groupby(level=0).size().reindex(range(npc)).to_numpy(float)
    ptyp = SP.groupby("pc")["ptype"].first().reindex(range(npc)).to_numpy()
    sp_ptr, sp_idx = csr(SP["ai"].to_numpy(), SP["pc"].to_numpy(), n)
    ns, aa_s, mind, top = pair_sponsor(sp_ptr, sp_idx, ai, bi, wP, dP)
    has = top >= 0
    tt = np.where(has, ptyp[np.maximum(top, 0)] if npc else "none", "none")
    out = pd.DataFrame({"n_shared": ns, "aa_s": aa_s, "log_min_dP": np.where(has, np.log(np.maximum(mind, 1)), 0.0)})
    for t_, col in zip(PTYPES, ["top_flash", "top_menu", "top_other"]):
        out[col] = (tt == t_).astype(int)
    out["top_top1_share"] = np.where(has, top1[np.maximum(top, 0)] if npc else 0.0, 0.0)
    out["top_log_nrecip"] = np.where(has, np.log1p(nrec[np.maximum(top, 0)]) if npc else 0.0, 0.0)
    out["spon_ratio_min"], out["spon_ratio_max"] = np.minimum(ratio[ai], ratio[bi]), np.maximum(ratio[ai], ratio[bi])
    out["no_shared"] = (~has).astype(int)
    return out


def tie_key(a_id, b_id):
    """Tie-break value in [0, 1): splitmix64 hash of (query id, candidate id); independent of row order and of the query set."""
    x = (np.asarray(a_id, np.int64).astype(np.uint64) << np.uint64(32)) ^ np.asarray(b_id, np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def query_metrics(Q, score):
    """Per-query MRR, hits@10, AP and hits@10 per common-recipient size bin (rank by score descending, ties by tie_key ascending)."""
    tb = tie_key(Q["a_id"].to_numpy(), Q["b_id"].to_numpy())
    cols = ["q", "y"] + (["bin"] if "bin" in Q else [])
    d = Q[cols].assign(s=score, tb=tb).sort_values(["q", "s", "tb"], ascending=[True, False, True])
    d["rank"] = d.groupby("q").cumcount() + 1
    g = d.groupby("q")
    out = {"mrr": g.apply(lambda x: 1.0 / x.loc[x["y"] == 1, "rank"].min()).to_numpy(),
           "hits10": g.apply(lambda x: float((x.loc[x["y"] == 1, "rank"] <= 10).any())).to_numpy(),
           "ap": g.apply(lambda x: float(np.mean(np.cumsum(x["y"].to_numpy())[x["y"].to_numpy() == 1] / x.loc[x["y"] == 1, "rank"].to_numpy()))).to_numpy(),
           "q": np.array(sorted(d["q"].unique()))}
    if "bin" in Q:
        for bname, _, _ in BINS:   # only that bin's positives (other positives removed), ranks recomputed
            dd = d[(d["y"] == 0) | (d["bin"] == bname)].copy()
            qs = dd.groupby("q")["y"].sum(); qs = qs[qs > 0].index
            dd = dd[dd["q"].isin(qs)]
            dd["rank"] = dd.groupby("q").cumcount() + 1
            out[f"hits10_{bname}"] = dd.groupby("q").apply(lambda x: float((x.loc[x["y"] == 1, "rank"] <= 10).any())).to_numpy()
    return out
