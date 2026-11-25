"""Leiden CPM communities of the weekly USDT transfer graph (link-prediction feature same_cpm, manuscript §3.7).

Chunk = blocks [chunk, chunk + 200,000). Graph: successful USDT rows of type transfer / transferFrom / internal_tx with amount < 1.9e18,
undirected, one edge per address pair (amounts summed, self-loops dropped), edge weight log1p(amount sum); nodes = addresses in ascending
id order.
Resolution: gamma = c · gamma_ref(chunk), gamma_ref = 2W / (N² − N) (W = sum of edge weights, N = nodes); c = 2^log2_gamma / gamma_ref(ref chunk),
so every chunk has the same resolution relative to its density (defaults: log2_gamma = −10, ref chunk 80,000,000).
Leiden (igraph community_leiden, objective CPM, n_iterations = −1: until stable) with Python's random seeded to `seed` before the call.
"""
import random

import numpy as np
import pandas as pd

from rdh_data_adapter.load.connection import get_duckdb_connection
from rdh_data_adapter.util.environment import get_db_path

CHUNK = 200_000
USDT_AMOUNT_CAP = 1.9e18
USDT_TYPES = ["transfer", "transferFrom", "internal_tx"]
LOG2_GAMMA, REF_CHUNK = -10.0, 80_000_000

_SQL = """
    SELECT from_addr AS src, to_addr AS dst, amount::DOUBLE AS weight, block_number
    FROM usdt_tx
    WHERE block_number BETWEEN ? AND ? AND success AND amount::DOUBLE < ?
      AND type::VARCHAR IN (SELECT unnest(?))
    ORDER BY block_number, from_addr, to_addr, amount
"""


def load_chunk(chunk: int, db_path=None, size: int = CHUNK):
    """Undirected deduplicated USDT graph of one chunk: n, local_to_global (address ids), lo/hi (pair end points, lo < hi), w (amount
    sums), self_idx/self_w (self loops). None when the chunk has no rows."""
    con = get_duckdb_connection(db_path or get_db_path(), memory_limit="8GB", threads=4)
    try:
        df = con.execute(_SQL, [chunk, chunk + size - 1, USDT_AMOUNT_CAP, USDT_TYPES]).df()
    finally:
        con.close()
    n_raw = len(df)
    if n_raw == 0:
        return None
    codes, uniques = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True), sort=True)
    src, dst = codes[:n_raw].astype(np.int64), codes[n_raw:].astype(np.int64)
    w = df["weight"].to_numpy(dtype=np.float64)
    del df, codes
    n = len(uniques)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    uk, inv = np.unique(lo * n + hi, return_inverse=True)
    pair_w = np.bincount(inv, weights=w, minlength=uk.size)
    plo, phi = uk // n, uk % n
    is_self = plo == phi
    return {"n": n, "n_raw": n_raw, "local_to_global": np.asarray(uniques), "lo": plo[~is_self], "hi": phi[~is_self], "w": pair_w[~is_self],
            "self_idx": plo[is_self], "self_w": pair_w[is_self]}


def gamma_ref(g) -> float:
    """2W / (N² − N) with W = sum of log1p edge weights (self-loops excluded)."""
    return 2.0 * float(np.log1p(g["w"]).sum()) / (g["n"] * g["n"] - g["n"])


def leiden_cpm(g, gamma: float, seed: int = 0, n_iterations: int = -1) -> np.ndarray:
    """Community label of each local node (index = local id)."""
    import igraph as ig
    G = ig.Graph(n=g["n"], edges=np.column_stack([g["lo"], g["hi"]]), directed=False)
    G.es["weight"] = np.log1p(g["w"]).tolist()
    random.seed(seed)
    r = G.community_leiden(objective_function="CPM", weights="weight", resolution=gamma, n_iterations=n_iterations)
    return np.asarray(r.membership, dtype=np.int64)


def chunk_labels(chunk: int, gamma_c: float, db_path=None, seed: int = 0) -> pd.DataFrame | None:
    """Labels of one chunk with gamma = gamma_c · gamma_ref(chunk). Columns cluster_id, address_id."""
    g = load_chunk(chunk, db_path)
    if g is None:
        return None
    labels = leiden_cpm(g, gamma_c * gamma_ref(g), seed)
    return pd.DataFrame({"cluster_id": labels, "address_id": g["local_to_global"]})


def gamma_c_of(ref_chunk: int = REF_CHUNK, log2_gamma: float = LOG2_GAMMA, db_path=None) -> float:
    """c = 2^log2_gamma / gamma_ref(ref_chunk)."""
    return (2.0 ** log2_gamma) / gamma_ref(load_chunk(ref_chunk, db_path))


def write_chunk_labels(chunk: int, gamma_c: float, out, db_path=None, seed: int = 0) -> str:
    """Writes <out>/labels/leiden_cpm_week_<chunk>.parquet unless it exists; returns a status line."""
    from pathlib import Path
    f = Path(out) / "labels" / f"leiden_cpm_week_{chunk}.parquet"
    if f.exists():
        return "kept"
    d = chunk_labels(chunk, gamma_c, db_path, seed)
    if d is None:
        return "empty"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.parquet"); d.to_parquet(tmp, index=False); tmp.replace(f)
    return f"{len(d):,} nodes, {d['cluster_id'].nunique():,} clusters"
