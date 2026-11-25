"""Delegation sponsorship tables of one window (link-prediction features S and candidate sources, manuscript §3.7).

Intervals: paired ENERGY delegations P -> r of the window (R1 FIFO pairing, rdh_data_adapter get_duckdb_delegate_graph_full) plus the
paired delegations that started in the 14-day warm-up and are still open at the window start; open (unpaired) delegations are dropped.
Interval [d, u] with u = d + duration. Intervals of the same receiver that overlap (ends included) are marked overlapping.
A USDT transfer (successful, type transfer) is sponsored by an interval when its sender is r and its block is in [d, u].
Provider type: flash-type (phi_P >= 0.9), else menu-type (mu_P >= mu0), else other (effect_menu_ratio defaults).

Outputs of window [start, start + 200,000):
  sponsored_nonambiguous  one row per (interval, transfer) of non-overlapping intervals: iid, P, r, t (token receiver), b (block), cls,
                          ptype, same_block; ordered by r, iid, P, b, tid, t (tid = transfer number in (block, tx_id, from, to, amount) order)
  sponsored_any_by_sender sender -> number of distinct sponsored transfers inside the window (overlapping intervals included)
"""
import json
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from rdh import RDHParams
from rdh.models.effect.menu_ratio import effect_menu_ratio
from rdh.rdh import PHI0
from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full
from rdh_data_adapter.util.environment import get_db_path

WSZ = 200_000
CLASSES = [("flash", 0, 100), ("m10", 150, 300), ("h1", 1150, 1300), ("d1", 27000, 30600)]


def dur_class(d):
    out = np.full(d.shape, "other", dtype=object)
    for name, lo, hi in CLASSES:
        out[(d >= lo) & (d < hi)] = name
    return out


def intervals(start: int, db_path=None, params: RDHParams | None = None):
    """Delegation intervals of window [start, start + WSZ) with provider type; returns (iv, facts)."""
    end = start + WSZ - 1
    p = params or RDHParams()
    g = get_duckdb_delegate_graph_full(start, end, db_path=db_path, warmup_open=True)
    dec = np.asarray(g["decompress_map"], np.int64)
    dur = g["dur"]
    open_mask = np.isnan(dur)
    facts = {"window": [start, end], "n_delegations": int(len(dur)), "n_open": int(open_mask.sum()), "open_share": float(open_mask.mean())}
    keep = ~open_mask
    iv = pd.DataFrame({"P": dec[g["src"][keep]], "r": dec[g["dst"][keep]], "d": g["bn"][keep].astype(np.int64),
                       "dur": dur[keep].astype(np.int64), "concurrent": g["concurrent"][keep]})
    wu = g["warmup"]
    iv = pd.concat([iv, pd.DataFrame({"P": wu["src"].to_numpy(np.int64), "r": wu["dst"].to_numpy(np.int64), "d": wu["bn"].to_numpy(np.int64),
                                      "dur": wu["dur"].to_numpy(np.int64), "concurrent": wu["concurrent"].to_numpy(bool)})], ignore_index=True)
    facts["n_warmup_open_intervals"] = int(len(wu))
    iv["u"] = iv["d"] + iv["dur"]
    iv["cls"] = dur_class(iv["dur"].to_numpy())
    # deterministic order: stable sort on every column (remaining ties are identical rows)
    iv = iv.sort_values(["r", "d", "u", "P", "dur", "concurrent"], kind="mergesort").reset_index(drop=True)
    iv["iid"] = np.arange(len(iv))
    same_prev = iv["r"].to_numpy()[1:] == iv["r"].to_numpy()[:-1]
    prev_max = iv.groupby("r")["u"].cummax().shift(1)
    first = np.r_[True, ~same_prev]
    ov_prev = (~first) & (iv["d"].to_numpy() <= prev_max.fillna(-1).to_numpy())
    ov_next = np.r_[ov_prev[1:], False]
    iv["overlap"] = ov_prev | ov_next
    facts.update(n_intervals=int(len(iv)), overlap_share=float(iv["overlap"].mean()))
    menu = effect_menu_ratio(g, flash_max=p.flash_max, rho=p.rho, k=p.k, min_n=p.min_n)
    del g
    phi = pd.Series(menu["phi"], index=dec[menu["id"]]); mu = pd.Series(menu["mu"], index=dec[menu["id"]])
    pt = pd.Series("other", index=phi.index, dtype=object)
    pt[(mu >= p.mu0).fillna(False)] = "menu-type"
    pt[phi >= PHI0] = "flash-type"
    iv["ptype"] = iv["P"].map(pt).fillna("other")
    return iv, facts


def build(start: int, out_dir, db_path=None, params: RDHParams | None = None) -> dict:
    """Writes sponsored_nonambiguous.parquet, sponsored_any_by_sender.parquet and facts.json of window [start, start + WSZ) to out_dir."""
    t0 = time.time()
    db = str(db_path or get_db_path())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    end = start + WSZ - 1
    iv, facts = intervals(start, db, params)
    tmp = out / "duckdb_tmp"; tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory = '{tmp}'"); con.execute("SET memory_limit = '8GB'"); con.execute("SET threads = 6")
        con.execute(f"ATTACH '{db}' AS db (READ_ONLY)")
        con.register("iv_df", iv[["iid", "P", "r", "d", "u", "overlap"]])
        con.execute("CREATE TEMP TABLE iv AS SELECT * FROM iv_df")
        max_u = int(iv["u"].max())
        con.execute(f"""CREATE TEMP TABLE tr AS
SELECT row_number() OVER (ORDER BY block_number, tx_id, from_addr, to_addr, amount) AS tid, from_addr AS s, to_addr AS t, block_number AS b
FROM db.usdt_tx
WHERE block_number BETWEEN {start} AND {max_u} AND success AND type = 'transfer'
  AND from_addr IN (SELECT DISTINCT r FROM iv)""")
        n_window_tr = con.execute(f"SELECT count(*) FROM db.usdt_tx WHERE block_number BETWEEN {start} AND {end} AND success AND type = 'transfer'").fetchone()[0]
        con.execute("""CREATE TEMP TABLE j AS
SELECT iv.iid, tr.tid, tr.b, tr.t, (tr.b = iv.d OR tr.b = iv.u) AS same_block
FROM iv JOIN tr ON tr.s = iv.r AND tr.b BETWEEN iv.d AND iv.u""")
        J = con.execute("SELECT * FROM j ORDER BY iid, tid").df()
    finally:
        con.close(); shutil.rmtree(tmp, ignore_errors=True)
    J = J.merge(iv[["iid", "P", "r", "d", "cls", "ptype", "overlap"]], on="iid")
    facts.update(n_window_transfers=int(n_window_tr), n_joined_rows=int(len(J)), same_block_share=float(J["same_block"].mean()),
                 ambiguous_rows_share=float(J["overlap"].mean()))
    Jw = J[(J["b"] >= start) & (J["b"] <= end)]
    facts["sponsored_transfers"] = int(Jw["tid"].nunique())
    Jw.groupby("r")["tid"].nunique().rename("sponsored_transfers").rename_axis("sender").reset_index() \
        .sort_values("sender", kind="mergesort").reset_index(drop=True).to_parquet(out / "sponsored_any_by_sender.parquet", index=False)
    Jn = J[~J["overlap"]].sort_values(["r", "iid", "P", "b", "tid", "t"], kind="mergesort")
    Jn[["iid", "P", "r", "t", "b", "cls", "ptype", "same_block"]].reset_index(drop=True).to_parquet(out / "sponsored_nonambiguous.parquet", index=False)
    facts["t_total_s"] = time.time() - t0
    (out / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    return facts
