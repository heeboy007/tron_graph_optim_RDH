"""RDH filters and link prediction for one confirmation repetition (default k = 403: G trained on window 403, evaluated on window 405).

Stages (each stage keeps its files under $RDH_DUMP_DIR/v1 and is skipped when they exist):
  f2           F2 controller-key sets of windows 303–421 (rdh.models.filter), c from configs/c_decision.json
  pairs        pair effect-size table of each window (rdh.pair_table); F3 = merge rule without F1 and F2(c) (rdh.rdh)
  table4       reference window (default 400 = blocks 80.0M–80.2M): merge-rule ablation over every delegation pair (rdh.ablation_table,
               manuscript Table 4 without the HHI row); the RDH (F3) row gives the merged pairs and the largest component
  sponsorship  sponsorship tables (model.ext_features.sponsorship)
  cpm          Leiden CPM labels of the USDT graph (model.ext_features.cpm)
  lp           query caches, hub-capped labels (H), G and the residual models for the frozen setting (configs/frozen_config.json) and the
               supplementary setting (configs/robust_DRDH_DS_L8.json); prints G vs residual model vs placebo.

    RDH_DUCKDB_DIR=<replica database 2a> .venv/bin/python -u main.py [--k=403] [--H=1000] [--ref=400] [--stages=f2,table4,pairs,sponsorship,cpm,lp]
The replica database holds windows 396–407 of usdt_tx, enough for k = 403.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WSZ = 200_000
SCAN_LO, SCAN_HI = 60_000_000, 85_000_000       # records of the F2 filter
F2_WINS, F2_EXTRA_HI = list(range(303, 422)), 422  # population: first ENERGY delegation up to the end of window 422
STAGES = ["f2", "table4", "pairs", "sponsorship", "cpm", "lp"]


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def log(m):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {m}", flush=True)


def stage_f2(store):
    from rdh.models.filter import ControllerRecords, f2_frame, f2_sets, lookahead_summary
    from rdh_data_adapter.load.permission import account_created, first_energy_delegation, permission_updates, signing_evidence
    f = store.f2 / "f2_epoch_chain_causal.parquet"
    if f.exists():
        return
    store.f2.mkdir(parents=True, exist_ok=True)
    fd = first_energy_delegation(SCAN_LO, (F2_EXTRA_HI + 1) * WSZ - 1)
    acc = set(fd["a"])
    active, owner, n46 = permission_updates(acc, SCAN_LO, SCAN_HI)
    R = ControllerRecords.from_frames(fd, active, owner, account_created(acc), signing_evidence(acc, SCAN_LO, SCAN_HI - 1), SCAN_HI, WSZ)
    res, pairs, pop = f2_sets(R, F2_WINS + [min(F2_WINS) - 1, F2_EXTRA_HI], "causal")     # population also of the two neighbouring windows
    pairs[pairs["w"].isin(F2_WINS)].to_parquet(store.f2 / "pairs_epoch_chain_causal.parquet", index=False)
    pd.Series(pop).rename_axis("w").rename("population").reset_index().to_parquet(store.f2 / "population_epoch_chain_causal.parquet", index=False)
    chk = {"accounts": len(acc), "type46_success_rows": n46, **lookahead_summary(R)}
    (store.f2 / "checks.json").write_text(json.dumps(chk, indent=1))
    f2_frame({w: res[w] for w in F2_WINS}).to_parquet(f, index=False)
    log(f"f2: {chk}")


def f2_by_window(store, c):
    f = pd.read_parquet(store.f2 / "f2_epoch_chain_causal.parquet", filters=[("c", "==", c)])
    return {int(w): frozenset(g["P"].astype(np.int64)) for w, g in f.groupby("w")}


def stage_table4(store, w, c, f2):
    """Merge-rule ablation of window w with F2(c); written to lp/results/table4_w<w>.csv and printed."""
    from rdh import ablation_table
    from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full
    out = store.result(f"table4_w{w}.csv")
    if not out.exists():
        t0 = time.time()
        g = get_duckdb_delegate_graph_full(w * WSZ, (w + 1) * WSZ - 1)
        T = ablation_table(g, w * WSZ, (w + 1) * WSZ - 1, f2.get(w, frozenset()), f2_label=f"F2({c})")
        del g; gc.collect()
        out.parent.mkdir(parents=True, exist_ok=True); T.to_csv(out, index=False)
        log(f"table4 w{w} ({time.time()-t0:.0f}s)")
    T = pd.read_csv(out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(T.to_string(index=False))
    r = T[T["rule"] == "RDH (F3)"].iloc[0]
    log(f"window {w}: RDH (F3) {r['merged_pairs']:,} merged pairs, largest component {r['max_comp']:,}; table -> {out}")


def stage_pairs(store, wins):
    from rdh import pair_table
    from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full
    for w in wins:
        f = store.pairs(w)
        if f.exists():
            continue
        t0 = time.time()
        g = get_duckdb_delegate_graph_full(w * WSZ, (w + 1) * WSZ - 1)
        p = pair_table(g, w * WSZ, (w + 1) * WSZ - 1)
        del g; gc.collect()
        f.parent.mkdir(parents=True, exist_ok=True); p.to_parquet(f, index=False)
        log(f"pairs w{w}: {len(p):,} pairs ({time.time()-t0:.0f}s)")


def stage_sponsorship(store, wins):
    from model.ext_features.sponsorship import build
    for w in wins:
        if (store.sponsor(w) / "facts.json").exists():
            continue
        f = build(w * WSZ, store.sponsor(w))
        gc.collect()
        log(f"sponsorship w{w}: {f['sponsored_transfers']:,} sponsored transfers ({f['t_total_s']:.0f}s)")


def stage_cpm(store, wins):
    from concurrent.futures import ProcessPoolExecutor
    from model.ext_features.cpm import LOG2_GAMMA, REF_CHUNK, gamma_c_of, write_chunk_labels
    todo = [w * WSZ for w in wins if not (store.cpm_labels / f"leiden_cpm_week_{w * WSZ}.parquet").exists()]
    if not todo:
        return
    gf = store.cpm / "gamma.json"
    if gf.exists():
        meta = json.loads(gf.read_text())
    else:
        meta = {"ref_chunk": REF_CHUNK, "log2_gamma": LOG2_GAMMA, "gamma_c": gamma_c_of(REF_CHUNK, LOG2_GAMMA)}
        store.cpm.mkdir(parents=True, exist_ok=True); gf.write_text(json.dumps(meta, indent=1))
    with ProcessPoolExecutor(3) as ex:
        for c, msg in zip(todo, ex.map(write_chunk_labels, todo, [meta["gamma_c"]] * len(todo), [store.cpm] * len(todo))):
            log(f"cpm {c}: {msg}")


def stage_lp(store, k, H, f2):
    from model.link_prediction.pipeline import confirm, f3_rep, hcache_file, plan, window_cache
    from rdh_data_adapter.load.connection import get_duckdb_connection
    from rdh_data_adapter.util.environment import get_db_path
    con = get_duckdb_connection(get_db_path(), memory_limit="12GB", threads=6, temp_directory=store.tmp / f"lp_{os.getpid()}")
    try:
        P = plan([k])
        for w, roles in P.items():
            if roles:
                window_cache(store, w, con, f3_rep(store, w, f2[w]), roles)
        for w, roles in P.items():
            for r, s in roles:
                hcache_file(store, f"w{w}_{r}_s{s}", con, H)
        for name, cfg_file in (("confirm_H{H}", "frozen_config.json"), ("confirm_robust_DRDH_DS_L8_H{H}", "robust_DRDH_DS_L8.json")):
            confirm(store, [k], con, f2, H, json.loads((ROOT / "configs" / cfg_file).read_text()), name.format(H=H))
    finally:
        con.close()


def report(store, k, H):
    """G vs residual model (setting) vs placebo on the evaluation queries of window k + 2: means and paired per-query differences."""
    rows, notes = [], []
    for name, label in ((f"confirm_H{H}", "frozen D_RDH, L=8"), (f"confirm_robust_DRDH_DS_L8_H{H}", "suppl. D_RDH+D_S, L=8")):
        r = json.loads((store.result(name) / f"k{k}.json").read_text())
        m = {x: {met: np.asarray(v[met], float) for met in ("mrr", "ap", "hits10")} for x, v in r["models"].items() if x in ("G", "setting", "placebo")}
        for met in ("mrr", "ap", "hits10"):
            g, s, p = m["G"][met], m["setting"][met], m["placebo"][met]
            rows.append({"setting": label, "metric": met, "G": g.mean(), "residual": s.mean(), "placebo": p.mean(),
                         "residual-G": (s - g).mean(), "residual-placebo": (s - p).mean(),
                         "q: residual>G": int((s > g).sum()), "q: residual<G": int((s < g).sum())})
        notes.append(f"{label}: evaluation queries {r['eval_queries']:,}, trees residual {r['trees']['setting']} / placebo {r['trees']['placebo']}")
    D = pd.DataFrame(rows)
    out = store.result(f"summary_k{k}_H{H}.csv"); D.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(D.to_string(index=False))
    print("\n".join(notes))
    log(f"summary -> {out}")


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from model.link_prediction.pipeline import Store
    k, H, ref = int(arg("k", 403)), int(arg("H", 1000)), int(arg("ref", 400))
    stages = arg("stages", ",".join(STAGES)).split(",")
    store = Store(Path(os.environ.get("RDH_DUMP_DIR") or ROOT / "dump") / "v1")
    c = int(json.loads((ROOT / "configs" / "c_decision.json").read_text())["c"])
    wins = list(range(k - 7, k + 3))                       # training pools (L <= 8), label window k + 1, evaluation window k + 2
    used = [w for w in wins if w != k + 1]                 # windows whose F3 / sponsorship / CPM inputs are read
    log(f"k = {k}, H = {H}, c = {c}, windows {used}, store {store.root}, database {os.environ.get('RDH_DUCKDB_DIR')}")
    t0 = time.time()
    if "f2" in stages:
        stage_f2(store)
    if "table4" in stages:
        stage_table4(store, ref, c, f2_by_window(store, c))
    if "pairs" in stages:
        stage_pairs(store, used)
    if "sponsorship" in stages:
        stage_sponsorship(store, used)
    if "cpm" in stages:
        stage_cpm(store, used)
    if "lp" in stages:
        stage_lp(store, k, H, f2_by_window(store, c))
        report(store, k, H)
    log(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
