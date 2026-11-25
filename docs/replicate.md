# How to replicate

The reference window (window 400 = blocks 80.0M–80.2M: merge-rule ablation, manuscript Table 4 without the HHI row) and one confirmation
repetition of the link-prediction study (k = 403: G trained on window 403, evaluated on window 405), from the replica database (record 2a)
and the settings files in `configs/` (record 1). Window k = blocks [k·200,000, (k+1)·200,000).

## Setup
```bash
uv sync --extra model --group full        # Python >= 3.13; locked versions in uv.lock
export RDH_DUCKDB_DIR=/path/to/block_replica_v14_2a.duckdb   # read-only
export RDH_DUMP_DIR=$PWD/dump              # intermediate files go to $RDH_DUMP_DIR/v1
.venv/bin/python -u main.py                # all stages; --stages=f2,table4,pairs,sponsorship,cpm,lp to run some of them
```
With the full database instead of the replica, `RDH_BLOCKSCAN_DIR` points to the block-scan database holding `perm_updates` and
`account_create` (the replica database holds every table).

## Stages and code

| stage | what | code |
|---|---|---|
| f2 | controller keys of every account with an ENERGY delegation (permission updates, creation, signing evidence; look-ahead check), F2(c) of windows 303–421 | `rdh_data_adapter/load/permission.py`, `rdh/models/filter/` |
| table4 | reference window (`--ref`, default 400): the merge rule over every delegation pair, removing the provider filter, μ, α, χ_out, χ_in one after another (cumulative rows) or one part of RDH at a time (leave-one-out rows) | `rdh/rdh.py` `ablation_table`, `comp_stats` |
| pairs | pair effect sizes of each window (R1 FIFO pairing with 14-day warm-up / horizon) | `rdh/rdh.py` `pair_table` |
| (F3) | merge rule without F1 (φ_P ≥ 0.9 or μ_P ≥ 0.61) and F2(c), components | `rdh/rdh.py` `f3_edges`, `component_rep` |
| sponsorship | USDT transfers of a receiver inside a delegation interval of a provider | `model/ext_features/sponsorship/` |
| cpm | Leiden CPM communities of the weekly USDT graph (γ = c·γ_ref, log2 γ = −10 at chunk 80,000,000) | `model/ext_features/cpm/` |
| lp | queries and features (T, D_RDH, D_S), hub-capped label (H = 1,000), G, residual model, placebo | `model/link_prediction/` |

Settings (record 1):
- `configs/c_decision.json`: c = 3. It comes from the decision rule (`rdh/models/filter/c_rule.py`) on the development windows 303–327.
  The rule reads every delegation count of those windows, which the replica database does not hold.
- `configs/frozen_config.json`: the frozen setting (D_RDH, L = 8), chosen on the development repetitions before the confirmation run.
- `configs/robust_DRDH_DS_L8.json`: the supplementary setting (D_RDH + D_S, L = 8).

## Output
The `table4` stage prints the ablation table of the reference window (pairs, addresses, components, largest component, largest out- /
in-degree, providers) and writes it to `$RDH_DUMP_DIR/v1/lp/results/table4_w400.csv`. Its first row, RDH (F3), gives the merged pairs and
the largest component of the reference window (460 and 21 with c = 3). The HHI row of Table 4 is not part of this repository.

`main.py` prints, for each setting, the mean query MRR / AP / hits@10 of G, of the residual model (G + boosting on the setting's
features) and of the placebo (same model on D rows permuted between queries), and the per-query differences. The table is also written to
`$RDH_DUMP_DIR/v1/lp/results/summary_k403_H1000.csv`; per-query metrics and candidate scores are in `lp/results/<setting>/`.
