# Link-prediction features

A pair (a, b) of USDT senders in window w (blocks [w·200,000, (w+1)·200,000)). Transfer graph of w: successful USDT `transfer` rows,
self-transfers dropped; degrees count distinct addresses. Label: a and b get a new common recipient in window w + 1 (with H = 1,000: at
least one such recipient has in-degree ≤ 1,000 in w + 1). Code: `model/link_prediction/features.py`.

Which model uses which group:

| model | features |
|---|---|
| G (global model) | T |
| residual model, frozen setting (`configs/frozen_config.json`) | D_RDH, on top of the G score |
| residual model, supplementary setting (`configs/robust_DRDH_DS_L8.json`) | D_RDH + D_S, on top of the G score |

## T: transfer graph

| feature | definition |
|---|---|
| `log_out_min`, `log_out_max` | log(1 + out-degree), smaller / larger of a and b |
| `log_in_min`, `log_in_max` | log(1 + in-degree), smaller / larger of a and b |
| `cn` | common neighbours in the undirected graph |
| `aa_t` | Adamic–Adar: Σ 1 / log(degree) over common neighbours of degree ≥ 2 |
| `jaccard` | cn / (degree(a) + degree(b) − cn) |
| `direct` | 1 if a and b are linked (either direction) |
| `same_cpm` | 1 if a and b are in the same Leiden CPM community of window w (`model/ext_features/cpm`) |

## D_RDH: RDH structure (F3 components of window w)

| feature | definition |
|---|---|
| `<t>_d` for the 8 T features above except `same_cpm` | t on the graph where every F3 component is merged into one node (edges inside a component dropped), minus t on the original graph |
| `lcs_min`, `lcs_max` | log(component size), smaller / larger of a and b (size 1 outside a component) |
| `in_any` | 1 if a or b is in an F3 component of size ≥ 2 |
| `in_both` | 1 if both are |

All D_RDH features are 0 when neither a nor b is in a component.
F3 = merge rule without providers in F1 (φ_P ≥ 0.9 or μ_P ≥ 0.61) or F2(c) (`rdh/rdh.py`, `rdh/models/filter`).

## D_S: delegation sponsorship (window w, `model/ext_features/sponsorship`)

A sponsor of x is a provider P whose non-overlapping delegation interval to x covers a transfer sent by x. d_P = number of distinct
receivers P sponsors in w. The "top" shared sponsor is the shared sponsor with the smallest d_P (ties: smaller id).

| feature | definition |
|---|---|
| `n_shared` | number of sponsors shared by a and b |
| `aa_s` | Σ 1 / log d_P over shared sponsors with 2 ≤ d_P ≤ 1,000 |
| `log_min_dP` | log d_P of the top shared sponsor |
| `top_flash`, `top_menu`, `top_other` | type of the top shared sponsor: flash-type (φ_P ≥ 0.9), menu-type (μ_P ≥ 0.61), other |
| `top_top1_share` | share of the top sponsor's sponsored transfers that go to its most frequent token receiver |
| `top_log_nrecip` | log(1 + number of distinct token receivers of the top sponsor's sponsored transfers) |
| `spon_ratio_min`, `spon_ratio_max` | sponsored transfers / transfer rows sent in w (capped at 1; overlapping intervals included), smaller / larger of a and b |
| `no_shared` | 1 if a and b share no sponsor |

`log_min_dP`, `top_*` are 0 when there is no shared sponsor.

## Other columns of the query caches (not features)

`q` (query sender), `a_id`, `b_id` (address ids), `y` (label), `bin` (bin of the smallest in-degree in w + 1 among the common recipients of a positive: B1 ≤ 10,
B2 ≤ 100, B3 ≤ 1,000, B4 > 1,000), `src_2hop`, `src_spon` (candidate came from 2-hop neighbours / shared sponsors).
