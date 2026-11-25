"""F2 controller-key rule and F1/F3 merge filters on small synthetic records (no database)."""
import numpy as np
import pandas as pd

from rdh.models.filter import ControllerRecords, f2_sets
from rdh.models.filter.controller import decide_keys
from rdh.rdh import component_rep, f3_edges

W = 100   # window size of the synthetic records: window w = blocks [100 w, 100 w + 100)


def records(fd, hist=None, born=None, ev=None, scan_hi=1_000):
    ev_rows = [(P, s, b) for P, xs in (ev or {}).items() for s, b in xs]
    evd = pd.DataFrame(ev_rows, columns=["P", "s", "bmin"]).astype(np.int64)
    fdf = pd.DataFrame({"a": list(fd), "fd": list(fd.values())}).astype(np.int64)
    return ControllerRecords.from_frames(fdf, hist or {}, {}, born or {}, evd, scan_hi, W)


def test_state_of_last_update_before_window_plus_updates_inside():
    hist = [(150, frozenset({7})), (250, frozenset({8})), (420, frozenset({9}))]
    # window 3 = [300, 400): last update before (250 -> {8}); nothing inside
    assert decide_keys(3, hist, [], [], None, 150, "causal", W, 1_000)[0] == {8}
    # window 2 = [200, 300): state {7} plus the update inside ({8})
    assert decide_keys(2, hist, [], [], None, 150, "causal", W, 1_000)[0] == {7, 8}


def test_signing_evidence_stops_at_first_update_and_window_end():
    ev = [(5, 120), (6, 380)]
    # no update before window 1; created never -> evidence before min(first update 500, window end + 1 = 200)
    assert decide_keys(1, [], [], ev, None, 500, "causal", W, 1_000)[0] == {5}
    # created before the window end -> no key from evidence
    assert decide_keys(1, [], [], ev, 50, 500, "causal", W, 1_000)[0] == frozenset()


def test_f2_counts_key_scale_in_population_and_passes_lookahead():
    # accounts 1, 2, 3 share key 9 through signing evidence; account 3 enters the population only in window 2
    R = records({1: 10, 2: 20, 3: 210}, ev={1: [(9, 50)], 2: [(9, 60)], 3: [(9, 220)]})
    res, pairs, pop = f2_sets(R, [1, 2], "causal", cs=[2, 3])
    assert pop == {1: 2, 2: 3}
    assert res[1][2] == {1, 2} and res[1][3] == set()
    assert res[2][3] == {1, 2, 3}
    assert R.n_violations == 0 and R.n_checks == 5


def test_f3_drops_f1_and_f2_providers():
    pairs = pd.DataFrame({"P_addr": [10, 10, 20, 30, 40], "r_addr": [11, 12, 21, 31, 41], "s_in": [0.5] * 5, "s_out": [0.5] * 5,
                          "active": [9, 9, 9, 9, 3], "mu_P": [np.nan, np.nan, 0.7, np.nan, np.nan], "phi_P": [0.1, 0.1, 0.1, 0.95, 0.1]})
    # 20: mu >= mu0 (F1 and outside the merge rule); 30: phi >= 0.9 (F1); 40: active <= A
    assert f3_edges(pairs).tolist() == [[10, 11], [10, 12]]
    assert f3_edges(pairs, f2={10}).tolist() == []
    rep = component_rep(f3_edges(pairs))
    assert rep.tolist() == [[10, 10], [11, 10], [12, 10]]
