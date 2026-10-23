"""
실제 DuckDB 데이터(R1 짝짓기 결과)로 menu_ratio의 numpy 구현을 검증한다.
provider마다 반복문으로 계산하는 단순한 참조 구현과 모든 provider에서 값이 같아야 한다.

실행: uv run --group demo pytest tests/test_menu_ratio.py -v
"""
import math
from collections import Counter

import numpy as np
import pytest

from rdh.models.effect.menu_ratio import menu_ratio, effect_menu_ratio

pytestmark = pytest.mark.timeout(1800)

PARAMS = [
    dict(flash_max=100, rho=0.05, k=3, min_n=50),   # 기본값
    dict(flash_max=100, rho=0.02, k=1, min_n=20),
]


def reference_menu_ratio(p, dur, flash_max, rho, k, min_n):
    order = np.argsort(p, kind="stable")
    p, dur = p[order], dur[order].astype(np.int64)
    ids, starts = np.unique(p, return_index=True)
    ends = np.r_[starts[1:], p.size]
    log_step = math.log1p(rho)

    out = {key: [] for key in ("n_paired", "phi", "m", "mu")}
    for s, e in zip(starts, ends):
        d = dur[s:e]
        flash = d < flash_max
        nf = d[~flash]
        counts = Counter(int(math.floor(math.log(x) / log_step)) for x in nf)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        menu = {b + off for b, _ in top for off in (-1, 0, 1)}
        covered = sum(cnt for b, cnt in counts.items() if b in menu)
        m = int(nf.size)
        out["n_paired"].append(d.size)
        out["phi"].append(flash.mean())
        out["m"].append(m)
        out["mu"].append(covered / m if m >= min_n else np.nan)
    return {"id": ids, **{key: np.array(v) for key, v in out.items()}}


@pytest.fixture(scope="session")
def paired(full_du_graph):
    mask = ~np.isnan(full_du_graph["dur"])
    return full_du_graph["src"][mask], full_du_graph["dur"][mask]


@pytest.mark.parametrize("params", PARAMS, ids=lambda d: f"rho{d['rho']}_k{d['k']}")
def test_matches_reference(paired, params):
    p, dur = paired
    got = menu_ratio(p, dur, **params)
    ref = reference_menu_ratio(p, dur, **params)
    np.testing.assert_array_equal(got["id"], ref["id"])
    np.testing.assert_array_equal(got["n_paired"], ref["n_paired"])
    np.testing.assert_array_equal(got["m"], ref["m"])
    np.testing.assert_allclose(got["phi"], ref["phi"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["mu"], ref["mu"], rtol=0, atol=1e-12, equal_nan=True)


def test_graph_wrapper_excludes_open(full_du_graph, paired):
    got = effect_menu_ratio(full_du_graph)
    assert got["n_paired"].sum() == paired[0].size == np.count_nonzero(~np.isnan(full_du_graph["dur"]))


def test_value_ranges(paired):
    r = menu_ratio(*paired)
    assert ((r["phi"] >= 0) & (r["phi"] <= 1)).all()
    assert (r["m"] <= r["n_paired"]).all()
    valid = ~np.isnan(r["mu"])
    assert (r["m"][valid] >= 50).all() and (r["m"][~valid] < 50).all()
    assert ((r["mu"][valid] > 0) & (r["mu"][valid] <= 1)).all()


def test_rejects_open_nan(full_du_graph):
    with pytest.raises(AssertionError):
        menu_ratio(full_du_graph["src"], full_du_graph["dur"])
