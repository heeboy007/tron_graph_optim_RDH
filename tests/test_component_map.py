"""
component map 공개 API 검증.
- 합성 예제: 병합 조건(active, mu 제외, e 미달), 대표 주소 = 최소 주소, condense_edges, 결정성, 저장/읽기
- 실제 DB(없으면 skip): 기준 설정 회귀값, tau >= 0.5에서 최대 차수 <= 1

실행: uv run --group full pytest tests/test_component_map.py -v
"""
import numpy as np
import pandas as pd
import pytest

from rdh import (
    RDHParams, component_map_from_events, component_map_from_graph, condense_edges, save_as_pq, load_from_pq,
)
from rdh.models.effect.merge import merge_mask, components
from conftest import BLOCK_START, BLOCK_END

START, END = 0, 11_999          # 1200블록 버킷 10개
BUCKET = 1200
PARAMS = RDHParams(A=5, mu0=0.61, tau=0.5, bucket=BUCKET, flash_max=100, rho=0.05, k=3, min_n=5)


def _delegations(src, dst, buckets, dur, offset=10, amt=1_000_000):
    rows = []
    for b in buckets:
        bn = b * BUCKET + offset
        rows += [(src, dst, amt, bn, 1), (src, dst, amt, bn + dur, -1)]
    return rows


def synthetic_events():
    rows = []
    rows += _delegations(10, 20, range(7), dur=10)                 # flash만 -> mu NaN, 독점 쌍 -> 병합
    rows += _delegations(20, 15, range(7), dur=10, offset=50)      # 20 -> 15도 병합 -> {10, 15, 20}, rep 10
    rows += _delegations(30, 40, range(7), dur=1200)               # 1시간 메뉴만 -> mu = 1 >= 0.61 -> 제외
    rows += _delegations(70, 60, range(7), dur=10)                 # 70 -> 60: s_in = 14/21
    rows += _delegations(70, 60, range(7), dur=10, offset=600)
    rows += _delegations(50, 60, range(7), dur=10)                 # 50 -> 60: s_in = 7/21 < 0.5 -> 제외
    rows += _delegations(80, 90, range(5), dur=10)                 # active = 5, A = 5 -> 제외
    df = pd.DataFrame(rows, columns=["src", "dst", "amt", "bn", "is_d"])
    df = df.iloc[np.lexsort((df["bn"], df["amt"], df["dst"], df["src"]))].reset_index(drop=True)
    return {c: df[c].to_numpy() for c in df.columns}


@pytest.fixture(scope="module")
def cmap():
    return component_map_from_events(**synthetic_events(), start=START, end=END, params=PARAMS)


# ---------- 합성 예제 ----------

def test_merge_conditions(cmap):
    merged = set(zip(cmap.merged_pairs["src"], cmap.merged_pairs["dst"]))
    assert merged == {(10, 20), (20, 15), (70, 60)}
    assert (30, 40) not in merged, "mu_P >= mu0 provider는 제외"
    assert (50, 60) not in merged, "e <= tau 쌍은 제외"
    assert (80, 90) not in merged, "active <= A 쌍은 제외"
    row = cmap.merged_pairs.set_index(["src", "dst"]).loc[(70, 60)]
    assert row["s_in"] == pytest.approx(14 / 21) and row["s_out"] == 1.0 and row["active"] == 7


def test_mu_condition_can_be_disabled():
    cm = component_map_from_events(**synthetic_events(), start=START, end=END,
                                   params=RDHParams(**{**PARAMS.to_dict(), "mu0": None}))
    assert (30, 40) in set(zip(cm.merged_pairs["src"], cm.merged_pairs["dst"]))


def test_representative_is_min_address(cmap):
    got = dict(zip(cmap.addr.tolist(), cmap.rep.tolist()))
    assert got == {10: 10, 15: 10, 20: 10, 60: 60, 70: 60}
    np.testing.assert_array_equal(cmap.comp_id, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(cmap.comp_size, [3, 3, 3, 2, 2])
    for dtype_arr in (cmap.addr, cmap.rep, cmap.comp_id, cmap.comp_size):
        assert dtype_arr.dtype == np.int64
    assert cmap.summary == {"merged_pairs": 3, "merged_nodes": 5, "components": 2,
                            "max_comp": 3, "max_out_deg": 1, "max_in_deg": 1}


def test_deterministic(cmap):
    again = component_map_from_events(**synthetic_events(), start=START, end=END, params=PARAMS)
    np.testing.assert_array_equal(again.addr, cmap.addr)
    np.testing.assert_array_equal(again.comp_id, cmap.comp_id)
    pd.testing.assert_frame_equal(again.merged_pairs, cmap.merged_pairs)


def test_unsorted_input_rejected():
    ev = synthetic_events()
    shuffled = {k: v[::-1].copy() for k, v in ev.items()}
    with pytest.raises(AssertionError):
        component_map_from_events(**shuffled, start=START, end=END, params=PARAMS)


def test_condense_edges(cmap):
    src = np.array([20, 10, 20, 99, 99, 60])
    dst = np.array([15, 99, 99, 70, 98, 70])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    s, d, ws, n_internal = condense_edges(src, dst, cmap, weight=w)
    assert n_internal == 2                       # 20 -> 15, 60 -> 70은 같은 컴포넌트 안
    assert list(zip(s.tolist(), d.tolist(), ws.tolist())) == [(10, 99, 5.0), (99, 60, 4.0), (99, 98, 5.0)]

    s, d, cnt, _ = condense_edges(src, dst, cmap)  # weight 없으면 간선 수
    assert cnt.dtype == np.int64 and dict(zip(zip(s.tolist(), d.tolist()), cnt.tolist()))[(10, 99)] == 2

    s, d, _, n_internal = condense_edges(src, dst, cmap, drop_internal=False)
    assert n_internal == 2 and (10, 10) in set(zip(s.tolist(), d.tolist()))


def test_save_load_roundtrip(cmap, tmp_path):
    save_as_pq(tmp_path / "cmap", cmap)
    back = load_from_pq(tmp_path / "cmap")
    for f in ("addr", "rep", "comp_id", "comp_size"):
        np.testing.assert_array_equal(getattr(back, f), getattr(cmap, f))
    pd.testing.assert_frame_equal(back.merged_pairs, cmap.merged_pairs)
    assert back.params == cmap.params and back.summary == cmap.summary
    assert (back.start, back.end) == (START, END)


def test_merge_mask_tau_one_requires_exact():
    s = np.array([1.0, 0.99, 1.0]); a = np.array([9, 9, 9]); mu = np.full(3, np.nan)
    np.testing.assert_array_equal(merge_mask(s, s, a, mu, A=5, mu0=0.61, tau=1.0), [True, False, True])


def test_components_empty():
    addr, rep = components(np.array([0]), np.array([1]), np.array([False]), np.array([5, 7]))
    assert addr.size == 0 and rep.size == 0


# ---------- 실제 DB (없으면 skip) ----------

def test_regression_reference_setting(full_du_graph):
    cm = component_map_from_graph(full_du_graph, BLOCK_START, BLOCK_END, RDHParams(A=5, mu0=0.61, tau=0.05))
    assert cm.summary["merged_pairs"] == 1_125
    assert cm.summary["max_comp"] == 70 and cm.summary["components"] == 287
    assert (cm.summary["max_out_deg"], cm.summary["max_in_deg"]) == (10, 8)


@pytest.mark.parametrize("tau", [0.5, 0.8, 1.0])
@pytest.mark.parametrize("mu0", [0.61, None])
def test_high_tau_degree_at_most_one(full_du_graph, tau, mu0):
    cm = component_map_from_graph(full_du_graph, BLOCK_START, BLOCK_END, RDHParams(A=5, mu0=mu0, tau=tau))
    assert cm.summary["max_out_deg"] <= 1 and cm.summary["max_in_deg"] <= 1
    assert (cm.comp_size == 2).all(), "차수가 1 이하면 컴포넌트는 모두 쌍이다"
