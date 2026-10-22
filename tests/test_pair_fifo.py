"""
실제 DuckDB 데이터로 delegate-undelegate R1 FIFO 짝짓기(pair_fifo)를 검증한다.

기준값은 같은 알고리즘의 DuckDB SQL 구현(window function 기반)으로
블록 80,000,000 ~ 80,200,000, warm-up/horizon 403,200블록, ENERGY에서 얻은 값이다.

실행: uv run --group demo pytest tests/test_pair_fifo.py -v
"""
import numpy as np
import pytest

from conftest import BLOCK_START, BLOCK_END

HORIZON_BLOCKS = 403_200
pytestmark = pytest.mark.timeout(900)

# SQL 구현 기준값
EXPECTED_DELEGATES = 9_412_994
EXPECTED_OPEN = 243_161
EXPECTED_CONCURRENT = 0.0466
EXPECTED_UNIQUE_PAIRS = 3_905_329
EXPECTED_CLASSES = {  # [lo, hi) 블록
    (0, 100): 6_158_702,
    (150, 300): 786_629,
    (1_150, 1_300): 635_957,
    (27_000, 30_600): 105_265,
}


def _sorted_edges(graph):
    src = graph["decompress_map"][graph["src"]]
    dst = graph["decompress_map"][graph["dst"]]
    order = np.lexsort((graph["bn"], dst, src))
    return src[order], dst[order], graph["bn"][order]


def test_same_delegates_as_delegate_only_graph(full_du_graph, full_graph):
    """짝짓기 결과에 창 안의 delegate가 빠지거나 중복 없이 그대로 남아야 한다."""
    assert full_du_graph["l_edge_count"] == full_graph["l_edge_count"]
    for a, b in zip(_sorted_edges(full_du_graph), _sorted_edges(full_graph)):
        np.testing.assert_array_equal(a, b)


def test_array_shapes_consistent(full_du_graph):
    n = full_du_graph["l_edge_count"]
    for key in ("src", "dst", "bn", "amt", "dur", "concurrent", "uk_inv"):
        assert full_du_graph[key].shape == (n,), key
    assert full_du_graph["uk_cnt"].sum() == n


def test_bn_within_window(full_du_graph):
    bn = full_du_graph["bn"]
    assert bn.min() >= BLOCK_START and bn.max() <= BLOCK_END


def test_duration_range(full_du_graph):
    dur, bn = full_du_graph["dur"], full_du_graph["bn"]
    matched = ~np.isnan(dur)
    assert (dur[matched] >= 0).all()
    assert (dur[matched] == np.round(dur[matched])).all(), "dur는 블록 차이(정수)여야 한다"
    # 회수 블록은 horizon 끝을 넘을 수 없다
    assert (bn[matched] + dur[matched] <= BLOCK_END + HORIZON_BLOCKS).all()


def test_matches_sql_reference(full_du_graph):
    dur = full_du_graph["dur"]
    assert dur.size == EXPECTED_DELEGATES
    assert int(np.isnan(dur).sum()) == EXPECTED_OPEN
    assert round(float(full_du_graph["concurrent"].mean()), 4) == EXPECTED_CONCURRENT
    assert full_du_graph["uk"].size == EXPECTED_UNIQUE_PAIRS
    d = dur[~np.isnan(dur)]
    for (lo, hi), expected in EXPECTED_CLASSES.items():
        assert int(((d >= lo) & (d < hi)).sum()) == expected, (lo, hi)
