import sys
from pathlib import Path

import numpy as np
import pytest

# spawn 워커도 부모의 sys.path를 물려받으므로 repo root를 넣어두면 rdh/dataset import가 된다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdh.models.stochastic.cdf import build_cdf_r
from rdh.pipeline.stochastic.split import split_dataset

# main.py와 동일한 블록 구간
BLOCK_START, BLOCK_END = 80_000_000, 80_200_000
SIMPLIFY_THRESHOLD, ANALYTIC_THRESHOLD = 5, 20000


@pytest.fixture(scope="session")
def full_graph():
    # 창 안의 ENERGY delegate만 (undelegate 짝짓기 없음)
    from dataset.load.resource_graph import get_duckdb_delegate_graph
    return get_duckdb_delegate_graph(BLOCK_START, BLOCK_END)


@pytest.fixture(scope="session")
def full_du_graph():
    # warm-up/horizon 포함 delegate-undelegate R1 짝짓기 결과 (dur, concurrent 포함)
    from dataset.load.resource_graph import get_duckdb_delegate_graph_full
    return get_duckdb_delegate_graph_full(BLOCK_START, BLOCK_END)


@pytest.fixture(scope="session")
def g_recipient_count(full_graph):
    # main.py와 같이 recipient count는 전체 데이터셋으로 만든다
    return np.bincount(full_graph["dst"], minlength=full_graph["g_node_count"])


@pytest.fixture(scope="session")
def cdf_r(g_recipient_count):
    return build_cdf_r(g_recipient_count)


@pytest.fixture(scope="session")
def mc_graph(full_graph):
    split = split_dataset(full_graph, SIMPLIFY_THRESHOLD, ANALYTIC_THRESHOLD)
    return split["mc"]


@pytest.fixture(scope="session")
def mc_providers(mc_graph):
    return np.unique(mc_graph["src"])
