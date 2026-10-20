"""
실제 DuckDB 데이터의 split["mc"]만을 사용해 multicore p_value를 검증한다.

실행: uv run --group demo pytest tests -v
"""
import numpy as np
import pytest

from conftest import SIMPLIFY_THRESHOLD, ANALYTIC_THRESHOLD
from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from rdh.prob_model.stats import graph_ss
from rdh.pipeline.monte_carlo import monte_carlo_p_value
from rdh.pipeline.monte_carlo_multicore import monte_carlo_p_value_multicore

N_MC = 16
pytestmark = pytest.mark.timeout(900)


@pytest.fixture(scope="session")
def single_core_p(mc_graph, g_recipient_count):
    return monte_carlo_p_value(mc_graph, g_recipient_count, n_monte_carlo=N_MC)


@pytest.fixture(scope="session", params=[2, 4, 8])
def multi_core_p(request, mc_graph, g_recipient_count, tmp_path_factory):
    workers = request.param
    base_path = tmp_path_factory.mktemp(f"mc_w{workers}")
    p = monte_carlo_p_value_multicore(mc_graph, g_recipient_count, base_path, n_monte_carlo=N_MC, workers=workers)
    return workers, base_path, p


# ---------- 입력 데이터(mc split) 계약 ----------

def test_mc_split_only_contains_mc_providers(mc_graph, full_graph):
    assert mc_graph["src"].shape == mc_graph["dst"].shape
    assert mc_graph["src"].size > 0
    assert mc_graph["node_count"] == full_graph["node_count"]
    _, counts = np.unique(mc_graph["src"], return_counts=True)
    assert (counts > SIMPLIFY_THRESHOLD).all()
    assert (counts <= ANALYTIC_THRESHOLD).all()
    assert (mc_graph["src"] != mc_graph["dst"]).all()


def test_recipient_count_matches_full_graph(g_recipient_count, full_graph):
    assert g_recipient_count.shape == (full_graph["node_count"],)
    assert g_recipient_count.sum() == full_graph["dst"].size
    assert (g_recipient_count >= 0).all()


def test_cdf_r_is_valid(cdf_r, full_graph):
    assert cdf_r.shape == (full_graph["node_count"],)
    assert (np.diff(cdf_r) >= 0).all(), "cdf가 단조증가가 아니면 searchsorted 결과가 정의되지 않는다"
    assert cdf_r[0] >= 0
    assert cdf_r[-1] == 1.0


def test_cdf_r_matches_recipient_mass(cdf_r, g_recipient_count):
    mass = np.diff(np.concatenate(([0.0], cdf_r)))
    np.testing.assert_allclose(mass, g_recipient_count / g_recipient_count.sum(), rtol=0, atol=1e-12)
    assert ((mass > 0) == (g_recipient_count > 0)).all()


# ---------- 샘플러 ----------

def test_sampler_range_and_no_self_loop(mc_graph, cdf_r):
    rng = np.random.default_rng(np.random.SeedSequence(42, spawn_key=(0,)))
    src = mc_graph["src"]
    sample = sample_from_cdf_r(src=src, cdf=cdf_r, size=(src.size,), rng=rng)
    assert sample.min() >= 0
    assert sample.max() < mc_graph["node_count"], "범위 밖 index는 ss에서 다른 provider로 조용히 섞인다"
    assert (sample != src).all()


def test_sampler_never_picks_zero_mass_recipient(mc_graph, cdf_r):
    mass = np.diff(np.concatenate(([0.0], cdf_r)))
    rng = np.random.default_rng(np.random.SeedSequence(42, spawn_key=(1,)))
    src = mc_graph["src"]
    sample = sample_from_cdf_r(src=src, cdf=cdf_r, size=(src.size,), rng=rng)
    assert (mass[sample] > 0).all()


# ---------- multicore p_value ----------

def test_multicore_equals_single_core(multi_core_p, single_core_p):
    workers, _, p = multi_core_p
    np.testing.assert_array_equal(p, single_core_p, err_msg=f"workers={workers}")


def test_p_value_range_and_grid(multi_core_p, mc_graph):
    _, _, p = multi_core_p
    assert p.shape == (mc_graph["node_count"],)
    assert np.isfinite(p).all()
    assert (p >= 1 / (N_MC + 1)).all() and (p <= 1).all()
    k = p * (N_MC + 1)
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)


def test_mc_providers_are_actually_tested(multi_core_p, mc_graph, mc_providers):
    _, _, p = multi_core_p
    observed = graph_ss(mc_graph)
    assert (observed[mc_providers] > 0).all()
    # 전부 1이면 검정이 사실상 동작하지 않은 것
    assert (p[mc_providers] < 1.0).any()


def test_non_mc_nodes_are_p_one(multi_core_p, mc_graph, mc_providers):
    """현재 동작 고정: mc provider가 아닌 노드는 p=1 (검정 대상 아님과 구분 불가)."""
    _, _, p = multi_core_p
    non_mc = np.ones(mc_graph["node_count"], dtype=bool)
    non_mc[mc_providers] = False
    assert (p[non_mc] == 1.0).all()


def test_worker_files_written_to_given_base_path(multi_core_p, mc_graph, g_recipient_count, cdf_r):
    _, base_path, _ = multi_core_p
    np.testing.assert_array_equal(np.load(base_path / "provider.npy"), mc_graph["src"])
    np.testing.assert_array_equal(np.load(base_path / "g_recipient_count.npy"), g_recipient_count)
    # 함수 내부에서 만든 cdf가 g_recipient_count로 만든 cdf와 같아야 한다
    np.testing.assert_array_equal(np.load(base_path / "cdf_r.npy"), cdf_r)
    np.testing.assert_array_equal(np.load(base_path / "cdf_r.npy"), build_cdf_r(g_recipient_count))
    np.testing.assert_array_equal(np.load(base_path / "observed_ss.npy"), graph_ss(mc_graph))
