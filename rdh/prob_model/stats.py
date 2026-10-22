import numpy as np
from ..graph.resource_graph_type import SimpleGraph

def graph_hhi(graph: SimpleGraph) -> np.ndarray:
    return hhi(graph["src"], graph["dst"], graph["g_node_count"])

def graph_ss(graph: SimpleGraph) -> np.ndarray:
    return ss(graph["src"], graph["dst"], graph["g_node_count"])

def ss(provider: np.ndarray, recipient: np.ndarray, g_node_count: int) -> np.ndarray:
    # 0. check if has same shape(it should...)
    assert provider.shape == recipient.shape

    # 1: get unique pairs
    key = provider.astype(np.int64) * g_node_count + recipient.astype(np.int64)
    unique_keys, unique_counts = np.unique(key, return_counts=True, sorted=False)
    unique_keys_provider_map = unique_keys // g_node_count

    # 2: calculate ss
    return np.bincount(unique_keys_provider_map, weights=unique_counts ** 2, minlength=g_node_count)

def hhi(provider: np.ndarray, recipient: np.ndarray, g_node_count: int, assertion: bool = False) -> np.ndarray:
    # 0. check if has same shape(it should...)
    assert provider.shape == recipient.shape

    # 1: get unique pairs
    key = provider.astype(np.int64) * g_node_count + recipient.astype(np.int64)
    unique_keys, unique_counts = np.unique(key, return_counts=True, sorted=False)
    # 2: calculate portion
    unique_keys_provider_map = unique_keys // g_node_count
    provider_events = np.bincount(unique_keys_provider_map, weights=unique_counts, minlength=g_node_count)
    portion = unique_counts / provider_events[unique_keys_provider_map]
    # 3: calculate hhi per provider id(compressed)
    calculated_hhi = np.bincount(unique_keys_provider_map, weights=portion ** 2, minlength=g_node_count)

    if assertion:
        # 4. check if hhi looks sane
        active = provider_events > 0
        min_hhi = 1 / np.bincount(unique_keys_provider_map, minlength=g_node_count)[active] - 1e-12
        assert (calculated_hhi[active] <= 1 + 1e-12).all()
        assert (calculated_hhi[active] >= min_hhi).all()
        assert np.allclose(np.bincount(unique_keys_provider_map, weights=portion, minlength=g_node_count)[active], 1.0)

    return calculated_hhi
