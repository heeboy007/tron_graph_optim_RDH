import numpy as np
from ..graph.resource_graph_type import ResourceGraph

def observed_hhi(graph: ResourceGraph) -> np.ndarray:
    return hhi(graph["src"], graph["dst"], graph["node_count"])

def hhi(provider: np.ndarray, recipient: np.ndarray, node_count: int, assertion: bool = False):
    # 0. check if recipients are correct, or just start
    if assertion:
        assert provider.shape == recipient.shape

    # 1: get unique pairs
    key = provider.astype(np.int64) * node_count + recipient.astype(np.int64)
    unique_keys, unique_counts = np.unique(key, return_counts=True, sorted=False)
    # 2: calculate portion
    unique_keys_provider_map = unique_keys // node_count
    provider_events = np.bincount(unique_keys_provider_map, weights=unique_counts, minlength=node_count)
    portion = unique_counts / provider_events[unique_keys_provider_map]
    # 3: calculate hhi per provider id(compressed)
    calculated_hhi = np.bincount(unique_keys_provider_map, weights=portion ** 2, minlength=node_count)

    if assertion:
        # 4. check if hhi looks sane
        active = provider_events > 0
        min_hhi = 1 / np.bincount(unique_keys_provider_map, minlength=node_count)[active] - 1e-12
        assert (calculated_hhi[active] <= 1 + 1e-12).all()
        assert (calculated_hhi[active] >= min_hhi).all()
        assert np.allclose(np.bincount(unique_keys_provider_map, weights=portion, minlength=node_count)[active], 1.0)

    return calculated_hhi
