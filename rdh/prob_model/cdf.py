import numpy as np

from ..graph.resource_graph import ResourceGraph

def get_rng(seed=42):
    return np.random.default_rng(seed)

def build_cdf_r(graph:ResourceGraph) -> np.ndarray:
    recipient_count = np.bincount(graph["dst"], minlength=graph["node_count"])
    p_r = (recipient_count / recipient_count.sum()).astype(np.float64)
    return np.cumsum(p_r)

def sample_from_cdf_r(size, cdf: np.ndarray):
    rng = get_rng()
    return np.searchsorted(cdf, rng.uniform(size=size))