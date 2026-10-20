import numpy as np

from ..graph.resource_graph_type import SimpleGraph

def build_cdf_r(graph: SimpleGraph) -> np.ndarray:
    recipient_count = np.bincount(graph["dst"], minlength=graph["node_count"])
    p_r = (recipient_count / recipient_count.sum()).astype(np.float64)
    return np.cumsum(p_r)

def sample_from_cdf_r(src: np.ndarray, cdf: np.ndarray, size, rng):
    null_sample = np.searchsorted(cdf, rng.uniform(size=size))

    retry = 0
    loop = src == null_sample
    new_sample_needed = np.sum(loop)
    while new_sample_needed > 0:
        null_sample[loop] = np.searchsorted(cdf, rng.uniform(size=(new_sample_needed, )))

        loop = src == null_sample
        new_sample_needed = np.sum(loop)
        retry += 1

    assert (null_sample != src).all()
    return null_sample