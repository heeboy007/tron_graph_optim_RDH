import numpy as np
from typing import TypedDict

class SimpleGraph(TypedDict, total=False):
    src: np.ndarray
    dst: np.ndarray
    g_node_count: int
    l_edge_count: int

class ResourceGraph(SimpleGraph):
    bn: np.ndarray
    decompress_map: np.ndarray
    unique_pairs: np.ndarray
    unique_pair_inv: np.ndarray

class SplitGraph(TypedDict):
    strong: SimpleGraph
    mc: SimpleGraph
    analytic: SimpleGraph