import numpy as np
from typing import TypedDict

class SimpleGraph(TypedDict, total=False):
    src: np.ndarray
    dst: np.ndarray
    g_node_count: int
    l_edge_count: int

class ResourceGraph(SimpleGraph):
    bn: np.ndarray
    amt: np.ndarray
    decompress_map: np.ndarray
    dur: np.array
    concurrent: np.array
    uk: np.ndarray
    uk_inv: np.ndarray
    uk_cnt: np.ndarray
    uk_src: np.ndarray
    uk_dst: np.ndarray

class SplitGraph(TypedDict):
    strong: SimpleGraph
    mc: SimpleGraph
    analytic: SimpleGraph