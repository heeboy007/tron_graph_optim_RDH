import numpy as np
from typing import TypedDict

class SimpleGraph(TypedDict, total=False):
    src: np.ndarray
    dst: np.ndarray
    g_node_count: int
    l_edge_count: int

class SplitGraph(TypedDict):
    strong: SimpleGraph
    mc: SimpleGraph
    analytic: SimpleGraph

class ResourceGraph(SimpleGraph, total=False):
    bn: np.ndarray
    decompress_map: np.ndarray