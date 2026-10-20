import numpy as np
from typing import TypedDict

class SimpleGraph(TypedDict):
    src: np.ndarray
    dst: np.ndarray
    node_count: int

class SplitGraph(TypedDict):
    strong: SimpleGraph
    mc: SimpleGraph
    analytic: SimpleGraph

class ResourceGraph(SimpleGraph, total=False):
    src: np.ndarray
    dst: np.ndarray
    query_time: float
    conversion_time: float
    records: int
    decompress_map: np.ndarray
    node_count: int