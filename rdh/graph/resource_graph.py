import numpy as np
from typing import TypedDict

class ResourceGraph(TypedDict, total=False):
    src: np.ndarray
    dst: np.ndarray
    query_time: float
    conversion_time: float
    records: int
    decompress_map: np.ndarray
    node_count: int

