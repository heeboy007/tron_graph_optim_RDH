import numpy as np
from typing import TypedDict

from rdh.graph.resource_graph_type import ResourceGraph

class Coverage(TypedDict, total=False):
    active: np.ndarray
    kappa: np.ndarray

def effect_coverage(graph: ResourceGraph, start, end, bucket = 1200) -> Coverage:
    #this is for ceiling stuff double minus is part of the calculation
    end_bucket_index = -(-(end - start + 1) // bucket) 
    bn_bucket = (graph["bn"] - start) // bucket
    assert bn_bucket.min() >= 0 and bn_bucket.max() < end_bucket_index

    #여기도 미리 만들어둔 쌍 관련 자료를 사용.
    unique_pair_count = graph["uk"].shape[0]
    pb = np.unique(graph["uk_inv"].astype(np.int64) * end_bucket_index + bn_bucket)
    active = np.bincount(pb // end_bucket_index, minlength=unique_pair_count)   # 쌍별 활동 버킷 수
    kappa = active / end_bucket_index

    return {
        "active": active,
        "kappa": kappa
    }