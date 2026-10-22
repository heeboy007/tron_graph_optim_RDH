import numpy as np
from typing import TypedDict

from rdh.graph.resource_graph_type import ResourceGraph

class Coverage(TypedDict):
    active: np.ndarray
    kappa: np.ndarray
    n_pair: np.ndarray

def effect_coverage(graph: ResourceGraph, start, end, bucket = 1200) -> Coverage:
    #this is for ceiling stuff double minus is part of the calculation
    end_bucket_index = -(-(end - start + 1) // bucket) 
    bn_bucket = (graph["bn"] - start) // bucket
    assert bn_bucket.min() >= 0 and bn_bucket.max() < end_bucket_index
    
    pair_key = graph["src"].astype(np.int64) * graph["g_node_count"] + graph["dst"].astype(np.int64)
    pairs, pair_idx = np.unique(pair_key, return_inverse=True)
    unique_pair_count = len(pairs)

    #여기도 결국은 쌍을 만듬...
    pb = np.unique(pair_idx.astype(np.int64) * end_bucket_index + bn_bucket)
    active = np.bincount(pb // end_bucket_index, minlength=unique_pair_count)   # 쌍별 활동 버킷 수

    #src = pairs // graph["g_node_count"]
    #dst = pairs % graph["g_node_count"]
    kappa = active / end_bucket_index
    n_pair = np.bincount(pair_idx, minlength=unique_pair_count)   # 이벤트 수 (비교용)

    return {
        "active": active,
        "kappa": kappa,
        "n_pair": n_pair
    }