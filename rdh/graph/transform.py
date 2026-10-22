import numpy as np

from .resource_graph_type import ResourceGraph

def compress_graph(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    assert src.shape == dst.shape

    all_ids = np.concatenate((src, dst))
    g_edge_count = src.shape[0]
    decompress_map, inv = np.unique(all_ids, return_inverse=True)
    return (inv[:g_edge_count], inv[g_edge_count:], decompress_map, decompress_map.shape[0])

def decompress_graph(graph: ResourceGraph) -> ResourceGraph:
    if "decompress_map" not in graph:
        raise ValueError("ResourceGraph doesn't have the decompress map. Is it already compressed?")
    graph["src"] = graph["decompress_map"][graph["src"]]
    graph["dst"] = graph["decompress_map"][graph["dst"]]
    del graph["decompress_map"]

def pair_tx_unique(graph: ResourceGraph):
    key = graph["src"].astype(np.int64) * graph['g_node_count'] + graph["dst"].astype(np.int64)
    uk, uk_inv, uk_cnt = np.unique(key, return_inverse=True, return_counts=True)
    uk_src = uk // graph['g_node_count']
    uk_dst = uk % graph['g_node_count']

    graph["uk"] = uk
    graph["uk_inv"] = uk_inv
    graph["uk_src"] = uk_src.astype(np.int32)
    graph["uk_dst"] = uk_dst.astype(np.int32)
    graph["uk_cnt"] = uk_cnt.astype(np.int32)
