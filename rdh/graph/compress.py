import numpy as np

from .resource_graph_type import ResourceGraph

def compress_graph(graph: ResourceGraph) -> ResourceGraph:
    if graph["src"].shape[0] != graph["dst"].shape[0]:
        raise ValueError("ResourceGraph's length of src and dst should be the same.")
    m = graph["src"].shape[0]
    all_ids = np.concatenate((graph["src"], graph["dst"]))
    decompress_map, inv = np.unique(all_ids, return_inverse=True)
    src_inv, dst_inv = inv[:m], inv[m:]
    graph["src"] = src_inv
    graph["dst"] = dst_inv
    graph["decompress_map"] = decompress_map
    graph["node_count"] = decompress_map.shape[0]

def decompress_graph(graph: ResourceGraph) -> ResourceGraph:
    if "decompress_map" not in graph:
        raise ValueError("ResourceGraph doesn't have the decompress map. Is it already compressed?")
    graph["src"] = graph["decompress_map"][graph["src"]]
    graph["dst"] = graph["decompress_map"][graph["dst"]]
    del graph["decompress_map"]