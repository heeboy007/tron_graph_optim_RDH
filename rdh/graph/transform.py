import numpy as np

from .resource_graph_type import ResourceGraph
from .component_map_type import ComponentMap

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

def paired_delegate_graph(paired) -> ResourceGraph:
    """pair_fifo의 결과(창 안의 delegate, src/dst는 원래 주소)를 압축된 ResourceGraph로 바꾼다."""
    src, dst, decompress_map, g_node_count = compress_graph(paired["src"].to_numpy(), paired["dst"].to_numpy())
    graph = {
        "src": src,
        "dst": dst,
        "bn": paired["bn"].to_numpy(),
        "amt": paired["amt"].to_numpy(),
        "dur": paired["dur"].to_numpy(np.float64),
        "concurrent": paired["concurrent"].to_numpy(bool),
        "g_node_count": g_node_count,
        "l_edge_count": src.shape[0],
        "decompress_map": decompress_map
    }
    pair_tx_unique(graph)
    return graph

def condense_edges(src_addr, dst_addr, cmap: ComponentMap, weight=None, drop_internal=True):
    """간선의 끝점을 컴포넌트 대표 주소로 바꾸고 같은 (src, dst) 간선을 합친다.
    같은 컴포넌트 안의 간선(양 끝이 같은 컴포넌트)은 drop_internal이면 버린다. 맵에 없는 주소는 그대로 둔다.
    반환: (src, dst, weight_sum, n_internal). weight가 None이면 weight_sum은 간선 수다."""
    src_addr = np.asarray(src_addr, dtype=np.int64)
    dst_addr = np.asarray(dst_addr, dtype=np.int64)
    assert src_addr.shape == dst_addr.shape
    w = np.ones(src_addr.shape, np.int64) if weight is None else np.asarray(weight)
    assert w.shape == src_addr.shape

    s, d = cmap.to_rep(src_addr), cmap.to_rep(dst_addr)
    in_comp = np.isin(src_addr, cmap.addr)
    internal = in_comp & (s == d)
    n_internal = int(internal.sum())
    if drop_internal:
        keep = ~internal
        s, d, w = s[keep], d[keep], w[keep]

    if s.size == 0:
        return s, d, w[:0], n_internal
    pairs, inv = np.unique(np.stack((s, d), axis=1), axis=0, return_inverse=True)
    weight_sum = np.bincount(inv.ravel(), weights=w, minlength=pairs.shape[0])
    if weight is None or np.issubdtype(w.dtype, np.integer):
        weight_sum = weight_sum.astype(np.int64)
    return pairs[:, 0], pairs[:, 1], weight_sum, n_internal
