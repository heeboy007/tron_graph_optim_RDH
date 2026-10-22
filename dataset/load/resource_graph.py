import time

from rdh.graph.resource_graph_type import ResourceGraph
from rdh.graph.transform import compress_graph, pair_tx_unique
from dataset.util.environment import get_environ
from dataset.load.connection import get_duckdb_connection
from dataset.load.convert import convert_dataframe

def get_duckdb_resource_graph(start, end) -> tuple[ResourceGraph, ResourceGraph]:
    environ = get_environ()
    con = get_duckdb_connection(environ["db_path"])

    #fetch block_number, undelegate graph alongside delegate
    sql = """
    SELECT 
        from_addr AS src, 
        to_addr AS dst,
        block_number AS bn,
        type == 'delegate' AS is_d
    FROM resource_tx
    WHERE block_number BETWEEN ? AND ? 
        AND success 
        AND (type == 'delegate' OR type == 'undelegate')
        AND res_type == 'ENERGY'
    ORDER BY from_addr, to_addr, block_number
    """
    try:
        print(f"[duckdb] fetching un-deduped, un-weighted resource graph of {start}~{end}")
        t0 = time.perf_counter()
        df = con.execute(sql, [start, end]).df()
        t1 = time.perf_counter()

        r_src, r_dst, bn, is_d = convert_dataframe(df)
        c_src, c_dst, decompress_map, g_node_count = compress_graph(r_src, r_dst)
        d_graph_l_edge_count = is_d.sum()

        d_graph = {
            "src": c_src[is_d],
            "dst": c_dst[is_d],
            "bn": bn[is_d],
            "g_node_count": g_node_count,
            "l_edge_count": d_graph_l_edge_count,
            "decompress_map": decompress_map
        }
        pair_tx_unique(d_graph)

        u_graph = {
            "src": r_src[~is_d],
            "dst": r_dst[~is_d],
            "bn": bn[~is_d],
            "g_node_count": g_node_count,
            "l_edge_count": c_src.shape[0] - d_graph_l_edge_count
        }
        pair_tx_unique(u_graph)

        t2 = time.perf_counter()
        print(f"[duckdb] query time : {t1 - t0}, conversion time: {t2 - t1}, " +\
              f"ec : D/{d_graph['l_edge_count']} U/{u_graph['l_edge_count']} gn: {g_node_count}")

        # check if data doens't contain any loops
        assert (c_src == c_dst).sum() == 0
        assert (c_dst.shape == c_dst.shape and c_dst.shape == bn.shape and bn.shape == is_d.shape)
    finally:
        con.close()
    return (d_graph, u_graph)