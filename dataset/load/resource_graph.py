import time

from rdh.graph.resource_graph_type import ResourceGraph
from rdh.graph.transform import compress_graph, pair_tx_unique
from rdh.pipeline.effect.pair_delegate_undelegate import pair_fifo
from dataset.util.environment import get_environ
from dataset.load.connection import get_duckdb_connection

def get_duckdb_delegate_graph(start, end) -> ResourceGraph:
    environ = get_environ()
    con = get_duckdb_connection(environ["db_path"])

    #fetch block_number, undelegate graph alongside delegate
    sql = """
    SELECT 
        from_addr AS src, 
        to_addr AS dst,
        block_number AS bn
    FROM resource_tx
    WHERE block_number BETWEEN ? AND ? 
        AND success 
        AND type == 'delegate'
        AND res_type == 'ENERGY'
    ORDER BY from_addr, to_addr, block_number
    """
    try:
        print(f"[duckdb] fetching delegate resource graph(without amount or undelegate) of {start}~{end}")
        t0 = time.perf_counter()
        df = con.execute(sql, [start, end]).df()
        t1 = time.perf_counter()

        r_src = df["src"].to_numpy()
        r_dst = df["dst"].to_numpy()
        bn = df['bn'].to_numpy()
        c_src, c_dst, decompress_map, g_node_count = compress_graph(r_src, r_dst)

        d_graph = {
            "src": c_src,
            "dst": c_dst,
            "bn": bn,
            "g_node_count": g_node_count,
            "l_edge_count": c_src.shape[0],
            "decompress_map": decompress_map
        }
        pair_tx_unique(d_graph)

        t2 = time.perf_counter()
        print(f"[duckdb] query time : {t1 - t0}, conversion time: {t2 - t1}, " + \
              f"ec : D/{d_graph['l_edge_count']} gn: {g_node_count}")

        # check if data doens't contain any loops
        assert (c_src == c_dst).sum() == 0
        assert (c_dst.shape == c_dst.shape and c_dst.shape == bn.shape)
    finally:
        con.close()
    return d_graph

def get_duckdb_delegate_graph_full(start, end) -> ResourceGraph:
    environ = get_environ()
    con = get_duckdb_connection(environ["db_path"])

    # with warm-up and horizon matching
    # 14 days, converted to seconds divided by 3 seconds per block, so 14 days amount of blocks.
    c_horizon = (14 * 24 * 60 * 60) // 3 
    c_start, c_end = start - c_horizon, end + c_horizon 

    #fetch block_number, undelegate graph alongside delegate
    sql = """
    SELECT 
        from_addr AS src, 
        to_addr AS dst,
        amount AS amt,
        block_number AS bn,
        CASE WHEN type == 'delegate' THEN 1 ELSE -1 END AS is_d
    FROM resource_tx
    WHERE block_number BETWEEN ? AND ? 
        AND success 
        AND (type == 'delegate' OR type == 'undelegate')
        AND res_type == 'ENERGY'
    ORDER BY from_addr, to_addr, amount, block_number, time, tx_id
    """
    try:
        print(f"[duckdb] fetching D/U graph of {start} ~ {end} (horizon {c_start} ~ {c_end})")
        t0 = time.perf_counter()
        
        df = con.execute(sql, [c_start, c_end]).df()
        t1 = time.perf_counter()

        df = pair_fifo(df, start, end)
        t2 = time.perf_counter()

        src, dst, decompress_map, g_node_count = compress_graph(df["src"].to_numpy(), df["dst"].to_numpy())

        graph = {
            "src": src,
            "dst": dst,
            "bn": df['bn'].to_numpy(),
            "amt": df['amt'].to_numpy(),
            "dur": df['dur'].to_numpy(),
            "concurrent": df['concurrent'].to_numpy(),
            "g_node_count": g_node_count,
            "l_edge_count": src.shape[0],
            "decompress_map": decompress_map
        }
        pair_tx_unique(graph)

        t3 = time.perf_counter()
        print(f"[duckdb] query time : {t1 - t0}, paring time: {t2 - t1}, conversion time: {t3 - t2}" + \
              f"ec : D/{graph['l_edge_count']} gn: {g_node_count}")

        # check if data doens't contain any loops
        assert (src == dst).sum() == 0
    finally:
        con.close()
    return graph