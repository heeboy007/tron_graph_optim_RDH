import time

import numpy as np
import pandas as pd

from rdh.graph.resource_graph_type import ResourceGraph
from rdh.graph.transform import compress_graph, pair_tx_unique, paired_delegate_graph
from rdh.pipeline.effect.pair_delegate_undelegate import pair_fifo_arrays
from rdh_data_adapter.util.environment import get_db_path
from rdh_data_adapter.load.connection import get_duckdb_connection

def get_duckdb_delegate_graph(start, end, db_path=None) -> ResourceGraph:
    con = get_duckdb_connection(db_path or get_db_path())

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

# 짝짓기용 쿼리의 DuckDB memory_limit. 정렬이 이 한도까지 메모리를 쓰고, 넘으면 디스크로 내보낸다.
# 창 80.0M~80.2M 측정: 1GB 19s / 최대 RSS 4.9GiB, 2GB 14s / 6.3GiB, 4GB 11s / 8.7GiB
PAIRING_MEMORY_LIMIT = "1GB"

def get_duckdb_delegate_graph_full(start, end, db_path=None, memory_limit: str = PAIRING_MEMORY_LIMIT) -> ResourceGraph:
    con = get_duckdb_connection(db_path or get_db_path(), memory_limit=memory_limit)

    # with warm-up and horizon matching
    # 14 days, converted to seconds divided by 3 seconds per block, so 14 days amount of blocks.
    c_horizon = (14 * 24 * 60 * 60) // 3 
    c_start, c_end = start - c_horizon, end + c_horizon 

    #fetch block_number, undelegate graph alongside delegate
    # 결과를 바꾸지 않고 행을 줄이는 두 조건 (짝짓기는 (src, dst, amt) 묶음마다 독립):
    # 1. 창 안에 delegate가 하나도 없는 묶음은 뺀다.
    # 2. 창 이후의 delegate는 뺀다. FIFO이므로 창 안 delegate가 모두 회수되기 전에는 회수되지 않는다.
    sql = """
    WITH ev AS (
        SELECT
            from_addr AS src,
            to_addr AS dst,
            amount AS amt,
            block_number AS bn,
            time,
            tx_id,
            CASE WHEN type == 'delegate' THEN 1 ELSE -1 END::TINYINT AS is_d
        FROM resource_tx
        WHERE block_number BETWEEN $c_start AND $c_end
            AND success
            AND (type == 'delegate' OR type == 'undelegate')
            AND res_type == 'ENERGY'
    ), win_groups AS (
        SELECT DISTINCT src, dst, amt FROM ev WHERE is_d = 1 AND bn BETWEEN $start AND $end
    )
    SELECT src, dst, amt, bn, is_d
    FROM ev SEMI JOIN win_groups USING (src, dst, amt)
    WHERE NOT (is_d = 1 AND bn > $end)
    ORDER BY src, dst, amt, bn, time, tx_id
    """
    try:
        print(f"[duckdb] fetching D/U graph of {start} ~ {end} (horizon {c_start} ~ {c_end})")
        t0 = time.perf_counter()
        
        # .df()는 DuckDB 결과와 pandas 사본이 동시에 생기므로 numpy로 바로 받는다
        cols = con.execute(sql, {"c_start": c_start, "c_end": c_end, "start": start, "end": end}).fetchnumpy()
        ev = {c: np.asarray(cols[c]) for c in ("src", "dst", "amt", "bn", "is_d")}
        del cols
        t1 = time.perf_counter()

        d_idx, u_bn, concurrent = pair_fifo_arrays(**ev, start_bn=start, end_bn=end)
        bn = ev["bn"][d_idx]
        dur = np.where(u_bn < 0, np.nan, u_bn - bn)
        paired = pd.DataFrame({"src": ev["src"][d_idx], "dst": ev["dst"][d_idx], "amt": ev["amt"][d_idx],
                               "bn": bn, "dur": dur, "concurrent": concurrent})
        del ev, d_idx, u_bn
        t2 = time.perf_counter()

        graph = paired_delegate_graph(paired)
        src, dst, g_node_count = graph["src"], graph["dst"], graph["g_node_count"]

        t3 = time.perf_counter()
        print(f"[duckdb] query time : {t1 - t0}, paring time: {t2 - t1}, conversion time: {t3 - t2}" + \
              f"ec : D/{graph['l_edge_count']} gn: {g_node_count}")

        # check if data doens't contain any loops
        assert (src == dst).sum() == 0
    finally:
        con.close()
    return graph