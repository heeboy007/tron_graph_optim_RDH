import time
import duckdb
from pathlib import Path

from .graph.resource_graph import ResourceGraph
from .graph.compress import compress_graph
from .util.environment import get_environ
from .util.trace import trace

def get_duckdb_connection(db_path, memory_limit: str = "8GB", threads: int = 4):
    con = duckdb.connect(str(Path(db_path).expanduser()), read_only=True)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    return con

@trace
def get_duckdb_resource_graph(start, end) -> ResourceGraph:
    environ = get_environ()
    con = get_duckdb_connection(environ["db_path"])
    sql = """
    SELECT 
        from_addr AS src, 
        to_addr AS dst
    FROM resource_tx
    WHERE block_number BETWEEN ? AND ? AND success AND type == 'delegate'
    ORDER BY from_addr, to_addr
    """
    try:
        print(f"fetching un-deduped, un-weighted resource graph of {start}~{end}")
        t0 = time.perf_counter()
        df = con.execute(sql, [start, end]).df()
        t1 = time.perf_counter()
        src = df["src"].to_numpy()
        dst = df["dst"].to_numpy()
        t2 = time.perf_counter()
        print(f"query time : {t1 - t0}, conversion time: {t2 - t1}, records : {len(df)}")

        # check if data doens't contain any loops
        assert (src == dst).sum() == 0
    finally:
        con.close()
    graph = {
        "src": src,
        "dst": dst,
        "query_time": t1 - t0,
        "conversion_time": t2 - t1,
        "records": len(df)
    }
    compress_graph(graph)
    return graph