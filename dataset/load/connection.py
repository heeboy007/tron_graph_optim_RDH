import duckdb
from pathlib import Path

def get_duckdb_connection(db_path, memory_limit: str = "8GB", threads: int = 4):
    con = duckdb.connect(str(Path(db_path).expanduser()), read_only=True)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    return con