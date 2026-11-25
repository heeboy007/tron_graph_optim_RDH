import os
import duckdb
from pathlib import Path

def get_duckdb_connection(db_path, memory_limit: str = "8GB", threads: int = 4, temp_directory=None):
    """Read-only DuckDB connection. Spill files go to temp_directory, or to the environment variable RDH_DUCKDB_TEMP_DIR when the
    argument is not given (DuckDB default otherwise: <db>.tmp next to the database file, shared by every process)."""
    con = duckdb.connect(str(Path(db_path).expanduser()), read_only=True)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    tmp = temp_directory or os.environ.get("RDH_DUCKDB_TEMP_DIR")
    if tmp:
        Path(tmp).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp}'")
    return con
