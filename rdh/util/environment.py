import os
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict

load_dotenv()

class RDHEnvrionment(TypedDict):
    db_path: Path
    trace_path: Path
    trace_mem: bool

def get_environ() -> RDHEnvrionment:
    #var 1
    db_path = Path(os.environ.get("RDH_DUCKDB_DIR"))
    if not db_path:
        raise ValueError("RDH_DUCKDB_DIR 환경변수(tron.db 파일 경로)를 설정하세요.")

    trace_path = Path(os.environ.get("RDH_TRACE_DIR"))
    if not trace_path:
        raise ValueError("RDH_TRACE_DIR 환경변수(tron.db 파일 경로)를 설정하세요.")

    trace_mem = os.environ.get("RDH_TRACE_MEM") == 'True'
    return { 
        "db_path": db_path, 
        "trace_path": trace_path,
        "trace_mem": trace_mem
    }