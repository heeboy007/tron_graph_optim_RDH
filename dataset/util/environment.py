from pathlib import Path
from typing import TypedDict
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
    IS_PRODUCTION = False
except ImportError:
    IS_PRODUCTION = True

class RDHEnvrionment(TypedDict):
    db_path: Path
    trace_path: Path
    trace_mem: bool
    tmp_path: Path

def get_environ() -> RDHEnvrionment:
    db_path = os.environ.get("RDH_DUCKDB_DIR")
    if not db_path:
        raise ValueError("RDH_DUCKDB_DIR 환경변수(tron.db 파일 경로)를 설정하세요.")
    
    tmp_path = os.environ.get("RDH_TMP_DIR")
    if not tmp_path:
        raise ValueError("RDH_TMP_DIR 환경변수(tmp for multicore 경로)를 설정하세요.")

    trace_path = os.environ.get("RDH_TRACE_DIR")
    if not trace_path:
        raise ValueError("RDH_TRACE_DIR 환경변수(trace 저장 경로)를 설정하세요.")
    
    trace_mem = False if IS_PRODUCTION else os.environ.get("RDH_TRACE_MEM") == 'True'
    return { 
        "db_path": Path(db_path),
        "tmp_path": Path(tmp_path),
        "trace_path": Path(trace_path),
        "trace_mem": trace_mem
    }