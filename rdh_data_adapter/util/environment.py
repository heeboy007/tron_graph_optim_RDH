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
    trace_mem: bool
    dump_path: Path

def get_db_path() -> Path:
    # DB만 필요할 때: RDH_DUCKDB_DIR만 요구한다
    db_path = os.environ.get("RDH_DUCKDB_DIR")
    if not db_path:
        raise ValueError("RDH_DUCKDB_DIR 환경변수(tron.db 파일 경로)를 설정하세요.")
    return Path(db_path)

def get_environ() -> RDHEnvrionment:
    db_path = os.environ.get("RDH_DUCKDB_DIR")
    if not db_path:
        raise ValueError("RDH_DUCKDB_DIR 환경변수(tron.db 파일 경로)를 설정하세요.")
    
    dump_path = os.environ.get("RDH_DUMP_DIR")
    if not dump_path:
        raise ValueError("RDH_DUMP_DIR 환경변수(tmp for multicore, datasets etc 경로)를 설정하세요.")
    
    trace_mem = False if IS_PRODUCTION else os.environ.get("RDH_TRACE_MEM") == 'True'
    return { 
        "db_path": Path(db_path),
        "dump_path": Path(dump_path),
        "trace_mem": trace_mem
    }