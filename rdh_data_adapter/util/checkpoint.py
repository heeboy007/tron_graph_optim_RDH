import json
import numpy as np
from datetime import datetime as dt
from pathlib import Path

from rdh_data_adapter.util.environment import get_environ

def save_as_json(name: str, stats: dict, vary_by_time = False) -> Path:
    environ = get_environ()
    timestamp = f"_{dt.now():%Y%m%d_%H%M%S}" if vary_by_time else ""
    path = environ['dump_path'] / f"{name}{timestamp}.json"
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    return path

def load_from_json(name: str) -> np.ndarray:
    environ = get_environ()
    path = environ['dump_path'] / f"{name}.json"
    with open(path, 'r', encoding="utf-8") as f:
        stats = json.load(f)
    return stats

def save_as_npy(name: str, na: np.ndarray, vary_by_time = False) -> Path:
    environ = get_environ()
    timestamp = f"_{dt.now():%Y%m%d_%H%M%S}" if vary_by_time else ""
    path = environ['dump_path'] / f"{name}{timestamp}.npy"
    np.save(path, na)
    return path

def load_from_npy(name: str) -> np.ndarray:
    environ = get_environ()
    path = environ['dump_path'] / f"{name}.npy"
    return np.load(path, mmap_mode='r')
# ComponentMap 저장/읽기는 rdh.io에 있다 (기본 설치만으로 다른 repo에서도 읽을 수 있도록)
from rdh.io import save_as_pq, load_from_pq  # noqa: E402,F401
