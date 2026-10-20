import json
from datetime import datetime as dt

from dataset.util.environment import get_environ

def save_as_json(name: str, stats: dict, vary_by_time = False):
    environ = get_environ()
    timestamp = f"_{dt.now():%Y%m%d_%H%M%S}" if vary_by_time else ""
    path = environ['tmp_path'] / f"{name}{timestamp}.json"
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)