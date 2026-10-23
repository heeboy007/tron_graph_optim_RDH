"""ComponentMap 저장/읽기: path(폴더) 안에 address_map.parquet, merged_pairs.parquet, meta.json."""
import json
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .graph.component_map_type import ComponentMap, RDHParams

DIST_NAME = "tron-graph-optim-rdh"

def _package_version() -> str | None:
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        return None

def _git_commit() -> str | None:
    # 소스 checkout에서 실행할 때만 얻을 수 있다 (pip 설치본이면 None)
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None

def save_as_pq(path, cmap: ComponentMap) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "addr": cmap.addr, "rep": cmap.rep, "comp_id": cmap.comp_id, "comp_size": cmap.comp_size,
    }).to_parquet(path / "address_map.parquet", index=False)
    cmap.merged_pairs.to_parquet(path / "merged_pairs.parquet", index=False)
    meta = {
        "params": cmap.params.to_dict(), "start": cmap.start, "end": cmap.end, "summary": cmap.summary,
        "package_version": _package_version(), "git_commit": _git_commit(),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path

def load_from_pq(path) -> ComponentMap:
    path = Path(path)
    am = pd.read_parquet(path / "address_map.parquet")
    meta = json.loads((path / "meta.json").read_text())
    col = lambda c: am[c].to_numpy(np.int64)
    return ComponentMap(
        addr=col("addr"), rep=col("rep"), comp_id=col("comp_id"), comp_size=col("comp_size"),
        merged_pairs=pd.read_parquet(path / "merged_pairs.parquet"),
        params=RDHParams(**meta["params"]), start=meta["start"], end=meta["end"], summary=meta["summary"],
    )
