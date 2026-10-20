import time
import numpy as np
from pathlib import Path
from multiprocessing import get_context, Pool

from ..graph.resource_graph_type import SimpleGraph
from .monte_carlo import monte_carlo_acc_sample
from ..prob_model.stats import graph_ss

_MATERIAL = None

def init_worker(base_path: Path, keys: tuple, kwargs: dict): 
    global _MATERIAL
    # .npy가 아니라면 mmap_mode 없이 일반 로드로 바꾸세요
    args = ({
        key: np.load(base_path / f"{key}.npy", mmap_mode="r") for key in keys
    }, kwargs)
    _MATERIAL = {k: v for d in args for k, v in d.items()}

def save_for_worker(base_path: Path, provider: np.ndarray, observed_ss: np.ndarray, cdf_r: np.ndarray):
    np.save(base_path / "provider.npy", provider)
    np.save(base_path / "observed_ss.npy", observed_ss)
    np.save(base_path / "cdf_r.npy", cdf_r)

def material_wrapper(task_id: int):
    return monte_carlo_acc_sample(
        provider=_MATERIAL["provider"],
        observed_ss=_MATERIAL["observed_ss"],
        cdf_r=_MATERIAL["cdf_r"],
        node_count=_MATERIAL["node_count"],
        task_id=task_id
    )

def monte_carlo_p_value_multicore(
        graph: SimpleGraph, 
        cdf_r: np.ndarray, 
        base_path: Path, 
        n_monte_carlo = 2000, workers = 8):
    assert n_monte_carlo % workers == 0

    observed_ss_mc = graph_ss(graph)
    edges_count = graph["src"].shape[0]

    ctx = get_context("spawn")   # OS 무관하게 동작을 통일
    save_for_worker(
        base_path,
        provider=graph["src"],
        observed_ss=observed_ss_mc,
        cdf_r=cdf_r
    )
    keys = ("provider", "observed_ss", "cdf_r")
    kwargs = {
        "edges_count": edges_count,
        "node_count": graph["node_count"]
    }
    
    done = 0
    exceed = np.zeros((graph["node_count"], ))
    with ctx.Pool(workers, initializer=init_worker, initargs=(base_path, keys, kwargs)) as pool:
        for part_exceed in pool.imap_unordered(material_wrapper, range(n_monte_carlo)):
            exceed += part_exceed
            done += 1
            print(f"{done}/{n_monte_carlo} tasks")
    
    return (exceed + 1) / (n_monte_carlo + 1)