import json
import time
import tracemalloc
from collections import defaultdict
from typing import Callable
from datetime import datetime as dt
from functools import wraps

from .environment import get_environ

def create_trace_deco():
    """
        acc_time은 같은 @trace가 존재할시, 그 trace의 time은 배타적으로 추적한다.
        ex) a, b에 둘다 @trace적용시,
        a {
            ... 뭔가 a관련거 ...
            b()
            ... 뭔가 a관련거 ...
        }
        이런식일 경우 a의 acc_time에서는 b의 실행시간이 제외된다.
    """
    environ = get_environ()
    json_path = environ["trace_path"] / f"trace_{dt.now():%Y%m%d_%H%M%S}.json"
    trace_memory = environ["trace_mem"]
    func_stack = []
    stats = defaultdict(lambda: {"calls": 0, "peak": 0, "acc_time": 0.0, "t0": 0.0})

    def deco_trace_mem_ver(fn: Callable):
        func_key = fn.__module__ + ':' + fn.__qualname__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if len(func_stack) == 0: #is empty?
                #only the toppest can call trace start.
                tracemalloc.start()
            else:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                #update a bit early
                parent = stats[func_stack[-1]]
                parent["peak"] = max(parent["peak"], peak)
                parent["acc_time"] += time.perf_counter() - parent["t0"]
            func_stack.append(func_key)

            #set t0, increment call
            curr = stats[func_key]
            curr["t0"] = time.perf_counter()
            curr["calls"] += 1

            try:
                return fn(*args, **kwargs)
            finally:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()

                #update values
                curr["peak"] = max(curr["peak"], peak)
                curr["acc_time"] += time.perf_counter() - curr["t0"]

                #only the toppest can call trace stop.
                func_stack.pop()
                if len(func_stack) == 0:
                    tracemalloc.stop()
                else:
                    stats[func_stack[-1]]["t0"] = time.perf_counter()
            
        return wrapper

    def deco_non_trace_mem_ver(fn: Callable):
        func_key = fn.__module__ + '_' + fn.__qualname__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if len(func_stack) > 0:
                #update a bit early
                parent = stats[func_stack[-1]]
                parent["acc_time"] += time.perf_counter() - parent["t0"]
            func_stack.append(func_key)

            #set t0, increment call
            curr = stats[func_key]
            curr["t0"] = time.perf_counter()
            curr["calls"] += 1

            try:
                return fn(*args, **kwargs)
            finally:
                #update values
                curr["acc_time"] += time.perf_counter() - curr["t0"]

                #only the toppest can call trace stop.
                func_stack.pop()
                if len(func_stack) > 0:
                    stats[func_stack[-1]]["t0"] = time.perf_counter()
        return wrapper

    def print_stats():
        result = {
            k: {kk: vv for kk, vv in v.items() if kk != "t0"}
            for k, v in stats.items()
        }
        print(result)

    def save_stats():
        result = {
            k: {kk: vv for kk, vv in v.items() if kk != "t0"}
            for k, v in stats.items()
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    deco = deco_trace_mem_ver if trace_memory else deco_non_trace_mem_ver
    return (deco, print_stats, save_stats)

trace, print_stats, save_stats = create_trace_deco()
