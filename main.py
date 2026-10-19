import numpy as np
import time

from rdh.dataset import get_duckdb_resource_graph
from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from rdh.prob_model.stats import hhi
from rdh.util.trace import print_stats

if __name__ == "__main__":
    start, end = 80_000_000, 80_200_000
    graph = get_duckdb_resource_graph(start, end)
    observed_hhi = hhi(graph)
    edges_count = graph["src"].shape[0]

    cdf_r = build_cdf_r(graph)

    t = time.perf_counter()
    sample_recipient = sample_from_cdf_r(size=(edges_count, ), cdf=cdf_r)
    sample_hhi = hhi(graph, override_recipients=sample_recipient)

    t_sample_hhi_once = time.perf_counter() - t
    print(f"t_sample_hhi_once: {t_sample_hhi_once}")
    print(sample_hhi)
    print_stats()
    
