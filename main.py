import numpy as np

from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from rdh.prob_model.stats import observed_hhi
from rdh.pipeline.split import split_dataset
from rdh.pipeline.strong import rdh_strong_edges
from dataset.duckdb_parse import get_duckdb_resource_graph
from dataset.util.trace import trace, print_stats, save_stats

if __name__ == "__main__":
    start, end = 80_000_000, 80_200_000
    graph = get_duckdb_resource_graph(start, end)
    observed_hhi = observed_hhi(graph)
    # edges_count = graph["src"].shape[0]

    cdf_r = build_cdf_r(graph)

    # sample_recipient = sample_from_cdf_r(size=(edges_count, ), cdf=cdf_r)
    # sample_hhi = hhi(graph, override_recipients=sample_recipient)

    # 1: get unique pairs
    split = split_dataset(graph)
    node_count = graph["node_count"]

    rdh_strong_edges = rdh_strong_edges(split["strong"], node_count)
    #calculated_hhi = np.bincount(unique_keys_provider_map, weights=portion ** 2, minlength=node_count)
    
    
    

    
