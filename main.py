import numpy as np

from rdh.prob_model.cdf import build_cdf_r
from rdh.pipeline.split import split_dataset
from rdh.pipeline.monte_carlo_multicore import monte_carlo_p_value_multicore
from dataset.duckdb_parse import get_duckdb_resource_graph
from dataset.util.environment import get_environ
from dataset.util.trace import trace, print_stats

if __name__ == "__main__":
    environ = get_environ()
    start, end = 80_000_000, 80_200_000
    graph = get_duckdb_resource_graph(start, end)

    # this cdf must be calculated with entire dataset
    cdf_r = build_cdf_r(graph)

    # split dataset for suited pipleines
    split = split_dataset(graph)
    
    monte_carlo_p_value_multicore = trace(monte_carlo_p_value_multicore)
    p_values = monte_carlo_p_value_multicore(split["mc"], cdf_r, environ["tmp_path"])
    print(p_values)
    print_stats()

    
    
    

    
