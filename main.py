import numpy as np

from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from rdh.pipeline.split import split_dataset
from rdh.pipeline.monte_carlo_multicore import monte_carlo_p_value_multicore
from dataset.duckdb_parse import get_duckdb_resource_graph
from dataset.util.environment import get_environ
from dataset.util.trace import trace, print_stats
from dataset.plot.analysis_plot_qq import z_mean_var_qq_plots
from dataset.util.checkpoint import load_from_json

import pandas as pd, matplotlib.pyplot as plt

if __name__ == "__main__":
    environ = get_environ()
    start, end = 80_000_000, 80_200_000
    graph = get_duckdb_resource_graph(start, end)

    # # this cdf must be calculated with entire dataset
    # g_recipient_count = np.bincount(graph["dst"], minlength=graph["node_count"])

    # # split dataset for suited pipleines
    # split = split_dataset(graph)
    # result = provider_mean_variance_closed_form_check(split["analytic"], g_recipient_count)
    # print(result)
    # save_as_json("mean_variance_cf_check", result)
    provider_anlysis = load_from_json('mean_variance_cf_check')
    z_mean_var_qq_plots(provider_anlysis, graph)
    
    

    

    
    
    

    
