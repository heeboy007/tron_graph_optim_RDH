import numpy as np

from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from rdh.prob_model.stats import ss
from rdh.pipeline.split import split_dataset_range
from rdh.pipeline.monte_carlo_multicore import monte_carlo_p_value_multicore
from dataset.duckdb_parse import get_duckdb_resource_graph
from dataset.plot.analysis_plot import dist_comparison_plots
from dataset.util.environment import get_environ
from dataset.util.trace import trace, print_stats
from dataset.util.checkpoint import load_from_json, save_as_json, load_from_npy, save_as_npy

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    environ = get_environ()
    start, end = 80_000_000, 80_200_000
    global_graph = get_duckdb_resource_graph(start, end)

    # this cdf must be calculated with entire dataset
    #g_recipient_count = np.bincount(global_graph["dst"], minlength=global_graph["node_count"])
    raw_ass = ss(global_graph['src'], global_graph['dst'], global_graph['node_count'])
    
    

    
    