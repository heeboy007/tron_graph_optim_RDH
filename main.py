from rdh.models.stochastic.cdf import build_cdf_r, sample_from_cdf_r
from rdh.models.stochastic.stats import ss
from rdh.pipeline.split import split_dataset_range
from rdh.pipeline.monte_carlo_multicore import monte_carlo_p_value_multicore
from rdh.models.analytic.closed import closed_ss_mean_var
from dataset.load.resource_graph import get_duckdb_resource_graph
from dataset.analysis.checks import z_credibility_check
from dataset.analysis.check_plots import z_credibility_plot
from dataset.analysis.data_plots import plot_coverage_pairs
from dataset.util.environment import get_environ
from dataset.util.trace import trace, print_stats
from dataset.util.checkpoint import load_from_json, save_as_json, load_from_npy, save_as_npy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LogNorm

if __name__ == "__main__":
    environ = get_environ()
    start, end = 80_000_000, 80_200_000
    d_graph, u_graph = get_duckdb_resource_graph(start, end)

    plot_coverage_pairs(d_graph, start, end)

    