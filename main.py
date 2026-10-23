from rdh.models.stochastic.cdf import build_cdf_r, sample_from_cdf_r
from rdh.models.stochastic.stats import ss
from rdh.models.effect.coverage import effect_coverage
from rdh.models.effect.concentration import effect_concentration
from rdh.pipeline.stochastic.split import split_dataset_range
from rdh.pipeline.stochastic.monte_carlo_multicore import monte_carlo_p_value_multicore
from rdh.models.analytic.closed import closed_ss_mean_var
from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full
from rdh_data_adapter.analysis.checks import z_credibility_check
from rdh_data_adapter.analysis.check_plots import z_credibility_plot
from rdh_data_adapter.analysis.data_plots import plot_coverage_pairs, plot_concentration_s, plot_duration_hist
from rdh_data_adapter.util.environment import get_environ
from rdh_data_adapter.util.trace import trace, print_stats
from rdh_data_adapter.util.checkpoint import load_from_json, save_as_json, load_from_npy, save_as_npy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    environ = get_environ()

    start, end = 80_000_000, 80_200_000
    d_graph = get_duckdb_delegate_graph_full(start, end)
    plot_duration_hist(d_graph)

    