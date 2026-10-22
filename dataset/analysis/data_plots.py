import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from rdh.models.effect.coverage import effect_coverage
from rdh.graph.resource_graph_type import ResourceGraph

def plot_providers(graph: ResourceGraph):
    provider_counts = np.bincount(np.bincount(graph["src"], minlength=graph["g_node_count"]))
    provider_counts = provider_counts[provider_counts > 0]
    counts, _ = np.histogram(provider_counts, bins=[1, 5, 10, 50, 100, 1000, np.inf])
    labels = ["1~4", "5~9", "10~49", "50~99", "100~999", "1000+"]
        
    plt.figure(figsize=(8, 5))
    bar = plt.bar(labels, counts)

    for i, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, counts[i], ha='center', va='bottom', size = 12)

    plt.xlabel('Value Ranges')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.title('histogram of provider event counts')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()
    return bar

def plot_coverage_pairs(graph: ResourceGraph, start, end):
    coverage = effect_coverage(graph, start, end)
    
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    a_bin = np.linspace(0, 175, 30)
    n_bin = np.linspace(0, 1000, 30)

    h = axs.hist2d(coverage["active"], graph["uk_cnt"], bins=[a_bin, n_bin], norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=axs, label="pairs")
    axs.set_xlabel("active(by blocks coverage)"); 
    axs.set_ylabel("n_pair(unique pair counts)"); 
    axs.set_title(f"active, n_pair 2d hist")

    fig.tight_layout()
    plt.show()

def plot_duration_hist(graph: ResourceGraph):
    # dur: 짝지어진 위임의 블록 차이, 짝이 없는(open) 위임은 NaN
    dur = graph["dur"]
    matched = dur[~np.isnan(dur)]
    n_open, n_zero = np.isnan(dur).sum(), (matched == 0).sum()

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    d_bin = np.unique(np.round(np.logspace(0, 5, 400)))  # 1 ~ 100,000블록, 정수 경계

    axs.hist(matched[matched >= 1], bins=d_bin)
    for blocks, label in ((20, "1m"), (200, "10m"), (1200, "1h"), (28800, "1d")):
        axs.axvline(blocks, c="k", ls="--", lw=0.8)
        axs.text(blocks, 1, f" {label}", transform=axs.get_xaxis_transform(), va="top", fontsize=9)
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel("dur (blocks)")
    axs.set_ylabel("delegations")
    axs.set_title(f"delegation duration hist (n={dur.size:,}, dur=0: {n_zero:,}, open: {n_open:,} not shown)")

    fig.tight_layout()
    plt.show()

def plot_concentration_s(s_in: np.ndarray, s_out: np.ndarray):
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    sbin = np.linspace(0, 1, 50)

    h = axs.hist2d(s_in, s_out, bins=[sbin, sbin], norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=axs, label="pairs")
    axs.set_xlabel("s_in"); 
    axs.set_ylabel("s_out"); 
    axs.set_title(f"s_in, s_out 2d hist")

    fig.tight_layout()
    plt.show()