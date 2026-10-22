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
    
    fig, axs = plt.subplots(1, 1, figsize=(13, 5))
    a_bin = np.linspace(0, 175, 30)
    n_bin = np.linspace(0, 1000, 30)

    h = axs.hist2d(coverage["active"], graph["uk_cnt"], bins=[a_bin, n_bin], norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=axs, label="pairs")
    axs.set_xlabel("active(by blocks coverage)"); 
    axs.set_ylabel("n_pair(unique pair counts)"); 
    axs.set_title(f"active, n_pair 2d hist")

    fig.tight_layout()
    plt.show()