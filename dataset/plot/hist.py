import numpy as np
import matplotlib.pyplot as plt

from rdh.graph.resource_graph import ResourceGraph

def plot_providers(graph: ResourceGraph):
    provider_counts = np.bincount(np.bincount(graph["src"], minlength=graph["node_count"]))
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