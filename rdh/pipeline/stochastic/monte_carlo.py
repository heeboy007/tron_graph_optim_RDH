import numpy as np

from ...models.stochastic.cdf import build_cdf_r, sample_from_cdf_r
from ...models.stochastic.stats import graph_ss, ss
from ...graph.resource_graph_type import SimpleGraph

# single-core
def monte_carlo_p_value(graph: SimpleGraph, g_recipient_count: np.ndarray, n_monte_carlo = 8000):
    observed_ss_mc = graph_ss(graph)
    cdf_r = build_cdf_r(g_recipient_count)
    
    exceed = np.zeros((graph["g_node_count"], ), dtype=np.int32)
    for task_id in range(n_monte_carlo):
        exceed += monte_carlo_acc_sample(
            provider=graph["src"],
            observed_ss=observed_ss_mc,
            cdf_r=cdf_r,
            g_node_count=graph["g_node_count"],
            task_id=task_id
        )
    
    return (exceed + 1) / (n_monte_carlo + 1)

#sampling
def monte_carlo_acc_sample(
        provider: np.ndarray,
        observed_ss: np.ndarray, 
        cdf_r: np.ndarray, 
        g_node_count: int,
        task_id: int,
        base_seed = 42):
    rng = np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(task_id, )))
    edges_count = provider.shape[0]
    
    sample_recipients = sample_from_cdf_r(src=provider, cdf=cdf_r, size=(edges_count, ), rng=rng)
    null_ss_sample = ss(provider, sample_recipients, g_node_count)
    return (null_ss_sample >= observed_ss).astype(np.int32)