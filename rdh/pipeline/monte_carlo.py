import numpy as np

from ..prob_model.cdf import sample_from_cdf_r
from ..prob_model.stats import graph_ss, ss
from ..graph.resource_graph_type import SimpleGraph

# single-core
def monte_carlo_p_value(graph: SimpleGraph, cdf_r: np.ndarray, n_monte_carlo = 20000):
    observed_ss_mc = graph_ss(graph)
    
    exceed = np.zeros((graph["node_count"], ), dtype=np.int32)
    for task_id in range(n_monte_carlo):
        exceed += monte_carlo_acc_sample(
            provider=graph["src"],
            observed_ss=observed_ss_mc,
            cdf_r=cdf_r,
            node_count=graph["node_count"],
            task_id=task_id
        )
    
    return (exceed + 1) / (n_monte_carlo + 1)

#sampling
def monte_carlo_acc_sample(
        provider: np.ndarray,
        observed_ss: np.ndarray, 
        cdf_r: np.ndarray, 
        node_count: int,
        task_id: int,
        base_seed = 42):
    rng = np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(task_id, )))
    edges_count = provider.shape[0]
    
    sample_recipients = sample_from_cdf_r(src=provider, cdf=cdf_r, size=(edges_count, ), rng=rng)
    null_ss_sample = ss(provider, sample_recipients, node_count)
    return (null_ss_sample >= observed_ss).astype(np.int32)
