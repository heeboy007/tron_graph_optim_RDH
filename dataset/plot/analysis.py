import numpy as np

from rdh.graph.resource_graph_type import SimpleGraph
from rdh.pipeline.split import split_dataset_range
from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from dataset.util.checkpoint import save_as_json, save_as_npy

def provider_mean_variance_closed_form_check(graph: SimpleGraph, g_recipient_count: np.ndarray, base_seed: int = 42):
    # is sorted
    analytic_providers = np.unique(graph["src"], sorted=True)
    provider_anlysis = {}
    for provider in analytic_providers:
        rng = np.random.default_rng([base_seed, provider])
        # 'l' represents 'local'
        l_recipients = graph["dst"][graph["src"] == provider]
        l_recipient_count = np.bincount(l_recipients, minlength=graph["node_count"])
        curr_recipient_count = g_recipient_count - l_recipient_count
        cdf_l = build_cdf_r(curr_recipient_count)

        # 샘플러가 r != src로 재추출하는 분포와 일치 시켜야함.
        p = curr_recipient_count.astype(np.float64)
        p[provider] = 0.0          
        p /= p.sum()

        n = l_recipients.shape[0]
        h = (p ** 2).sum()
        t = (p ** 3).sum()
        closed_mean = n + n * (n - 1) * h
        closed_var = 2 * n * (n - 1) * h * (1 - h) + 4 * n * (n - 1) * (n - 2) * (t - h * h)

        n_sim = 2000
        sample_l = sample_from_cdf_r(np.full(l_recipients.shape, provider), cdf_l, (n_sim, l_recipients.shape[0]), rng)
        sample_sss = np.apply_along_axis(lambda a: ((np.unique(a, return_counts=True)[1]) ** 2).sum(), axis=1, arr=sample_l)

        mc_mean = sample_sss.mean()
        mc_var = sample_sss.var(ddof=1)
        mc_std = sample_sss.std(ddof=1)

        se_mean = mc_std / np.sqrt(n_sim)
        z_mean = (mc_mean - closed_mean) / se_mean
        rel_var = mc_var / closed_var - 1
        kurt = ((sample_sss - mc_mean) ** 4).mean() / sample_sss.var() ** 2
        se_rel_var = np.sqrt((kurt - 1) / n_sim)
        provider_anlysis[provider.item()] = {
            'provider': provider.item(),
            'n': n,
            'z_mean': z_mean.item(),
            'rel_var': rel_var.item(),
            'se_rel_var': se_rel_var.item(),
            'c_mean': closed_mean.item(),
            'c_var': closed_var.item(),
            's_mean': mc_mean.item(),
            's_var': mc_var.item()
        }
    return provider_anlysis

def distribution_check(global_graph: SimpleGraph, g_recipient_count: np.ndarray, base_seed: int = 42):
    split = split_dataset_range(global_graph, split_range=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    base_seed = 42
    provider_anlysis = {}
    for s, graph in split.items():
        print(f"split : {s} src / dst : {graph['src'].shape}")
        samples = 3

        #skip 0~100
        if s <= 0: continue

        pre_rng = np.random.default_rng([base_seed, s])

        providers = np.unique(graph["src"], sorted=True)
        sample_prodviders = providers[pre_rng.choice(providers.shape[0], min(samples, providers.shape[0]), replace=False)]
        
        for provider in sample_prodviders:
            pro_rng = np.random.default_rng([base_seed, provider])
            # 'l' represents 'local'
            l_recipients = graph["dst"][graph["src"] == provider]
            l_recipient_count = np.bincount(l_recipients, minlength=graph["node_count"])
            curr_recipient_count = g_recipient_count - l_recipient_count
            cdf_l = build_cdf_r(curr_recipient_count)

            p = curr_recipient_count.astype(np.float64)
            p[provider] = 0.0          
            p /= p.sum()
    
            n = l_recipients.shape[0]
            h = (p ** 2).sum()
            t = (p ** 3).sum()
            closed_mean = n + n * (n - 1) * h
            closed_var = 2 * n * (n - 1) * h * (1 - h) + 4 * n * (n - 1) * (n - 2) * (t - h * h)

            n_sim = 8000
            sample_l = sample_from_cdf_r(np.full(l_recipients.shape, provider), cdf_l, (n_sim, l_recipients.shape[0]), pro_rng)
            sample_sss = np.apply_along_axis(lambda a: ((np.unique(a, return_counts=True)[1]) ** 2).sum(), axis=1, arr=sample_l)
            save_as_npy(f"sample_sss_{s}_{provider}", sample_sss)
            provider_anlysis[provider.item()] = {
                'provider': provider.item(),
                'n': n,
                'c_mean': closed_mean.item(),
                'c_var': closed_var.item(),
            }
    save_as_json('distribution_cl_mv', provider_anlysis)