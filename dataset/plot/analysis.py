import numpy as np

from rdh.graph.resource_graph_type import SimpleGraph
from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r

def provider_mean_variance_closed_form_check(graph: SimpleGraph, g_recipient_count: np.ndarray):
    # is sorted
    analytic_providers = np.unique(graph["src"], sorted=True)
    provider_anlysis = {}
    for provider in analytic_providers:
        rng = np.random.default_rng([42, provider])
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