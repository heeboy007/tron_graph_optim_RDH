"""Link prediction (manuscript §3.7, §6.1): does RDH structure (F3 components) add to a transfer-graph model G?

    features   candidate pairs, transfer-graph features T, sponsorship features S, ranking queries, query metrics
    graph      sparse pair kernels (numba) and USDT transfer edges of a window
    ranking    G, residual boosting, placebo, early stopping
    pipeline   query caches, hub-capped labels, one repetition (Store = folder layout of intermediate files)
"""
