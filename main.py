from rdh.prob_model.cdf import build_cdf_r, sample_from_cdf_r
from rdh.prob_model.stats import hhi
from dataset.duckdb_parse import get_duckdb_resource_graph
from dataset.util.trace import trace, print_stats, save_stats

if __name__ == "__main__":
    hhi = trace(hhi)

    start, end = 80_000_000, 80_200_000
    graph = get_duckdb_resource_graph(start, end)
    observed_hhi = hhi(graph)
    edges_count = graph["src"].shape[0]

    cdf_r = build_cdf_r(graph)

    sample_recipient = sample_from_cdf_r(size=(edges_count, ), cdf=cdf_r)
    sample_hhi = hhi(graph, override_recipients=sample_recipient)
    
    print(sample_hhi)
    print_stats()
    save_stats()
    
