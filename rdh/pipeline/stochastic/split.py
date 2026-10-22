import numpy as np

from ...graph.resource_graph_type import ResourceGraph, SplitGraph, SimpleGraph

def split_dataset(graph: ResourceGraph, simplify_threshold: int = 5, analytic_threshold: int = 20000) -> SplitGraph:
    _, inv, unique_provider_counts = np.unique(graph["src"], return_inverse=True, return_counts=True)
    statistically_unreliable = (unique_provider_counts <= simplify_threshold)[inv]
    analytic_analysis = (unique_provider_counts > analytic_threshold)[inv]
    monte_carlo_analysis = ~(statistically_unreliable | analytic_analysis)

    return {
        'strong': { 
            'src': graph["src"][statistically_unreliable],
            'dst': graph["dst"][statistically_unreliable],
            'g_node_count': graph["g_node_count"],
            'l_edge_count': statistically_unreliable.sum()
        }, 
        'mc': { 
            'src': graph["src"][monte_carlo_analysis],
            'dst': graph["dst"][monte_carlo_analysis],
            'g_node_count': graph["g_node_count"],
            'l_edge_count': monte_carlo_analysis.sum()
        },
        'analytic': {
            'src': graph["src"][analytic_analysis],
            'dst': graph["dst"][analytic_analysis],
            'g_node_count': graph["g_node_count"],
            'l_edge_count': analytic_analysis.sum()
        }
    }

def split_dataset_range(graph: ResourceGraph, split_range: tuple[int]) -> dict[int, SimpleGraph]:
    _, inv, unique_provider_counts = np.unique(graph["src"], return_inverse=True, return_counts=True)
    range_bool_maps = [(t < unique_provider_counts)[inv] for t in (0, *split_range)]
    dataset = {}
    for idx, r_low in enumerate((0, *split_range)):
        if idx == len(split_range): #last one, should be treated as inf
            boolean_index = range_bool_maps[idx]
        else:
            boolean_index = range_bool_maps[idx] ^ range_bool_maps[idx + 1]
        
        dataset[r_low] = {
            'src': graph["src"][boolean_index],
            'dst': graph["dst"][boolean_index],
            'g_node_count': graph["g_node_count"],
            'l_edge_count': boolean_index.sum()
        }
    return dataset

