import numpy as np

from ..graph.resource_graph_type import ResourceGraph, SplitGraph

def split_dataset(graph: ResourceGraph, simplify_threshold: int = 5, analytic_threshold: int = 20000) -> SplitGraph:
    _, inv, unique_provider_counts = np.unique(graph["src"], return_inverse=True, return_counts=True)
    statistically_unreliable = (unique_provider_counts <= simplify_threshold)[inv]
    analytic_analysis = (unique_provider_counts > analytic_threshold)[inv]
    monte_carlo_analysis = ~(statistically_unreliable | analytic_analysis)

    return {
        'strong': { 
            'src': graph["src"][statistically_unreliable],
            'dst': graph["dst"][statistically_unreliable]
        }, 
        'mc': { 
            'src': graph["src"][monte_carlo_analysis],
            'dst': graph["dst"][monte_carlo_analysis]
        },
        'analytic': {
            'src': graph["src"][analytic_analysis],
            'dst': graph["dst"][analytic_analysis]
        }
    }

