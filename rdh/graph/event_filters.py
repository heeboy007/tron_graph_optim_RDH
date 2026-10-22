
import numpy as np

from .resource_graph_type import ResourceGraph, SimpleGraph

def detect_single_providers(graph: ResourceGraph) -> SimpleGraph:
    # if this is true, there is only one provider-recipient match.
    diff = graph['uk_src'][:-1] != graph['uk_src'][1:]
    is_single = np.concatenate(([True], diff)) & np.concatenate((diff, [True]))

    return {
        'src': graph['uk_src'][is_single][::-1],
        'dst': graph['uk_dst'][is_single][::-1],
        'g_node_count': graph['g_node_count'],
        'l_edge_count': is_single.sum()
    }