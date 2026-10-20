
import numpy as np

from ..graph.resource_graph_type import SimpleGraph

def rdh_strong_edges(graph: SimpleGraph) -> SimpleGraph:
    key = graph["src"].astype(np.int64) * graph['node_count'] + graph["dst"].astype(np.int64)
    unique_keys = np.unique(key)

    #re-extract 
    unique_keys_provider_map = unique_keys // graph['node_count']
    unique_keys_recipient_map = unique_keys % graph['node_count']
    
    diff = unique_keys_provider_map[:-1] != unique_keys_provider_map[1:]
    # if this is true, there is only one provider-recipient match.
    is_single = np.concatenate(([True], diff)) & np.concatenate((diff, [True]))

    return {
        'src': unique_keys_provider_map[is_single][::-1],
        'dst': unique_keys_recipient_map[is_single][::-1],
        'node_count': graph['node_count']
    }
    