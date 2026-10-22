import numpy as np
from typing import TypedDict

from rdh.graph.resource_graph_type import ResourceGraph

class Concentration(TypedDict, total=False):
    s_out: np.ndarray
    s_in: np.ndarray

def effect_concentration(graph: ResourceGraph) -> Concentration:
    provider_events = np.bincount(graph["uk_src"], weights=graph["uk_cnt"], minlength=graph["g_node_count"])
    recipient_events = np.bincount(graph["uk_dst"], weights=graph["uk_cnt"], minlength=graph["g_node_count"])

    s_out = graph["uk_cnt"] / provider_events[graph["uk_src"]]
    s_in = graph["uk_cnt"] / recipient_events[graph["uk_dst"]]

    assert ((s_in > 0) & (s_in <= 1)).all() and ((s_out > 0) & (s_out <= 1)).all()

    return {
        's_out': s_out,
        's_in': s_in
    }
