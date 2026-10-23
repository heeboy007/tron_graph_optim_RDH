from rdh import ComponentMap, RDHParams, component_map_from_graph
from rdh_data_adapter.load.resource_graph import get_duckdb_delegate_graph_full

def build_component_map(start: int, end: int, params: RDHParams | None = None, db_path=None) -> ComponentMap:
    #just a quick function
    graph = get_duckdb_delegate_graph_full(start, end, db_path=db_path)
    return component_map_from_graph(graph, start, end, params)
