"""
effect size 문턱으로 RDH 컴포넌트 맵을 만드는 공개 API.

- component_map_from_graph: 이미 로드된 ResourceGraph(짝짓기 결과)에서 계산한다.
- component_map_from_events: 정렬된 원시 delegate/undelegate 이벤트에서 짝짓기(pair_fifo)부터 계산한다.
"""
import numpy as np
import pandas as pd

from .graph.component_map_type import ComponentMap, RDHParams
from .graph.resource_graph_type import ResourceGraph
from .graph.transform import paired_delegate_graph
from .models.effect.concentration import effect_concentration
from .models.effect.coverage import effect_coverage
from .models.effect.menu_ratio import effect_menu_ratio
from .models.effect.merge import merge_mask, components
from .pipeline.effect.pair_delegate_undelegate import pair_fifo

def component_map_from_graph(graph: ResourceGraph, start: int, end: int, params: RDHParams | None = None) -> ComponentMap:
    """graph: get_duckdb_delegate_graph_full / paired_delegate_graph의 반환 형식.
    (창 [start, end] 안의 delegate, 압축된 src/dst, bn, dur(open은 NaN), uk_* 필드, decompress_map)"""
    params = params or RDHParams()
    n = graph["g_node_count"]

    conc = effect_concentration(graph)
    cov = effect_coverage(graph, start, end, bucket=params.bucket)
    menu = effect_menu_ratio(graph, flash_max=params.flash_max, rho=params.rho, k=params.k, min_n=params.min_n)
    mu_node = np.full(n, np.nan)
    mu_node[menu["id"]] = menu["mu"]
    mu_pair = mu_node[graph["uk_src"]]

    mask = merge_mask(conc["s_in"], conc["s_out"], cov["active"], mu_pair, params.A, params.mu0, params.tau)
    addr, rep = components(graph["uk_src"], graph["uk_dst"], mask, graph["decompress_map"])

    reps, comp_id, comp_size = np.unique(rep, return_inverse=True, return_counts=True)
    decompress = np.asarray(graph["decompress_map"], dtype=np.int64)
    m_src, m_dst = graph["uk_src"][mask], graph["uk_dst"][mask]
    merged_pairs = pd.DataFrame({
        "src": decompress[m_src], "dst": decompress[m_dst],
        "s_in": conc["s_in"][mask], "s_out": conc["s_out"][mask],
        "active": np.asarray(cov["active"])[mask].astype(np.int64), "mu_P": mu_pair[mask],
    }).sort_values(["src", "dst"], ignore_index=True)

    summary = {
        "merged_pairs": int(mask.sum()),
        "merged_nodes": int(addr.size),
        "components": int(reps.size),
        "max_comp": int(comp_size.max()) if comp_size.size else 0,
        "max_out_deg": int(np.bincount(m_src).max()) if m_src.size else 0,
        "max_in_deg": int(np.bincount(m_dst).max()) if m_dst.size else 0,
    }
    return ComponentMap(
        addr=addr.astype(np.int64), rep=rep.astype(np.int64),
        comp_id=comp_id.astype(np.int64), comp_size=comp_size[comp_id].astype(np.int64),
        merged_pairs=merged_pairs, params=params, start=int(start), end=int(end), summary=summary,
    )

def component_map_from_events(src, dst, amt, bn, is_d, start: int, end: int, params: RDHParams | None = None) -> ComponentMap:
    """원시 ENERGY delegate(is_d = 1) / undelegate(is_d = -1) 이벤트에서 계산한다. src/dst는 원래 주소 id.

    - 입력은 (src, dst, amt, bn, time, tx_id) 순으로 정렬되어 있어야 한다. 이 함수는 (src, dst, amt, bn) 순서만 확인할 수 있고,
      같은 블록 안의 순서(time, tx_id)는 호출하는 쪽이 맞춰야 한다.
    - 짝짓기가 맞으려면 창 앞뒤의 이벤트(warm-up / horizon, 기본 데이터 로더는 각 403,200블록)를 함께 넣어야 한다.
      [start, end] 안의 delegate만 컴포넌트 계산에 쓰인다.
    """
    df = pd.DataFrame({
        "src": np.asarray(src, np.int64), "dst": np.asarray(dst, np.int64),
        "amt": np.asarray(amt, np.int64), "bn": np.asarray(bn, np.int64), "is_d": np.asarray(is_d, np.int64),
    })
    assert np.isin(df["is_d"], (1, -1)).all(), "is_d는 1(delegate) 또는 -1(undelegate)이어야 한다"
    assert (df["src"] != df["dst"]).all(), "자기 자신에게의 위임은 없어야 한다"
    order = np.lexsort((df["bn"], df["amt"], df["dst"], df["src"]))
    assert (order == np.arange(len(df))).all(), "입력이 (src, dst, amt, bn) 순으로 정렬되어 있지 않다"

    paired = pair_fifo(df, start, end)
    return component_map_from_graph(paired_delegate_graph(paired), start, end, params)
