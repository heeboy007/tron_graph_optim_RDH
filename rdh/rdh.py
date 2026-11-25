"""RDH merge filters F1 and F3 (manuscript §3.4).

Merge rule (rdh.models.effect.merge): pair (P, r) merges when active > A, (mu_P < mu0 or NaN) and min(s_in, s_out) > tau.
F1: providers flagged by their own delegation profile — transaction-unit type (phi_P >= phi0) or menu type (mu_P >= mu0).
F3: merge rule without the pairs of providers in F1 or in F2 (controller-key filter, rdh.models.filter), for one window.

    pairs = pair_table(graph, start, end)                  # graph: rdh_data_adapter get_duckdb_delegate_graph_full(start, end)
    edges = f3_edges(pairs, f2_of_window)                  # merged (P, r) pairs under F3
    rep = component_rep(edges)                             # (address, representative = smallest address of its component)
    table = ablation_table(graph, start, end, f2_of_window)  # Table 4 rule rows (all pairs of the window)
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .graph.component_map_type import RDHParams
from .graph.resource_graph_type import ResourceGraph
from .models.effect.concentration import effect_concentration
from .models.effect.coverage import effect_coverage
from .models.effect.menu_ratio import effect_menu_ratio

PHI0 = 0.9                 # transaction-unit (flash) type: share of flash delegations >= PHI0
PAIR_ACTIVE_MIN = 3        # pair_table keeps pairs with active > PAIR_ACTIVE_MIN (superset of the merge rule's active > A)


def pair_table(graph: ResourceGraph, start: int, end: int, params: RDHParams | None = None, active_min: int = PAIR_ACTIVE_MIN) -> pd.DataFrame:
    """Pair-level effect sizes of one window (same metric functions and parameters as component_map_from_graph).
    Columns: P_addr, r_addr, n_pair, n_P, n_in, s_in, s_out, active, mu_P, phi_P, m_P; pairs with active > active_min, in (P, r) order."""
    params = params or RDHParams()
    N = graph["g_node_count"]
    conc = effect_concentration(graph)
    cov = effect_coverage(graph, start, end, bucket=params.bucket)
    menu = effect_menu_ratio(graph, flash_max=params.flash_max, rho=params.rho, k=params.k, min_n=params.min_n)
    mu_node = np.full(N, np.nan); phi_node = np.full(N, np.nan); m_node = np.zeros(N, np.int64)
    mu_node[menu["id"]], phi_node[menu["id"]], m_node[menu["id"]] = menu["mu"], menu["phi"], menu["m"]
    n_P = np.bincount(graph["uk_src"], weights=graph["uk_cnt"], minlength=N).astype(np.int64)
    n_in = np.bincount(graph["uk_dst"], weights=graph["uk_cnt"], minlength=N).astype(np.int64)
    active = np.asarray(cov["active"]).astype(np.int64)
    decompress = np.asarray(graph["decompress_map"], dtype=np.int64)
    P, r = graph["uk_src"], graph["uk_dst"]
    keep = active > active_min
    return pd.DataFrame({
        "P_addr": decompress[P[keep]], "r_addr": decompress[r[keep]], "n_pair": graph["uk_cnt"][keep].astype(np.int64),
        "n_P": n_P[P[keep]], "n_in": n_in[r[keep]], "s_in": conc["s_in"][keep], "s_out": conc["s_out"][keep],
        "active": active[keep], "mu_P": mu_node[P[keep]], "phi_P": phi_node[P[keep]], "m_P": m_node[P[keep]],
    })


def base_mask(pairs: pd.DataFrame, params: RDHParams | None = None) -> np.ndarray:
    """Merge rule on a pair table (rdh.models.effect.merge.merge_mask with mu0 set)."""
    p = params or RDHParams()
    mu = pairs["mu_P"].to_numpy()
    return ((pairs["active"].to_numpy() > p.A) & (np.isnan(mu) | (mu < p.mu0))
            & (np.minimum(pairs["s_in"].to_numpy(), pairs["s_out"].to_numpy()) > p.tau))


def f1_mask(pairs: pd.DataFrame, params: RDHParams | None = None, phi0: float = PHI0) -> np.ndarray:
    """Pairs whose provider is in F1 (phi_P >= phi0 or mu_P >= mu0)."""
    p = params or RDHParams()
    return (pairs["phi_P"].to_numpy() >= phi0) | (pairs["mu_P"].to_numpy() >= p.mu0)


def f3_edges(pairs: pd.DataFrame, f2=(), params: RDHParams | None = None, phi0: float = PHI0) -> np.ndarray:
    """Merged (P, r) pairs under F3: merge rule without providers in F1 or in f2 (iterable of provider ids). Array (m, 2) int64."""
    P = pairs["P_addr"].to_numpy(np.int64)
    f2 = np.fromiter(f2, np.int64)
    excl = f1_mask(pairs, params, phi0) | np.isin(P, f2)
    return np.c_[P, pairs["r_addr"].to_numpy(np.int64)][base_mask(pairs, params) & ~excl]


def comp_stats(src: np.ndarray, dst: np.ndarray, n: int) -> dict:
    """Merged pairs (node indices < n) -> pairs, addresses, components (size >= 2), largest component, largest out- / in-degree."""
    if src.size == 0:
        return {"merged_pairs": 0, "merged_nodes": 0, "components": 0, "max_comp": 0, "max_out_deg": 0, "max_in_deg": 0}
    _, lab = connected_components(csr_matrix((np.ones(src.size), (src, dst)), shape=(n, n)), directed=False)
    nodes = np.unique(np.concatenate([src, dst]))
    sizes = np.bincount(lab[nodes]); sizes = sizes[sizes > 0]
    return {"merged_pairs": int(src.size), "merged_nodes": int(nodes.size), "components": int(sizes.size), "max_comp": int(sizes.max()),
            "max_out_deg": int(np.bincount(src).max()), "max_in_deg": int(np.bincount(dst).max())}


def ablation_table(graph: ResourceGraph, start: int, end: int, f2=(), params: RDHParams | None = None, phi0: float = PHI0,
                   f2_label: str = "F2") -> pd.DataFrame:
    """Merge-rule ablation of one window over every delegation pair (manuscript Table 4 without the HHI row).
    Cumulative rows remove, in order, the provider filter (F1 and f2), μ, α, χ_out and χ_in; leave-one-out rows drop one part of RDH (F3).
    Columns: kind, rule, merged_pairs, merged_nodes, components, max_comp, max_out_deg, max_in_deg, providers."""
    p = params or RDHParams()
    N = graph["g_node_count"]
    conc = effect_concentration(graph)
    cov = effect_coverage(graph, start, end, bucket=p.bucket)
    menu = effect_menu_ratio(graph, flash_max=p.flash_max, rho=p.rho, k=p.k, min_n=p.min_n)
    mu_node = np.full(N, np.nan); phi_node = np.full(N, np.nan)
    mu_node[menu["id"]], phi_node[menu["id"]] = menu["mu"], menu["phi"]
    P, R = graph["uk_src"].astype(np.int64), graph["uk_dst"].astype(np.int64)
    dec = np.asarray(graph["decompress_map"], np.int64)
    mu = mu_node[P]
    f1 = ((phi_node >= phi0) | (mu_node >= p.mu0))[P]
    f2 = np.isin(dec, np.fromiter(f2, np.int64))[P]
    a_ok, mu_ok = np.asarray(cov["active"]) > p.A, np.isnan(mu) | (mu < p.mu0)
    so_ok, si_ok = conc["s_out"] > p.tau, conc["s_in"] > p.tau
    pf = ~(f1 | f2)
    rows = [("cumulative", "RDH (F3)", a_ok & mu_ok & so_ok & si_ok & pf),
            ("cumulative", "− provider filter (F0)", a_ok & mu_ok & so_ok & si_ok),
            ("cumulative", "− μ", a_ok & so_ok & si_ok),
            ("cumulative", "− α", so_ok & si_ok),
            ("cumulative", "− χ_out", si_ok),
            ("cumulative", "− χ_in (no rules)", np.ones(P.size, bool)),
            ("leave-one-out", "RDH − F1 only", a_ok & mu_ok & so_ok & si_ok & ~f2),
            ("leave-one-out", f"RDH − {f2_label} only", a_ok & mu_ok & so_ok & si_ok & ~f1),
            ("leave-one-out", "RDH − μ only", a_ok & so_ok & si_ok & pf),
            ("leave-one-out", "RDH − α only", mu_ok & so_ok & si_ok & pf),
            ("leave-one-out", "RDH − χ_out only", a_ok & mu_ok & si_ok & pf),
            ("leave-one-out", "RDH − χ_in only", a_ok & mu_ok & so_ok & pf)]
    return pd.DataFrame([{"kind": kind, "rule": rule, **comp_stats(P[m], R[m], N), "providers": int(np.unique(P[m]).size)}
                         for kind, rule, m in rows])


def component_rep(edges: np.ndarray) -> np.ndarray:
    """Merged edges -> (address, representative = smallest address in the component) for components of size >= 2."""
    if len(edges) == 0:
        return np.empty((0, 2), np.int64)
    nodes, inv = np.unique(np.concatenate([edges[:, 0], edges[:, 1]]), return_inverse=True)
    m = len(edges)
    _, lab = connected_components(csr_matrix((np.ones(m), (inv[:m], inv[m:])), shape=(nodes.size, nodes.size)), directed=False)
    return np.c_[nodes, pd.Series(nodes).groupby(lab).transform("min").to_numpy(np.int64)]
