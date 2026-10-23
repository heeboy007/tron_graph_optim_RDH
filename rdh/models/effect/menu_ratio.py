import numpy as np
from typing import TypedDict

from rdh.graph.resource_graph_type import ResourceGraph

class MenuRatio(TypedDict, total=False):
    id: np.ndarray
    n_paired: np.ndarray
    phi: np.ndarray
    m: np.ndarray
    mu: np.ndarray

def menu_ratio(p, dur, flash_max=100, rho=0.05, k=3, min_n=50) -> MenuRatio:
    """p: provider id (정수), dur: 짝지어진 위임의 지속 시간(블록, 정수). open(짝 없음)은 빼고 넣는다.
    반환: provider id, n_paired, phi(flash 비율), m(flash 아닌 짝 수), mu(자기 메뉴 집중도, m<min_n이면 NaN)"""
    p = np.asarray(p)
    dur = np.asarray(dur)
    # NaN(open)이 int64로 바뀌면 조용히 엉뚱한 값이 되므로 미리 막는다
    assert p.shape == dur.shape
    assert not np.isnan(dur.astype(np.float64)).any(), "open(NaN)은 빼고 넣어야 한다"
    assert (dur >= 0).all() and flash_max >= 1
    dur = dur.astype(np.int64)

    ids, pi = np.unique(p, return_inverse=True)
    P = len(ids)

    n_paired = np.bincount(pi, minlength=P)
    is_flash = dur < flash_max
    phi = np.bincount(pi, weights=is_flash, minlength=P) / n_paired

    # flash 아닌 위임: 상대폭 rho의 로그 빈
    nf = ~is_flash
    if not nf.any():
        # flash 아닌 짝이 하나도 없으면 m = 0, mu는 모두 NaN
        return {"id": ids, "n_paired": n_paired, "phi": phi, "m": np.zeros(P, np.int64), "mu": np.full(P, np.nan)}
    b = np.floor(np.log(dur[nf]) / np.log1p(rho)).astype(np.int64)
    b = b - b.min() + 1                      # 1부터 시작 (±1 이웃이 음수가 되지 않게)
    B = int(b.max()) + 2                     # provider당 빈 개수 (+1 이웃 자리 포함)
    key = pi[nf].astype(np.int64) * B + b

    uk, c = np.unique(key, return_counts=True)          # (provider, 빈)별 개수, provider 순 정렬
    prov, bins = uk // B, uk % B
    m = np.bincount(prov, weights=c, minlength=P).astype(np.int64)

    # provider 안에서 개수 내림차순 순위 (동률은 빈 번호가 작은 쪽 우선)
    order = np.lexsort((bins, -c, prov))
    prov_sorted = prov[order]
    first = np.searchsorted(prov_sorted, np.arange(P))  # 각 provider의 시작 위치
    rank = np.arange(len(order)) - first[prov_sorted]
    top_keys = uk[order][rank < k]

    # 상위 k개 빈과 양옆 빈 → 메뉴 집합
    menu_keys = np.unique(np.concatenate([top_keys - 1, top_keys, top_keys + 1]))
    in_menu = np.isin(uk, menu_keys, assume_unique=True)
    covered = np.bincount(prov, weights=c * in_menu, minlength=P)

    with np.errstate(invalid="ignore", divide="ignore"):
        mu = covered / m
    mu[m < min_n] = np.nan
    return {"id": ids, "n_paired": n_paired, "phi": phi, "m": m, "mu": mu}

def effect_menu_ratio(graph: ResourceGraph, **kwargs) -> MenuRatio:
    # dur이 NaN인 open 위임은 빼고 계산한다. id는 압축된 provider id.
    paired = ~np.isnan(graph["dur"])
    return menu_ratio(graph["src"][paired], graph["dur"][paired], **kwargs)
