import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def merge_mask(s_in, s_out, active, mu_pair, A, mu0, tau) -> np.ndarray:
    """쌍별 병합 여부. active > A, (mu_P < mu0 또는 mu_P가 NaN), e = min(s_in, s_out) > tau.
    mu0가 None이면 mu 조건을 쓰지 않고, tau >= 1.0이면 e == 1.0을 요구한다."""
    s_in, s_out = np.asarray(s_in, np.float64), np.asarray(s_out, np.float64)
    active, mu_pair = np.asarray(active), np.asarray(mu_pair, np.float64)
    assert s_in.shape == s_out.shape == active.shape == mu_pair.shape

    mask = active > A
    if mu0 is not None:
        mask &= np.isnan(mu_pair) | (mu_pair < mu0)
    e = np.minimum(s_in, s_out)
    mask &= (e == 1.0) if tau >= 1.0 else (e > tau)
    return mask

def components(uk_src, uk_dst, mask, decompress_map) -> tuple[np.ndarray, np.ndarray]:
    """병합 간선(uk_src[mask] - uk_dst[mask], 압축 id)으로 무방향 연결 컴포넌트를 구한다.
    크기 2 이상 컴포넌트의 원래 주소(addr)와 대표 주소(rep = 컴포넌트 내 최소 주소)를 (rep, addr) 순으로 돌려준다."""
    decompress_map = np.asarray(decompress_map, dtype=np.int64)
    src, dst = np.asarray(uk_src)[mask], np.asarray(uk_dst)[mask]
    if src.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    n = decompress_map.shape[0]
    adj = csr_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)

    nodes = np.unique(np.concatenate((src, dst)))     # 병합 간선에 참여한 노드 = 크기 2 이상 컴포넌트의 노드
    addr = decompress_map[nodes]
    lab = labels[nodes]
    rep_by_label = np.full(labels.max() + 1, np.iinfo(np.int64).max)
    np.minimum.at(rep_by_label, lab, addr)
    rep = rep_by_label[lab]

    order = np.lexsort((addr, rep))
    return addr[order], rep[order]
