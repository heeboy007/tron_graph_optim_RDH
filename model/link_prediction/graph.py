"""Sparse pair kernels (numba) and transfer-edge loading of the link-prediction stages."""
import numba as nb
import numpy as np

WSZ = 200_000


@nb.njit(parallel=True, cache=True)
def pair_inter(ptr, idx, a, b, w):
    """Intersection size of sorted CSR rows a[i], b[i] and the sum of w over the common elements."""
    n = a.size
    cnt = np.zeros(n, np.int64); ws = np.zeros(n)
    for i in nb.prange(n):
        p, pe = ptr[a[i]], ptr[a[i] + 1]; q, qe = ptr[b[i]], ptr[b[i] + 1]
        c = 0; s = 0.0
        while p < pe and q < qe:
            x, y = idx[p], idx[q]
            if x == y:
                c += 1; s += w[x]; p += 1; q += 1
            elif x < y:
                p += 1
            else:
                q += 1
        cnt[i] = c; ws[i] = s
    return cnt, ws


@nb.njit(parallel=True, cache=True)
def pair_has(ptr, idx, a, b):
    n = a.size
    out = np.zeros(n, np.bool_)
    for i in nb.prange(n):
        lo, hi = ptr[a[i]], ptr[a[i] + 1]
        while lo < hi:
            mid = (lo + hi) // 2
            if idx[mid] < b[i]:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo < ptr[a[i] + 1] and idx[lo] == b[i]
    return out


@nb.njit(parallel=True, cache=True)
def pair_sponsor(ptr, idx, a, b, w, dP):
    """Shared sponsors: count, Adamic–Adar sum, smallest d_P, code of the sponsor with the smallest d_P (-1 = none)."""
    n = a.size
    cnt = np.zeros(n, np.int64); aa = np.zeros(n); mind = np.zeros(n); top = np.full(n, -1, np.int64)
    for i in nb.prange(n):
        p, pe = ptr[a[i]], ptr[a[i] + 1]; q, qe = ptr[b[i]], ptr[b[i] + 1]
        c = 0; s = 0.0; md = 1e18; t = -1
        while p < pe and q < qe:
            x, y = idx[p], idx[q]
            if x == y:
                c += 1; s += w[x]
                if dP[x] < md:
                    md = dP[x]; t = x
                p += 1; q += 1
            elif x < y:
                p += 1
            else:
                q += 1
        cnt[i] = c; aa[i] = s; top[i] = t; mind[i] = md if c > 0 else 0.0
    return cnt, aa, mind, top


def csr(u, v, n):
    k = np.unique(u.astype(np.int64) * n + v.astype(np.int64))
    uu, vv = k // n, k % n
    return np.searchsorted(uu, np.arange(n + 1)).astype(np.int64), vv.astype(np.int64)


def load_edges(con, k):
    """Successful USDT ``transfer`` rows (from, to) of window k = blocks [k·200,000, (k+1)·200,000)."""
    lo, hi = k * WSZ, (k + 1) * WSZ - 1
    return con.execute(f"SELECT from_addr AS f, to_addr AS t FROM usdt_tx WHERE block_number BETWEEN {lo} AND {hi} AND success AND type = 'transfer'").df()
