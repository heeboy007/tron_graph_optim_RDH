import numpy as np
import pandas as pd

def pair_fifo_arrays(src, dst, amt, bn, is_d, start_bn: int, end_bn: int):
    """R1 FIFO 짝짓기 (같은 src, dst, amt 묶음 안에서 시간순).
    입력은 (src, dst, amt, bn, time, tx_id) 순으로 정렬된 배열이다 (is_d: delegate 1 / undelegate -1).
    반환: 창 [start_bn, end_bn] 안 delegate의 (index, u_bn, concurrent). u_bn은 짝이 없으면 -1.

    묶음 안 누적합 S에 대해 열린 위임 수 open = S - min(0, 누적 최솟값)이고,
    직전 open > 0인 undelegate만 회수에 성공한다. FIFO이므로 묶음 안 k번째 성공 회수가 k번째 delegate의 짝이다."""
    n = len(src)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, bool)

    # 묶음 시작 위치와 묶음 번호 (정렬되어 있으므로 인접 원소 비교로 충분)
    start = np.empty(n, bool)
    start[0] = True
    np.not_equal(src[1:], src[:-1], out=start[1:])
    start[1:] |= dst[1:] != dst[:-1]
    start[1:] |= amt[1:] != amt[:-1]
    g = np.cumsum(start, dtype=np.int64) - 1
    first = np.flatnonzero(start)                     # 묶음 번호 -> 첫 원소 위치

    x = np.asarray(is_d, dtype=np.int64)
    S = np.cumsum(x)
    S -= (S - x)[first][g]                            # 묶음 안의 누적합

    # 묶음별 누적 최솟값: 묶음 번호가 커질수록 큰 값을 빼서 이전 묶음의 최솟값이 섞이지 않게 함
    big = 2 * int(np.abs(S).max()) + 2
    shift = g * big
    runmin = np.minimum.accumulate(S - shift)
    runmin += shift
    del shift
    np.minimum(runmin, 0, out=runmin)
    S -= runmin                                       # S는 이제 open_after
    del runmin
    open_before = np.empty(n, np.int64)
    open_before[0] = 0
    open_before[1:] = S[:-1]
    open_before[start] = 0
    del S, start

    d_idx = np.flatnonzero(x == 1)
    u_idx = np.flatnonzero((x == -1) & (open_before > 0))   # 성공한 회수만
    del x

    # 묶음 안 순번: 정렬된 묶음 번호에서 그 묶음이 처음 나온 위치를 뺀다
    def rank_in_group(idx):
        gi = g[idx]
        return gi, np.arange(idx.size) - np.searchsorted(gi, gi)

    dg, dk = rank_in_group(d_idx)
    in_window = (bn[d_idx] >= start_bn) & (bn[d_idx] <= end_bn)
    d_idx, dg, dk = d_idx[in_window], dg[in_window], dk[in_window]
    concurrent = open_before[d_idx] > 0
    del open_before

    ug, uk = rank_in_group(u_idx)
    del g
    # (묶음 번호, 순번)이 같은 성공 회수를 찾는다. 두 키 모두 정렬되어 있다.
    ukey = ug * (n + 1) + uk
    dkey = dg * (n + 1) + dk
    u_bn = np.full(d_idx.size, -1, np.int64)
    if ukey.size:
        pos = np.minimum(np.searchsorted(ukey, dkey), ukey.size - 1)
        hit = ukey[pos] == dkey
        u_bn[hit] = np.asarray(bn)[u_idx[pos[hit]]]
    return d_idx, u_bn, concurrent

def pair_fifo(df: pd.DataFrame, start_bn: int, end_bn: int) -> pd.DataFrame:
    # 이미 (src, dst, amt, bn, time, tx_id) 순으로 정렬은 되어 있다고 가정함.
    cols = {c: df[c].to_numpy() for c in ("src", "dst", "amt", "bn", "is_d")}
    d_idx, u_bn, concurrent = pair_fifo_arrays(**cols, start_bn=start_bn, end_bn=end_bn)
    bn = cols["bn"][d_idx]
    u_bn = np.where(u_bn < 0, np.nan, u_bn)
    return pd.DataFrame({
        "src": cols["src"][d_idx], "dst": cols["dst"][d_idx], "amt": cols["amt"][d_idx], "bn": bn,
        "u_bn": u_bn, "dur": u_bn - bn, "concurrent": concurrent,
    })
