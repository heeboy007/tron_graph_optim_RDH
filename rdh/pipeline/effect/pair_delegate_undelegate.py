import numpy as np
import pandas as pd

def pair_fifo(df: pd.DataFrame, start_bn: int, end_bn: int) -> pd.DataFrame:
    # 이미 정렬은 되어 있다고 가정함.
    g = df.groupby(["src", "dst", "amt"], sort=False).ngroup().to_numpy()  
    x = df["is_d"].to_numpy(np.int64)
    n = len(x)

    start = np.r_[True, g[1:] != g[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0)) # 각 행이 속한 묶음의 시작 위치
    cs = np.cumsum(x)
    S = cs - (cs[first] - x[first]) # 묶음 안의 누적합

    # 묶음별 누적 최솟값: 묶음 번호가 커질수록 큰 값을 빼서 이전 묶음의 최솟값이 섞이지 않게 함
    BIG = 2 * int(np.abs(S).max()) + 2
    runmin = np.minimum.accumulate(S - g * BIG) + g * BIG
    open_after = S - np.minimum(0, runmin)
    open_before = np.where(start, 0, np.r_[0, open_after[:-1]])

    df["g"], df["open_before"] = g, open_before
    d = df[df["is_d"] == 1].copy()
    u = df[(df["is_d"] == -1) & (df["open_before"] > 0)].copy() # 성공한 회수만
    d["k"] = d.groupby("g").cumcount()
    u["k"] = u.groupby("g").cumcount()
    d["concurrent"] = d["open_before"] > 0

    out = d.merge(u[["g", "k", "bn"]].rename(columns={"bn": "u_bn"}), on=["g", "k"], how="left")
    out = out[(out["bn"] >= start_bn) & (out["bn"] <= end_bn)]
    out["dur"] = out["u_bn"] - out["bn"]
    return out[["src", "dst", "amt", "bn", "u_bn", "dur", "concurrent"]]
