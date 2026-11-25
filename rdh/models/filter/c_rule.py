"""Decision rule of the F2 scale threshold c (manuscript §3.4).

Per key-size bin (number of accounts holding the key: 1, 2, 3–4, 5–9, 10–49, 50+), median over the bin's accounts of the account's median
weekly ENERGY delegation count (development windows, windows with > 0 delegations). c = lower edge of the bin after the largest adjacent
ratio (first maximum). Two key-size bases: pooled (accounts per key over all development windows) and window median (median over windows
of the accounts per key); when they differ the window median is used.
"""
import numpy as np
import pandas as pd

SIZE_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"), (10, 49, "10–49"), (50, 10 ** 9, "50+")]


def c_rule(pairs: pd.DataFrame, dp: pd.DataFrame, wins):
    """pairs: (w, P, s) account keys (rdh.models.filter.controller.f2_sets); dp: (w, P, n_d_energy) per-window delegation counts.
    Returns (table rows per basis, {basis: c})."""
    spd = pairs[pairs["w"].isin(wins)]
    dpd = dp[dp["w"].isin(wins) & (dp["n_d_energy"] > 0)]
    dmd = dpd.groupby("P")["n_d_energy"].median()
    ced = spd[["P", "s"]].drop_duplicates()
    rows, cs = [], {}
    for basis, npv in (("pooled", ced.groupby("s")["P"].nunique()), ("window median", spd.groupby(["w", "s"])["P"].nunique().groupby("s").median())):
        meds = []
        for lo, hi, lab in SIZE_BINS:
            ks = npv[(npv >= lo) & (npv <= hi)].index
            ps = set(ced.loc[ced["s"].isin(ks), "P"])
            meds.append((lab, lo, len(ks), len(ps), float(dmd.reindex(list(ps)).dropna().median())))
        ratios = [meds[i + 1][4] / meds[i][4] for i in range(len(meds) - 1)]
        ib = int(np.nanargmax(ratios))
        cs[basis] = int(meds[ib + 1][1])
        rows.append({"basis": basis, "bins": [{"bin": l, "keys": nk, "accounts": na, "median_weekly_delegations": m} for l, _, nk, na, m in meds],
                     "adjacent_ratios": ratios, "largest_ratio_between": [meds[ib][0], meds[ib + 1][0]], "c": cs[basis]})
    return rows, cs


def decide_c(cs: dict) -> int:
    return cs["window median"] if cs["pooled"] != cs["window median"] else cs["pooled"]
