"""F2: providers that share a controller key (manuscript §3.4, epoch_chain_causal).

Population of window w (blocks [w·wsz, (w+1)·wsz)): every account with at least one successful ENERGY delegation in [scan_lo, end of w].
Keys of an account in window w (records with block in [scan_lo, scan_hi)):
  - permission updates (type 46; external keys of every active permission, plus the owner permission for the "causal_owner" mode):
    the state of the last update before the window start, plus every update inside the window;
  - no update before the window: an account created (type 0) before the window end has no key; otherwise the keys that signed a
    successful delegate / undelegate of the account (signing evidence) before min(first update, window end + 1).
Key scale in window w = number of population accounts of w holding the key. F2(c) of window w = population accounts holding a key of
scale >= c.

Causality: every key decision is recomputed with every record after the window end removed and must be equal (look-ahead check).
The "retro" mode (sensitivity: signing evidence of the whole first epoch, creation at any block) has no look-ahead check.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MODES = {"causal": "epoch_chain_causal", "causal_owner": "epoch_chain_causal_owner", "retro": "epoch_chain"}


@dataclass
class ControllerRecords:
    """Records of the population accounts (address ids). Build with rdh_data_adapter.load.permission (see from_frames)."""
    fd: dict                  # account -> first successful ENERGY delegation block
    hist: dict                # account -> [(block, frozenset active keys)] successful permission updates, in order
    ohist: dict               # account -> [(block, frozenset owner keys)]
    born: dict                # account -> creation block
    ev: dict                  # account -> [(key, first signing block)]
    scan_hi: int              # records end (exclusive); signing evidence of accounts without an update stops here
    wsz: int = 200_000
    accounts: list = field(default_factory=list)
    first_upd: dict = field(default_factory=dict)
    n_checks: int = 0
    n_violations: int = 0
    max_used: dict = field(default_factory=dict)

    def __post_init__(self):
        self.accounts = sorted(self.fd)
        self.first_upd = {a: h[0][0] for a, h in self.hist.items()}

    @classmethod
    def from_frames(cls, fd: pd.DataFrame, active: dict, owner: dict, born: dict, evidence: pd.DataFrame, scan_hi: int, wsz: int = 200_000):
        """fd: columns a, fd; active / owner: {account: [(block, keys)]}; born: {account: block}; evidence: columns P, s, bmin."""
        ev = defaultdict(list)
        for P, s, b in zip(evidence["P"].to_numpy(), evidence["s"].to_numpy(), evidence["bmin"].to_numpy()):
            ev[int(P)].append((int(s), int(b)))
        return cls(fd=dict(zip(fd["a"], fd["fd"])), hist=active, ohist=owner, born=born, ev=dict(ev), scan_hi=scan_hi, wsz=wsz)


def window_state(w, hist, born_a, wsz):
    lo, hi = w * wsz, (w + 1) * wsz
    prev = [s for b, s in hist if b < lo]
    inside = [s for b, s in hist if lo <= b < hi]
    if prev:
        src, sa = "update", prev[-1]
    elif born_a is not None and born_a < lo:
        src, sa = "created", frozenset()
    else:
        src, sa = "unknown", frozenset()
    return src, sa, frozenset().union(sa, *inside)


def decide_keys(w, hist, ohist, ev, born_a, first_upd, mode, wsz, scan_hi):
    """Keys of one account in window w from the given records; returns (keys, largest record block used)."""
    end = (w + 1) * wsz - 1
    src, _, sb = window_state(w, hist, born_a, wsz)
    used = [b for b, _ in hist if b <= end]
    if src == "unknown":
        if born_a is not None and (mode == "retro" or born_a <= end):
            fk = frozenset(); used.append(born_a)
        else:
            cut = first_upd if first_upd is not None else scan_hi
            lim = cut if mode == "retro" else min(cut, end + 1)
            sel = [(k, b) for k, b in ev if b < lim]
            fk = frozenset(k for k, _ in sel); used += [b for _, b in sel]
        k = fk | sb
    else:
        k = sb
        if src == "created":
            used.append(born_a)
    if mode == "causal_owner":
        k = k | window_state(w, ohist, born_a, wsz)[2]
    return k, (max(used) if used else None)


def window_keys(R: ControllerRecords, a, w, mode):
    end = (w + 1) * R.wsz - 1
    hist, ohist, ev, born_a, fu = R.hist.get(a, []), R.ohist.get(a, []), R.ev.get(a, []), R.born.get(a), R.first_upd.get(a)
    r, mb = decide_keys(w, hist, ohist, ev, born_a, fu, mode, R.wsz, R.scan_hi)
    if mode == "retro":
        return r
    # look-ahead check: drop every record after the window end (updates, evidence, creation, first update) and decide again
    r2, _ = decide_keys(w, [x for x in hist if x[0] <= end], [x for x in ohist if x[0] <= end], [x for x in ev if x[1] <= end],
                        born_a if born_a is not None and born_a <= end else None, fu if fu is not None and fu <= end else None, mode,
                        R.wsz, R.scan_hi)
    R.n_checks += 1
    if r2 != r:
        R.n_violations += 1
    assert r2 == r and (mb is None or mb <= end), ("look-ahead", a, w, mode)
    R.max_used[w] = max(R.max_used.get(w, 0), mb or 0)
    return r


def f2_sets(R: ControllerRecords, wins, mode="causal", cs=range(2, 11)):
    """F2(c) for each window and c. Returns (res {w: {c: set of accounts}}, pairs frame (w, P, s) of the account keys, population size per w)."""
    res, rows, pop_n = {}, [], {}
    for w in wins:
        end = (w + 1) * R.wsz - 1
        members = [a for a in R.accounts if R.fd[a] <= end]
        pop_n[w] = len(members)
        # accounts without update history and without signing evidence have no key in any window (skipped; not counted as checks)
        st = {a: window_keys(R, a, w, mode) for a in members if a in R.hist or a in R.ev}
        st = {a: k for a, k in st.items() if k}
        n = defaultdict(int)
        for ks in st.values():
            for k in ks:
                n[k] += 1
        res[w] = {}
        for c in cs:
            big = {k for k, x in n.items() if x >= c}
            res[w][c] = {a for a, ks in st.items() if ks & big}
        x = np.array([(a, k) for a, ks in st.items() for k in ks], dtype=np.int64).reshape(-1, 2)
        rows.append(pd.DataFrame({"w": np.full(len(x), w, np.int32), "P": x[:, 0], "s": x[:, 1]}))
    pairs = pd.concat(rows, ignore_index=True).sort_values(["w", "P", "s"]).reset_index(drop=True)
    return res, pairs, pop_n


def f2_frame(res) -> pd.DataFrame:
    """{w: {c: set}} -> frame (c, w, P) ordered by w, c, P."""
    return pd.DataFrame([{"c": c, "w": w, "P": p} for w in sorted(res) for c in res[w] for p in sorted(res[w][c])],
                        columns=["c", "w", "P"]).astype({"c": np.int8, "w": np.int16, "P": np.int64})


def lookahead_summary(R: ControllerRecords) -> dict:
    return {"lookahead_checks": R.n_checks, "lookahead_violations": R.n_violations,
            "lookahead_max_block_minus_window_end_max": int(max(R.max_used[w] - ((w + 1) * R.wsz - 1) for w in R.max_used)) if R.max_used else None}
