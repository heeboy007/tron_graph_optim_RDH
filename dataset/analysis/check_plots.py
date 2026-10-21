import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

from rdh.graph.resource_graph_type import SimpleGraph

def prep(r):
    n = r["n"]
    U = (np.asarray(r["sss"]) - n) / 2
    mu, sd = (r["c_mean"] - n) / 2, np.sqrt(r["c_var"]) / 2
    return U, mu, sd, (U - mu) / sd

def tail_curve_gamma(U, mu, sd, min_exceed=20):
    B = len(U); k_, th = mu**2 / sd**2, sd**2 / mu          # 감마: 평균 mu, 분산 sd^2에 맞춤
    us = np.unique(np.ceil(mu + np.linspace(0.5, 3.6, 40) * sd))
    xs, rN, rG = [], [], []
    for u in us:
        k = (U >= u).sum()
        if k < min_exceed: break
        emp = k / B
        xs.append((u - mu) / sd)
        rN.append(emp / stats.norm.sf((u - 0.5 - mu) / sd))
        rG.append(emp / stats.gamma.sf(u - 0.5, a=k_, scale=th))
    return np.array(xs), np.array(rN), np.array(rG)

def zhist(U, mu, sd, ax):
    edges = np.arange(np.floor(U.min()) - 0.5, np.ceil(U.max()) + 1.5, 1.0) if sd < 30 else np.linspace(U.min(), U.max(), 61)
    dens, e = np.histogram(U, bins=edges, density=True)
    ax.step((((e[:-1] + e[1:]) / 2) - mu) / sd, dens * sd, where="mid")
    x = np.linspace(-4, 6, 400); ax.plot(x, stats.norm.pdf(x), "k", lw=1)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)

def z_mean_var_qq_plots(provider_anlysis: dict, graph: SimpleGraph):
    df = pd.DataFrame(provider_anlysis).T.astype(float)
    df["z_variance"] = df["rel_var"] / df["se_rel_var"]
    df["share"] = df["n"] / graph["src"].shape[0]          # 전체 이벤트 수는 직접 채우세요

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for i, c in enumerate(["z_mean", "z_variance"]):
        sc = ax[i,0].scatter(df["n"], df[c], c=np.log10(df["share"]), s=8)
        ax[i,0].set_xscale("log")
        ax[i,0].axhline(0, c="k")
        for y in (-3, -2, 2, 3): 
            ax[i,0].axhline(y, ls="--", c="gray")
        ax[i,0].set_title(c) 
        fig.colorbar(sc, ax=ax[i,0], label="log10 share")
        #PP plots line of y=x, set plot range to -3~3 -> just make them QQ-plots
        stats.probplot(df[c], dist="norm", plot=ax[i,1])
        ax[i,1].axline((0, 0), slope=1.0)
        ax[i,1].set_xlim(left=-3, right=3)
        ax[i,1].set_ylim(bottom=-3, top=3)
        ax[i,1].set_title(f"Normal Q-Q plot of {c}") 
    
    fig.tight_layout()

    ax[0,2].errorbar(df["n"], df["rel_var"], yerr=2*df["se_rel_var"], fmt=".", alpha=.5)
    ax[0,2].set_xscale("log")
    ax[0,2].axhline(0, c="k") 
    ax[0,2].set_title("rel_var ± 2se")
    w = 1/df["se_rel_var"]**2
    print("mean z_mean", df.z_mean.mean(), "±", 1/np.sqrt(len(df)))
    print("weighted rel_var", (w*df.rel_var).sum()/w.sum(), "±", 1/np.sqrt(w.sum()))
    plt.show()

def dist_comparison_plots(records: dict):
    rows = []
    for r in records.values():
        U, mu, sd, z = prep(r)
        rows.append(dict(split=r["split_idx"], n=r["n"], provider=r["provider"], mu_U=mu, sd_U=sd,
                        mean_z=z.mean(), var_z=z.var(ddof=1), skew=stats.skew(z), exkurt=stats.kurtosis(z),
                        f2=(z >= 2).mean(), f3=(z >= 3).mean()))
    df = pd.DataFrame(rows).sort_values("n")
    print(df.round(4).to_string(index=False))

    splits = sorted(df["split"].unique())

    # ---- 그림 1: 구간별 최소 n provider의 표준화 히스토그램 ----
    fig, axs = plt.subplots(3, 3, figsize=(14, 11)); axs = axs.ravel()
    for ax, s in zip(axs, splits):
        r = min((r for r in records.values() if r["split_idx"] == s), key=lambda r: r["n"])
        U, mu, sd, _ = prep(r); zhist(U, mu, sd, ax); ax.set_title(f"split {s}  n={r['n']}  sd_U={sd:.1f}")
    fig.tight_layout()

    # ---- 그림 2: 꼬리 비율 (구간별 색) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(splits)))
    for r in records.values():
        x, rN, rG = tail_curve_gamma(*prep(r)[:3])
        if len(x):
            c = cmap[splits.index(r["split_idx"])]
            ax[0].plot(x, rN, color=c, alpha=.7); ax[1].plot(x, rG, color=c, alpha=.7)
    for a, t in zip(ax, ["normal", "gamma (mean/var matched)"]):
        a.axhline(1, c="k"); a.axhline(2, c="r", ls="--"); a.set_yscale("log"); a.set_title(t); a.set_xlabel("z")
    fig.tight_layout()

    # ---- 그림 3: 왜도·초과 첨도 vs n ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, c, band in zip(ax, ["skew", "exkurt"], [0.027, 0.055]):
        a.scatter(df["n"], df[c], c=np.log10(df["n"]), s=14); a.set_xscale("log")
        a.axhline(0, c="k"); a.axhspan(-2*band, 2*band, color="gray", alpha=.2); a.set_title(c)
    fig.tight_layout()

    plt.show()
