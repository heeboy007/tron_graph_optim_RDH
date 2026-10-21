import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

from rdh.graph.resource_graph_type import SimpleGraph

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