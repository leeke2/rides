import glob

import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import pandas as pd

df = pd.concat(
    pd.read_json(fn) for fn in glob.glob("results/results_ts_E30_R1_*_C*.json")
)

mpl_style.use("pub.mplstyle")

fig = plt.figure(figsize=(8.94, 7))
ax = fig.gca()

(
    df.groupby(["demand_factor", "seed"])
    .agg({"objective": "mean", "initial_max_load_factor": "first"})
    .assign(initial_max_load_factor=lambda x: x.initial_max_load_factor / 100)
    .reset_index()
    .plot.scatter(
        x="initial_max_load_factor",
        y="objective",
        marker="x",
        s=50,
        linewidth=1,
        ax=ax,
        c="k",
        alpha=0.7,
    )
)

ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-0.2, 2.2)
ax.set_ylabel("Objective")
ax.set_xlabel("Max load factor (without LSS)")

plt.tight_layout()
fig.savefig("figures/fig_a8.pdf")
