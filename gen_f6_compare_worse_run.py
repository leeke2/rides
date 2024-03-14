import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import pandas as pd

rl = pd.read_json("results/results_rl_E30_R1_DF1.0_NC_20240304014029.json")
ga = pd.read_json("results/results_ga_E30_R10_DF1.0_NC_20240304014633.json")
ts = pd.read_json("results/results_ts_E30_R10_DF1.0_NC_20240304014246.json")

rl.loc[rl.objective > 1, "objective"] = 1.0
ga.loc[ga.objective > 1, "objective"] = 1.0
ts.loc[ts.objective > 1, "objective"] = 1.0

sorted_indices = (
    rl.groupby("seed")
    .objective.mean()
    .sort_values(ascending=False)
    .reset_index()
    .assign(
        sorted_idx=lambda x: range(len(x)),
        position=lambda x: (x.sorted_idx * 3) * 0.75,
    )
    .rename(columns={"objective": "objective_rl"})
)

mpl_style.use("pub.mplstyle")

data_ga = (
    ga.groupby("seed")
    .agg({"objective": "max"})
    .merge(sorted_indices, left_on="seed", right_on="seed")
    .assign(objective_diff=lambda x: -x.objective_rl + x.objective)
    .sort_values(by="sorted_idx")
)

data_ts = (
    ts.groupby("seed")
    .agg({"objective": "max"})
    .merge(sorted_indices, left_on="seed", right_on="seed")
    .assign(objective_diff=lambda x: -x.objective_rl + x.objective)
    .sort_values(by="sorted_idx")
)

fig = plt.figure(figsize=(8.94, 5.34))
ax = fig.gca()
ax.bar(
    range(30),
    data_ts.objective_diff,
    width=0.5,
    bottom=data_ts.objective_rl,
    color=["g" if d > 0 else "r" for d in data_ts.objective_diff],
)

ax.scatter(
    range(30), sorted_indices.objective_rl, marker="x", color="k", s=50, linewidth=1
)

ax.set_ylim((0.775, 1.025))
ax.set_xlim((-0.5, 30.5))
ax.set_xticks(range(30))
ax.set_xlabel("Problem instances")
ax.set_ylabel("Objective")

plt.tight_layout()
fig.savefig("figures/fig_6a.pdf")


fig = plt.figure(figsize=(8.94, 5.34))
ax = fig.gca()
ax.bar(
    range(30),
    data_ga.objective_diff,
    width=0.5,
    bottom=data_ga.objective_rl,
    color=["g" if d > 0 else "r" for d in data_ga.objective_diff],
)

ax.scatter(
    range(30), sorted_indices.objective_rl, marker="x", color="k", s=50, linewidth=1
)

ax.set_ylim((0.775, 1.025))
ax.set_xlim((-0.5, 30.5))
ax.set_xticks(range(30))
ax.set_xlabel("Problem instances")
ax.set_ylabel("Objective")

plt.tight_layout()
fig.savefig("figures/fig_6b.pdf")
