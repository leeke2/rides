import pandas as pd

c = pd.read_json("results/results_ts_E30_R10_DF1.0_C_20240304015504.json")
nc = pd.read_json("results/results_ts_E30_R10_DF1.0_NC_20240304074129.json")

nc_mean_objective = (
    nc.groupby("seed")
    .objective.mean()
    .reset_index()
    .rename(columns={"objective": "objective_nc"})
)

print(
    pd.concat([c, nc])
    .groupby(["congested", "seed"])
    .agg({"objective": "mean"})
    .reset_index()
    .merge(nc_mean_objective, left_on="seed", right_on="seed")
    .assign(objective_diff=lambda x: x.objective - x.objective_nc)
    .groupby("congested")
    .agg(
        {"objective": ["min", "mean", "max"], "objective_diff": ["min", "mean", "max"]}
    )
    .round(4)
)
