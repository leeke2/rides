import pandas as pd

rl = pd.read_json("results/results_rl_E30_R1_DF1.0_NC_20240304014029.json")
ts = pd.read_json("results/results_ts_E30_R10_DF1.0_NC_20240304074129.json")
ga = pd.read_json("results/results_ga_E30_R10_DF1.0_NC_20240304014633.json")

rl_mean_objective = (
    rl.groupby("seed")
    .objective.mean()
    .reset_index()
    .rename(columns={"objective": "objective_rl"})
)

print(
    pd.concat([rl, ts, ga])
    .merge(rl_mean_objective, left_on="seed", right_on="seed")
    .assign(objective_diff=lambda x: x.objective - x.objective_rl)
    .groupby(["method", "seed"])
    .agg({"objective": "mean", "objective_diff": "max", "time": "mean"})
    .reset_index()
    .groupby("method")
    .agg(
        {
            "objective": ["min", "mean", "max"],
            "objective_diff": ["min", "mean", "max"],
            "time": ["min", "mean", "max"],
        }
    )
    .round(4)
)
