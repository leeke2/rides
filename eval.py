import argparse
import itertools
import json
from datetime import datetime
from time import time

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange
from geneticalgorithm import geneticalgorithm as ga
from tqdm import tqdm
from tram import mat_linear_assign, mat_linear_congested_assign

from net import CNN, load_ckpt, to_tensor
from tabu import tabu_search


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("--nenvs", type=int, default=20)
    parser.add_argument("--nruns", type=int, default=5)
    parser.add_argument(
        "--congested",
        default=True,
        type=lambda x: (
            True
            if x.lower() in ["true", "yes", "1", "t"]
            else (False if x.lower() in ["false", "no", "0", "f"] else None)
        ),
    )
    parser.add_argument("--demand_factor", type=float, default=1)
    args = parser.parse_args()

    run_experiment(args)


def to_str(item):
    if isinstance(item, dict):
        return {key: to_str(val) for key, val in item.items()}

    if isinstance(item, list):
        return list(map(to_str, item))

    if isinstance(item, tuple):
        return tuple(map(to_str, item))

    return str(item)


def frequency(travel_time, stops, nbuses):
    trip_time = sum(
        travel_time[from_][to_] for from_, to_ in zip(stops[:-1], stops[1:])
    )

    return nbuses / trip_time


def tabu_search_experiment(env):
    params = {"tabu_tenure": 10, "max_epochs": 100, "patience": 50, "nneighbours": 10}
    sol, _ = tabu_search(env.unwrapped.sol, **params)

    inst = env.inst
    ttd = sol._ttd

    pct_improved = np.sum((ttd < inst.base_ttd) * inst.demand) / np.sum(inst.demand)
    pct_worsened = np.sum((ttd > inst.base_ttd) * inst.demand) / np.sum(inst.demand)

    mean_improved = np.sum(
        (((inst.base_ttd - ttd) / inst.base_ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened = np.sum(
        (((ttd - inst.base_ttd) / inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    mean_improved_abs = np.sum(
        ((inst.base_ttd - ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened_abs = np.sum(
        ((ttd - inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    return {
        "alignment": sol._lss.stops,
        "allocation": sol._lss.nbuses,
        "objective": sol._obj,
        "initial_max_load_factor": (
            -float("inf")
            if not inst.congested
            else np.max(inst.base_flow / (inst._oris.frequency * inst.capacity) * 100)
        ),
        "pct_demand_improved": pct_improved,
        "pct_demand_worsened": pct_worsened,
        "mean_relative_travel_time_improved": mean_improved,
        "mean_relative_travel_time_worsened": mean_worsened,
        "mean_absolute_travel_time_improved": mean_improved_abs,
        "mean_absolute_travel_time_worsened": mean_worsened_abs,
    }


def genetic_algorithm_experiment(env):
    congested = env.inst.congested

    params = {
        "max_num_iteration": 100,
        "population_size": 10,
        "mutation_probability": 0.1,
        "elit_ratio": 0.01,
        "crossover_probability": 0.5,
        "parents_portion": 0.3,
        "crossover_type": "two_point",
        "max_iteration_without_improv": 50,
    }

    # params = {
    #     "max_num_iteration": 3000,
    #     "population_size": 100,
    #     "mutation_probability": 0.1,
    #     "elit_ratio": 0.01,
    #     "crossover_probability": 0.5,
    #     "parents_portion": 0.3,
    #     "crossover_type": "uniform",
    #     "max_iteration_without_improv": None,
    # }

    inst = env.inst

    def objective_fn(sol):
        if sol[-1] == 0:
            return 1.0

        nbuses = [inst.nbuses - sol[-1], sol[-1]]
        alignments = [
            list(range(inst.nstops)),
            [i for i, served in enumerate(sol[:-1]) if served == 1],
        ]

        frequencies = [
            frequency(inst.travel_time, alg, nb)
            for (alg, nb) in zip(alignments, nbuses)
        ]

        if congested:
            assignment = mat_linear_congested_assign(
                alignments,
                frequencies,
                inst.travel_time,
                inst.demand,
                inst.capacity,
                max_iters=100,
            )
        else:
            assignment = mat_linear_assign(
                alignments, frequencies, inst.travel_time, inst.demand
            )

        return assignment[2] / inst.base_obj

    varbound = np.array([[0, 1]] * inst.nstops + [[0, inst.nbuses - 1]])

    model = ga(
        function=objective_fn,
        dimension=inst.nstops + 1,
        variable_type="int",
        variable_boundaries=varbound,
        algorithm_parameters=params,
        convergence_curve=False,
        progress_bar=False,
    )
    model.run()

    sol = model.output_dict["variable"]
    obj = model.output_dict["function"]

    alignments = [
        list(range(inst.nstops)),
        [i for i, served in enumerate(sol[:-1]) if served == 1],
    ]
    nbuses = [inst.nbuses - sol[-1], sol[-1]]

    frequencies = [
        frequency(inst.travel_time, alg, nb) for (alg, nb) in zip(alignments, nbuses)
    ]

    if congested:
        assignment = mat_linear_congested_assign(
            alignments,
            frequencies,
            inst.travel_time,
            inst.demand,
            inst.capacity,
            max_iters=100,
        )
    else:
        assignment = mat_linear_assign(
            alignments, frequencies, inst.travel_time, inst.demand
        )

    ttd = assignment[0]

    pct_improved = np.sum((ttd < inst.base_ttd) * inst.demand) / np.sum(inst.demand)
    pct_worsened = np.sum((ttd > inst.base_ttd) * inst.demand) / np.sum(inst.demand)

    mean_improved = np.sum(
        (((inst.base_ttd - ttd) / inst.base_ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened = np.sum(
        (((ttd - inst.base_ttd) / inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    mean_improved_abs = np.sum(
        ((inst.base_ttd - ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened_abs = np.sum(
        ((ttd - inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    return {
        "alignment": alignments[-1],
        "allocation": nbuses[-1],
        "objective": obj,
        "initial_max_load_factor": (
            -float("inf")
            if not inst.congested
            else np.max(inst.base_flow / (inst._oris.frequency * inst.capacity) * 100)
        ),
        "pct_demand_improved": pct_improved,
        "pct_demand_worsened": pct_worsened,
        "mean_relative_travel_time_improved": mean_improved,
        "mean_relative_travel_time_worsened": mean_worsened,
        "mean_absolute_travel_time_improved": mean_improved_abs,
        "mean_absolute_travel_time_worsened": mean_worsened_abs,
    }


def reinforcement_learning_experiment(net, env, seed):
    def process_state(state, nbuses, action_mask):
        edges_selected = state["od_demand"] * 0.0
        alignment = np.arange(1, state["od_demand"].shape[0] + 1)[
            state["stops_lss"]
        ].tolist()

        for from_, to_ in zip(alignment[:-1], alignment[1:]):
            edges_selected[from_ - 1, to_ - 1] = 1

        edges_selected *= state["nbuses_lss"] / nbuses

        edges = np.stack(
            [
                state["od_demand"],
                state["link_travel_time"] / np.max(state["link_travel_time"]),
                state["base_od_travel_time"] / np.max(state["link_travel_time"]),
                state["od_travel_time"] / np.max(state["link_travel_time"]),
                np.nan_to_num(
                    state["od_travel_time"] / state["base_od_travel_time"], 0.0
                ),
                state["od_travel_time"]
                * state["od_demand"]
                / np.max(state["link_travel_time"]),
                edges_selected,
            ],
            axis=-1,
        )

        mask_1d = np.expand_dims(state["stops_lss"], axis=0)
        mask_2d = mask_1d | mask_1d.T
        mask_2d = np.expand_dims(mask_2d, axis=-1)

        edges = np.concatenate((edges * mask_2d, edges * ~mask_2d), axis=-1)

        edges = rearrange(edges, "h w c -> c h w")

        return edges.astype(np.float32), action_mask.astype(np.bool_)

    state, info = env.reset(seed=seed)
    state, mask = process_state(state, env.inst.nbuses, info["action_mask"])
    state = {"obs": state, "mask": mask}
    done = False

    while not done:
        action = torch.max(net(to_tensor(state)), dim=-1)[1].item()
        state, _, done, *_, info = env.step(action)
        state, mask = process_state(state, env.inst.nbuses, info["action_mask"])
        state = {"obs": state, "mask": mask}

    sol = env.sol
    inst = env.inst
    ttd = sol._ttd

    pct_improved = np.sum((ttd < inst.base_ttd) * inst.demand) / np.sum(inst.demand)
    pct_worsened = np.sum((ttd > inst.base_ttd) * inst.demand) / np.sum(inst.demand)

    mean_improved = np.sum(
        (((inst.base_ttd - ttd) / inst.base_ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened = np.sum(
        (((ttd - inst.base_ttd) / inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    mean_improved_abs = np.sum(
        ((inst.base_ttd - ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened_abs = np.sum(
        ((ttd - inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    return {
        "alignment": sol._lss.stops,
        "allocation": sol._lss.nbuses,
        "objective": sol._obj,
        "initial_max_load_factor": (
            -float("inf")
            if not inst.congested
            else np.max(inst.base_flow / (inst._oris.frequency * inst.capacity) * 100)
        ),
        "pct_demand_improved": pct_improved,
        "pct_demand_worsened": pct_worsened,
        "mean_relative_travel_time_improved": mean_improved,
        "mean_relative_travel_time_worsened": mean_worsened,
        "mean_absolute_travel_time_improved": mean_improved_abs,
        "mean_absolute_travel_time_worsened": mean_worsened_abs,
    }


def brute_force(env):
    inst = env.inst
    congested = inst.congested

    def objective_fn(sol):
        if sol[-1] == 0:
            return 1.0

        nbuses = [inst.nbuses - sol[-1], sol[-1]]
        alignments = [
            list(range(inst.nstops)),
            [i for i, served in enumerate(sol[:-1]) if served == 1],
        ]

        frequencies = [
            frequency(inst.travel_time, alg, nb)
            for (alg, nb) in zip(alignments, nbuses)
        ]

        if congested:
            assignment = mat_linear_congested_assign(
                alignments,
                frequencies,
                inst.travel_time,
                inst.demand,
                inst.capacity,
                max_iters=100,
            )
        else:
            assignment = mat_linear_assign(
                alignments, frequencies, inst.travel_time, inst.demand
            )

        return assignment[2] / inst.base_obj

    ncombs = 2**inst.nstops
    print(ncombs)

    alignments = itertools.product([0, 1], repeat=inst.nstops)

    cur_obj = 1.0
    cur_sol = [0] * (inst.nstops + 1)

    for alignment in tqdm(alignments, total=ncombs):
        if sum(alignment) < 2:
            continue

        for allocation in range(1, inst.nbuses):
            sol = list(alignment) + [allocation]
            obj = objective_fn(sol)

            if obj < cur_obj:
                cur_obj = obj
                cur_sol = sol

    alignments = [
        list(range(inst.nstops)),
        [i for i, served in enumerate(cur_sol[:-1]) if served == 1],
    ]
    nbuses = [inst.nbuses - sol[-1], sol[-1]]

    frequencies = [
        frequency(inst.travel_time, alg, nb) for (alg, nb) in zip(alignments, nbuses)
    ]

    if congested:
        assignment = mat_linear_congested_assign(
            alignments,
            frequencies,
            inst.travel_time,
            inst.demand,
            inst.capacity,
            max_iters=100,
        )
    else:
        assignment = mat_linear_assign(
            alignments, frequencies, inst.travel_time, inst.demand
        )

    ttd = assignment[0]

    pct_improved = np.sum((ttd < inst.base_ttd) * inst.demand) / np.sum(inst.demand)
    pct_worsened = np.sum((ttd > inst.base_ttd) * inst.demand) / np.sum(inst.demand)

    mean_improved = np.sum(
        (((inst.base_ttd - ttd) / inst.base_ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened = np.sum(
        (((ttd - inst.base_ttd) / inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    mean_improved_abs = np.sum(
        ((inst.base_ttd - ttd) * inst.demand)[ttd < inst.base_ttd]
        / pct_improved
        / np.sum(inst.demand)
    )
    mean_worsened_abs = np.sum(
        ((ttd - inst.base_ttd) * inst.demand)[ttd > inst.base_ttd]
        / pct_worsened
        / np.sum(inst.demand)
    )

    return {
        "alignment": alignments[-1],
        "allocation": nbuses[-1],
        "objective": cur_obj,
        "initial_max_load_factor": (
            -float("inf")
            if not inst.congested
            else np.max(inst.base_flow / (inst._oris.frequency * inst.capacity) * 100)
        ),
        "pct_demand_improved": pct_improved,
        "pct_demand_worsened": pct_worsened,
        "mean_relative_travel_time_improved": mean_improved,
        "mean_relative_travel_time_worsened": mean_worsened,
        "mean_absolute_travel_time_improved": mean_improved_abs,
        "mean_absolute_travel_time_worsened": mean_worsened_abs,
    }


def run_experiment(args):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results = []

    env = gym.make(
        "rides_env:Rides-v0",
        max_iters=100,
        congested=args.congested,
        demand_factor=args.demand_factor * 1.25,
    )

    net = CNN((14, 45, 45), 47)
    load_ckpt("results/epoch=3625-step=3626000.ckpt", net)

    for seed in tqdm(range(args.nenvs)):
        env.reset(seed=seed)

        for run in range(args.nruns):
            t0 = time()

            if args.method == "ts":
                result = tabu_search_experiment(env)
            elif args.method == "ga":
                result = genetic_algorithm_experiment(env)
            elif args.method == "rl":
                result = reinforcement_learning_experiment(net, env, seed)
            elif args.method == "bf":
                result = brute_force(env)
            else:
                raise ValueError()

            if (
                len(result["alignment"]) < 2
                or result["allocation"] < 1
                or result["objective"] > 1.0
            ):
                result["alignment"] = []
                result["allocation"] = 0
                result["objective"] = 1.0
                result["pct_demand_improved"] = 0.0
                result["pct_demand_worsened"] = 0.0
                result["mean_relative_travel_time_improved"] = 0.0
                result["mean_relative_travel_time_worsened"] = 0.0
                result["mean_absolute_travel_time_improved"] = 0.0
                result["mean_absolute_travel_time_worsened"] = 0.0

            results.append(
                {
                    "seed": seed,
                    "run": run,
                    "method": args.method,
                    "demand_factor": args.demand_factor,
                    "congested": args.congested,
                    "time": time() - t0,
                    **result,
                }
            )

    out = json.dumps(to_str(results), indent=4)

    congested = "C" if args.congested else "NC"

    with open(
        f"results_{args.method}_E{args.nenvs}_R{args.nruns}_DF{args.demand_factor}_{congested}_{timestamp}.json",
        "w+",
    ) as f:
        f.write(out)


if __name__ == "__main__":
    cli_main()
