import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style as mpl_style


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fns", nargs="*")
    parser.add_argument("--sort_instances_by_objective", "-s", type=str, default="ts")
    parser.add_argument("--grouped_by", "-g", type=str, default="method")
    parser.add_argument("--plot_stylesheet", "-p", type=str, default="pub.mplstyle")
    parser.add_argument("--ymin", type=float, default=0.775)
    parser.add_argument("--ymax", type=float, default=1.025)
    parser.add_argument("--output_prefix", "-o", type=str, default="lssdp_c")
    parser.add_argument("--legends", "-l", type=str, nargs="*", default=[])
    parser.add_argument(
        "--individual_plots",
        "-i",
        type=lambda x: (
            True
            if x.lower() in ["true", "yes", "1", "t"]
            else (False if x.lower() in ["false", "no", "0", "f"] else None)
        ),
        default=False,
    )

    args = parser.parse_args()

    if len(args.fns) == 0:
        raise RuntimeError("No result files found!")

    return args


def read_results(fns) -> pd.DataFrame:
    dfs = []

    for fn in fns:
        _, method, *_ = fn.split(os.path.sep)[-1].split("_")
        dfs.append(pd.read_json(fn).assign(method=method))

    return pd.concat(dfs)


def prepare_boxplot_data(data, grouped_by, grouped_by_val, sort_by="ts"):
    data.loc[data.objective > 1, "objective"] = 1.0

    sorted_indices = (
        data[lambda x: x[grouped_by] == sort_by]
        .groupby("seed")
        .objective.mean()
        .sort_values(ascending=False)
        .reset_index()
        .assign(sorted_idx=lambda x: range(len(x)))
    )

    return (
        data[lambda x: x[grouped_by] == grouped_by_val]
        .merge(sorted_indices[["seed", "sorted_idx"]], left_on="seed", right_on="seed")
        .groupby("sorted_idx")
        .agg({"objective": list})
        .values.T.tolist()
    )[0]


def plot_boxplot(data, legend=None, figsize=(8.94, 5.34), ylim=(0.775, 1.025)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.boxplot(data, whis=(0, 100))
    ax.set_ylim(ylim)
    ax.set_xlabel("Problem instances")
    ax.set_ylabel("Objective")

    if legend is not None:
        ax.legend([legend.upper()])

    return fig, ax


def plot_boxplot_side_by_side(
    data,
    legend=None,
    figsize=(8.94 * 2, 7),
    ylim=(0.775, 1.025),
    colors=["b", "g", "k", "r"],
):

    colors = iter(colors)

    num_groups = sum(1 for key, value in data.items() if len(value[0]) > 1)
    scatter_group_exists = any(
        True if len(value[0]) == 1 else False for key, value in data.items()
    )
    idx_group = -1

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    num_items = 0
    xmax = 0
    handles = []

    for _, d in data.items():
        idx_group += 1

        if len(d) == 0:
            continue

        if len(d[0]) == 1:
            idx_group -= 1
            x = [
                i * 0.75 * (num_groups + 1) + (num_groups - 1) * 0.75 / 2
                for i in range(num_items)
            ]

            y = [item[0] for item in d]

            handle = ax.scatter(x, y, marker="x", color="k", s=50, linewidth=1)
            handles.append(handle)
        else:
            positions = [(i * num_groups + i + idx_group) * 0.75 for i in range(len(d))]
            xmax = max(xmax, positions[-1])
            num_items = max(num_items, len(d))

            c = next(colors)
            handle = ax.boxplot(
                d,
                whis=(0, 100),
                positions=positions,
                widths=0.5,
                boxprops={"color": c, "linewidth": min(2, 2 / num_groups)},
                medianprops={"color": c, "linewidth": min(2, 2 / num_groups)},
                whiskerprops={"color": c},
                capprops={"color": c},
            )

            handles.append(handle["boxes"][0])

    ax.set_ylim(ylim)
    ax.set_xlabel("Problem instances")
    ax.set_ylabel("Objective")
    ax.set_xlim((-0.75, xmax + 0.75))
    ax.set_xticks(
        [
            i * 0.75 * (num_groups + 1) + (num_groups - 1) * 0.75 / 2
            for i in range(num_items)
        ]
    )

    if legend is not None:
        ax.legend(handles, [item.upper() for item in legend])

    return fig, ax


def cli_main() -> None:
    args = get_cli_args()

    results = read_results(args.fns)

    if Path(args.plot_stylesheet).exists():
        mpl_style.use(args.plot_stylesheet)
        logging.info(f"Loaded stylesheet {args.plot_stylesheet}!")
    else:
        logging.warning(f"Matplotlib stylesheet not found at {args.plot_stylesheet}!")

    if args.individual_plots:
        for group in results[args.grouped_by].unique():
            data = prepare_boxplot_data(
                results, args.grouped_by, group, args.sort_instances_by_objective
            )

            fig, ax = plot_boxplot(
                data, legend=group.upper(), ylim=(args.ymin, args.ymax)
            )

            ytickslocs = [y for y in ax.get_yticks() if y > args.ymin and y < args.ymax]
            ax.set_yticks(ytickslocs, [f"{y:.2f}" for y in ytickslocs])

        plt.tight_layout()
        fig.savefig(f"{args.output_prefix}_{group}.pdf", dpi=200)
    else:
        data = {
            group: prepare_boxplot_data(
                results, args.grouped_by, group, args.sort_instances_by_objective
            )
            for group in results[args.grouped_by].unique()
        }

        fig, ax = plot_boxplot_side_by_side(
            data,
            legend=(
                results[args.grouped_by].unique()
                if len(args.legends) == 0
                else args.legends
            ),
            ylim=(args.ymin, args.ymax),
        )

        ytickslocs = [y for y in ax.get_yticks() if y > args.ymin and y < args.ymax]
        ax.set_yticks(ytickslocs, [f"{y:.2f}" for y in ytickslocs])

        group = "_".join(data.keys())

        plt.tight_layout()
        fig.savefig(f"{args.output_prefix}_{group}.pdf", dpi=200)


if __name__ == "__main__":
    cli_main()
