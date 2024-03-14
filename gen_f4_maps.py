import math
import multiprocessing as mp
from functools import reduce
from time import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import taxicab as tc


def mp_func(x):
    return (x[0], tc.shortest_path(*x[1:])[1])


def get_best_offset(x, y, all_plot_coords, offset=0.01):
    def d(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    o = offset
    p = math.sqrt(offset**2 / 2)
    options = [(o, 0), (p, p), (0, o), (-p, p), (-o, 0), (-p, -p), (0, -o), (p, -p)]
    values = [
        sum(d(x1, y1, x + dx, y + dy) <= offset for y1, x1 in all_plot_coords)
        for dx, dy in options
    ]

    selected = -1
    selected_value = len(all_plot_coords)

    for i in range(len(options)):
        if values[i] < selected_value:
            selected = i
            selected_value = values[i]

    return x + options[selected][0], y + options[selected][1]


def plot_mat(mat, ax):
    matsize = mat.shape[0]
    mask = 1 - np.triu(np.ones_like(mat), k=1)
    mat /= np.max(mat)

    ax.imshow(
        np.ma.array(mat, mask=mask).filled(fill_value=-float("inf")), vmin=0, vmax=1
    )

    # Major ticks
    ax.set_xticks(np.arange(-0.5, matsize, 5))
    ax.set_yticks(np.arange(-0.5, matsize, 5))

    # Labels for major ticks
    ax.set_xticklabels([], minor=True)
    ax.set_yticklabels([], minor=True)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, matsize, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matsize, 1), minor=True)

    ax.grid(which="minor", color="w", linestyle="-", alpha=0.3, linewidth=1)
    ax.grid(which="major", color="w", linewidth=1.5)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        top=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    for i in range(0, matsize, 5):
        ax.text(
            i - 0.75,
            i + 0.75,
            i + 1,
            rotation=-45,
            ha="center",
            va="center",
            fontfamily="serif",
            fontsize="small",
        )

    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)

    ax.set_xlabel(
        "Arr",
        fontfamily="serif",
        fontsize="small",
    )
    ax.set_ylabel(
        "Dep",
        fontfamily="serif",
        fontsize="small",
    )
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_label_position("top")


def plot_map(env, ax):
    df = pd.read_csv(".rides_sgnetwork/singapore_stops_grouped.csv").set_index(
        "BusStopCode"
    )

    coords = df.loc[env.unwrapped.inst.stops][["Latitude", "Longitude"]]
    w, e = coords["Longitude"].min(), coords["Longitude"].max()
    s, n = coords["Latitude"].min(), coords["Latitude"].max()
    coords = coords.values.tolist()

    dx = e - w
    dy = n - s
    d = max(dx, dy)

    py = dy * 0.4
    px = dx * 0.4

    pxp = (1.2 * d - dx) / 2
    pyp = (1.2 * d - dy) / 2

    G = ox.graph_from_bbox(
        bbox=(n + py, s - py, e + px, w - px), network_type="drive", simplify=False
    )
    G_plot = ox.graph_from_bbox(
        bbox=(n + pyp, s - pyp, e + pxp, w - pxp),
        network_type="drive",
        custom_filter='["highway"~"primary|secondary|motorway|motorway_link|trunk_link|primary_link|secondary_link"]',
    )

    with mp.Pool() as pool:
        edges = {
            key: [(node["y"], node["x"]) for n in val for node in [G.nodes[n]]]
            for key, val in pool.map(
                mp_func,
                [
                    (f"{org}-{dst}", G, coords[org], coords[dst])
                    for org, dst in zip(
                        range(env.unwrapped.inst.nstops - 1),
                        range(1, env.unwrapped.inst.nstops),
                    )
                ],
            )
        }

    edges_lss = {
        f"{org}-{dst}": (
            edges[f"{org}-{dst}"]
            if f"{org}-{dst}" in edges
            else [
                (node["y"], node["x"])
                for n in tc.shortest_path(G, coords[org], coords[dst])[1]
                for node in [G.nodes[n]]
            ]
        )
        for org, dst in zip(env.sol._lss.stops[:-1], env.sol._lss.stops[1:])
    }

    ttmat = np.load(".rides_sgnetwork/ttmat.npy")
    indices = df.loc[env.unwrapped.inst.stops]["index"].tolist()

    shorter_segments = []

    for org, dst in zip(env.sol._lss.stops[:-1], env.sol._lss.stops[1:]):
        if dst == org + 1:
            continue

        v = ttmat[indices[org]][indices[dst]]
        for i in range(org, dst):
            v -= ttmat[indices[i]][indices[i + 1]]
            if v < 0:
                break

        if v < 0:
            shorter_segments.append((org, dst))

    all_plot_coords = list(reduce(lambda a, b: a + b, edges.values()))
    if len(edges_lss) > 0:
        all_plot_coords += reduce(lambda a, b: a + b, edges_lss.values())

        ox.plot_graph(
            G_plot,
            bgcolor="#ffffff",
            node_size=0,
            edge_color="#000000",
            edge_alpha=0.05,
            ax=ax,
            show=False,
        )

    if env.sol._obj < 1.0:
        for a, b in shorter_segments:
            y, x = zip(*edges_lss[f"{a}-{b}"])
            ax.plot(x, y, "orange", lw=7.5, alpha=0.5)

    y, x = zip(*reduce(lambda a, b: a + b, edges.values()))
    ax.plot(x, y, "k", lw=0.5)

    y, x = list(zip(*coords))
    ax.scatter(x, y, s=3, c="k")

    if env.sol._obj < 1.0:
        y, x = zip(*reduce(lambda a, b: a + b, edges_lss.values()))
        ax.plot(x, y, "b", lw=0.75)

        y, x = list(zip(*[coords[i] for i in env.sol._lss.stops]))
        ax.scatter(x, y, s=20, lw=1, facecolors="w", edgecolor="b", zorder=10)

    for i in range(0, env.unwrapped.inst.nstops, 5):
        xytext = get_best_offset(
            coords[i][1], coords[i][0], all_plot_coords, offset=0.01
        )
        ax.annotate(
            i + 1,
            xy=coords[i][::-1],
            xytext=xytext,
            arrowprops=dict(arrowstyle="-", lw=0.5),
            bbox=dict(boxstyle="square", pad=0, fc="w", ec="none"),
            ha="center",
            va="center",
            fontfamily="serif",
            fontsize="small",
        )

    label = (
        env.unwrapped.inst.name.split(" ")[0] + f" (obj={min(env.sol._obj, 1.0):.4f})"
    )
    ax.text(
        e + pxp - 1.2 * d * 0.025,
        n + pyp - 1.2 * d * 0.025,
        label,
        ha="right",
        va="top",
        fontfamily="serif",
    )


def plot_env(env):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8.94, 2.98))

    plot_map(env, ax[0])
    plot_mat(env.unwrapped.inst.travel_time, ax[1])
    plot_mat(env.unwrapped.inst.demand, ax[2])

    return fig


if __name__ == "__main__":
    results = pd.read_json("results/results_rl_E30_R1_DF1.0_NC_20240304014029.json")

    env = gym.make(
        "rides_env:Rides-v0", congested=False, demand_factor=1.0, max_iters=100
    )

    for _, row in results.iterrows():
        try:
            _ = env.reset(seed=row.seed)
            if row.seed not in [3, 6, 11]:
                continue

            if row.allocation > 0 and len(row.alignment) >= 2:
                for item in row.alignment:
                    _ = env.step(int(item) + 1)

                for _ in range(row.allocation - 1):
                    _ = env.step(46)

            fig = plot_env(env)
            plt.tight_layout()
            fig.savefig(f"env_{row.seed}.pdf", dpi=300)
        except:
            continue
