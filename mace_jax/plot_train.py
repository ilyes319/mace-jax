import argparse
import glob
import json
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

fig_width = 2.5
fig_height = 2.1

plt.rcParams.update({"font.size": 6})

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def parse_training_results(path: str) -> List[dict]:
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["path"] = os.path.dirname(path)
            d["name"] = os.path.basename(path).split(".")[0]
            results.append(d)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mace training statistics")
    parser.add_argument(
        "--path", help="path to results file or directory", required=True
    )
    return parser.parse_args()


def plot(data: pd.DataFrame, output_path: str) -> None:
    data = data[data["interval"] > 0]

    valid_data = data[data["mode"] == "eval_valid"]
    train_data = data[data["mode"] == "eval_train"]
    test_data = data[data["mode"] == "eval_test"]

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(2 * fig_width, fig_height), constrained_layout=True
    )

    ax = axes[0]
    if len(valid_data) > 0:
        ax.plot(
            valid_data["interval"],
            valid_data["loss"],
            color=colors[0],
            zorder=1,
            label="Validation",
        )
    if len(train_data) > 0:
        ax.plot(
            train_data["interval"],
            train_data["loss"],
            color=colors[1],
            zorder=1,
            label="Training",
        )
    if len(test_data) > 0:
        ax.plot(
            test_data["interval"],
            test_data["loss"],
            color=colors[2],
            zorder=1,
            label="Test",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interval")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    if len(valid_data) > 0:
        ax.plot(
            valid_data["interval"],
            1000 * valid_data["mae_f"],
            color=colors[0],
            zorder=1,
            label="MAE Force [meV/A]",
        )
    if len(train_data) > 0:
        ax.plot(
            train_data["interval"],
            1000 * train_data["mae_f"],
            color=colors[1],
            zorder=1,
            label="MAE Force [meV/A]",
        )
    if len(test_data) > 0:
        ax.plot(
            test_data["interval"],
            1000 * test_data["mae_f"],
            color=colors[2],
            zorder=1,
            label="MAE Force [meV/A]",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interval")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)


def get_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    paths = glob.glob(os.path.join(path, "*.metrics"))

    if len(paths) == 0:
        raise RuntimeError(f"Cannot find results in '{path}'")

    return paths


def main():
    args = parse_args()
    data = pd.DataFrame(
        results
        for path in get_paths(args.path)
        for results in parse_training_results(path)
    )

    for (path, name), group in data.groupby(["path", "name"]):
        plot(group, output_path=f"{path}/{name}.png")


if __name__ == "__main__":
    main()
