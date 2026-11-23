"""Plot sealed summary tables only; never open counts, truth, or fitted models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
MODES = {"capped600": {"label": "600-iteration cap", "marker": "s", "line": "--", "offset": -.045},
         "convergence": {"label": "Continued optimization", "marker": "o", "line": "-", "offset": .045}}
METRICS = (("max_covariance_relative_error", "Cross-assay count covariance"),
           ("max_restricted_transport_relative_error", "Log-rate transport on the true source subspace"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sealed_tables(run_dir: Path) -> tuple[list[dict], dict]:
    seal_path = run_dir / "evaluation_seal.json"
    if not seal_path.is_file():
        raise FileNotFoundError("Evaluation is not sealed; plotting is disabled")
    seal = json.loads(seal_path.read_text())
    for name in ("comparison.tsv", "aggregate.tsv"):
        expected = seal["outputs"].get(name)
        if expected is None or sha(run_dir / name) != expected:
            raise ValueError(f"Summary table is missing from the seal or changed: {name}")
    with (run_dir / "comparison.tsv").open(newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    rows = []
    keys = set()
    for item in raw:
        row = {"world": item["world"], "dimension": int(item["dimension"]),
               "n_fit": int(item["n_fit"]), "mode": item["optimizer_mode"],
               "gradient_converged": item["gradient_converged"] == "True"}
        key = (row["world"], row["n_fit"], row["mode"])
        if (key in keys or row["dimension"] not in COLORS or row["mode"] not in MODES
                or row["n_fit"] not in (3000, 12000)
                or item["gradient_converged"] not in ("True", "False")):
            raise ValueError("Unexpected or duplicate selected-fit row")
        keys.add(key)
        for metric, _ in METRICS:
            row[metric] = float(item[metric])
            if not np.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError("Selected-fit recovery error is missing, negative, or nonfinite")
        rows.append(row)
    for dimension in COLORS:
        expected_worlds = {f"d{dimension}_rep{replicate}" for replicate in range(3)}
        for size in (3000, 12000):
            for mode in MODES:
                group = [row for row in rows if (row["dimension"], row["n_fit"], row["mode"]) == (dimension, size, mode)]
                if len(group) != 3 or {row["world"] for row in group} != expected_worlds:
                    raise ValueError("Each dimension/sample-size/mode must retain all three frozen worlds")
    provenance = {"evaluation_seal_sha256": sha(seal_path),
                  "comparison_sha256": seal["outputs"]["comparison.tsv"],
                  "aggregate_sha256": seal["outputs"]["aggregate.tsv"]}
    return rows, provenance


def draw(rows: list[dict]):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "count-transport-matched-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6))
    fig.subplots_adjust(left=.075, right=.985, bottom=.32, top=.79, wspace=.24)
    fig.suptitle("Cross-assay transport in an exact-model simulation", fontsize=17, y=.98)
    fig.text(.5, .933, "Known source dimension  |  3 independent worlds per dimension  |  3 assays", ha="center", fontsize=11)
    dimension_handles = [Line2D([0], [0], color=color, lw=2.5, label=f"{dimension} source{'s' if dimension > 1 else ''}")
                         for dimension, color in COLORS.items()]
    mode_handles = [Line2D([0], [0], color="#333333", ls=value["line"], marker=value["marker"],
                           markerfacecolor="white" if mode == "capped600" else "#333333",
                           label=value["label"], lw=1.8, markersize=6)
                    for mode, value in MODES.items()]
    fig.legend(handles=dimension_handles, loc="upper center", bbox_to_anchor=(.29, .89), ncol=3, frameon=False)
    fig.legend(handles=mode_handles, loc="upper center", bbox_to_anchor=(.77, .89), ncol=2, frameon=False)
    all_positive = all(row[metric] > 0 for row in rows for metric, _ in METRICS)
    for ax, (metric, title) in zip(axes, METRICS):
        for dimension, color in COLORS.items():
            dimension_offset = (dimension - 2) * .23
            for mode, style in MODES.items():
                median_x, median_y = [], []
                for location, size in enumerate((3000, 12000)):
                    group = sorted((row for row in rows if (row["dimension"], row["n_fit"], row["mode"]) ==
                                    (dimension, size, mode)), key=lambda row: row["world"])
                    values = np.array([row[metric] for row in group])
                    center = location + dimension_offset + style["offset"]
                    x = center + np.array([-.018, 0, .018])
                    ax.scatter(x, values, marker=style["marker"], s=42, linewidths=1.3,
                               edgecolors=color, facecolors="white" if mode == "capped600" else color,
                               alpha=.8, zorder=4)
                    not_converged = [i for i, row in enumerate(group) if not row["gradient_converged"]]
                    if not_converged:
                        ax.scatter(x[not_converged], values[not_converged], marker="x", s=36,
                                   color="#222222", linewidths=1.1, zorder=5)
                    median_x.append(center)
                    median_y.append(float(np.median(values)))
                ax.plot(median_x, median_y, color=color, ls=style["line"], lw=2.2, alpha=.9,
                        marker="_", markersize=14, markeredgewidth=2.5, zorder=3)
        upper = max(.16, 1.35 * max(row[metric] for row in rows))
        lower = min(.01, .7 * min(row[metric] for row in rows)) if all_positive else 0.
        if all_positive:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=.01, linscale=.5)
        ax.set_ylim(lower, upper)
        first_power = int(np.floor(np.log10(lower))) if all_positive else -2
        ticks = ([] if all_positive else [0.]) + [
            multiplier * 10. ** power for power in range(first_power, int(np.ceil(np.log10(upper))) + 1)
            for multiplier in (1, 3) if lower <= multiplier * 10. ** power < upper]
        ax.set_yticks(sorted(set(ticks)))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.axhline(.10, color="#666666", lw=1.2, ls=":", zorder=1)
        ax.text(.985, .10, "10% reference", color="#555555", fontsize=9,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
        ax.set_xlim(-.43, 1.43)
        ax.set_xticks([0, 1], ["3,000", "12,000"])
        ax.set_xlabel("Training cells per assay", labelpad=9)
        ax.set_ylabel("Worst-pair relative error (lower is better)")
        ax.set_title(title, pad=15)
        ax.grid(axis="y", color="#d9d9d9", lw=.65, alpha=.65)
        ax.set_axisbelow(True)
    footnotes = [
        "Each small point is one world; lines and horizontal bars show the median of three worlds. All selected fits are retained.",
        "Black crosses mark failure to meet the gradient criterion. Continuation has no total iteration cap; numerical failure can still stop it.",
        "Numerical convergence does not imply a global optimum. Estimation failure does not disprove the identifiability theorem.",
        "Favorable exact-model conditions and known dimension; only three replicates. This tests transport, not causal directions.",
        "The 10% line covers only the error criteria; full recovery also requires numerical and predictive checks. " +
        ("Vertical axes use logarithmic scales." if all_positive else "Vertical scales are logarithmic above 1%."),
    ]
    for y, text in zip((.21, .174, .138, .102, .066), footnotes):
        fig.text(.075, y, text, ha="left", va="center", fontsize=9, color="#444444")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    targets = [run_dir / "recovery.png", run_dir / "recovery.svg"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Recovery figures already exist; preserve the existing derived outputs")
    rows, provenance = load_sealed_tables(run_dir)
    fig = draw(rows)
    description = "Derived only from sealed comparison.tsv; " + json.dumps(provenance, sort_keys=True)
    with targets[0].open("xb") as handle:
        fig.savefig(handle, format="png", dpi=240, facecolor="white", metadata={"Description": description})
    with targets[1].open("xb") as handle:
        fig.savefig(handle, format="svg", facecolor="white", metadata={"Description": description, "Date": None})
    plt.close(fig)
    print(json.dumps({"figures": [str(path) for path in targets], **provenance}))


if __name__ == "__main__":
    main()
