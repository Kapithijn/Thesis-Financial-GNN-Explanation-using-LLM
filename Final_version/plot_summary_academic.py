#!/usr/bin/env python3
"""Fast academic plots from final-run summary JSON files.

This script is designed for the final elliptic result tree:

    outputs/23417336/task_*/run_*/results_summary_*.json

It uses the summary files rather than the raw prompt/response files, which makes
plotting much faster and produces compact thesis-ready figures and CSV tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSIFICATION_EXPERIMENTS = [
    "embedding_classification",
    "embedding_classification_explainer_subgraph",
    "raw_graph_reasoning",
]

RECONSTRUCTION_EXPERIMENTS = [
    "reconstruction_1hop",
    "reconstruction_1hop_embed_expl",
    "reconstruction_1hop_no_gnn",
]

BASELINE_EXPERIMENTS = [
    "baseline_random",
    "baseline_cosine",
    "baseline_feature",
]

GNN_ORDER = ["GCN", "GAT", "GIN", "GraphSAGE"]
LLM_ORDER = [
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]

TASK_LLM = {
    "task_0": "Qwen/Qwen2.5-3B",
    "task_1": "Qwen/Qwen3.5-4B",
    "task_2": "Qwen/Qwen3.5-9B",
    "task_3": "Qwen/Qwen3.5-27B",
    "task_4": "Qwen/Qwen3.5-35B-A3B",
}

LABELS = {
    "embedding_classification": "Embedding classification",
    "embedding_classification_explainer_subgraph": "Embedding + explainer nodes",
    "raw_graph_reasoning": "Raw graph reasoning",
    "reconstruction_1hop": "1-hop reconstruction",
    "reconstruction_1hop_embed_expl": "1-hop + explainer text",
    "reconstruction_1hop_no_gnn": "1-hop without GNN embedding",
    "baseline_random": "Random baseline",
    "baseline_cosine": "Embedding cosine baseline",
    "baseline_feature": "Feature-distance baseline",
}

SHORT_LABELS = {
    "embedding_classification": "Embedding",
    "embedding_classification_explainer_subgraph": "Embedding +\nexplainer nodes",
    "raw_graph_reasoning": "Raw graph",
    "reconstruction_1hop": "Embedding",
    "reconstruction_1hop_embed_expl": "Embedding +\nexplainer text",
    "reconstruction_1hop_no_gnn": "No GNN\nembedding",
    "baseline_random": "Random",
    "baseline_cosine": "Cosine",
    "baseline_feature": "Feature\ndistance",
}

REFERENCE_LABELS = {
    "ground_truth": "True 1-hop neighbors",
    "explainer": "GNNExplainer subgraph",
}

CLASS_METRICS = ["accuracy", "precision", "recall", "f1"]
RECON_METRICS = ["precision", "recall", "f1", "jaccard", "overlap", "edit_distance"]

COLORS = {
    "blue": "#2A6FBB",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#7B4EA3",
    "red": "#D55E00",
    "gray": "#6B7280",
    "dark": "#111827",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def short_llm(name: Any) -> str:
    value = "" if name is None else str(name)
    return value.split("/")[-1] if value else "unknown"


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def normalize_experiment(name: str) -> tuple[str, str]:
    if name.endswith("_explainer"):
        base = name.removesuffix("_explainer")
        if base in RECONSTRUCTION_EXPERIMENTS or base in BASELINE_EXPERIMENTS:
            return base, "explainer"
    return name, "ground_truth"


def ordered_unique(values: list[Any], preferred: list[str] | None = None) -> list[str]:
    unique = list(dict.fromkeys(str(v) for v in values if v is not None))
    if preferred is None:
        return sorted(unique)
    ordered = [value for value in preferred if value in unique]
    ordered.extend(sorted(value for value in unique if value not in ordered))
    return ordered


def mean_rows(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        result = {key: value for key, value in zip(group_keys, group)}
        result["groups_n"] = len(items)
        for metric in metrics:
            values = [safe_float(item.get(metric)) for item in items]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                result[metric] = math.nan
                result[f"{metric}_sd"] = math.nan
                result[f"{metric}_se"] = math.nan
                continue
            result[metric] = mean(values)
            result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
            result[f"{metric}_se"] = result[f"{metric}_sd"] / math.sqrt(len(values)) if len(values) > 1 else 0.0
        out.append(result)
    return out


def load_summary_rows(root: Path, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    files = sorted(root.glob("task_*/run_*/results_summary_*.json"))
    print(f"Found {len(files)} summary files", flush=True)

    for idx, path in enumerate(files, 1):
        task = path.parents[1].name
        run = path.parent.name
        run_index = int(run.split("_")[-1]) if run.startswith("run_") else None
        file_experiment = path.name.removeprefix("results_summary_").removesuffix(".json")
        experiment, reference = normalize_experiment(file_experiment)
        llm = TASK_LLM.get(task, "unknown")

        payload = json.loads(path.read_text())
        added = 0
        if isinstance(payload, dict) and any("|" in key for key in payload):
            for key, metrics in payload.items():
                parts = key.split("|")
                if len(parts) != 4:
                    continue
                key_experiment, model, row_dataset, key_llm = parts
                if row_dataset != dataset:
                    continue
                item = {
                    "family": "classification",
                    "reference": "ground_truth",
                    "experiment": key_experiment,
                    "task": task,
                    "run": run,
                    "run_index": run_index,
                    "model": model,
                    "llm": key_llm,
                    "file": str(path),
                }
                item.update(metrics)
                rows.append(item)
                added += 1
        elif isinstance(payload, dict):
            family = "reconstruction" if experiment in RECONSTRUCTION_EXPERIMENTS else "baseline"
            item = {
                "family": family,
                "reference": reference,
                "experiment": experiment,
                "task": task,
                "run": run,
                "run_index": run_index,
                "model": "all_gnns",
                "llm": llm if family == "reconstruction" else "structural_baseline",
                "file": str(path),
            }
            item.update(payload)
            rows.append(item)
            added = 1

        inventory.append(
            {
                "task": task,
                "run": run,
                "run_index": run_index,
                "file_experiment": file_experiment,
                "experiment": experiment,
                "reference": reference,
                "rows_added": added,
                "path": str(path),
            }
        )
        if idx % 25 == 0:
            print(f"Loaded {idx}/{len(files)} summary files", flush=True)
    return rows, inventory


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = ["family", "reference", "experiment", "task", "run", "run_index", "model", "llm", "groups_n"]
    fieldnames = preferred + sorted({key for row in rows for key in row.keys()} - set(preferred))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix(
    rows: list[dict[str, Any]],
    row_key: str,
    col_key: str,
    value_key: str,
    row_order: list[str],
    col_order: list[str],
) -> np.ndarray:
    lookup = {(str(row.get(row_key)), str(row.get(col_key))): safe_float(row.get(value_key)) for row in rows}
    data = np.full((len(row_order), len(col_order)), np.nan)
    for i, rvalue in enumerate(row_order):
        for j, cvalue in enumerate(col_order):
            data[i, j] = lookup.get((rvalue, cvalue), math.nan)
    return data


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    rows: list[str],
    cols: list[str],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".2f",
    cmap: str = "cividis",
) -> None:
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left", pad=8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticklabels(rows)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "NA" if math.isnan(value) else format(value, fmt)
            if math.isfinite(value) and vmax > vmin:
                normalized = (value - vmin) / (vmax - vmin)
                color = "white" if normalized < 0.45 else "#111827"
            else:
                color = "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def add_caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=8, color="#374151", wrap=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_inventory(path: Path, inventory: list[dict[str, Any]]) -> None:
    tasks = ordered_unique([row["task"] for row in inventory], [f"task_{i}" for i in range(5)])
    runs = ordered_unique([row["run"] for row in inventory], ["run_1", "run_2", "run_3"])
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in inventory:
        counts[(row["task"], row["run"])] += 1
    data = np.array([[counts[(task, run)] for run in runs] for task in tasks], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    fig.suptitle("Result-file completeness", fontweight="bold")
    draw_heatmap(ax, data, tasks, runs, "Summary JSON files per task/run", vmin=0, vmax=max(15, np.nanmax(data)), fmt=".0f", cmap="Blues")
    add_caption(fig, "A complete task/run has 15 summary files: 3 classification, 3 reconstruction, 3 reconstruction-vs-explainer, and 6 baseline files.")
    save_figure(fig, path)


def plot_classification(path: Path, class_rows: list[dict[str, Any]], dataset: str) -> None:
    overall = mean_rows(class_rows, ["experiment", "model", "llm"], CLASS_METRICS)
    exp_llm = mean_rows(overall, ["experiment", "llm"], ["accuracy", "f1", "precision", "recall"])
    experiments = [exp for exp in CLASSIFICATION_EXPERIMENTS if any(row["experiment"] == exp for row in exp_llm)]
    llms = ordered_unique([row["llm"] for row in exp_llm], LLM_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), constrained_layout=True)
    fig.suptitle(f"Classification results on {dataset}", fontweight="bold")
    for ax, metric, title in zip(
        axes.ravel(),
        ["accuracy", "f1", "precision", "recall"],
        ["Accuracy", "Illicit-class F1", "Precision", "Recall"],
    ):
        draw_heatmap(
            ax,
            matrix(exp_llm, "experiment", "llm", metric, experiments, llms),
            [SHORT_LABELS[exp] for exp in experiments],
            [short_llm(llm) for llm in llms],
            title,
            cmap="cividis",
        )
    add_caption(fig, "Values average over GNN architectures and three runs. Precision, recall, and F1 target the illicit class.")
    save_figure(fig, path)

    for experiment in experiments:
        plot_classification_experiment(path.parent / f"11_classification_{experiment}.png", experiment, overall, class_rows, dataset)


def plot_classification_experiment(
    path: Path,
    experiment: str,
    overall: list[dict[str, Any]],
    by_run: list[dict[str, Any]],
    dataset: str,
) -> None:
    rows = [row for row in overall if row["experiment"] == experiment]
    run_rows = [row for row in by_run if row["experiment"] == experiment]
    models = ordered_unique([row["model"] for row in rows], GNN_ORDER)
    llms = ordered_unique([row["llm"] for row in rows], LLM_ORDER)
    model_summary = mean_rows(run_rows, ["model"], ["accuracy", "f1"])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    fig.suptitle(f"{LABELS[experiment]} on {dataset}", fontweight="bold")
    draw_heatmap(axes[0], matrix(rows, "model", "llm", "accuracy", models, llms), models, [short_llm(llm) for llm in llms], "Accuracy")
    draw_heatmap(axes[1], matrix(rows, "model", "llm", "f1", models, llms), models, [short_llm(llm) for llm in llms], "F1")

    ax = axes[2]
    x = np.arange(len(models))
    lookup = {row["model"]: row for row in model_summary}
    values = [safe_float(lookup.get(model, {}).get("accuracy")) for model in models]
    errors = [safe_float(lookup.get(model, {}).get("accuracy_se"), 0.0) for model in models]
    ax.bar(x, values, yerr=errors, capsize=3, color=COLORS["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean accuracy by GNN", loc="left")
    ax.grid(axis="y", alpha=0.25)
    add_caption(fig, "Error bars denote standard error over LLM/task/run groups.")
    save_figure(fig, path)


def plot_reconstruction(path: Path, recon_rows: list[dict[str, Any]], dataset: str) -> None:
    overall = mean_rows(recon_rows, ["reference", "experiment", "llm"], RECON_METRICS)
    exp_rows = mean_rows(overall, ["reference", "experiment"], ["f1", "jaccard", "precision", "recall"])
    experiments = RECONSTRUCTION_EXPERIMENTS + BASELINE_EXPERIMENTS
    x = np.arange(len(experiments))
    width = 0.36
    gt = {row["experiment"]: row for row in exp_rows if row["reference"] == "ground_truth"}
    explainer = {row["experiment"]: row for row in exp_rows if row["reference"] == "explainer"}

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.8), constrained_layout=True)
    fig.suptitle(f"Subgraph reconstruction results on {dataset}", fontweight="bold")
    for ax, (metric, title) in zip(
        axes.ravel(),
        [("f1", "F1"), ("jaccard", "Jaccard"), ("precision", "Precision"), ("recall", "Recall")],
    ):
        gt_values = [safe_float(gt.get(exp, {}).get(metric)) for exp in experiments]
        gt_errors = [safe_float(gt.get(exp, {}).get(f"{metric}_se"), 0.0) for exp in experiments]
        exp_values = [safe_float(explainer.get(exp, {}).get(metric)) for exp in experiments]
        exp_errors = [safe_float(explainer.get(exp, {}).get(f"{metric}_se"), 0.0) for exp in experiments]

        ax.bar(
            x - width / 2,
            gt_values,
            width,
            yerr=gt_errors,
            capsize=3,
            color=COLORS["green"],
            label="True 1-hop neighbors",
        )
        ax.bar(
            x + width / 2,
            exp_values,
            width,
            yerr=exp_errors,
            capsize=3,
            color=COLORS["purple"],
            label="GNNExplainer subgraph",
        )
        ax.set_title(title, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[exp] for exp in experiments], rotation=24, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(title)
        for xpos, value in zip(x - width / 2, gt_values):
            if math.isfinite(value) and value >= 0.18:
                ax.text(xpos, min(value + 0.025, 0.98), f"{value:.2f}", ha="center", fontsize=7)
        for xpos, value in zip(x + width / 2, exp_values):
            if math.isfinite(value) and value >= 0.18:
                ax.text(xpos, min(value + 0.025, 0.98), f"{value:.2f}", ha="center", fontsize=7)
        if metric == "f1":
            ax.legend(frameon=False, loc="upper left")

    add_caption(
        fig,
        "Bars compare LLM and baseline-selected subgraphs against true neighbors and against GNNExplainer-selected nodes. "
        "Error bars denote standard error over runs/LLM tasks represented in the summary files.",
    )
    save_figure(fig, path)

    llm_rows = [row for row in overall if row["experiment"] in RECONSTRUCTION_EXPERIMENTS]
    plot_reconstruction_by_llm(path.parent / "21_reconstruction_by_llm.png", llm_rows, dataset)


def plot_reconstruction_by_llm(path: Path, rows: list[dict[str, Any]], dataset: str) -> None:
    llms = ordered_unique([row["llm"] for row in rows], LLM_ORDER)
    experiments = RECONSTRUCTION_EXPERIMENTS
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.8), constrained_layout=True)
    fig.suptitle(f"LLM reconstruction F1 by model size on {dataset}", fontweight="bold")
    for ax, reference in zip(axes, ["ground_truth", "explainer"]):
        ref_rows = [row for row in rows if row["reference"] == reference]
        for experiment, color in zip(experiments, [COLORS["blue"], COLORS["purple"], COLORS["orange"]]):
            lookup = {row["llm"]: row for row in ref_rows if row["experiment"] == experiment}
            x = np.arange(len(llms))
            values = [safe_float(lookup.get(llm, {}).get("f1")) for llm in llms]
            errors = [safe_float(lookup.get(llm, {}).get("f1_se"), 0.0) for llm in llms]
            ax.errorbar(x, values, yerr=errors, marker="o", linewidth=1.8, capsize=3, color=color, label=SHORT_LABELS[experiment])
        ax.set_title(REFERENCE_LABELS[reference], loc="left")
        ax.set_xticks(np.arange(len(llms)))
        ax.set_xticklabels([short_llm(llm) for llm in llms], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    add_caption(fig, "Error bars denote standard error over three runs for each LLM-specific task.")
    save_figure(fig, path)


def plot_topline(path: Path, class_rows: list[dict[str, Any]], recon_rows: list[dict[str, Any]], dataset: str) -> None:
    class_top = mean_rows(class_rows, ["family", "experiment"], ["accuracy", "f1"])
    recon_top = mean_rows(recon_rows, ["family", "reference", "experiment"], ["f1", "jaccard"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    fig.suptitle(f"Final experiment overview on {dataset}", fontweight="bold")

    ax = axes[0]
    experiments = CLASSIFICATION_EXPERIMENTS
    lookup = {row["experiment"]: row for row in class_top}
    x = np.arange(len(experiments))
    ax.bar(x - 0.18, [safe_float(lookup.get(exp, {}).get("accuracy")) for exp in experiments], 0.36, color=COLORS["blue"], label="Accuracy")
    ax.bar(x + 0.18, [safe_float(lookup.get(exp, {}).get("f1")) for exp in experiments], 0.36, color=COLORS["orange"], label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[exp] for exp in experiments], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Classification", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    experiments = RECONSTRUCTION_EXPERIMENTS + BASELINE_EXPERIMENTS
    width = 0.36
    gt = {row["experiment"]: row for row in recon_top if row["reference"] == "ground_truth"}
    ex = {row["experiment"]: row for row in recon_top if row["reference"] == "explainer"}
    x = np.arange(len(experiments))
    ax.bar(x - width / 2, [safe_float(gt.get(exp, {}).get("f1")) for exp in experiments], width, color=COLORS["green"], label="True neighbors")
    ax.bar(x + width / 2, [safe_float(ex.get(exp, {}).get("f1")) for exp in experiments], width, color=COLORS["purple"], label="GNNExplainer")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[exp] for exp in experiments], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Subgraph reconstruction F1", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    add_caption(fig, "Topline values average over all available LLMs, GNNs, and runs represented in the summary files.")
    save_figure(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/23417336"))
    parser.add_argument("--dataset", default="elliptic")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    configure_style()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or Path(__file__).resolve().parent / f"summary_plots_{input_dir.name}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, inventory = load_summary_rows(input_dir, args.dataset)
    class_rows = [row for row in rows if row["family"] == "classification"]
    recon_rows = [row for row in rows if row["family"] in {"reconstruction", "baseline"}]

    class_overall = mean_rows(class_rows, ["experiment", "model", "llm"], CLASS_METRICS)
    recon_overall = mean_rows(recon_rows, ["family", "reference", "experiment", "llm"], RECON_METRICS)
    topline = mean_rows(class_rows, ["family", "experiment"], ["accuracy", "f1"]) + mean_rows(
        recon_rows, ["family", "reference", "experiment"], ["f1", "jaccard", "precision", "recall"]
    )

    write_csv(output_dir / "summary_file_inventory.csv", inventory)
    write_csv(output_dir / "classification_summary_metrics.csv", class_overall)
    write_csv(output_dir / "reconstruction_summary_metrics.csv", recon_overall)
    write_csv(output_dir / "topline_summary_metrics.csv", topline)

    plot_inventory(output_dir / "00_result_inventory.png", inventory)
    plot_topline(output_dir / "01_topline_overview.png", class_rows, recon_rows, args.dataset)
    plot_classification(output_dir / "10_classification_overview.png", class_rows, args.dataset)
    plot_reconstruction(output_dir / "20_reconstruction_overview.png", recon_rows, args.dataset)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Academic Summary Plots",
                "",
                f"Input: `{input_dir}`",
                f"Dataset: `{args.dataset}`",
                "",
                "These figures are generated from `results_summary_*.json` files.",
                "Classification plots retain GNN, LLM, and run information.",
                "Reconstruction summary files are already averaged over GNNs by the pipeline, so these plots show LLM/run and experiment-level results.",
                "The `_explainer` files are plotted as a separate GNNExplainer-subgraph reference.",
                "",
            ]
        )
    )

    print(f"Loaded {len(rows)} metric rows from {len(inventory)} summary files")
    print(f"Wrote plots and CSV tables to {output_dir}")
    for path in sorted(output_dir.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
