#!/usr/bin/env python3
"""Plot final pipeline outputs by experiment, GNN, and LLM.

The script reads the raw JSON files produced by the extended pipeline and
generates publication-friendly diagnostic plots. It intentionally recomputes
metrics from raw rows so every plot can include sample counts, parse coverage,
and benchmark comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSIFICATION_EXPERIMENTS = [
    "embedding_classification",
    "raw_graph_reasoning",
]

RECONSTRUCTION_EXPERIMENTS = [
    "reconstruction_1hop",
    "reconstruction_1hop_embed_expl",
    "reconstruction_1hop_no_gnn",
]

RECONSTRUCTION_BASELINES = [
    "baseline_random",
    "baseline_cosine",
    "baseline_feature",
]

EXPERIMENT_LABELS = {
    "embedding_classification": "GNN embedding -> LLM class",
    "raw_graph_reasoning": "Raw graph context -> LLM class",
    "reconstruction_1hop": "Embedding -> LLM 1-hop reconstruction",
    "reconstruction_1hop_embed_expl": "Embedding + explanation -> LLM 1-hop reconstruction",
    "reconstruction_1hop_no_gnn": "No-GNN candidate prompt -> LLM 1-hop reconstruction",
    "baseline_random": "Random structural baseline",
    "baseline_cosine": "Embedding/candidate cosine baseline",
    "baseline_feature": "Raw-feature distance baseline",
    "gnn_prediction": "GNN prediction benchmark",
}

TASK_LLM_FALLBACK = {
    "task_0": "Qwen/Qwen2.5-3B",
    "task_1": "Qwen/Qwen3.5-4B",
    "task_2": "Qwen/Qwen3.5-9B",
    "task_3": "Qwen/Qwen3.5-27B",
    "task_4": "Qwen/Qwen3.5-35B-A3B",
}

GNN_ORDER = ["GCN", "GAT", "GIN", "GraphSAGE"]
METRICS_HIGHER_BETTER = ["accuracy", "precision", "recall", "f1", "jaccard", "overlap"]


def short_llm(name: str | None) -> str:
    if not name:
        return "unknown"
    value = str(name).split("/")[-1]
    return value.replace("-Instruct", "")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else None
    if isinstance(value, float):
        return int(value) if value in (0.0, 1.0) else None

    text = str(value).strip().lower()
    if not text or text == "unknown":
        return None
    text = text.strip("`*_ \t\r\n\"'.,;:()[]{}")
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    if text == "licit":
        return 0
    if text == "illicit":
        return 1

    patterns = [
        r"predicted[_\s-]*class\s*(?:is|=|:)?\s*([01])\b",
        r"\b(?:answer|prediction|label|classification|class)\s*(?:is|=|:)?\s*([01])\b",
        r"predicted[_\s-]*class\s*(?:is|=|:)?\s*(licit|illicit)\b",
        r"\b(?:answer|prediction|label|classification|class)\s*(?:is|=|:)?\s*(licit|illicit)\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            continue
        token = matches[-1].group(1).lower()
        if token in {"0", "1"}:
            return int(token)
        if token in {"licit", "illicit"}:
            return 0 if token == "licit" else 1
    return None


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "n": 0,
            "valid_n": 0,
            "parse_rate": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "llm_positive_rate": 0.0,
            "target_positive_rate": 0.0,
        }

    valid_n = correct = tp = fp = fn = positive_pred = target_pos = 0
    for row in rows:
        true = parse_binary_label(row.get("target_class"))
        pred = parse_binary_label(row.get("llm_pred"))
        if true == 1:
            target_pos += 1
        if pred is not None:
            valid_n += 1
            if pred == 1:
                positive_pred += 1
        if true is not None and pred is not None and true == pred:
            correct += 1
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred != 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": total,
        "valid_n": valid_n,
        "parse_rate": valid_n / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "llm_positive_rate": positive_pred / valid_n if valid_n else 0.0,
        "target_positive_rate": target_pos / total,
    }


def gnn_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    correct = tp = fp = fn = positive_pred = target_pos = 0
    for row in rows:
        true = parse_binary_label(row.get("target_class"))
        pred = parse_binary_label(row.get("gnn_pred"))
        if true == 1:
            target_pos += 1
        if pred == 1:
            positive_pred += 1
        if true is not None and pred is not None and true == pred:
            correct += 1
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred != 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": total,
        "valid_n": total,
        "parse_rate": 1.0,
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "llm_positive_rate": positive_pred / total if total else 0.0,
        "target_positive_rate": target_pos / total if total else 0.0,
    }


def reconstruction_row_metrics(row: dict[str, Any]) -> dict[str, float]:
    true_neighbors = set(int(v) for v in (row.get("true_neighbors") or []))
    predicted = set(int(v) for v in (row.get("predicted_neighbors") or []))
    intersection = true_neighbors.intersection(predicted)
    union = true_neighbors.union(predicted)
    precision = len(intersection) / len(predicted) if predicted else 0.0
    recall = len(intersection) / len(true_neighbors) if true_neighbors else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    jaccard = len(intersection) / len(union) if union else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "overlap": len(intersection) / max(1, len(true_neighbors)),
        "edit_distance": float(len(union) - len(intersection)),
        "true_size": float(len(true_neighbors)),
        "pred_size": float(len(predicted)),
        "empty_pred_rate": 1.0 if not predicted else 0.0,
        "exact_match_rate": 1.0 if true_neighbors == predicted else 0.0,
    }


def reconstruction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "overlap": 0.0,
            "edit_distance": 0.0,
            "true_size": 0.0,
            "pred_size": 0.0,
            "empty_pred_rate": 0.0,
            "exact_match_rate": 0.0,
        }
    per_row = [reconstruction_row_metrics(row) for row in rows]
    metrics = {"n": len(rows)}
    for key in per_row[0]:
        metrics[key] = mean(item[key] for item in per_row)
    return metrics


def discover_raw_files(root: Path) -> list[tuple[Path, str, str, str]]:
    files = []
    for path in root.glob("task_*/run_*/results_raw_*.json"):
        task = path.parents[1].name
        run = path.parent.name
        experiment = path.name.removeprefix("results_raw_").removesuffix(".json")
        files.append((path, task, run, experiment))
    return sorted(files)


def load_rows(root: Path, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for path, task, run, experiment in discover_raw_files(root):
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            inventory.append(
                {
                    "path": str(path),
                    "task": task,
                    "run": run,
                    "experiment": experiment,
                    "rows_total": 0,
                    "rows_dataset": 0,
                    "status": f"read_error: {exc}",
                }
            )
            continue
        if not isinstance(payload, list):
            continue

        dataset_rows = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("dataset") != dataset:
                continue
            item = dict(row)
            item["experiment"] = item.get("experiment") or experiment
            item["task"] = task
            item["run"] = run
            item["run_index"] = int(run.split("_")[-1]) if run.startswith("run_") else None
            item["llm"] = item.get("llm") or TASK_LLM_FALLBACK.get(task)
            dataset_rows.append(item)
        records.extend(dataset_rows)
        inventory.append(
            {
                "path": str(path),
                "task": task,
                "run": run,
                "experiment": experiment,
                "rows_total": len(payload),
                "rows_dataset": len(dataset_rows),
                "status": "ok",
            }
        )
    return records, inventory


def aggregate_groups(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metric_fn,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    out = []
    for group, items in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        result = {key: value for key, value in zip(group_keys, group)}
        result.update(metric_fn(items))
        out.append(result)
    return out


def mean_rows(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out = []
    for group, items in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        result = {key: value for key, value in zip(group_keys, group)}
        result["groups_n"] = len(items)
        for metric in metric_keys:
            values = [safe_float(item.get(metric), math.nan) for item in items]
            values = [value for value in values if not math.isnan(value)]
            result[metric] = mean(values) if values else math.nan
            result[f"{metric}_sd"] = pstdev(values) if len(values) > 1 else 0.0
        result["n"] = int(sum(safe_float(item.get("n")) for item in items))
        out.append(result)
    return out


def estimate_task_run_groups(overall_rows: list[dict[str, Any]], models: list[str]) -> int:
    """Recover task/run count from rows already averaged by model and LLM."""
    if not overall_rows or not models:
        return 0
    grouped_runs = sum(int(safe_float(row.get("groups_n"))) for row in overall_rows)
    return int(round(grouped_runs / max(1, len(models))))


def ordered_unique(values: list[str], preferred: list[str] | None = None) -> list[str]:
    unique = list(dict.fromkeys(v for v in values if v is not None))
    if preferred is None:
        return sorted(unique)
    ordered = [value for value in preferred if value in unique]
    ordered.extend(sorted(value for value in unique if value not in ordered))
    return ordered


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def heatmap_matrix(
    rows: list[dict[str, Any]],
    row_key: str,
    col_key: str,
    value_key: str,
    row_order: list[str],
    col_order: list[str],
) -> np.ndarray:
    lookup = {(str(row.get(row_key)), str(row.get(col_key))): safe_float(row.get(value_key), math.nan) for row in rows}
    matrix = np.full((len(row_order), len(col_order)), np.nan)
    for i, rvalue in enumerate(row_order):
        for j, cvalue in enumerate(col_order):
            matrix[i, j] = lookup.get((rvalue, cvalue), np.nan)
    return matrix


def draw_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".2f",
    cmap: str = "viridis",
) -> None:
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([short_llm(label) for label in col_labels], rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                text = "NA"
                color = "#666666"
            else:
                text = format(value, fmt)
                color = "white" if value > (vmin + vmax) / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def add_caption(fig, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
        wrap=True,
    )


def plot_classification_experiment(
    out_path: Path,
    experiment: str,
    class_overall: list[dict[str, Any]],
    gnn_overall: list[dict[str, Any]],
    dataset: str,
) -> None:
    exp_rows = [row for row in class_overall if row.get("experiment") == experiment]
    benchmark_exp = "raw_graph_reasoning" if experiment == "embedding_classification" else "embedding_classification"
    bench_rows = [row for row in class_overall if row.get("experiment") == benchmark_exp]
    models = ordered_unique([str(row.get("model")) for row in exp_rows], GNN_ORDER)
    llms = ordered_unique([str(row.get("llm")) for row in exp_rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"{EXPERIMENT_LABELS[experiment]} on {dataset}", fontsize=16, weight="bold")

    draw_heatmap(
        axes[0, 0],
        heatmap_matrix(exp_rows, "model", "llm", "accuracy", models, llms),
        models,
        llms,
        "LLM accuracy vs ground truth",
    )
    draw_heatmap(
        axes[0, 1],
        heatmap_matrix(exp_rows, "model", "llm", "f1", models, llms),
        models,
        llms,
        "LLM illicit-class F1",
    )

    ax = axes[1, 0]
    x = np.arange(len(models))
    width = 0.26
    exp_by_model = {str(row.get("model")): row for row in mean_rows(exp_rows, ["model"], ["accuracy"])}
    bench_by_model = {str(row.get("model")): row for row in mean_rows(bench_rows, ["model"], ["accuracy"])}
    gnn_by_model = {str(row.get("model")): row for row in gnn_overall}
    ax.bar(
        x - width,
        [safe_float(exp_by_model.get(model, {}).get("accuracy"), math.nan) for model in models],
        width,
        label=EXPERIMENT_LABELS[experiment],
        color="#3B82F6",
    )
    ax.bar(
        x,
        [safe_float(bench_by_model.get(model, {}).get("accuracy"), math.nan) for model in models],
        width,
        label=EXPERIMENT_LABELS[benchmark_exp],
        color="#F59E0B",
    )
    ax.bar(
        x + width,
        [safe_float(gnn_by_model.get(model, {}).get("accuracy"), math.nan) for model in models],
        width,
        label="GNN prediction benchmark",
        color="#10B981",
    )
    ax.set_title("Accuracy benchmark next to experiment", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    metrics = ["parse_rate", "precision", "recall", "llm_positive_rate", "target_positive_rate"]
    exp_summary = mean_rows(exp_rows, ["experiment"], metrics)[0] if exp_rows else {}
    values = [safe_float(exp_summary.get(metric), 0.0) for metric in metrics]
    labels = ["parse\ncoverage", "precision", "recall", "LLM\npositive", "target\npositive"]
    colors = ["#64748B", "#2563EB", "#7C3AED", "#EF4444", "#14B8A6"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Behavior summary averaged over GNN x LLM x runs", fontsize=11, weight="bold")
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)

    n_total = int(sum(safe_float(row.get("n")) for row in exp_rows))
    run_count = estimate_task_run_groups(exp_rows, models)
    add_caption(
        fig,
        f"Rows: {n_total}; completed task/run groups: {run_count}. "
        "Accuracy treats unparseable LLM answers as incorrect; parse coverage shows how often a binary label was recovered.",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_experiment(
    out_path: Path,
    experiment: str,
    reconstruction_overall: list[dict[str, Any]],
    baseline_overall: list[dict[str, Any]],
    dataset: str,
) -> None:
    exp_rows = [row for row in reconstruction_overall if row.get("experiment") == experiment]
    models = ordered_unique([str(row.get("model")) for row in exp_rows], GNN_ORDER)
    llms = ordered_unique([str(row.get("llm")) for row in exp_rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"{EXPERIMENT_LABELS[experiment]} on {dataset}", fontsize=16, weight="bold")

    draw_heatmap(
        axes[0, 0],
        heatmap_matrix(exp_rows, "model", "llm", "f1", models, llms),
        models,
        llms,
        "LLM reconstruction F1",
    )
    draw_heatmap(
        axes[0, 1],
        heatmap_matrix(exp_rows, "model", "llm", "jaccard", models, llms),
        models,
        llms,
        "LLM reconstruction Jaccard",
    )

    ax = axes[1, 0]
    x = np.arange(len(models))
    width = 0.18
    exp_by_model = {str(row.get("model")): row for row in mean_rows(exp_rows, ["model"], ["f1"])}
    ax.bar(
        x - 1.5 * width,
        [safe_float(exp_by_model.get(model, {}).get("f1"), math.nan) for model in models],
        width,
        label=EXPERIMENT_LABELS[experiment],
        color="#3B82F6",
    )
    baseline_colors = {
        "baseline_random": "#94A3B8",
        "baseline_cosine": "#F59E0B",
        "baseline_feature": "#10B981",
    }
    for offset, baseline in zip([-0.5, 0.5, 1.5], RECONSTRUCTION_BASELINES):
        rows = [row for row in baseline_overall if row.get("experiment") == baseline]
        by_model = {str(row.get("model")): row for row in mean_rows(rows, ["model"], ["f1"])}
        ax.bar(
            x + offset * width,
            [safe_float(by_model.get(model, {}).get("f1"), math.nan) for model in models],
            width,
            label=EXPERIMENT_LABELS[baseline],
            color=baseline_colors[baseline],
        )
    ax.set_title("F1 benchmark next to experiment", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    metrics = ["precision", "recall", "overlap", "empty_pred_rate", "exact_match_rate"]
    exp_summary = mean_rows(exp_rows, ["experiment"], metrics)[0] if exp_rows else {}
    values = [safe_float(exp_summary.get(metric), 0.0) for metric in metrics]
    labels = ["precision", "recall", "overlap", "empty\nprediction", "exact\nmatch"]
    colors = ["#2563EB", "#7C3AED", "#14B8A6", "#EF4444", "#10B981"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Behavior summary averaged over GNN x LLM x runs", fontsize=11, weight="bold")
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)

    n_total = int(sum(safe_float(row.get("n")) for row in exp_rows))
    avg_true = mean([safe_float(row.get("true_size")) for row in exp_rows]) if exp_rows else 0.0
    avg_pred = mean([safe_float(row.get("pred_size")) for row in exp_rows]) if exp_rows else 0.0
    run_count = estimate_task_run_groups(exp_rows, models)
    add_caption(
        fig,
        f"Rows: {n_total}; completed task/run groups: {run_count}; "
        f"mean true neighbors: {avg_true:.2f}; mean predicted neighbors: {avg_pred:.2f}. "
        "Benchmarks are structural non-LLM methods computed on the same candidate sets.",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classification_overview(out_path: Path, class_overall: list[dict[str, Any]], gnn_overall: list[dict[str, Any]], dataset: str) -> None:
    rows = mean_rows(class_overall, ["experiment", "llm"], ["accuracy", "f1", "parse_rate"])
    llms = ordered_unique([str(row.get("llm")) for row in rows])
    experiments = CLASSIFICATION_EXPERIMENTS

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(f"Classification Overview on {dataset}", fontsize=15, weight="bold")
    for ax, metric, title in zip(axes, ["accuracy", "f1", "parse_rate"], ["Accuracy", "Illicit-class F1", "Parse coverage"]):
        matrix = heatmap_matrix(rows, "experiment", "llm", metric, experiments, llms)
        draw_heatmap(ax, matrix, [EXPERIMENT_LABELS[e] for e in experiments], llms, title)
    gnn_acc = mean([safe_float(row.get("accuracy")) for row in gnn_overall]) if gnn_overall else 0.0
    add_caption(fig, f"Mean GNN prediction benchmark accuracy across GNNs/runs: {gnn_acc:.2f}. Values average over all GNN architectures.")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_overview(out_path: Path, reconstruction_overall: list[dict[str, Any]], baseline_overall: list[dict[str, Any]], dataset: str) -> None:
    all_rows = reconstruction_overall + baseline_overall
    rows = mean_rows(all_rows, ["experiment", "model"], ["f1", "jaccard", "recall"])
    experiments = RECONSTRUCTION_EXPERIMENTS + RECONSTRUCTION_BASELINES
    models = ordered_unique([str(row.get("model")) for row in rows], GNN_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    fig.suptitle(f"Reconstruction Overview on {dataset}", fontsize=15, weight="bold")
    for ax, metric, title in zip(axes, ["f1", "jaccard", "recall"], ["F1", "Jaccard", "Recall"]):
        matrix = heatmap_matrix(rows, "experiment", "model", metric, experiments, models)
        draw_heatmap(ax, matrix, [EXPERIMENT_LABELS[e] for e in experiments], models, title)
    add_caption(fig, "LLM reconstruction rows are averaged across LLMs and runs; structural baselines are averaged across runs.")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_metrics(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    classification_rows = [row for row in records if row.get("experiment") in CLASSIFICATION_EXPERIMENTS]
    reconstruction_rows = [row for row in records if row.get("experiment") in RECONSTRUCTION_EXPERIMENTS]
    baseline_rows = [row for row in records if row.get("experiment") in RECONSTRUCTION_BASELINES]

    class_by_run = aggregate_groups(
        classification_rows,
        ["experiment", "task", "run", "model", "llm"],
        classification_metrics,
    )
    gnn_by_run = []
    for experiment in CLASSIFICATION_EXPERIMENTS:
        rows = [row for row in classification_rows if row.get("experiment") == experiment]
        gnn_by_run.extend(
            aggregate_groups(
                rows,
                ["task", "run", "model"],
                gnn_metrics,
            )
        )
    for row in gnn_by_run:
        row["experiment"] = "gnn_prediction"
        row["llm"] = "GNN"

    reconstruction_by_run = aggregate_groups(
        reconstruction_rows,
        ["experiment", "task", "run", "model", "llm"],
        reconstruction_metrics,
    )
    baseline_by_run = aggregate_groups(
        baseline_rows,
        ["experiment", "task", "run", "model"],
        reconstruction_metrics,
    )
    for row in baseline_by_run:
        row["llm"] = "structural_baseline"

    class_overall = mean_rows(
        class_by_run,
        ["experiment", "model", "llm"],
        ["accuracy", "precision", "recall", "f1", "parse_rate", "llm_positive_rate", "target_positive_rate"],
    )
    gnn_overall = mean_rows(
        gnn_by_run,
        ["experiment", "model", "llm"],
        ["accuracy", "precision", "recall", "f1", "parse_rate", "llm_positive_rate", "target_positive_rate"],
    )
    reconstruction_overall = mean_rows(
        reconstruction_by_run,
        ["experiment", "model", "llm"],
        ["precision", "recall", "f1", "jaccard", "overlap", "edit_distance", "true_size", "pred_size", "empty_pred_rate", "exact_match_rate"],
    )
    baseline_overall = mean_rows(
        baseline_by_run,
        ["experiment", "model"],
        ["precision", "recall", "f1", "jaccard", "overlap", "edit_distance", "true_size", "pred_size", "empty_pred_rate", "exact_match_rate"],
    )
    return class_by_run + gnn_by_run, class_overall + gnn_overall, reconstruction_overall + baseline_overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/thijnvanveen/Desktop/Scriptie/Code/Explanation-of-GNN-using-LLM/outputs/Final_run_elliptic"),
        help="Root output directory containing task_*/run_* result files.",
    )
    parser.add_argument("--dataset", default="elliptic", help="Dataset to plot.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots and CSV summaries. Defaults to INPUT_DIR/plots_by_experiment.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "plots_by_experiment").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, inventory = load_rows(input_dir, args.dataset)
    if not records:
        raise SystemExit(f"No raw rows found for dataset '{args.dataset}' in {input_dir}")

    class_by_run_and_gnn, class_overall_all, reconstruction_overall_all = build_metrics(records)
    class_overall = [row for row in class_overall_all if row.get("experiment") in CLASSIFICATION_EXPERIMENTS]
    gnn_overall = [row for row in class_overall_all if row.get("experiment") == "gnn_prediction"]
    reconstruction_overall = [row for row in reconstruction_overall_all if row.get("experiment") in RECONSTRUCTION_EXPERIMENTS]
    baseline_overall = [row for row in reconstruction_overall_all if row.get("experiment") in RECONSTRUCTION_BASELINES]

    write_csv(output_dir / "run_inventory.csv", inventory)
    write_csv(output_dir / "classification_metrics_by_run.csv", class_by_run_and_gnn)
    write_csv(output_dir / "classification_metrics_overall.csv", class_overall_all)
    write_csv(output_dir / "reconstruction_metrics_overall.csv", reconstruction_overall_all)

    plot_classification_overview(output_dir / "00_classification_overview.png", class_overall, gnn_overall, args.dataset)
    plot_reconstruction_overview(output_dir / "00_reconstruction_overview.png", reconstruction_overall, baseline_overall, args.dataset)

    for idx, experiment in enumerate(CLASSIFICATION_EXPERIMENTS, start=1):
        plot_classification_experiment(
            output_dir / f"{idx:02d}_{experiment}.png",
            experiment,
            class_overall,
            gnn_overall,
            args.dataset,
        )
    for idx, experiment in enumerate(RECONSTRUCTION_EXPERIMENTS, start=3):
        plot_reconstruction_experiment(
            output_dir / f"{idx:02d}_{experiment}.png",
            experiment,
            reconstruction_overall,
            baseline_overall,
            args.dataset,
        )

    print(f"Read {len(records)} raw rows for dataset={args.dataset}")
    print(f"Wrote plots and summaries to {output_dir}")
    for path in sorted(output_dir.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
