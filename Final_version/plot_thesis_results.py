#!/usr/bin/env python3
"""Create thesis plots and tables from pipeline result JSON files.

The script scans one or more output directories for ``results_raw_*.json`` files,
recomputes metrics from the raw rows, averages over available runs or seeds, and
writes publication-oriented figures plus CSV and LaTeX tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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

DATASET_ORDER = ["elliptic", "ibm_aml_hi_small", "dgraphfin"]
DATASET_LABELS = {
    "elliptic": "Elliptic",
    "ibm_aml_hi_small": "AML",
    "dgraphfin": "DGraph",
}

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

EXPERIMENT_LABELS = {
    "embedding_classification": "Embedding fidelity",
    "embedding_classification_explainer_subgraph": "Embedding with explainer fidelity",
    "raw_graph_reasoning": "Raw graph fidelity",
    "reconstruction_1hop": "1-hop neighbourhood reconstruction",
    "reconstruction_1hop_embed_expl": "Reconstruction with explainer",
    "reconstruction_1hop_no_gnn": "Reconstruction without GNN",
    "baseline_random": "Random baseline",
    "baseline_cosine": "Cosine baseline",
    "baseline_feature": "Feature baseline",
}

TABLE_COLUMNS = {
    "embedding_classification": "Fidelity embedding",
    "embedding_classification_explainer_subgraph": "Fidelity explainer",
    "raw_graph_reasoning": "Fidelity raw graph",
    "reconstruction_1hop": "Recon",
    "reconstruction_1hop_embed_expl": "Recon explainer",
    "reconstruction_1hop_no_gnn": "Recon no GNN",
}

EXPLAINER_RECONSTRUCTION_COLUMNS = {
    "reconstruction_1hop": "GNNExplainer recon",
    "reconstruction_1hop_embed_expl": "GNNExplainer recon with explainer",
    "reconstruction_1hop_no_gnn": "GNNExplainer recon no GNN",
}

COLORS = {
    "blue": "#2F6DB2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#7B4EA3",
    "red": "#C44E52",
    "gray": "#6B7280",
    "dark": "#111827",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def short_llm(name: Any) -> str:
    if name is None:
        return "unknown"
    text = str(name).split("/")[-1]
    text = text.replace("-Instruct", "")
    text = text.replace("Qwen", "Qwen ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dataset_label(name: Any) -> str:
    return DATASET_LABELS.get(str(name), str(name))


def ordered_unique(values: list[Any], preferred: list[str] | None = None) -> list[str]:
    unique = list(dict.fromkeys(str(value) for value in values if value not in (None, "")))
    if preferred is None:
        return sorted(unique)
    ordered = [value for value in preferred if value in unique]
    ordered.extend(sorted(value for value in unique if value not in ordered))
    return ordered


def parse_binary_label(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        return int(value) if value in (0.0, 1.0) else None

    text = str(value).strip().lower()
    if not text or text == "unknown":
        return None
    text = text.strip("`*_ \t\r\n\"'.,;:()[]{}")
    if text in {"0", "0.0", "licit", "normal", "benign", "negative"}:
        return 0
    if text in {"1", "1.0", "illicit", "suspicious", "fraud", "fraudulent", "positive"}:
        return 1

    patterns = [
        r"predicted[_\s-]*class\s*(?:is|=|:)?\s*([01])\b",
        r"\b(?:answer|prediction|label|classification|class)\s*(?:is|=|:)?\s*([01])\b",
        r"predicted[_\s-]*class\s*(?:is|=|:)?\s*(licit|illicit|normal|suspicious|fraudulent|fraud)\b",
        r"\b(?:answer|prediction|label|classification|class)\s*(?:is|=|:)?\s*(licit|illicit|normal|suspicious|fraudulent|fraud)\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            continue
        token = matches[-1].group(1).lower()
        if token in {"0", "1"}:
            return int(token)
        if token in {"licit", "normal"}:
            return 0
        if token in {"illicit", "suspicious", "fraud", "fraudulent"}:
            return 1
    return None


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
        "llm_error_rate": 1.0 if row.get("llm_error") else 0.0,
    }


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return empty_classification_metrics()

    total = len(rows)
    valid_n = correct = tp = fp = fn = llm_errors = 0
    for row in rows:
        true = parse_binary_label(row.get("gnn_pred"))
        pred = parse_binary_label(row.get("llm_pred"))
        if row.get("llm_error"):
            llm_errors += 1
        if pred is not None:
            valid_n += 1
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
        "n": float(total),
        "valid_n": float(valid_n),
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_rate": valid_n / total,
        "llm_error_rate": llm_errors / total,
    }


def empty_classification_metrics() -> dict[str, float]:
    return {
        "n": 0.0,
        "valid_n": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "parse_rate": 0.0,
        "llm_error_rate": 0.0,
    }


def reconstruction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
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
            "llm_error_rate": 0.0,
        }

    per_row = [reconstruction_row_metrics(row) for row in rows]
    result = {"n": float(len(rows))}
    for key in per_row[0]:
        result[key] = mean(item[key] for item in per_row)
    return result


def normalize_experiment_from_filename(path: Path) -> tuple[str, str]:
    name = path.name.removeprefix("results_raw_").removesuffix(".json")
    if name.endswith("_explainer"):
        base = name.removesuffix("_explainer")
        if base in RECONSTRUCTION_EXPERIMENTS or base in BASELINE_EXPERIMENTS:
            return base, "explainer"
    return name, "ground_truth"


def infer_path_metadata(path: Path) -> dict[str, Any]:
    parts = list(path.parts)
    metadata: dict[str, Any] = {}
    for part in parts:
        if re.fullmatch(r"task_\d+", part):
            metadata["task"] = part
            metadata.setdefault("llm", TASK_LLM.get(part))
        elif re.fullmatch(r"run_\d+", part):
            metadata["run"] = part
            metadata["replicate_id"] = part
        elif re.fullmatch(r"seed_\d+", part):
            metadata["seed"] = int(part.split("_")[-1])
            metadata["replicate_id"] = part
        elif part in DATASET_ORDER:
            metadata.setdefault("dataset", part)
        elif part in GNN_ORDER:
            metadata.setdefault("model", part)
        elif part.startswith("Qwen_"):
            metadata.setdefault("llm", part.replace("Qwen_", "Qwen/"))
    return metadata


def load_raw_rows(input_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for input_dir in input_dirs:
        for path in sorted(input_dir.rglob("results_raw_*.json")):
            if "llm_prompt_io" in path.name or path in seen:
                continue
            seen.add(path)
            experiment, reference = normalize_experiment_from_filename(path)
            path_meta = infer_path_metadata(path)
            try:
                payload = json.loads(path.read_text())
            except Exception as exc:
                inventory.append({"path": str(path), "experiment": experiment, "status": f"read_error: {exc}", "rows": 0})
                continue
            if not isinstance(payload, list):
                inventory.append({"path": str(path), "experiment": experiment, "status": "not_list", "rows": 0})
                continue

            added = 0
            for item in payload:
                if not isinstance(item, dict):
                    continue
                family = "classification"
                if experiment in RECONSTRUCTION_EXPERIMENTS:
                    family = "reconstruction"
                elif experiment in BASELINE_EXPERIMENTS:
                    family = "baseline"

                row = dict(item)
                # Prompt and raw response strings can be very large. Metrics only
                # need parsed labels, predicted neighbor ids, and error flags.
                for bulky_key in ("prompt", "llm_raw_response", "llm_raw_output"):
                    row.pop(bulky_key, None)
                row["source_file"] = str(path)
                row["family"] = family
                row["reference"] = reference
                row["experiment"] = row.get("experiment") or experiment
                row["dataset"] = row.get("dataset") or path_meta.get("dataset") or "unknown"
                row["model"] = row.get("model") or path_meta.get("model") or "unknown"
                if family == "baseline":
                    row["llm"] = "structural_baseline"
                else:
                    row["llm"] = row.get("llm") or path_meta.get("llm") or "unknown"
                row["task"] = path_meta.get("task")
                row["run"] = path_meta.get("run")
                row["seed"] = path_meta.get("seed")
                row["replicate_id"] = path_meta.get("replicate_id") or path.parent.name
                rows.append(row)
                added += 1

            inventory.append(
                {
                    "path": str(path),
                    "experiment": experiment,
                    "reference": reference,
                    "family": "classification"
                    if experiment in CLASSIFICATION_EXPERIMENTS
                    else "reconstruction"
                    if experiment in RECONSTRUCTION_EXPERIMENTS
                    else "baseline",
                    "status": "ok",
                    "rows": len(payload),
                    "rows_added": added,
                }
            )
    return rows, inventory


def aggregate_by_replicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = row.get("family")
        if family == "classification" and row.get("experiment") not in CLASSIFICATION_EXPERIMENTS:
            continue
        if family == "reconstruction" and row.get("experiment") not in RECONSTRUCTION_EXPERIMENTS:
            continue
        if family == "baseline" and row.get("experiment") not in BASELINE_EXPERIMENTS:
            continue
        key = (
            row.get("family"),
            row.get("reference"),
            row.get("experiment"),
            row.get("dataset"),
            row.get("model"),
            row.get("llm"),
            row.get("replicate_id"),
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        family, reference, experiment, dataset, model, llm, replicate_id = key
        metrics = classification_metrics(items) if family == "classification" else reconstruction_metrics(items)
        result = {
            "family": family,
            "reference": reference,
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "llm": llm,
            "replicate_id": replicate_id,
            "source_files": len(set(row.get("source_file") for row in items)),
        }
        result.update(metrics)
        out.append(result)
    return out


def aggregate_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("family"),
            row.get("reference"),
            row.get("experiment"),
            row.get("dataset"),
            row.get("model"),
            row.get("llm"),
        )
        grouped[key].append(row)

    metric_keys = sorted({key for row in rows for key in row if key not in {
        "family",
        "reference",
        "experiment",
        "dataset",
        "model",
        "llm",
        "replicate_id",
        "source_files",
    }})

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        family, reference, experiment, dataset, model, llm = key
        result = {
            "family": family,
            "reference": reference,
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "llm": llm,
            "replicates": len(items),
            "n_total": int(sum(safe_float(item.get("n"), 0.0) for item in items)),
        }
        for metric in metric_keys:
            values = [safe_float(item.get(metric)) for item in items]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            result[metric] = mean(values)
            result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        out.append(result)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = [
        "family",
        "reference",
        "experiment",
        "dataset",
        "model",
        "llm",
        "replicate_id",
        "replicates",
        "n",
        "n_total",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "parse_rate",
        "empty_pred_rate",
        "llm_error_rate",
    ]
    fieldnames = preferred + sorted({key for row in rows for key in row} - set(preferred))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_simple_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def positive_table_dataset_label(dataset: Any) -> str:
    labels = {
        "elliptic": "Elliptic",
        "ibm_aml_hi_small": "IBM AML HI-Small",
        "dgraphfin": "DGraph",
    }
    return labels.get(str(dataset), str(dataset))


def positive_table_experiment_label(experiment: Any) -> str:
    labels = {
        "embedding_classification": "Embedding classification",
        "embedding_classification_explainer_subgraph": r"\shortstack[l]{Embedding classification\\explainer subgraph}",
        "raw_graph_reasoning": "Raw graph reasoning",
    }
    return labels.get(str(experiment), str(experiment))


def _summarize_count_values(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "sd": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(mean(values)),
        "sd": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def make_fidelity_positive_summary(raw_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    """Summarize true-label positives and GNN-predicted positives in fidelity targets.

    Raw classification rows repeat the same GNN prediction for each LLM. This
    diagnostic de-duplicates by dataset, experiment, GNN, replicate, and target
    node before counting positives, so LLM count does not affect the rates.
    """

    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in raw_rows:
        if row.get("reference") != "ground_truth":
            continue
        if row.get("family") != "classification":
            continue
        experiment = str(row.get("experiment"))
        if experiment not in CLASSIFICATION_EXPERIMENTS:
            continue
        key = (
            row.get("dataset"),
            experiment,
            row.get("model"),
            row.get("replicate_id"),
            row.get("target_node"),
        )
        dedup[key] = {
            "dataset": row.get("dataset"),
            "experiment": experiment,
            "model": row.get("model"),
            "replicate_id": row.get("replicate_id"),
            "target_node": row.get("target_node"),
            "target_class": parse_binary_label(row.get("target_class")),
            "gnn_pred": parse_binary_label(row.get("gnn_pred")),
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in dedup.values():
        group_key = (item["dataset"], item["experiment"], item["model"], item["replicate_id"])
        grouped[group_key].append(item)

    group_rows: list[dict[str, Any]] = []
    for (dataset, experiment, model, replicate_id), items in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        targets = len(items)
        true_labeled = sum(1 for item in items if item["target_class"] in (0, 1))
        true_pos = sum(1 for item in items if item["target_class"] == 1)
        gnn_labeled = sum(1 for item in items if item["gnn_pred"] in (0, 1))
        gnn_pos = sum(1 for item in items if item["gnn_pred"] == 1)
        group_rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "model": model,
                "replicate_id": replicate_id,
                "targets": targets,
                "true_labeled": true_labeled,
                "true_pos": true_pos,
                "true_pos_rate": true_pos / true_labeled if true_labeled else math.nan,
                "gnn_labeled": gnn_labeled,
                "gnn_pos": gnn_pos,
                "gnn_pos_rate": gnn_pos / gnn_labeled if gnn_labeled else math.nan,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for experiment in CLASSIFICATION_EXPERIMENTS:
            items = [row for row in group_rows if row["dataset"] == dataset and row["experiment"] == experiment]
            if not items:
                continue
            true_counts = [int(row["true_pos"]) for row in items]
            gnn_counts = [int(row["gnn_pos"]) for row in items]
            target_counts = [int(row["targets"]) for row in items]
            true_stats = _summarize_count_values(true_counts)
            gnn_stats = _summarize_count_values(gnn_counts)
            target_stats = _summarize_count_values(target_counts)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "experiment": experiment,
                    "groups": len(items),
                    "targets_mean": target_stats["mean"],
                    "targets_min": target_stats["min"],
                    "targets_max": target_stats["max"],
                    "true_pos_mean": true_stats["mean"],
                    "true_pos_sd": true_stats["sd"],
                    "true_pos_min": true_stats["min"],
                    "true_pos_max": true_stats["max"],
                    "true_pos_rate": (
                        mean(row["true_pos_rate"] for row in items if math.isfinite(row["true_pos_rate"]))
                        if any(math.isfinite(row["true_pos_rate"]) for row in items)
                        else math.nan
                    ),
                    "gnn_pos_mean": gnn_stats["mean"],
                    "gnn_pos_sd": gnn_stats["sd"],
                    "gnn_pos_min": gnn_stats["min"],
                    "gnn_pos_max": gnn_stats["max"],
                    "gnn_pos_rate": (
                        mean(row["gnn_pos_rate"] for row in items if math.isfinite(row["gnn_pos_rate"]))
                        if any(math.isfinite(row["gnn_pos_rate"]) for row in items)
                        else math.nan
                    ),
                }
            )

    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    group_fields = [
        "dataset",
        "experiment",
        "model",
        "replicate_id",
        "targets",
        "true_labeled",
        "true_pos",
        "true_pos_rate",
        "gnn_labeled",
        "gnn_pos",
        "gnn_pos_rate",
    ]
    summary_fields = [
        "dataset",
        "experiment",
        "groups",
        "targets_mean",
        "targets_min",
        "targets_max",
        "true_pos_mean",
        "true_pos_sd",
        "true_pos_min",
        "true_pos_max",
        "true_pos_rate",
        "gnn_pos_mean",
        "gnn_pos_sd",
        "gnn_pos_min",
        "gnn_pos_max",
        "gnn_pos_rate",
    ]
    write_simple_csv(table_dir / "fidelity_positive_counts_by_gnn_seed.csv", group_rows, group_fields)
    write_simple_csv(table_dir / "fidelity_positive_target_vs_gnn_summary.csv", summary_rows, summary_fields)

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Experiment & Targets & True pos. & True rate & GNN pos. & GNN rate \\",
        r"\midrule",
    ]
    for row in summary_rows:
        latex_lines.append(
            "{} & {} & {:.0f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                latex_escape(positive_table_dataset_label(row["dataset"])),
                positive_table_experiment_label(row["experiment"]),
                safe_float(row["targets_mean"]),
                safe_float(row["true_pos_mean"]),
                safe_float(row["true_pos_rate"]),
                safe_float(row["gnn_pos_mean"]),
                safe_float(row["gnn_pos_rate"]),
            )
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Positive-class distribution of the sampled target nodes and the corresponding GNN predictions for the classification fidelity experiments. Targets denotes the mean number of sampled target nodes per GNN-seed group. True pos. reports the number of target nodes with ground-truth class 1, while GNN pos. reports how many of these targets were predicted as class 1 by the GNN.}",
            r"\label{tab:positive_sample_summary}",
            r"\end{table}",
            "",
        ]
    )
    (table_dir / "fidelity_positive_target_vs_gnn_summary.tex").write_text("\n".join(latex_lines))

    def write_old_shape_positive_table(
        filename: str,
        prefix: str,
        caption: str,
        label: str,
    ) -> None:
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Dataset & Experiment & Targets & Mean pos. & SD pos. & Min pos. & Max pos. & Pos. rate \\",
            r"\midrule",
        ]
        for row in summary_rows:
            lines.append(
                "{} & {} & {:.0f} & {:.3f} & {:.3f} & {:.0f} & {:.0f} & {:.3f} \\\\".format(
                    latex_escape(positive_table_dataset_label(row["dataset"])),
                    positive_table_experiment_label(row["experiment"]),
                    safe_float(row["targets_mean"]),
                    safe_float(row[f"{prefix}_pos_mean"]),
                    safe_float(row[f"{prefix}_pos_sd"]),
                    safe_float(row[f"{prefix}_pos_min"]),
                    safe_float(row[f"{prefix}_pos_max"]),
                    safe_float(row[f"{prefix}_pos_rate"]),
                )
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{table}",
                "",
            ]
        )
        (table_dir / filename).write_text("\n".join(lines))

    write_old_shape_positive_table(
        "fidelity_positive_true_label_summary.tex",
        "true",
        "Ground-truth positive-class distribution of the sampled target nodes for the classification experiments. Targets denotes the number of sampled target nodes per GNN-seed group. Mean pos., SD pos., Min pos., and Max pos. report the number of ground-truth positive target nodes, while Pos. rate reports the corresponding average positive-class rate.",
        "tab:true_positive_sample_summary",
    )
    write_old_shape_positive_table(
        "fidelity_positive_gnn_prediction_summary.tex",
        "gnn",
        "GNN-predicted positive-class distribution of the sampled target nodes for the classification fidelity experiments. Targets denotes the number of sampled target nodes per GNN-seed group. Mean pos., SD pos., Min pos., and Max pos. report the number of targets predicted as positive by the GNN, while Pos. rate reports the corresponding average positive-class rate.",
        "tab:gnn_positive_sample_summary",
    )

    if not summary_rows:
        return

    x = np.arange(len(datasets))
    width = 0.34
    fig, axes = plt.subplots(1, len(CLASSIFICATION_EXPERIMENTS), figsize=(12.5, 3.6), sharey=True)
    if len(CLASSIFICATION_EXPERIMENTS) == 1:
        axes = [axes]
    for ax, experiment in zip(axes, CLASSIFICATION_EXPERIMENTS):
        subset = [row for row in summary_rows if row["experiment"] == experiment]
        lookup = {row["dataset"]: row for row in subset}
        true_rates = [safe_float(lookup.get(dataset, {}).get("true_pos_rate")) for dataset in datasets]
        gnn_rates = [safe_float(lookup.get(dataset, {}).get("gnn_pos_rate")) for dataset in datasets]
        ax.bar(x - width / 2, true_rates, width=width, label="True labels", color=COLORS["blue"])
        ax.bar(x + width / 2, gnn_rates, width=width, label="GNN predictions", color=COLORS["orange"])
        ax.set_title(EXPERIMENT_LABELS.get(experiment, experiment))
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_label(dataset) for dataset in datasets], rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Positive-class rate")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Positive-class target labels versus GNN positive predictions")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, figure_dir / "06_fidelity_positive_target_vs_gnn.png")


def matrix_from_rows(
    rows: list[dict[str, Any]],
    row_key: str,
    col_key: str,
    value_key: str,
    row_order: list[str],
    col_order: list[str],
) -> np.ndarray:
    lookup = {(str(row.get(row_key)), str(row.get(col_key))): safe_float(row.get(value_key)) for row in rows}
    out = np.full((len(row_order), len(col_order)), np.nan)
    for i, row_value in enumerate(row_order):
        for j, col_value in enumerate(col_order):
            out[i, j] = lookup.get((row_value, col_value), np.nan)
    return out


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "YlGnBu",
) -> None:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#F3F4F6")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left", pad=8)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if math.isfinite(value):
                text = f"{value:.2f}"
                color = "white" if value >= 0.55 else "#111827"
            else:
                text = "NA"
                color = "#6B7280"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_panels(
    mean_rows: list[dict[str, Any]],
    out_path: Path,
    experiment: str,
    title: str,
    datasets: list[str],
    metric: str = "f1",
    reference: str = "ground_truth",
) -> None:
    rows = [
        row
        for row in mean_rows
        if row.get("experiment") == experiment
        and row.get("reference") == reference
        and row.get("family") in {"classification", "reconstruction"}
    ]
    llms = ordered_unique([row.get("llm") for row in rows], LLM_ORDER)
    models = ordered_unique([row.get("model") for row in rows], GNN_ORDER)
    if not llms or not models:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.8), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle(title)
    for ax, dataset in zip(axes, datasets):
        panel_rows = [row for row in rows if row.get("dataset") == dataset]
        values = matrix_from_rows(panel_rows, "llm", "model", metric, llms, models)
        draw_heatmap(
            ax,
            values,
            [short_llm(llm) for llm in llms],
            models,
            dataset_label(dataset),
        )
        ax.set_xlabel("GNN")
        ax.set_ylabel("LLM")
    save_figure(fig, out_path)


def plot_reconstruction_baselines(
    mean_rows: list[dict[str, Any]],
    out_path: Path,
    datasets: list[str],
    reference: str = "ground_truth",
    title: str = "Best reconstruction F1 compared with baselines",
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 4.7), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle(title)
    labels = ["Best LLM", "Random", "Cosine", "Feature"]
    colors = [COLORS["blue"], COLORS["gray"], COLORS["orange"], COLORS["green"]]
    for ax, dataset in zip(axes, datasets):
        llm_rows = [
            row
            for row in mean_rows
            if row.get("dataset") == dataset
            and row.get("experiment") == "reconstruction_1hop"
            and row.get("reference") == reference
            and row.get("family") == "reconstruction"
        ]
        best_llm = None
        if llm_rows:
            finite_llm_rows = [row for row in llm_rows if math.isfinite(safe_float(row.get("f1")))]
            if finite_llm_rows:
                best_llm = max(finite_llm_rows, key=lambda row: safe_float(row.get("f1")))

        baseline_values = []
        for experiment in BASELINE_EXPERIMENTS:
            values = [
                safe_float(row.get("f1"))
                for row in mean_rows
                if row.get("dataset") == dataset
                and row.get("experiment") == experiment
                and row.get("reference") == reference
                and row.get("family") == "baseline"
            ]
            values = [value for value in values if math.isfinite(value)]
            baseline_values.append(max(values) if values else math.nan)
        best_llm_value = safe_float(best_llm.get("f1")) if best_llm else math.nan
        values = [best_llm_value] + baseline_values
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors)
        ax.set_title(dataset_label(dataset), loc="left")
        ax.set_xticks(x)
        tick_labels = labels.copy()
        if best_llm:
            tick_labels[0] = (
                f"Best LLM\n"
                f"{best_llm.get('model')}\n"
                f"{short_llm(best_llm.get('llm'))}"
            )
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if math.isfinite(value):
                ax.text(idx, min(value + 0.025, 0.98), f"{value:.2f}", ha="center", fontsize=8)
    save_figure(fig, out_path)


def plot_precision_recall(mean_rows: list[dict[str, Any]], out_path: Path, datasets: list[str]) -> None:
    rows = [
        row
        for row in mean_rows
        if row.get("experiment") == "reconstruction_1hop"
        and row.get("reference") == "ground_truth"
        and row.get("family") == "reconstruction"
    ]
    llms = ordered_unique([row.get("llm") for row in rows], LLM_ORDER)
    models = ordered_unique([row.get("model") for row in rows], GNN_ORDER)
    if not llms or not models:
        return

    fig, axes = plt.subplots(2, len(datasets), figsize=(5.2 * len(datasets), 8.2), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.suptitle("Reconstruction precision and recall by dataset, GNN, and LLM")
    for col, dataset in enumerate(datasets):
        panel_rows = [row for row in rows if row.get("dataset") == dataset]
        for row_idx, metric in enumerate(["precision", "recall"]):
            ax = axes[row_idx, col]
            values = matrix_from_rows(panel_rows, "llm", "model", metric, llms, models)
            draw_heatmap(
                ax,
                values,
                [short_llm(llm) for llm in llms],
                models,
                f"{dataset_label(dataset)} {metric}",
            )
            ax.set_xlabel("GNN")
            ax.set_ylabel("LLM")
    save_figure(fig, out_path)


def plot_behavior_rates(mean_rows: list[dict[str, Any]], out_path: Path, datasets: list[str]) -> None:
    llms = ordered_unique([row.get("llm") for row in mean_rows if row.get("family") != "baseline"], LLM_ORDER)
    if not llms:
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.0 * len(datasets), 7.2), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.suptitle("LLM output behavior")
    for col, dataset in enumerate(datasets):
        class_rows = [
            row
            for row in mean_rows
            if row.get("dataset") == dataset
            and row.get("experiment") == "embedding_classification"
            and row.get("family") == "classification"
        ]
        recon_rows = [
            row
            for row in mean_rows
            if row.get("dataset") == dataset
            and row.get("experiment") == "reconstruction_1hop"
            and row.get("family") == "reconstruction"
        ]
        for ax, rows, metric, ylabel in [
            (axes[0, col], class_rows, "parse_rate", "Parse rate"),
            (axes[1, col], recon_rows, "empty_pred_rate", "Empty prediction rate"),
        ]:
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                value = safe_float(row.get(metric))
                if math.isfinite(value):
                    grouped[str(row.get("llm"))].append(value)
            values = [mean(grouped[llm]) if grouped.get(llm) else math.nan for llm in llms]
            x = np.arange(len(llms))
            ax.bar(x, values, color=COLORS["purple"] if metric == "parse_rate" else COLORS["red"])
            ax.set_title(dataset_label(dataset), loc="left")
            ax.set_xticks(x)
            ax.set_xticklabels([short_llm(llm) for llm in llms], rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_path)


def make_dataset_tables(mean_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    try:
        import pandas as pd
    except Exception:
        pd = None

    table_rows: list[dict[str, Any]] = []
    for row in mean_rows:
        if row.get("reference") != "ground_truth":
            continue
        if row.get("family") not in {"classification", "reconstruction"}:
            continue
        experiment = str(row.get("experiment"))
        if experiment not in TABLE_COLUMNS:
            continue
        table_rows.append(
            {
                "dataset": row.get("dataset"),
                "GNN": row.get("model"),
                "LLM": short_llm(row.get("llm")),
                "experiment": TABLE_COLUMNS[experiment],
                "F1": safe_float(row.get("f1")),
            }
        )

    if pd is None:
        write_simple_csv(
            out_dir / "tables" / "dataset_experiment_f1_long.csv",
            table_rows,
            ["dataset", "GNN", "LLM", "experiment", "F1"],
        )
        ordered_cols = list(TABLE_COLUMNS.values())
        for dataset in datasets:
            subset = [row for row in table_rows if row["dataset"] == dataset]
            if not subset:
                continue
            combos = sorted(
                {(row["GNN"], row["LLM"]) for row in subset},
                key=lambda item: (
                    GNN_ORDER.index(item[0]) if item[0] in GNN_ORDER else len(GNN_ORDER),
                    item[1],
                ),
            )
            values = {(row["GNN"], row["LLM"], row["experiment"]): row["F1"] for row in subset}
            pivot_rows = []
            for gnn, llm in combos:
                pivot_row = {"GNN": gnn, "LLM": llm}
                for col in ordered_cols:
                    value = values.get((gnn, llm, col), math.nan)
                    pivot_row[col] = "" if not math.isfinite(value) else round(value, 3)
                pivot_rows.append(pivot_row)
            write_simple_csv(
                out_dir / "tables" / f"{dataset}_experiment_f1_table.csv",
                pivot_rows,
                ["GNN", "LLM"] + ordered_cols,
            )
        return

    df = pd.DataFrame(table_rows)
    if df.empty:
        return
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / "dataset_experiment_f1_long.csv", index=False)
    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index=["GNN", "LLM"], columns="experiment", values="F1", aggfunc="mean")
        ordered_cols = [label for label in TABLE_COLUMNS.values() if label in pivot.columns]
        pivot = pivot.reindex(columns=ordered_cols)
        pivot = pivot.sort_index()
        csv_path = out_dir / "tables" / f"{dataset}_experiment_f1_table.csv"
        tex_path = out_dir / "tables" / f"{dataset}_experiment_f1_table.tex"
        pivot.round(3).to_csv(csv_path)
        pivot.round(3).to_latex(tex_path, na_rep="", escape=True)


def make_explainer_reconstruction_tables(mean_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    try:
        import pandas as pd
    except Exception:
        pd = None

    table_rows: list[dict[str, Any]] = []
    for row in mean_rows:
        if row.get("reference") != "explainer":
            continue
        if row.get("family") != "reconstruction":
            continue
        experiment = str(row.get("experiment"))
        if experiment not in EXPLAINER_RECONSTRUCTION_COLUMNS:
            continue
        f1 = safe_float(row.get("f1"))
        if not math.isfinite(f1):
            continue
        table_rows.append(
            {
                "dataset": row.get("dataset"),
                "GNN": row.get("model"),
                "LLM": short_llm(row.get("llm")),
                "experiment": EXPLAINER_RECONSTRUCTION_COLUMNS[experiment],
                "F1": f1,
            }
        )

    ordered_cols = list(EXPLAINER_RECONSTRUCTION_COLUMNS.values())
    if pd is None:
        write_simple_csv(
            out_dir / "tables" / "dataset_explainer_reconstruction_f1_long.csv",
            table_rows,
            ["dataset", "GNN", "LLM", "experiment", "F1"],
        )
        for dataset in datasets:
            subset = [row for row in table_rows if row["dataset"] == dataset]
            if not subset:
                continue
            combos = sorted(
                {(row["GNN"], row["LLM"]) for row in subset},
                key=lambda item: (
                    GNN_ORDER.index(item[0]) if item[0] in GNN_ORDER else len(GNN_ORDER),
                    item[1],
                ),
            )
            values = {(row["GNN"], row["LLM"], row["experiment"]): row["F1"] for row in subset}
            pivot_rows = []
            for gnn, llm in combos:
                pivot_row = {"GNN": gnn, "LLM": llm}
                for col in ordered_cols:
                    value = values.get((gnn, llm, col), math.nan)
                    pivot_row[col] = "" if not math.isfinite(value) else round(value, 3)
                pivot_rows.append(pivot_row)
            write_simple_csv(
                out_dir / "tables" / f"{dataset}_explainer_reconstruction_f1_table.csv",
                pivot_rows,
                ["GNN", "LLM"] + ordered_cols,
            )
        return

    df = pd.DataFrame(table_rows)
    if df.empty:
        return
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / "dataset_explainer_reconstruction_f1_long.csv", index=False)
    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index=["GNN", "LLM"], columns="experiment", values="F1", aggfunc="mean")
        pivot = pivot.reindex(columns=[col for col in ordered_cols if col in pivot.columns])
        pivot = pivot.sort_index()
        csv_path = out_dir / "tables" / f"{dataset}_explainer_reconstruction_f1_table.csv"
        tex_path = out_dir / "tables" / f"{dataset}_explainer_reconstruction_f1_table.tex"
        pivot.round(3).to_csv(csv_path)
        pivot.round(3).to_latex(tex_path, na_rep="", escape=True)


def make_fidelity_accuracy_tables(mean_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    try:
        import pandas as pd
    except Exception:
        pd = None

    table_rows: list[dict[str, Any]] = []
    for row in mean_rows:
        if row.get("reference") != "ground_truth":
            continue
        if row.get("family") != "classification":
            continue
        experiment = str(row.get("experiment"))
        if experiment not in TABLE_COLUMNS:
            continue
        accuracy = safe_float(row.get("accuracy"))
        if not math.isfinite(accuracy):
            continue
        table_rows.append(
            {
                "dataset": row.get("dataset"),
                "GNN": row.get("model"),
                "LLM": short_llm(row.get("llm")),
                "experiment": TABLE_COLUMNS[experiment],
                "Accuracy": accuracy,
            }
        )

    ordered_cols = [TABLE_COLUMNS[experiment] for experiment in CLASSIFICATION_EXPERIMENTS]
    if pd is None:
        write_simple_csv(
            out_dir / "tables" / "dataset_fidelity_accuracy_long.csv",
            table_rows,
            ["dataset", "GNN", "LLM", "experiment", "Accuracy"],
        )
        for dataset in datasets:
            subset = [row for row in table_rows if row["dataset"] == dataset]
            if not subset:
                continue
            combos = sorted(
                {(row["GNN"], row["LLM"]) for row in subset},
                key=lambda item: (
                    GNN_ORDER.index(item[0]) if item[0] in GNN_ORDER else len(GNN_ORDER),
                    item[1],
                ),
            )
            values = {(row["GNN"], row["LLM"], row["experiment"]): row["Accuracy"] for row in subset}
            pivot_rows = []
            for gnn, llm in combos:
                pivot_row = {"GNN": gnn, "LLM": llm}
                for col in ordered_cols:
                    value = values.get((gnn, llm, col), math.nan)
                    pivot_row[col] = "" if not math.isfinite(value) else round(value, 3)
                pivot_rows.append(pivot_row)
            write_simple_csv(
                out_dir / "tables" / f"{dataset}_fidelity_accuracy_table.csv",
                pivot_rows,
                ["GNN", "LLM"] + ordered_cols,
            )
        return

    df = pd.DataFrame(table_rows)
    if df.empty:
        return
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / "dataset_fidelity_accuracy_long.csv", index=False)
    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index=["GNN", "LLM"], columns="experiment", values="Accuracy", aggfunc="mean")
        pivot = pivot.reindex(columns=[col for col in ordered_cols if col in pivot.columns])
        pivot = pivot.sort_index()
        csv_path = out_dir / "tables" / f"{dataset}_fidelity_accuracy_table.csv"
        tex_path = out_dir / "tables" / f"{dataset}_fidelity_accuracy_table.tex"
        pivot.round(3).to_csv(csv_path)
        pivot.round(3).to_latex(tex_path, na_rep="", escape=True)


def make_best_tables(mean_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    best_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for experiment in CLASSIFICATION_EXPERIMENTS + RECONSTRUCTION_EXPERIMENTS:
            candidates = [
                row
                for row in mean_rows
                if row.get("dataset") == dataset
                and row.get("experiment") == experiment
                and row.get("reference") == "ground_truth"
                and row.get("family") in {"classification", "reconstruction"}
                and math.isfinite(safe_float(row.get("f1")))
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: safe_float(row.get("f1")))
            best_rows.append(
                {
                    "dataset": dataset,
                    "experiment": experiment,
                    "GNN": best.get("model"),
                    "LLM": best.get("llm"),
                    "F1": safe_float(best.get("f1")),
                    "precision": safe_float(best.get("precision")),
                    "recall": safe_float(best.get("recall")),
                    "accuracy": safe_float(best.get("accuracy")),
                    "jaccard": safe_float(best.get("jaccard")),
                    "replicates": best.get("replicates"),
                    "n_total": best.get("n_total"),
                }
            )
    write_simple_csv(
        out_dir / "tables" / "best_by_dataset_experiment.csv",
        best_rows,
        ["dataset", "experiment", "GNN", "LLM", "F1", "precision", "recall", "accuracy", "jaccard", "replicates", "n_total"],
    )


def make_best_explainer_reconstruction_table(mean_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    best_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for experiment in RECONSTRUCTION_EXPERIMENTS:
            candidates = [
                row
                for row in mean_rows
                if row.get("dataset") == dataset
                and row.get("experiment") == experiment
                and row.get("reference") == "explainer"
                and row.get("family") == "reconstruction"
                and math.isfinite(safe_float(row.get("f1")))
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: safe_float(row.get("f1")))
            best_rows.append(
                {
                    "dataset": dataset,
                    "experiment": experiment,
                    "GNN": best.get("model"),
                    "LLM": best.get("llm"),
                    "F1": safe_float(best.get("f1")),
                    "precision": safe_float(best.get("precision")),
                    "recall": safe_float(best.get("recall")),
                    "jaccard": safe_float(best.get("jaccard")),
                    "replicates": best.get("replicates"),
                    "n_total": best.get("n_total"),
                }
            )
    write_simple_csv(
        out_dir / "tables" / "best_by_dataset_explainer_reconstruction.csv",
        best_rows,
        ["dataset", "experiment", "GNN", "LLM", "F1", "precision", "recall", "jaccard", "replicates", "n_total"],
    )


def make_baseline_table(mean_rows: list[dict[str, Any]], out_dir: Path, datasets: list[str]) -> None:
    for reference, filename in [
        ("ground_truth", "baseline_comparison_best.csv"),
        ("explainer", "baseline_comparison_best_explainer.csv"),
    ]:
        rows: list[dict[str, Any]] = []
        for dataset in datasets:
            for experiment in ["reconstruction_1hop"] + BASELINE_EXPERIMENTS:
                family = "reconstruction" if experiment == "reconstruction_1hop" else "baseline"
                candidates = [
                    row
                    for row in mean_rows
                    if row.get("dataset") == dataset
                    and row.get("experiment") == experiment
                    and row.get("reference") == reference
                    and row.get("family") == family
                    and math.isfinite(safe_float(row.get("f1")))
                ]
                if not candidates:
                    continue
                best = max(candidates, key=lambda row: safe_float(row.get("f1")))
                rows.append(
                    {
                        "dataset": dataset,
                        "reference": reference,
                        "method": EXPERIMENT_LABELS[experiment],
                        "GNN": best.get("model"),
                        "LLM": best.get("llm") if family == "reconstruction" else "",
                        "F1": safe_float(best.get("f1")),
                        "precision": safe_float(best.get("precision")),
                        "recall": safe_float(best.get("recall")),
                        "jaccard": safe_float(best.get("jaccard")),
                    }
                )
        write_simple_csv(
            out_dir / "tables" / filename,
            rows,
            ["dataset", "reference", "method", "GNN", "LLM", "F1", "precision", "recall", "jaccard"],
        )


def write_readme(out_dir: Path, input_dirs: list[Path]) -> None:
    lines = [
        "# Thesis Results",
        "",
        "Generated by `plot_thesis_results.py`.",
        "",
        "Input directories:",
    ]
    lines.extend(f"- `{path}`" for path in input_dirs)
    lines.extend(
        [
            "",
            "Main files:",
            "- `summary_mean_metrics.csv`: metrics averaged over runs or seeds.",
            "- `summary_by_replicate_metrics.csv`: one metric row per run or seed.",
            "- `tables/*_experiment_f1_table.csv`: dataset tables for the thesis.",
            "- `tables/*_experiment_f1_table.tex`: LaTeX versions of the dataset tables.",
            "- `tables/*_explainer_reconstruction_f1_table.csv`: GNNExplainer-subgraph reconstruction F1 tables.",
            "- `tables/*_explainer_reconstruction_f1_table.tex`: LaTeX versions of the GNNExplainer reconstruction tables.",
            "- `tables/*_fidelity_accuracy_table.csv`: dataset fidelity accuracy tables.",
            "- `tables/*_fidelity_accuracy_table.tex`: LaTeX versions of the fidelity accuracy tables.",
            "- `figures/01_fidelity_f1_heatmap.png`: F1 heatmaps for fidelity.",
            "- `figures/01b_fidelity_accuracy_heatmap.png`: accuracy heatmaps for fidelity.",
            "- `figures/02_neighbourhood_f1_heatmap.png`: F1 heatmaps for neighbourhood reconstruction.",
            "- `figures/02b_explainer_reconstruction_f1_heatmap.png`: F1 heatmaps for GNNExplainer-subgraph reconstruction.",
            "- `figures/03_reconstruction_baselines.png`: 1-hop neighbourhood reconstruction compared with baselines.",
            "- `figures/03b_explainer_reconstruction_baselines.png`: best GNNExplainer reconstruction method compared with baselines.",
            "- `figures/04_reconstruction_precision_recall.png`: precision and recall heatmaps.",
            "- `figures/05_llm_behavior_rates.png`: parse and empty-output behavior.",
            "- `figures/06_fidelity_positive_target_vs_gnn.png`: true positive target labels compared with GNN positive predictions.",
            "- `tables/fidelity_positive_target_vs_gnn_summary.tex`: LaTeX table for the fidelity target-positive diagnostic.",
            "- `tables/fidelity_positive_true_label_summary.tex`: old-shape LaTeX table for ground-truth target positives.",
            "- `tables/fidelity_positive_gnn_prediction_summary.tex`: old-shape LaTeX table for GNN-predicted positives.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="Output directory to scan. Pass multiple times for multiple final result trees.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("thesis_results"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASET_ORDER,
        help="Datasets and panel order for plots.",
    )
    parser.add_argument(
        "--fidelity-experiment",
        default="embedding_classification",
        choices=CLASSIFICATION_EXPERIMENTS,
        help="Classification experiment used for the main fidelity heatmap.",
    )
    parser.add_argument(
        "--neighbourhood-experiment",
        default="reconstruction_1hop",
        choices=RECONSTRUCTION_EXPERIMENTS,
        help="Reconstruction experiment used for the main neighbourhood heatmap.",
    )
    args = parser.parse_args()

    configure_style()
    input_dirs = [path.resolve() for path in args.input_dir]
    out_dir = args.output_dir.resolve()
    figure_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, inventory = load_raw_rows(input_dirs)
    by_replicate = aggregate_by_replicate(raw_rows)
    mean_rows = aggregate_mean(by_replicate)

    write_csv(out_dir / "result_file_inventory.csv", inventory)
    write_csv(out_dir / "summary_by_replicate_metrics.csv", by_replicate)
    write_csv(out_dir / "summary_mean_metrics.csv", mean_rows)
    make_dataset_tables(mean_rows, out_dir, args.datasets)
    make_explainer_reconstruction_tables(mean_rows, out_dir, args.datasets)
    make_fidelity_accuracy_tables(mean_rows, out_dir, args.datasets)
    make_best_tables(mean_rows, out_dir, args.datasets)
    make_best_explainer_reconstruction_table(mean_rows, out_dir, args.datasets)
    make_baseline_table(mean_rows, out_dir, args.datasets)
    make_fidelity_positive_summary(raw_rows, out_dir, args.datasets)

    plot_heatmap_panels(
        mean_rows,
        figure_dir / "01_fidelity_f1_heatmap.png",
        args.fidelity_experiment,
        "Fidelity F1 by dataset, GNN, and LLM",
        args.datasets,
        metric="f1",
    )
    plot_heatmap_panels(
        mean_rows,
        figure_dir / "01b_fidelity_accuracy_heatmap.png",
        args.fidelity_experiment,
        "Fidelity accuracy by dataset, GNN, and LLM",
        args.datasets,
        metric="accuracy",
    )
    plot_heatmap_panels(
        mean_rows,
        figure_dir / "02_neighbourhood_f1_heatmap.png",
        args.neighbourhood_experiment,
        "Neighbourhood reconstruction F1 by dataset, GNN, and LLM",
        args.datasets,
        metric="f1",
        reference="ground_truth",
    )
    plot_heatmap_panels(
        mean_rows,
        figure_dir / "02b_explainer_reconstruction_f1_heatmap.png",
        args.neighbourhood_experiment,
        "GNNExplainer-subgraph reconstruction F1 by dataset, GNN, and LLM",
        args.datasets,
        metric="f1",
        reference="explainer",
    )
    plot_reconstruction_baselines(
        mean_rows,
        figure_dir / "03_reconstruction_baselines.png",
        args.datasets,
        reference="ground_truth",
        title="1-hop neighbourhood reconstruction F1 compared with baselines",
    )
    plot_reconstruction_baselines(
        mean_rows,
        figure_dir / "03b_explainer_reconstruction_baselines.png",
        args.datasets,
        reference="explainer",
        title="Best GNNExplainer-subgraph reconstruction F1 compared with baselines",
    )
    plot_precision_recall(mean_rows, figure_dir / "04_reconstruction_precision_recall.png", args.datasets)
    plot_behavior_rates(mean_rows, figure_dir / "05_llm_behavior_rates.png", args.datasets)
    write_readme(out_dir, input_dirs)

    print(f"Raw rows: {len(raw_rows)}")
    print(f"Result files: {len(inventory)}")
    print(f"Replicate metric rows: {len(by_replicate)}")
    print(f"Mean metric rows: {len(mean_rows)}")
    print(f"Wrote thesis outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
