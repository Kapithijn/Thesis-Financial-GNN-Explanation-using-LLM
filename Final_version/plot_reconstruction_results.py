#!/usr/bin/env python3
"""Plot reconstruction benchmark summaries without requiring pandas/seaborn."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path


if "MPLCONFIGDIR" not in os.environ:
	cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
	cache_dir.mkdir(parents=True, exist_ok=True)
	os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib.pyplot as plt


METRICS = ["precision", "recall", "f1", "jaccard"]
MODEL_ORDER = ["GCN", "GAT", "GIN", "GraphSAGE"]
LLM_ORDER = [
	"Qwen/Qwen2.5-3B",
	"Qwen/Qwen3.5-4B",
	"Qwen/Qwen3.5-9B",
	"Qwen/Qwen3.5-27B",
	"Qwen/Qwen3.5-35B-A3B",
]


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("outputs/recon_elliptic_100nodes_allgnn_allllm_reconstruction_1hop"),
		help="Directory containing reconstruction summary files.",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=None,
		help="Directory for plots. Defaults to <input-dir>/plots.",
	)
	parser.add_argument("--metric", default="f1", choices=METRICS + ["overlap"], help="Primary heatmap metric.")
	return parser.parse_args()


def short_llm(name: str) -> str:
	return name.replace("Qwen/", "").replace("Qwen", "")


def ordered(values, preferred):
	seen = set(values)
	head = [value for value in preferred if value in seen]
	tail = sorted(seen - set(head))
	return head + tail


def as_float(value, default=0.0):
	try:
		return float(value)
	except Exception:
		return default


def as_int(value, default=0):
	try:
		return int(float(value))
	except Exception:
		return default


def load_by_model_llm(input_dir: Path):
	path = input_dir / "results_summary_reconstruction_1hop_by_model_llm.csv"
	if not path.exists():
		raise FileNotFoundError(f"Missing grouped summary: {path}")
	rows = []
	with path.open(newline="") as handle:
		for row in csv.DictReader(handle):
			clean = dict(row)
			for key in ["rows", "errors", "empty_predictions", "filtered_ids"]:
				clean[key] = as_int(clean.get(key))
			for key in METRICS + ["overlap", "edit_distance"]:
				clean[key] = as_float(clean.get(key))
			rows.append(clean)
	return rows


def load_json_dict(path: Path):
	if not path.exists():
		return {}
	obj = json.loads(path.read_text())
	return obj if isinstance(obj, dict) else {}


def save(fig, out_dir: Path, stem: str):
	out_dir.mkdir(parents=True, exist_ok=True)
	for suffix in [".png", ".pdf"]:
		fig.savefig(out_dir / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_metric_heatmap(rows, models, llms, metric, out_dir):
	lookup = {(row["model"], row["llm"]): row for row in rows}
	matrix = []
	for model in models:
		matrix.append([lookup.get((model, llm), {}).get(metric, math.nan) for llm in llms])

	fig, ax = plt.subplots(figsize=(max(8.0, 1.35 * len(llms)), max(3.6, 0.85 * len(models))))
	im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=max(0.35, max(v for row in matrix for v in row if not math.isnan(v))))
	ax.set_xticks(range(len(llms)), [short_llm(llm) for llm in llms], rotation=30, ha="right")
	ax.set_yticks(range(len(models)), models)
	ax.set_title(f"Elliptic reconstruction 1-hop: {metric} by GNN and LLM")
	ax.set_xlabel("LLM")
	ax.set_ylabel("GNN")
	for y, row in enumerate(matrix):
		for x, value in enumerate(row):
			label = "NA" if math.isnan(value) else f"{value:.3f}"
			ax.text(x, y, label, ha="center", va="center", fontsize=9)
	fig.colorbar(im, ax=ax, label=metric)
	save(fig, out_dir, f"reconstruction_1hop_{metric}_heatmap")


def plot_grouped_f1(rows, models, llms, explainer_summary, out_dir):
	lookup = {(row["model"], row["llm"]): row for row in rows}
	colors = plt.get_cmap("tab10").colors
	width = 0.82 / max(1, len(llms))
	x_positions = list(range(len(models)))

	fig, ax = plt.subplots(figsize=(10.5, 4.8))
	for idx, llm in enumerate(llms):
		offset = (idx - (len(llms) - 1) / 2) * width
		values = [lookup.get((model, llm), {}).get("f1", 0.0) for model in models]
		ax.bar(
			[x + offset for x in x_positions],
			values,
			width=width,
			label=short_llm(llm),
			color=colors[idx % len(colors)],
		)
	if "f1" in explainer_summary:
		ax.axhline(float(explainer_summary["f1"]), color="black", linestyle="--", linewidth=1.2, label="Explainer aggregate")
	ax.set_xticks(x_positions, models)
	ax.set_ylabel("F1")
	ax.set_ylim(0, 0.36)
	ax.set_title("Elliptic reconstruction 1-hop F1 by GNN and LLM")
	ax.legend(ncol=3, fontsize=8, frameon=False)
	ax.grid(axis="y", alpha=0.25)
	save(fig, out_dir, "reconstruction_1hop_f1_grouped")


def plot_llm_metric_averages(rows, llms, out_dir):
	averages = {}
	for llm in llms:
		llm_rows = [row for row in rows if row["llm"] == llm]
		averages[llm] = {
			metric: sum(row[metric] for row in llm_rows) / max(1, len(llm_rows))
			for metric in METRICS
		}

	width = 0.82 / len(METRICS)
	colors = plt.get_cmap("Set2").colors
	x_positions = list(range(len(llms)))
	fig, ax = plt.subplots(figsize=(10.5, 4.8))
	for idx, metric in enumerate(METRICS):
		offset = (idx - (len(METRICS) - 1) / 2) * width
		values = [averages[llm][metric] for llm in llms]
		ax.bar(
			[x + offset for x in x_positions],
			values,
			width=width,
			label=metric,
			color=colors[idx % len(colors)],
		)
	ax.set_xticks(x_positions, [short_llm(llm) for llm in llms], rotation=30, ha="right")
	ax.set_ylabel("Score averaged over GNNs")
	ax.set_ylim(0, 0.9)
	ax.set_title("Elliptic reconstruction 1-hop metrics averaged over GNNs")
	ax.legend(ncol=4, fontsize=8, frameon=False)
	ax.grid(axis="y", alpha=0.25)
	save(fig, out_dir, "reconstruction_1hop_metrics_by_llm")


def plot_empty_prediction_heatmap(rows, models, llms, out_dir):
	lookup = {(row["model"], row["llm"]): row for row in rows}
	matrix = []
	for model in models:
		row_values = []
		for llm in llms:
			row = lookup.get((model, llm), {})
			total = max(1, int(row.get("rows", 0)))
			row_values.append(float(row.get("empty_predictions", 0)) / total)
		matrix.append(row_values)

	fig, ax = plt.subplots(figsize=(max(8.0, 1.35 * len(llms)), max(3.6, 0.85 * len(models))))
	im = ax.imshow(matrix, cmap="OrRd", vmin=0.0, vmax=1.0)
	ax.set_xticks(range(len(llms)), [short_llm(llm) for llm in llms], rotation=30, ha="right")
	ax.set_yticks(range(len(models)), models)
	ax.set_title("Empty reconstruction predictions by GNN and LLM")
	ax.set_xlabel("LLM")
	ax.set_ylabel("GNN")
	for y, row in enumerate(matrix):
		for x, value in enumerate(row):
			ax.text(x, y, f"{100 * value:.0f}%", ha="center", va="center", fontsize=9)
	fig.colorbar(im, ax=ax, label="Empty predictions")
	save(fig, out_dir, "reconstruction_1hop_empty_predictions_heatmap")


def plot_aggregate_comparison(summary, explainer_summary, out_dir):
	fig, ax = plt.subplots(figsize=(7.6, 4.5))
	x_positions = list(range(len(METRICS)))
	width = 0.36
	llm_values = [float(summary.get(metric, 0.0)) for metric in METRICS]
	explainer_values = [float(explainer_summary.get(metric, 0.0)) for metric in METRICS]
	ax.bar([x - width / 2 for x in x_positions], llm_values, width=width, label="LLM reconstruction", color="#4c78a8")
	ax.bar([x + width / 2 for x in x_positions], explainer_values, width=width, label="Explainer-selected subgraph", color="#f58518")
	ax.set_xticks(x_positions, METRICS)
	ax.set_ylabel("Score")
	ax.set_ylim(0, max(0.2, max(llm_values + explainer_values) * 1.25))
	ax.set_title("Aggregate reconstruction quality")
	ax.legend(frameon=False)
	ax.grid(axis="y", alpha=0.25)
	save(fig, out_dir, "reconstruction_1hop_aggregate_comparison")


def main():
	args = parse_args()
	input_dir = args.input_dir.expanduser().resolve()
	out_dir = (args.out_dir or (input_dir / "plots")).expanduser().resolve()

	rows = load_by_model_llm(input_dir)
	models = ordered([row["model"] for row in rows], MODEL_ORDER)
	llms = ordered([row["llm"] for row in rows], LLM_ORDER)
	summary = load_json_dict(input_dir / "results_summary_reconstruction_1hop.json")
	explainer_summary = load_json_dict(input_dir / "results_summary_reconstruction_1hop_explainer.json")

	plot_metric_heatmap(rows, models, llms, args.metric, out_dir)
	plot_grouped_f1(rows, models, llms, explainer_summary, out_dir)
	plot_llm_metric_averages(rows, llms, out_dir)
	plot_empty_prediction_heatmap(rows, models, llms, out_dir)
	plot_aggregate_comparison(summary, explainer_summary, out_dir)

	print(f"Wrote plots to: {out_dir}")
	for path in sorted(out_dir.glob("*")):
		print(path)


if __name__ == "__main__":
	main()
