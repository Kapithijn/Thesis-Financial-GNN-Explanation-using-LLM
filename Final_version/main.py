"""Pipeline MAin.


The intended pipeline is:
1) Data loading + preprocessing
2) Model construction (GNN bundle)
3) Training (and optional saving)
4) Extraction for one or more target nodes (prediction + explainer masks + embedding + subgraph)
5) Prompt building + LLM inference
6) Evaluation (GNN prediction vs LLM prediction)

This file wires together the modules in `New_files/` into an end-to-end run.
"""
import torch

import copy

import argparse
import json
from pathlib import Path
from Data_File import load_dataset, preprocess, print_data_info
from GNN_Definition import build_model_bundle
from Train import train_all
from LLM_Module import (
	format_explanation,
	format_embedding,
	build_prompt,
	build_classification_prompt,
	build_raw_reasoning_prompt,
	build_neighbor_selection_prompt,
	run_inference_all,
	parse_neighbor_selection_response,
)
from Evalueation import aggregate_results, save_results, compute_classification_metrics, evaluate_reconstruction
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from Parallel_Extraction import extract_one

def parse_args(argv=None):
	"""Parse CLI arguments for running the pipeline."""
	parser = argparse.ArgumentParser(
		prog="pipeline",
		description=(
			"Run the end-to-end GNN → extraction → LLM → evaluation pipeline. "
			"See New_files/pipeline.md for intended behavior."
		),
	)

	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Path to a JSON/YAML config file describing datasets, models, LLMs, and hyperparameters.",
	)

	# Simple overrides (optional). If provided, they override config values.
	parser.add_argument("--datasets", nargs="*", default=None, help="Override dataset list (e.g., elliptic dgraphfin).")
	parser.add_argument("--models", nargs="*", default=None, help="Override GNN model list (subset of bundle keys).")
	parser.add_argument("--llms", nargs="*", default=None, help="Override LLM model list (HuggingFace model names/paths).")
	parser.add_argument(
		"--experiments",
		nargs="*",
		default=None,
		help=(
			"Override experiment list. Valid values include embedding_classification, "
			"embedding_classification_explainer_subgraph, raw_graph_reasoning, "
			"reconstruction_1hop, reconstruction_1hop_embed_expl, reconstruction_1hop_no_gnn, "
			"baseline_random, baseline_cosine, and baseline_feature."
		),
	)
	parser.add_argument("--target-nodes", nargs="*", type=int, default=None, help="Override target node ids for extraction.")
	parser.add_argument(
		"--num-target-nodes",
		type=int,
		default=None,
		help=(
			"Automatically select N target nodes (used when --target-nodes is not set). "
			"Selection pool and sampling can be controlled with --target-node-pool and --target-node-sampling."
		),
	)
	parser.add_argument(
		"--target-node-pool",
		choices=["test", "train", "val", "labeled", "all"],
		default=None,
		help="Pool to select target nodes from when using --num-target-nodes.",
	)
	parser.add_argument(
		"--target-node-sampling",
		choices=["random", "first"],
		default=None,
		help="How to select nodes from the pool when using --num-target-nodes.",
	)
	parser.add_argument(
		"--min-anomalous-target-ratio",
		type=float,
		default=None,
		help=(
			"Ensure at least this fraction of automatically sampled target nodes have "
			"the anomaly label when enough anomalous nodes are available. Use 0 to disable."
		),
	)
	parser.add_argument(
		"--target-anomaly-label",
		type=int,
		default=None,
		help="Label value treated as anomalous for balanced target-node sampling.",
	)
	parser.add_argument(
		"--target-normal-label",
		type=int,
		default=None,
		help="Label value treated as normal/negative for exact balanced target-node sampling.",
	)
	parser.add_argument(
		"--target-positive-count",
		type=int,
		default=None,
		help="Exact number of anomaly-label target nodes to sample when available.",
	)
	parser.add_argument(
		"--target-negative-count",
		type=int,
		default=None,
		help="Exact number of normal-label target nodes to sample when available.",
	)
	parser.add_argument("--num-hops", type=int, default=None, help="Override k for k-hop subgraph extraction.")
	parser.add_argument(
		"--explainer-scope",
		choices=["full", "local"],
		default=None,
		help="Run GNNExplainer on the full graph or on a local sampled subgraph.",
	)
	parser.add_argument(
		"--explainer-local-num-hops",
		type=int,
		default=None,
		help="Number of hops for local GNNExplainer subgraphs.",
	)
	parser.add_argument(
		"--explainer-max-nodes",
		type=int,
		default=None,
		help="Optional cap for nodes passed to local GNNExplainer.",
	)
	parser.add_argument(
		"--explainer-max-edges",
		type=int,
		default=None,
		help="Optional cap for edges passed to local GNNExplainer.",
	)
	parser.add_argument(
		"--subgraph-max-nodes",
		type=int,
		default=None,
		help="Optional cap for stored k-hop subgraph nodes. Omit to keep the full k-hop topology.",
	)
	parser.add_argument(
		"--subgraph-max-edges",
		type=int,
		default=None,
		help="Optional cap for stored k-hop subgraph edges. Omit to keep the full k-hop topology.",
	)
	parser.add_argument(
		"--include-subgraph-features",
		action="store_true",
		help="Store node feature matrices inside k-hop subgraph bundles.",
	)
	parser.add_argument(
		"--include-subgraph-labels",
		action="store_true",
		help="Store node labels inside k-hop subgraph bundles.",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=None,
		help="Override generation max_new_tokens for LLM responses.",
	)
	parser.add_argument(
		"--llm-batch-size",
		type=int,
		default=None,
		help="Number of prompts to generate per LLM batch (1 keeps sequential inference).",
	)
	parser.add_argument(
		"--thinking-budget",
		type=int,
		default=None,
		help=(
			"Optional Qwen chat-template thinking budget. Only takes effect together "
			"with --enable-thinking because thinking is disabled by default."
		),
	)
	parser.add_argument(
		"--disable-thinking",
		action="store_true",
		default=None,
		help="Disable Qwen chat-template thinking mode when supported by the tokenizer.",
	)
	parser.add_argument(
		"--enable-thinking",
		action="store_true",
		default=None,
		help="Explicitly enable Qwen chat-template thinking mode when supported by the tokenizer.",
	)
	parser.add_argument(
		"--reconstruction-max-candidates",
		type=int,
		default=None,
		help="Cap candidate nodes per reconstruction prompt to avoid very large LLM contexts.",
	)
	parser.add_argument(
		"--candidate-feature-max-length",
		type=int,
		default=None,
		help="Cap the number of raw feature values shown per candidate node in reconstruction prompts.",
	)
	parser.add_argument(
		"--candidate-embedding-max-length",
		type=int,
		default=None,
		help="Cap the number of embedding values shown per candidate node in reconstruction prompts.",
	)
	parser.add_argument(
		"--candidate-context-mode",
		choices=["pca", "summary", "vectors"],
		default=None,
		help=(
			"How to represent candidates in reconstruction prompts. "
			"'pca' uses globally fitted PCA coordinates, 'summary' uses compact "
			"similarity/distance statistics, and 'vectors' prints raw vectors."
		),
	)
	parser.add_argument(
		"--candidate-pca-components",
		type=int,
		default=None,
		help="Number of PCA components to use for feature/embedding vectors in reconstruction prompts.",
	)
	parser.add_argument(
		"--continue-on-llm-error",
		action="store_true",
		default=None,
		help="Log failed LLM prompts as error rows and continue instead of aborting the whole run.",
	)

	parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for artifacts/results.")
	parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Device selection.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
	parser.add_argument(
		"--num-runs",
		type=int,
		default=1,
		help="Number of full pipeline runs (use with --seed-base for variance sweeps).",
	)
	parser.add_argument(
		"--run-id",
		type=str,
		default=None,
		help="Optional run identifier (used in output dir templates).",
	)
	parser.add_argument(
		"--seed-base",
		type=int,
		default=None,
		help="Base seed for multi-run; run i uses seed_base + i.",
	)
	parser.add_argument(
		"--output-dir-template",
		type=str,
		default=None,
		help="Template for output dir; supports {base} and {run_id}.",
	)
	parser.add_argument(
		"--extract-workers",
		type=int,
		default=None,
		help="Number of worker processes for extraction (1 = sequential).",
	)
	parser.set_defaults(large_graph_cpu_fallback=None)
	fallback_group = parser.add_mutually_exclusive_group()
	fallback_group.add_argument(
		"--large-graph-cpu-fallback",
		action="store_true",
		dest="large_graph_cpu_fallback",
		help="Enable automatic CPU fallback for very large graphs and CUDA OOM retries.",
	)
	fallback_group.add_argument(
		"--no-large-graph-cpu-fallback",
		action="store_false",
		dest="large_graph_cpu_fallback",
		help="Disable automatic CPU fallback for very large graphs and CUDA OOM retries.",
	)

	# Stage flags.
	parser.add_argument("--skip-data", action="store_true", help="Skip dataset loading/preprocessing stage.")
	parser.add_argument("--skip-train", action="store_true", help="Skip training stage (expects trained weights to exist).")
	parser.add_argument("--skip-extract", action="store_true", help="Skip extraction stage (expects cached extractions to exist).")
	parser.add_argument("--skip-llm", action="store_true", help="Skip LLM inference stage (expects cached LLM outputs to exist).")
	parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage.")

	parser.add_argument("--dry-run", action="store_true", help="Print planned stages and exit without running.")
	return parser.parse_args(argv)


def load_config(path):
	"""Load a JSON or YAML config file into a dict."""
	if path is None:
		return {}

	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	suffix = config_path.suffix.lower()
	if suffix == ".json":
		with config_path.open("r", encoding="utf-8") as f:
			return json.load(f)

	if suffix in {".yml", ".yaml"}:
		try:
			import yaml  # type: ignore
		except ImportError as exc:
			raise ImportError(
				"YAML config requires PyYAML. Install with: pip install pyyaml"
			) from exc

		with config_path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
			return data if isinstance(data, dict) else {}

	raise ValueError(f"Unsupported config extension: {suffix} (use .json or .yaml)")


def default_config():
	"""Return a minimal default config dict."""
	return {
		"output_dir": "outputs",
		"datasets": ["elliptic"],#elliptic
		"models": ["GAT"],  # subset of bundle keys or None for all
		"llms": ["Qwen/Qwen2.5-0.5B-Instruct"],#Qwen/Qwen3.5-2B or Qwen/Qwen2.5-0.5B-Instruct Qwen2.5-3B-Instruct
		"experiments": [
				"embedding_classification",
				# "embedding_classification_explainer_subgraph",
				# "raw_graph_reasoning",
			# "reconstruction_1hop",
			# "reconstruction_1hop_embed_expl",
			# "reconstruction_1hop_no_gnn",
			# "baseline_random",
			# "baseline_cosine",
			# "baseline_feature",
		],
		"extract_workers": 2,
		"target_nodes": [],	
		"num_target_nodes": 2,
		"target_node_pool": "test",
		"target_node_sampling": "random",
		"min_anomalous_target_ratio": 0.2,
		"target_anomaly_label": 1,
		"target_normal_label": 0,
		"target_positive_count": None,
		"target_negative_count": None,
		"num_hops": 2, 
		"raw_graph_reasoning": {
			"condition": "raw_features_neighbors",
		},
			"reconstruction": {
				"candidate_ratio": 4,
				"explainer_top_k": 5,
				"explainer_min_score": None,
				"include_explanation_mask": False,
				"include_node_features": True,
				"include_candidate_features": True,
				"include_candidate_embeddings": True,
				"candidate_context_mode": "pca",
				"candidate_pca_components": 16,
				"candidate_feature_max_length": None,
				"candidate_embedding_max_length": None,
				"output_format": "json",
			},
			"prompt_explainer_subgraph": {
				"normalized_importance_threshold": 0.7,
				"top_k": 5,
				"fallback_top_k": 1,
			},
			"explanation": {
				"scope": "full",
				"local_num_hops": None,
				"max_nodes": None,
				"max_edges": None,
			},
			"subgraph": {
				"include": True,
				"max_nodes": None,
				"max_edges": None,
				"include_node_features": True,
				"include_node_labels": True,
			},
			"prompt": {
			"template": (
"You are helping compliance analysts understand and validate the prediction of a graph neural network (GNN) used for financial transaction fraud detection on a transaction graph.\n\n"
"You will see examples of GNN decisions with their correct class, then a new case to classify.\n"
"The explanation lists \"Index <id> with importance <score>\"; higher importance means more influence on the decision.\n\n"
"Class definition:\n"
"- 0 = licit (normal) transaction\n"
"- 1 = illicit (suspicious) transaction\n\n"
"Example 1\n"
"Explanation:\n"
"Top important edges/features:\n"
" - Index 172445 with importance 0.6319\n"
" - Index 156241 with importance 0.0000\n"
" - Index 156227 with importance 0.0000\n"
" - Index 156228 with importance 0.0000\n"
" - Index 156229 with importance 0.0000\n\n"
"Embedding:\n"
"embedding: [3.1249, 5.1401, 0.8050, 1.7747, 0.0000, 2.2467, 0.0000, 0.0000, 3.0109, 0.0000, 5.3093, 0.0000, 0.0000, 1.2692, 0.0000, 2.2743, 0.7417, 0.0000, 0.0000, 1.2539, 0.0000, 0.0000, 2.7624, 0.0000, 0.0000, 0.0000, 0.6356, 2.1562, 0.0000, 0.0000, 0.6110, 0.0000, 1.6118, 0.0000, 3.5684, 0.0000, 0.0000, 0.0000, 0.0000, 1.3218, 0.0000, 0.0000, 0.0000, 2.3571, 0.0000, 0.5580, 0.7710, 2.5565, 0.0000, 0.0000, 0.0000, 0.0511, 2.0591, 0.0480, 3.9548, 0.0000, 2.6845, 0.0000, 0.0000, 1.9812, 0.0000, 0.0000, 0.0000, 0.2535]\n\n"
"Subgraph:\n"
"Subgraph (k=2) with 2 nodes and 1 edges.\n\n"
"Correct label: 1 (illicit)\n\n"
"Example 2\n"
"Explanation:\n"
"Top important edges/features:\n"
" - Index 156242 with importance 0.0000\n"
" - Index 156228 with importance 0.0000\n"
" - Index 156229 with importance 0.0000\n"
" - Index 156230 with importance 0.0000\n"
" - Index 156231 with importance 0.0000\n\n"
"Embedding:\n"
"embedding: [8.4794, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 9.1156, 9.7832, 0.0000, 11.5688, 0.0000, 0.0000, 10.0358, 11.9583, 0.0000, 0.0000, 3.3227, 11.6687, 3.8521, 13.1241, 0.0000, 7.0590, 7.8354, 0.0000, 0.0000, 0.0000, 8.7543, 12.2876, 0.0000, 0.0827, 0.0000, 7.4203, 0.0000, 6.4304, 0.0000, 3.4102, 0.2344, 7.0176, 7.4691, 9.7183, 0.0000, 1.6065, 2.8625, 0.3167, 0.0000, 0.0000, 0.0000, 2.5734, 0.0000, 0.0000, 0.7634, 0.0000, 0.0000, 6.4666, 0.7579, 0.0000, 1.7948, 3.4339, 0.0000, 9.7507, 0.0000, 9.2003]\n\n"
"Subgraph:\n"
"Subgraph (k=2) with 1 nodes and 0 edges.\n\n"
"Correct label: 0 (licit)\n\n"
"Now classify the following case.\n\n"
"Explanation:\n{explanation}\n\n"
"Embedding:\n{embedding}\n\n"
"Subgraph:\n{subgraph}\n\n"
"Return your answer in exactly this format:\n"
"The predicted class is X\n"
"where X is either 0 (licit) or 1 (illicit). Do not output anything else.\n"
			),
			"embedding_max_length": None,
		},
		"prompt_raw_reasoning": {
			"template": (
"You are given a transaction node and optional graph context.\n\n"
"Class definition:\n"
"- 0 = licit (normal) transaction\n"
"- 1 = illicit (suspicious) transaction\n\n"
"Raw features for target node:\n{raw_features}\n\n"
"Neighbor feature table (if provided):\n{neighbor_table}\n\n"
"Edge list (if provided):\n{edge_list}\n\n"
"Return your answer in exactly this format:\n"
"The predicted class is X\n"
"where X is either 0 or 1. Do not output anything else."
			)
		},
		"prompt_reconstruction": {
			"template": (
"You are a strict JSON generator for a graph reconstruction benchmark.\n"
"Task: select the candidate node ids that are directly connected (1-hop neighbors) to the target node.\n"
"Rules:\n"
"- Output exactly one JSON object and nothing else.\n"
"- Do not explain, do not write markdown, and do not write code.\n"
"- selected_neighbors must contain only ids copied exactly from the candidate set.\n"
"- The answer can be empty, contain one id, or contain multiple ids.\n"
"- Do not copy the whole candidate set or a prefix of it unless every copied id is likely a direct neighbor.\n\n"
"Target embedding:\n{embedding}\n\n"
"Target raw features:\n{target_features}\n\n"
"Candidate node context:\n{candidate_context}\n\n"
"Candidate set (node ids):\n{candidates}\n\n"
"Required output format, using actual integer ids:\n"
"{{\"selected_neighbors\": [], \"confidence\": 0.0}}"
			)
		},
		"train": {
			"lr": 0.01,
			"epochs": 200,
			"print_every": 20,
			"patience": None,
		},
		"generation": {
			"max_new_tokens": 256,
			"llm_batch_size": 1,
			"disable_thinking": True,
			"continue_on_error": False,
		},
		"large_graph_cpu_fallback": True,
	}


def apply_cli_overrides(config, args):
	"""Apply CLI overrides on top of a config dict (in-place)."""
	if args.output_dir is not None:
		config["output_dir"] = args.output_dir
	if args.datasets is not None:
		config["datasets"] = args.datasets
	if args.models is not None:
		config["models"] = args.models
	if args.llms is not None:
		config["llms"] = args.llms
	if args.experiments is not None:
		config["experiments"] = args.experiments
	if args.target_nodes is not None:
		config["target_nodes"] = args.target_nodes
	if args.num_target_nodes is not None:
		config["num_target_nodes"] = int(args.num_target_nodes)
		if args.target_nodes is None:
			config["target_nodes"] = []
	if args.target_node_pool is not None:
		config["target_node_pool"] = str(args.target_node_pool)
	if args.target_node_sampling is not None:
		config["target_node_sampling"] = str(args.target_node_sampling)
	if getattr(args, "min_anomalous_target_ratio", None) is not None:
		config["min_anomalous_target_ratio"] = min(1.0, max(0.0, float(args.min_anomalous_target_ratio)))
	if getattr(args, "target_anomaly_label", None) is not None:
		config["target_anomaly_label"] = int(args.target_anomaly_label)
	if getattr(args, "target_normal_label", None) is not None:
		config["target_normal_label"] = int(args.target_normal_label)
	if getattr(args, "target_positive_count", None) is not None:
		config["target_positive_count"] = max(0, int(args.target_positive_count))
	if getattr(args, "target_negative_count", None) is not None:
		config["target_negative_count"] = max(0, int(args.target_negative_count))
	if args.num_hops is not None:
		config["num_hops"] = int(args.num_hops)
	if getattr(args, "explainer_scope", None) is not None:
		explanation = config.get("explanation")
		if not isinstance(explanation, dict):
			explanation = {}
		explanation["scope"] = str(args.explainer_scope)
		config["explanation"] = explanation
	if getattr(args, "explainer_local_num_hops", None) is not None:
		explanation = config.get("explanation")
		if not isinstance(explanation, dict):
			explanation = {}
		explanation["local_num_hops"] = max(1, int(args.explainer_local_num_hops))
		config["explanation"] = explanation
	if getattr(args, "explainer_max_nodes", None) is not None:
		explanation = config.get("explanation")
		if not isinstance(explanation, dict):
			explanation = {}
		explanation["max_nodes"] = max(1, int(args.explainer_max_nodes))
		config["explanation"] = explanation
	if getattr(args, "explainer_max_edges", None) is not None:
		explanation = config.get("explanation")
		if not isinstance(explanation, dict):
			explanation = {}
		explanation["max_edges"] = max(1, int(args.explainer_max_edges))
		config["explanation"] = explanation
	if getattr(args, "max_new_tokens", None) is not None:
		generation = config.get("generation")
		if not isinstance(generation, dict):
			generation = {}
		generation["max_new_tokens"] = max(1, int(args.max_new_tokens))
		config["generation"] = generation
	if getattr(args, "llm_batch_size", None) is not None:
		generation = config.get("generation")
		if not isinstance(generation, dict):
			generation = {}
		generation["llm_batch_size"] = max(1, int(args.llm_batch_size))
		config["generation"] = generation
	if getattr(args, "thinking_budget", None) is not None:
		generation = config.get("generation")
		if not isinstance(generation, dict):
			generation = {}
		generation["thinking_budget"] = max(1, int(args.thinking_budget))
		config["generation"] = generation
	if getattr(args, "enable_thinking", None):
		generation = config.get("generation")
		if not isinstance(generation, dict):
			generation = {}
		generation["enable_thinking"] = True
		generation["disable_thinking"] = False
		config["generation"] = generation
	if getattr(args, "disable_thinking", None):
		generation = config.get("generation")
		if not isinstance(generation, dict):
			generation = {}
		generation.pop("thinking_budget", None)
		generation["disable_thinking"] = True
		config["generation"] = generation
	if getattr(args, "reconstruction_max_candidates", None) is not None:
		reconstruction = config.get("reconstruction")
		if not isinstance(reconstruction, dict):
			reconstruction = {}
		reconstruction["max_candidates"] = max(1, int(args.reconstruction_max_candidates))
		config["reconstruction"] = reconstruction
	if getattr(args, "candidate_feature_max_length", None) is not None:
		reconstruction = config.get("reconstruction")
		if not isinstance(reconstruction, dict):
			reconstruction = {}
		reconstruction["candidate_feature_max_length"] = max(0, int(args.candidate_feature_max_length))
		config["reconstruction"] = reconstruction
	if getattr(args, "candidate_embedding_max_length", None) is not None:
		reconstruction = config.get("reconstruction")
		if not isinstance(reconstruction, dict):
			reconstruction = {}
		reconstruction["candidate_embedding_max_length"] = max(0, int(args.candidate_embedding_max_length))
		config["reconstruction"] = reconstruction
	if getattr(args, "candidate_context_mode", None) is not None:
		reconstruction = config.get("reconstruction")
		if not isinstance(reconstruction, dict):
			reconstruction = {}
		reconstruction["candidate_context_mode"] = str(args.candidate_context_mode)
		config["reconstruction"] = reconstruction
	if getattr(args, "candidate_pca_components", None) is not None:
		reconstruction = config.get("reconstruction")
		if not isinstance(reconstruction, dict):
			reconstruction = {}
		reconstruction["candidate_pca_components"] = max(1, int(args.candidate_pca_components))
		config["reconstruction"] = reconstruction
	if getattr(args, "continue_on_llm_error", None) is True:
		generation = config.get("generation")
		if not isinstance(generation, dict):
			generation = {}
		generation["continue_on_error"] = True
		config["generation"] = generation
	if args.seed is not None:
		config["seed"] = int(args.seed)
	if getattr(args, "extract_workers", None) is not None and args.extract_workers is not None:
		config["extract_workers"] = max(1, int(args.extract_workers))
	if getattr(args, "no_large_graph_cpu_fallback", False):
		config["large_graph_cpu_fallback"] = False
	elif getattr(args, "large_graph_cpu_fallback", None):
		config["large_graph_cpu_fallback"] = True
	if args.device is not None:
		config["device"] = args.device
	return config


def _resolve_base_output_dir(config, args):
	"""Resolve the base output directory before applying run-specific suffixes."""
	if args.output_dir is not None:
		return str(args.output_dir)
	if isinstance(config, dict) and config.get("output_dir") is not None:
		return str(config.get("output_dir"))
	return "outputs"


def _resolve_run_id(args, run_index):
	"""Return a run id string or None for the single-run default."""
	num_runs = int(getattr(args, "num_runs", 1) or 1)
	if num_runs <= 1:
		return str(args.run_id) if args.run_id is not None else None
	if args.run_id:
		return f"{args.run_id}_{run_index + 1}"
	return str(run_index + 1)


def _resolve_run_output_dir(config, args, run_id):
	"""Return the per-run output directory path (or None to keep defaults)."""
	output_template = getattr(args, "output_dir_template", None)
	if output_template:
		base_dir = _resolve_base_output_dir(config, args)
		safe_run_id = run_id if run_id is not None else "1"
		return output_template.format(base=base_dir, run_id=safe_run_id)

	num_runs = int(getattr(args, "num_runs", 1) or 1)
	if run_id is None and num_runs <= 1:
		return args.output_dir

	base_dir = _resolve_base_output_dir(config, args)
	if run_id is None:
		return base_dir
	return str(Path(base_dir) / f"run_{run_id}")


def _resolve_run_seed(config, args, run_index):
	"""Return per-run seed or None when not specified."""
	base = getattr(args, "seed_base", None)
	if base is None:
		base = getattr(args, "seed", None)
	if base is None and isinstance(config, dict):
		base = config.get("seed")
	if base is None:
		return None
	return int(base) + int(run_index)


def _candidate_target_nodes(data, pool_name: str):
	"""Return a 1D tensor of candidate node indices based on a pool selection."""
	pool = (pool_name or "").strip().lower() or "test"
	if pool in {"train", "val", "test"}:
		mask = getattr(data, f"{pool}_mask", None)
		if mask is None:
			raise ValueError(f"Dataset has no {pool}_mask; cannot sample from pool '{pool}'.")
		return mask.nonzero(as_tuple=False).view(-1)

	if pool == "labeled":
		y = getattr(data, "y", None)
		if y is None:
			return torch.arange(int(getattr(data, "num_nodes", 0)))
		return (y >= 0).nonzero(as_tuple=False).view(-1)

	if pool == "all":
		return torch.arange(int(getattr(data, "num_nodes", 0)))

	raise ValueError(f"Unknown target_node_pool: {pool_name} (expected test/train/val/labeled/all)")


def _select_target_nodes(config, data):
	"""Resolve target nodes from config (explicit list or auto-selection)."""
	explicit = config.get("target_nodes")
	if isinstance(explicit, (list, tuple)) and len(explicit) > 0:
		return [int(v) for v in explicit]

	count = config.get("num_target_nodes")
	if count is None:
		return []
	count = int(count)
	if count <= 0:
		return []

	pool = str(config.get("target_node_pool", "test"))
	sampling = str(config.get("target_node_sampling", "random")).strip().lower() or "random"

	candidates = _candidate_target_nodes(data, pool)
	candidates = candidates.detach().cpu()

	# Ensure candidates are labeled when labels exist.
	y = getattr(data, "y", None)
	if y is not None:
		y = y.detach().cpu()
		if y.numel() == int(getattr(data, "num_nodes", y.numel())):
			candidates = candidates[y[candidates] >= 0]

	if candidates.numel() == 0:
		raise ValueError(f"No candidate target nodes found (pool='{pool}').")

	def _sample_from_pool(pool_tensor, sample_count, generator=None):
		sample_count = int(sample_count)
		if sample_count <= 0:
			return pool_tensor[:0]
		if sample_count >= int(pool_tensor.numel()):
			return pool_tensor
		if sampling == "first":
			return pool_tensor[:sample_count]
		if generator is None:
			perm = torch.randperm(int(pool_tensor.numel()))
		else:
			perm = torch.randperm(int(pool_tensor.numel()), generator=generator)
		return pool_tensor[perm[:sample_count]]

	def _shuffle_selected(selected_tensor, generator=None):
		if sampling == "first" or int(selected_tensor.numel()) <= 1:
			return selected_tensor
		if generator is None:
			perm = torch.randperm(int(selected_tensor.numel()))
		else:
			perm = torch.randperm(int(selected_tensor.numel()), generator=generator)
		return selected_tensor[perm]

	seed = config.get("seed")
	generator = None
	if seed is not None:
		generator = torch.Generator(device="cpu")
		generator.manual_seed(int(seed))

	if count >= int(candidates.numel()):
		selected = candidates
	else:
		min_anomaly_ratio = float(config.get("min_anomalous_target_ratio", 0.0) or 0.0)
		anomaly_label = int(config.get("target_anomaly_label", 1))
		normal_label = int(config.get("target_normal_label", 0))
		target_positive_count = config.get("target_positive_count")
		target_negative_count = config.get("target_negative_count")
		selected = None
		if (target_positive_count is not None or target_negative_count is not None) and y is not None:
			positive_needed = 0 if target_positive_count is None else max(0, int(target_positive_count))
			negative_needed = 0 if target_negative_count is None else max(0, int(target_negative_count))
			if positive_needed + negative_needed > count:
				print(
					f"Warning: requested {positive_needed} positive and {negative_needed} negative targets "
					f"but num_target_nodes={count}; trimming the larger class request."
				)
				while positive_needed + negative_needed > count:
					if positive_needed >= negative_needed and positive_needed > 0:
						positive_needed -= 1
					elif negative_needed > 0:
						negative_needed -= 1
					else:
						break

			positive_candidates = candidates[y[candidates] == anomaly_label]
			negative_candidates = candidates[y[candidates] == normal_label]
			selected_positive_count = min(positive_needed, int(positive_candidates.numel()))
			selected_negative_count = min(negative_needed, int(negative_candidates.numel()))
			if selected_positive_count < positive_needed:
				print(
					f"Warning: requested {positive_needed} target nodes with label={anomaly_label} "
					f"but only {int(positive_candidates.numel())} are available in pool='{pool}'."
				)
			if selected_negative_count < negative_needed:
				print(
					f"Warning: requested {negative_needed} target nodes with label={normal_label} "
					f"but only {int(negative_candidates.numel())} are available in pool='{pool}'."
				)

			positive_selected = _sample_from_pool(positive_candidates, selected_positive_count, generator)
			negative_selected = _sample_from_pool(negative_candidates, selected_negative_count, generator)
			selected_parts = [part for part in (positive_selected, negative_selected) if int(part.numel()) > 0]
			selected = torch.cat(selected_parts, dim=0) if selected_parts else candidates[:0]

			remaining_needed = count - int(selected.numel())
			if remaining_needed > 0:
				if int(selected.numel()) > 0:
					remaining_mask = ~torch.isin(candidates, selected)
					remaining_pool = candidates[remaining_mask]
				else:
					remaining_pool = candidates
				selected = torch.cat([selected, _sample_from_pool(remaining_pool, remaining_needed, generator)], dim=0)

			selected = _shuffle_selected(selected[:count], generator)
			actual_positive = int((y[selected] == anomaly_label).sum().item())
			actual_negative = int((y[selected] == normal_label).sum().item())
			print(
				f"Target sampling: selected {actual_positive} label={anomaly_label} and "
				f"{actual_negative} label={normal_label} nodes out of {int(selected.numel())} "
				f"from pool='{pool}' (requested positive={positive_needed}, negative={negative_needed})."
			)

		if selected is None and min_anomaly_ratio > 0.0 and y is not None:
			anomaly_candidates = candidates[y[candidates] == anomaly_label]
			required_anomalies = int(torch.ceil(torch.tensor(float(count) * min_anomaly_ratio)).item())
			required_anomalies = min(count, max(0, required_anomalies))
			available_anomalies = int(anomaly_candidates.numel())
			selected_anomaly_count = min(required_anomalies, available_anomalies)
			if required_anomalies > 0 and available_anomalies == 0:
				print(
					f"Warning: no anomalous target nodes with label={anomaly_label} "
					f"found in pool='{pool}'; falling back to unstratified target sampling."
				)
			elif selected_anomaly_count < required_anomalies:
				print(
					f"Warning: requested at least {required_anomalies}/{count} anomalous target nodes "
					f"but only {available_anomalies} are available in pool='{pool}'."
				)

			if selected_anomaly_count > 0:
				anomaly_selected = _sample_from_pool(anomaly_candidates, selected_anomaly_count, generator)
				if int(anomaly_selected.numel()) > 0:
					remaining_mask = ~torch.isin(candidates, anomaly_selected)
					remaining_pool = candidates[remaining_mask]
				else:
					remaining_pool = candidates
				remaining_needed = count - int(anomaly_selected.numel())
				remaining_selected = _sample_from_pool(remaining_pool, remaining_needed, generator)
				selected = torch.cat([anomaly_selected, remaining_selected], dim=0)
				selected = _shuffle_selected(selected[:count], generator)
				actual_anomalies = int((y[selected] == anomaly_label).sum().item())
				print(
					f"Target sampling: selected {actual_anomalies}/{int(selected.numel())} "
					f"anomalous nodes with label={anomaly_label} from pool='{pool}' "
					f"(minimum ratio={min_anomaly_ratio:.2f})."
				)

		if selected is None:
			selected = _sample_from_pool(candidates, count, generator)

	return [int(v.item()) for v in selected]


def resolve_device(device):
	"""Resolve a device string (auto → cuda/mps/cpu)."""
	if device is None or device == "auto":
		try:
			import torch
		except Exception:
			return "cpu"

		if torch.cuda.is_available():
			return "cuda"
		if torch.backends.mps.is_available():
			return "mps"
		return "cpu"
	return device


def run_data_stage(config):
	"""Load and preprocess datasets specified in the config."""
	dataset_names = config.get("datasets", [])
	dataset_kwargs = config.get("dataset_kwargs", {})
	should_print = bool(config.get("print_data_info", True))

	datasets = {}
	skipped = []
	for dataset_name in dataset_names:
		kwargs = {}
		if isinstance(dataset_kwargs, dict):
			kwargs = dataset_kwargs.get(dataset_name, {}) or {}

		try:
			data = load_dataset(dataset_name, **kwargs)
		except FileNotFoundError as exc:
			skipped.append((dataset_name, str(exc)))
			continue
		data = preprocess(data)
		if should_print:
			print_data_info(data)
		datasets[dataset_name] = data

	if skipped:
		for dataset_name, message in skipped:
			print(f"Skipping dataset '{dataset_name}': {message}")

	if not datasets:
		raise FileNotFoundError("No requested datasets could be loaded. Check that the raw dataset files are present.")

	return datasets


def _build_model_config_for_data(config, data):
	"""Infer model dimensions from a specific dataset."""
	model_config = dict(config)
	if model_config.get("in_channels") is None:
		model_config["in_channels"] = int(getattr(data, "num_node_features", 0))
	if model_config.get("out_channels") is None:
		labels = getattr(data, "y", None)
		if labels is None:
			raise ValueError("Cannot infer out_channels because dataset has no 'y' labels.")
		try:
			import torch
			valid = labels[labels >= 0]
			if valid.numel() == 0:
				raise ValueError("Cannot infer out_channels because all labels are negative/unlabeled.")
			model_config["out_channels"] = int(valid.max().item() + 1)
		except Exception:
			model_config["out_channels"] = int(labels.max().item() + 1)
	return model_config


def _resolve_runtime_device_for_data(requested_device, data, enable_large_graph_cpu_fallback=True):
	"""Prefer CPU for very large graphs when CUDA is requested."""
	device = resolve_device(requested_device)
	if device != "cuda" or not enable_large_graph_cpu_fallback:
		return device

	num_nodes = int(getattr(data, "num_nodes", 0) or 0)
	num_edges = int(data.edge_index.size(1)) if getattr(data, "edge_index", None) is not None else 0
	if num_nodes >= 1_000_000 or num_edges >= 2_000_000:
		return "cpu"
	return device


def run_model_build_stage(config, datasets):
	"""Build one model bundle per dataset and optionally select a subset by name."""

	if not datasets:
		raise ValueError("No datasets provided. Run the data stage first.")

	selected_names = config.get("models")
	bundles = {}
	for dataset_name, data in datasets.items():
		model_config = _build_model_config_for_data(config, data)
		bundle = build_model_bundle(model_config)

		if selected_names:
			selected = {}
			for model_name in selected_names:
				if model_name not in bundle:
					available = ", ".join(bundle.keys())
					raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
				selected[model_name] = bundle[model_name]
			bundle = selected

		bundles[dataset_name] = bundle

	return bundles


def run_training_stage(config, model_bundle, datasets):
	"""Train all model–dataset combinations and return training histories."""

	train_cfg = config.get("train", {})
	histories = {}
	enable_large_graph_cpu_fallback = bool(config.get("large_graph_cpu_fallback", True))
	for dataset_name, data in datasets.items():
		bundle = model_bundle.get(dataset_name)
		if bundle is None:
			continue
		device = _resolve_runtime_device_for_data(
			config.get("device"),
			data,
			enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
		)
		per_dataset_histories = train_all(bundle, {dataset_name: data}, train_cfg, device=device)
		histories[dataset_name] = per_dataset_histories
	return {"histories": histories}


def _print_local_explainer_cap_summary(records):
	"""Print whether local GNNExplainer node/edge caps were active."""
	summaries = {}
	for record in records:
		bundle = record.get("bundle") or {}
		explanation = bundle.get("explanation") or {}
		if explanation.get("explanation_scope") != "local":
			continue
		dataset_name = record.get("dataset") or bundle.get("dataset") or "unknown_dataset"
		model_name = record.get("model") or bundle.get("model") or "unknown_model"
		key = (str(dataset_name), str(model_name))
		item = summaries.setdefault(
			key,
			{
				"count": 0,
				"node_hits": 0,
				"edge_hits": 0,
				"max_nodes_cap": explanation.get("local_max_nodes"),
				"max_edges_cap": explanation.get("local_max_edges"),
				"max_nodes_before": 0,
				"max_edges_before": 0,
				"max_edges_before_edge_cap": 0,
				"max_nodes_after": 0,
				"max_edges_after": 0,
			},
		)
		item["count"] += 1
		item["node_hits"] += int(bool(explanation.get("hit_max_nodes", False)))
		item["edge_hits"] += int(bool(explanation.get("hit_max_edges", False)))
		item["max_nodes_before"] = max(
			item["max_nodes_before"],
			int(explanation.get("local_num_nodes_before_cap", 0) or 0),
		)
		item["max_edges_before"] = max(
			item["max_edges_before"],
			int(explanation.get("local_num_edges_before_cap", 0) or 0),
		)
		item["max_edges_before_edge_cap"] = max(
			item["max_edges_before_edge_cap"],
			int(explanation.get("local_num_edges_before_edge_cap", 0) or 0),
		)
		item["max_nodes_after"] = max(
			item["max_nodes_after"],
			int(explanation.get("local_num_nodes_after_cap", explanation.get("local_num_nodes", 0)) or 0),
		)
		item["max_edges_after"] = max(
			item["max_edges_after"],
			int(explanation.get("local_num_edges_after_cap", explanation.get("local_message_edges", 0)) or 0),
		)

	if not summaries:
		return

	print("Local GNNExplainer cap summary:")
	for (dataset_name, model_name), item in sorted(summaries.items()):
		count = max(1, int(item["count"]))
		print(
			f"- {dataset_name}|{model_name}: "
			f"node_cap_hits={item['node_hits']}/{count}, "
			f"edge_cap_hits={item['edge_hits']}/{count}, "
			f"max_before_nodes={item['max_nodes_before']}, "
			f"max_before_edges={item['max_edges_before']}, "
			f"max_before_edge_cap_edges={item['max_edges_before_edge_cap']}, "
			f"max_after_nodes={item['max_nodes_after']}, "
			f"max_after_edges={item['max_edges_after']}, "
			f"caps=(nodes={item['max_nodes_cap']}, edges={item['max_edges_cap']})"
		)


def _local_explainer_cap_fields(bundle):
	"""Return local GNNExplainer cap diagnostics for saved raw result rows."""
	explanation = (bundle or {}).get("explanation") or {}
	keys = [
		"explanation_scope",
		"local_num_hops",
		"local_max_nodes",
		"local_max_edges",
		"local_num_nodes_before_cap",
		"local_num_edges_before_cap",
		"local_num_edges_before_edge_cap",
		"local_num_nodes_after_cap",
		"local_num_edges_after_cap",
		"local_message_edges",
		"local_explanation_edges",
		"hit_max_nodes",
		"hit_max_edges",
	]
	return {key: explanation.get(key) for key in keys if key in explanation}


def run_extraction_stage(config, model_bundle, datasets):
	"""Run extraction (prediction/explanation/embedding/subgraph) for target nodes."""
	from Extracion import (
		extract_all,
		build_candidate_set,
		get_node_embedding_table,
		get_node_feature_table,
		get_one_hop_neighbors,
		compute_node_embedding_cache,
	)

	num_hops = int(config.get("num_hops", 2))
	enable_large_graph_cpu_fallback = bool(config.get("large_graph_cpu_fallback", True))

	try:
		from tqdm.auto import tqdm  # type: ignore
	except Exception:
		tqdm = None

	# Resolve and freeze the target node lists once per dataset (important when sampling is random).
	target_nodes_by_dataset = {}
	total = 0
	for dataset_name, data in datasets.items():
		if dataset_name not in model_bundle:
			continue
		device = _resolve_runtime_device_for_data(
			config.get("device"),
			data,
			enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
		)
		target_nodes = _select_target_nodes(config, data)
		if not target_nodes:
			raise ValueError(
				"No target nodes specified. Set config['target_nodes'], pass --target-nodes, "
				"or use --num-target-nodes with an appropriate pool (e.g., --target-node-pool test)."
			)
		target_nodes_by_dataset[dataset_name] = [int(v) for v in target_nodes]
		total += len(target_nodes_by_dataset[dataset_name]) * len(model_bundle[dataset_name])

	if total == 0:
		return []

	print(
		f"Starting extraction for {total} node(s) "
		f"({len(target_nodes_by_dataset)} dataset(s) × variable model counts)."
	)
	print("Note: this stage can be slow because GNNExplainer runs per target node.")

	workers = int(config.get("extract_workers", 1) or 1)
	if workers < 1:
		workers = 1
	recon_cfg_for_extraction = config.get("reconstruction", {}) or {}
	explainer_top_k = recon_cfg_for_extraction.get("explainer_top_k", 5)
	explainer_min_score = recon_cfg_for_extraction.get("explainer_min_score")
	explanation_cfg = config.get("explanation", {}) or {}
	explainer_scope = str(explanation_cfg.get("scope", "full") or "full").strip().lower()
	explainer_local_num_hops = explanation_cfg.get("local_num_hops")
	if explainer_local_num_hops is None:
		explainer_local_num_hops = num_hops
	explainer_max_nodes = explanation_cfg.get("max_nodes")
	explainer_max_edges = explanation_cfg.get("max_edges")
	if explainer_scope in {"local", "sampled", "subgraph"}:
		print(
			"Local GNNExplainer enabled: "
			f"num_hops={int(explainer_local_num_hops)}, "
			f"max_nodes={explainer_max_nodes}, max_edges={explainer_max_edges}"
		)

	def attach_candidate_context(bundle, data, model, embedding_cache=None):
		candidate_set = bundle.get("candidate_set") or {}
		candidate_ids = [int(v) for v in (candidate_set.get("candidates", []) or [])]
		bundle["candidate_feature_table"] = get_node_feature_table(data, candidate_ids)
		if model is None or not bool(recon_cfg_for_extraction.get("include_candidate_embeddings", True)):
			bundle["candidate_embedding_table"] = {"node_ids": [], "embeddings": [], "embedding_dim": 0}
			return
		try:
			bundle["candidate_embedding_table"] = get_node_embedding_table(
				model,
				data,
				candidate_ids,
				embedding_cache=embedding_cache,
			)
		except Exception as exc:
			print(f"Warning: could not compute candidate embeddings for reconstruction prompt: {exc}")
			bundle["candidate_embedding_table"] = {
				"node_ids": [],
				"embeddings": [],
				"embedding_dim": 0,
				"error": str(exc),
			}

	if workers > 1:
		requested_device = resolve_device(config.get("device"))
		print(f"Parallel extraction enabled: {workers} worker process(es).")
		print("Note: each worker loads its own copy of the graph and model (higher RAM use).")
		if requested_device == "cuda" and not enable_large_graph_cpu_fallback:
			print("Warning: parallel extraction on GPU can cause high memory use; reduce workers if needed.")

	progress_bar = None
	if tqdm is not None and total > 0:
		progress_bar = tqdm(total=total, desc="Extraction", unit="node")

	completed = 0
	records = []
	if workers == 1:
		for dataset_name, data in datasets.items():
			bundle = model_bundle.get(dataset_name)
			if bundle is None:
				continue
			device = _resolve_runtime_device_for_data(
				config.get("device"),
				data,
				enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
			)
			for model in bundle.values():
				model.to(device)
			if hasattr(data, "to"):
				data = data.to(device)
				datasets[dataset_name] = data
			target_nodes = target_nodes_by_dataset[dataset_name]
			for model_name, model in bundle.items():
				embedding_cache = None
				try:
					print(f"Computing reusable node embeddings for {dataset_name}|{model_name}...")
					embedding_cache = compute_node_embedding_cache(model, data)
					if embedding_cache is not None:
						print(
							f"Cached node embeddings for {dataset_name}|{model_name}: "
							f"shape={tuple(embedding_cache.shape)} device={embedding_cache.device}"
						)
				except Exception as exc:
					print(f"Warning: could not compute reusable node embeddings for {dataset_name}|{model_name}: {exc}")
					embedding_cache = None
				for node_id in target_nodes:
					if progress_bar is not None:
						progress_bar.set_postfix_str(f"{model_name}|{dataset_name}|node={node_id}")
						progress_bar.refresh()
					else:
						# Fallback progress indicator (prints ~20 times max).
						step = max(1, total // 20)
						if completed == 0 or completed % step == 0:
							pct = 100.0 * completed / total
							print(f"Extraction progress: {completed}/{total} ({pct:.1f}%)")

					bundle = extract_all(
						model,
						data,
						node_id,
						num_hops=num_hops,
						include_candidate_set=False,
						explainer_top_k=explainer_top_k,
						explainer_min_score=explainer_min_score,
						explanation_scope=explainer_scope,
						explanation_num_hops=int(explainer_local_num_hops),
						explanation_max_nodes=explainer_max_nodes,
						explanation_max_edges=explainer_max_edges,
						embedding_cache=embedding_cache,
						seed=(int(config.get("seed")) + int(node_id)) if config.get("seed") is not None else None,
					)
					recon_cfg = config.get("reconstruction", {}) or {}
					candidate_ratio = recon_cfg.get("candidate_ratio", 4)
					max_candidates = recon_cfg.get("max_candidates")
					seed = config.get("seed")
					neighbors = get_one_hop_neighbors(data, node_id)
					bundle["candidate_set"] = build_candidate_set(
						data,
						node_id,
						neighbors,
						candidate_ratio=candidate_ratio,
						max_candidates=max_candidates,
						seed=seed,
					)
					attach_candidate_context(bundle, data, model, embedding_cache=embedding_cache)
					bundle["dataset"] = dataset_name
					bundle["model"] = model_name
					records.append(
						{
							"dataset": dataset_name,
							"model": model_name,
							"target_node": int(node_id),
							"bundle": bundle,
						}
					)

					completed += 1
					if progress_bar is not None:
						progress_bar.update(1)
				del embedding_cache
	else:


		output_dir = Path(str(config.get("output_dir", "outputs")))
		tmp_dir = output_dir / "_tmp_parallel_extraction"
		tmp_dir.mkdir(parents=True, exist_ok=True)

		# Save trained weights so each worker can load them without pickling the whole model.
		state_paths = {}
		model_configs = {}
		for dataset_name, bundle in model_bundle.items():
			data = datasets.get(dataset_name)
			if data is None:
				continue
			device = _resolve_runtime_device_for_data(
				config.get("device"),
				data,
				enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
			)
			model_configs[dataset_name] = _build_model_config_for_data(config, data)
			for model_name, model in bundle.items():
				path = tmp_dir / f"{dataset_name}__{model_name}.pt"
				torch.save(model.state_dict(), path)
				state_paths[(dataset_name, model_name)] = str(path)

		dataset_kwargs_all = config.get("dataset_kwargs", {})
		if not isinstance(dataset_kwargs_all, dict):
			dataset_kwargs_all = {}

		seed = config.get("seed")
		per_worker_threads = 1

		futures = []
		ctx = mp.get_context("spawn")
		with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
			for dataset_name in datasets.keys():
				dataset_kwargs = dataset_kwargs_all.get(dataset_name, {})
				bundle = model_bundle.get(dataset_name)
				if bundle is None:
					continue
				device = _resolve_runtime_device_for_data(
					config.get("device"),
					datasets[dataset_name],
					enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
				)
				model_config = model_configs[dataset_name]
				for model_name in bundle.keys():
					state_dict_path = state_paths[(dataset_name, model_name)]
					for node_id in target_nodes_by_dataset[dataset_name]:
						futures.append(
							executor.submit(
								extract_one,
								dataset_name,
								dataset_kwargs,
								model_name,
								model_config,
								state_dict_path,
								int(node_id),
								int(num_hops),
								device=device,
								torch_num_threads=per_worker_threads,
								seed=int(seed) if seed is not None else None,
								explainer_top_k=explainer_top_k,
								explainer_min_score=explainer_min_score,
								explanation_scope=explainer_scope,
								explanation_num_hops=int(explainer_local_num_hops),
								explanation_max_nodes=explainer_max_nodes,
								explanation_max_edges=explainer_max_edges,
							)
						)

			for future in as_completed(futures):
				record = future.result()
				bundle = record.get("bundle") or {}
				recon_cfg = config.get("reconstruction", {}) or {}
				candidate_ratio = recon_cfg.get("candidate_ratio", 4)
				max_candidates = recon_cfg.get("max_candidates")
				seed = config.get("seed")
				data = datasets.get(record.get("dataset"))
				if data is not None:
					neighbors = get_one_hop_neighbors(data, record.get("target_node"))
					bundle["candidate_set"] = build_candidate_set(
						data,
						record.get("target_node"),
						neighbors,
						candidate_ratio=candidate_ratio,
						max_candidates=max_candidates,
						seed=seed,
					)
					model_for_context = (model_bundle.get(record.get("dataset")) or {}).get(record.get("model"))
					attach_candidate_context(bundle, data, model_for_context)
				bundle["dataset"] = record.get("dataset")
				bundle["model"] = record.get("model")
				record["bundle"] = bundle
				records.append(record)
				completed += 1
				if progress_bar is not None:
					progress_bar.set_postfix_str(
						f"{record.get('model')}|{record.get('dataset')}|node={record.get('target_node')}"
					)
					progress_bar.update(1)
				else:
					step = max(1, total // 20)
					if completed == 1 or completed % step == 0 or completed == total:
						pct = 100.0 * completed / total
						print(f"Extraction progress: {completed}/{total} ({pct:.1f}%)")

	if progress_bar is not None:
		progress_bar.close()

	_print_local_explainer_cap_summary(records)
	return records


def _format_subgraph_text(subgraph):
	"""Convert an extracted subgraph (often a dict) into readable text."""
	if isinstance(subgraph, dict):
		num_nodes = subgraph.get("num_nodes", "unknown")
		num_edges = subgraph.get("num_edges", "unknown")
		num_hops = subgraph.get("num_hops", "unknown")
		return f"Subgraph (k={num_hops}) with {num_nodes} nodes and {num_edges} edges."
	return str(subgraph)


def _format_explainer_subgraph_text(
	bundle,
	normalized_importance_threshold=0.7,
	top_k=5,
	fallback_top_k=1,
):
	"""Format a compact GNNExplainer-derived subgraph for classification prompts."""
	edges = list(bundle.get("explainer_edges") or [])
	target_node = bundle.get("target_node")
	if not edges:
		neighbors = bundle.get("explainer_neighbors") or []
		if not neighbors:
			return "Explainer subgraph: no explainer-selected edges available."
		nodes = sorted({int(v) for v in neighbors + ([target_node] if target_node is not None else [])})
		return (
			"Explainer subgraph: edge scores unavailable; using selected neighbor nodes only.\n"
			f"Nodes: {', '.join(str(v) for v in nodes)}"
		)

	def edge_score(edge):
		return (
			float(edge.get("normalized_importance", 0.0) or 0.0),
			float(edge.get("importance", 0.0) or 0.0),
		)

	edges.sort(key=edge_score, reverse=True)
	threshold = float(normalized_importance_threshold)
	selected = [edge for edge in edges if float(edge.get("normalized_importance", 0.0) or 0.0) >= threshold]
	if not selected:
		selected = edges[: max(1, int(fallback_top_k))]
	if top_k is not None:
		selected = selected[: max(0, int(top_k))]

	if not selected:
		return "Explainer subgraph: no explainer-selected edges available."

	nodes = set()
	for edge in selected:
		nodes.add(int(edge.get("source")))
		nodes.add(int(edge.get("target")))
	if target_node is not None:
		nodes.add(int(target_node))

	lines = [
		f"Explainer subgraph (normalized importance >= {threshold:.2f}; fallback top {int(fallback_top_k)}):",
		"Nodes: " + ", ".join(str(v) for v in sorted(nodes)),
		"Edges:",
	]
	for edge in selected:
		lines.append(
			" - "
			f"{int(edge.get('source'))} -> {int(edge.get('target'))} "
			f"(importance={float(edge.get('importance', 0.0) or 0.0):.4f}, "
			f"normalized={float(edge.get('normalized_importance', 0.0) or 0.0):.4f})"
		)
	return "\n".join(lines)


def _format_raw_features_text(raw_features):
	if raw_features is None:
		return "(none)"
	return "[" + ", ".join(f"{float(v):.4f}" for v in raw_features) + "]"


def _format_numeric_vector(values, max_length=None):
	if values is None:
		return "(unavailable)"
	try:
		flat_values = list(values.flatten()) if hasattr(values, "flatten") else list(values)
	except TypeError:
		flat_values = [values]

	original_len = len(flat_values)
	if max_length is not None:
		max_length = max(0, int(max_length))
		flat_values = flat_values[:max_length]

	text = "[" + ", ".join(f"{float(v):.4f}" for v in flat_values) + "]"
	if max_length is not None and original_len > max_length:
		text += f" (truncated to {max_length} of {original_len})"
	return text


def _as_float_vector(values):
	if values is None:
		return None
	try:
		import numpy as np
	except Exception:
		np = None

	if np is not None:
		try:
			vector = np.asarray(values, dtype=float).reshape(-1)
			if vector.size == 0:
				return None
			return vector
		except Exception:
			return None

	try:
		vector = [float(v) for v in values]
	except Exception:
		return None
	return vector if vector else None


def _vector_similarity_metrics(target_values, candidate_values):
	try:
		import numpy as np
	except Exception:
		return None

	target = _as_float_vector(target_values)
	candidate = _as_float_vector(candidate_values)
	if target is None or candidate is None:
		return None

	target = np.asarray(target, dtype=float).reshape(-1)
	candidate = np.asarray(candidate, dtype=float).reshape(-1)
	dim = min(int(target.size), int(candidate.size))
	if dim <= 0:
		return None

	target = target[:dim]
	candidate = candidate[:dim]
	diff = target - candidate
	target_norm = float(np.linalg.norm(target))
	candidate_norm = float(np.linalg.norm(candidate))
	denom = target_norm * candidate_norm
	cosine = float(np.dot(target, candidate) / denom) if denom > 0.0 else 0.0
	return {
		"dim": dim,
		"cosine": cosine,
		"l2": float(np.linalg.norm(diff)),
		"mean_abs_diff": float(np.mean(np.abs(diff))),
	}


def _format_metric_value(value):
	if value is None:
		return "unavailable"
	return f"{float(value):.4f}"


def _collect_table_vectors(table, value_key):
	if not isinstance(table, dict):
		return []
	values = table.get(value_key, [])
	if values is None:
		return []
	return [value for value in values]


def _fit_pca_model(vectors, n_components, label):
	try:
		import numpy as np
	except Exception:
		return {"error": "numpy_unavailable", "label": label}

	rows = []
	original_dim = None
	for values in vectors:
		vector = _as_float_vector(values)
		if vector is None:
			continue
		vector = np.asarray(vector, dtype=float).reshape(-1)
		if vector.size == 0:
			continue
		if original_dim is None:
			original_dim = int(vector.size)
		if int(vector.size) != original_dim:
			continue
		rows.append(vector)

	if not rows or original_dim is None:
		return {"error": "no_vectors", "label": label}

	matrix = np.vstack(rows)
	component_count = min(max(1, int(n_components)), int(original_dim), int(matrix.shape[0]))
	mean = matrix.mean(axis=0)

	if matrix.shape[0] < 2:
		components = np.eye(original_dim, dtype=float)[:component_count]
		explained_ratio = []
	else:
		centered = matrix - mean
		try:
			_, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
		except Exception as exc:
			return {"error": str(exc), "label": label}
		components = vt[:component_count]
		variance = (singular_values ** 2) / max(1, matrix.shape[0] - 1)
		total_variance = float(variance.sum())
		if total_variance > 0.0:
			explained_ratio = (variance[:component_count] / total_variance).tolist()
		else:
			explained_ratio = []

	return {
		"label": label,
		"mean": mean,
		"components": components,
		"n_components": int(component_count),
		"original_dim": int(original_dim),
		"fit_samples": int(matrix.shape[0]),
		"explained_variance_ratio": explained_ratio,
	}


def _project_with_pca(values, pca_model):
	if not isinstance(pca_model, dict) or pca_model.get("components") is None:
		return None
	vector = _as_float_vector(values)
	if vector is None:
		return None
	try:
		import numpy as np
	except Exception:
		return None
	vector = np.asarray(vector, dtype=float).reshape(-1)
	original_dim = int(pca_model.get("original_dim", 0) or 0)
	if original_dim <= 0 or int(vector.size) != original_dim:
		return None
	mean = pca_model.get("mean")
	components = pca_model.get("components")
	if mean is None or components is None:
		return None
	return (vector - mean) @ components.T


def _format_pca_projection(values, pca_model, label):
	projected = _project_with_pca(values, pca_model)
	if projected is None:
		error = pca_model.get("error") if isinstance(pca_model, dict) else None
		return f"{label}: (PCA unavailable{f'; {error}' if error else ''})"
	return f"{label}: " + _format_numeric_vector(projected)


def _build_reconstruction_pca_models(extraction_records, reconstruction_cfg):
	component_count = int(reconstruction_cfg.get("candidate_pca_components", 16) or 16)
	feature_vectors = []
	embedding_vectors = []

	for record in extraction_records:
		bundle = record.get("bundle") or {}
		raw_features = bundle.get("raw_features")
		if raw_features is not None:
			feature_vectors.append(raw_features)
		feature_vectors.extend(_collect_table_vectors(bundle.get("candidate_feature_table"), "features"))

		embedding = (bundle.get("embedding") or {}).get("embedding")
		if embedding is not None:
			embedding_vectors.append(embedding)
		embedding_vectors.extend(_collect_table_vectors(bundle.get("candidate_embedding_table"), "embeddings"))

	return {
		"features": _fit_pca_model(feature_vectors, component_count, "raw_features"),
		"embeddings": _fit_pca_model(embedding_vectors, component_count, "embeddings"),
	}


def _reconstruction_pca_group_key(record):
	return (str(record.get("dataset", "")), str(record.get("model", "")))


def _build_reconstruction_pca_models_by_group(extraction_records, reconstruction_cfg):
	groups = {}
	for record in extraction_records:
		groups.setdefault(_reconstruction_pca_group_key(record), []).append(record)

	models_by_group = {}
	for key, records in groups.items():
		models = _build_reconstruction_pca_models(records, reconstruction_cfg)
		group_label = f"{key[0]}|{key[1]}"
		for pca_model in models.values():
			if isinstance(pca_model, dict):
				pca_model["fit_group"] = group_label
		models_by_group[key] = models
	return models_by_group


def _pca_models_for_record(models_by_group, record):
	if not models_by_group:
		return None
	return models_by_group.get(_reconstruction_pca_group_key(record))


def _format_pca_model_note(pca_model, label):
	if not isinstance(pca_model, dict) or pca_model.get("components") is None:
		error = pca_model.get("error") if isinstance(pca_model, dict) else "unavailable"
		return f"{label} PCA unavailable ({error})"
	ratios = pca_model.get("explained_variance_ratio") or []
	explained = sum(float(v) for v in ratios) if ratios else 0.0
	group = pca_model.get("fit_group")
	group_text = f" fit_group={group}" if group else ""
	return (
		f"{label} PCA-{int(pca_model.get('n_components', 0))}{group_text} "
		f"fit_samples={int(pca_model.get('fit_samples', 0))} "
		f"original_dim={int(pca_model.get('original_dim', 0))} "
		f"explained_variance={explained:.4f}"
	)


def _table_value_map(table, value_key):
	if not isinstance(table, dict):
		return {}
	node_ids = table.get("node_ids", table.get("neighbor_ids", []))
	values = table.get(value_key, [])
	if node_ids is None:
		node_ids = []
	if values is None:
		values = []
	mapped = {}
	for idx, node_id in enumerate(node_ids):
		if idx < len(values):
			mapped[int(node_id)] = values[idx]
	return mapped


def _format_candidate_context_text(
	bundle,
	include_features=True,
	include_embeddings=True,
	feature_max_length=None,
	embedding_max_length=None,
	mode="summary",
	pca_models=None,
):
	candidate_set = bundle.get("candidate_set") or {}
	candidates = [int(v) for v in (candidate_set.get("candidates", []) or [])]
	if not candidates:
		return "(none)"

	feature_map = _table_value_map(bundle.get("candidate_feature_table"), "features") if include_features else {}
	embedding_map = _table_value_map(bundle.get("candidate_embedding_table"), "embeddings") if include_embeddings else {}
	mode = str(mode or "summary").strip().lower()

	if mode == "pca":
		pca_models = pca_models or {}
		feature_pca = pca_models.get("features", {})
		embedding_pca = pca_models.get("embeddings", {})
		lines = [
			"PCA coordinates use one basis fitted over target and candidate vectors for this dataset/model group.",
		]
		if include_features:
			lines.append(_format_pca_model_note(feature_pca, "raw_feature"))
		if include_embeddings:
			lines.append(_format_pca_model_note(embedding_pca, "embedding"))
		for node_id in candidates:
			parts = [f"node {node_id}"]
			if include_features:
				parts.append(_format_pca_projection(feature_map.get(node_id), feature_pca, "raw_feature_pca"))
			if include_embeddings:
				parts.append(_format_pca_projection(embedding_map.get(node_id), embedding_pca, "embedding_pca"))
			lines.append(": ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0])
		return "\n".join(lines)

	if mode == "summary":
		target_features = bundle.get("raw_features")
		target_embedding = (bundle.get("embedding") or {}).get("embedding")
		lines = [
			"Each row gives compact comparisons to the target node. Higher cosine and lower L2/mean_abs_diff indicate more similar nodes."
		]
		for node_id in candidates:
			parts = [f"node {node_id}"]
			if include_features:
				metrics = _vector_similarity_metrics(target_features, feature_map.get(node_id))
				if metrics is None:
					parts.append("raw_feature_similarity=unavailable")
				else:
					parts.extend(
						[
							f"raw_feature_dim={metrics['dim']}",
							f"raw_feature_cosine={_format_metric_value(metrics['cosine'])}",
							f"raw_feature_l2={_format_metric_value(metrics['l2'])}",
							f"raw_feature_mean_abs_diff={_format_metric_value(metrics['mean_abs_diff'])}",
						]
					)
			if include_embeddings:
				metrics = _vector_similarity_metrics(target_embedding, embedding_map.get(node_id))
				if metrics is None:
					parts.append("embedding_similarity=unavailable")
				else:
					parts.extend(
						[
							f"embedding_dim={metrics['dim']}",
							f"embedding_cosine={_format_metric_value(metrics['cosine'])}",
							f"embedding_l2={_format_metric_value(metrics['l2'])}",
							f"embedding_mean_abs_diff={_format_metric_value(metrics['mean_abs_diff'])}",
						]
					)
			lines.append(": ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0])
		return "\n".join(lines)

	lines = []
	for node_id in candidates:
		parts = [f"node {node_id}"]
		if include_features:
			parts.append(
				"raw_features="
				+ _format_numeric_vector(feature_map.get(node_id), max_length=feature_max_length)
			)
		if include_embeddings:
			parts.append(
				"embedding="
				+ _format_numeric_vector(embedding_map.get(node_id), max_length=embedding_max_length)
			)
		lines.append(": ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0])
	return "\n".join(lines)


def _format_neighbor_table_text(neighbor_table):
	if not neighbor_table:
		return "(none)"
	neighbor_ids = neighbor_table.get("neighbor_ids", [])
	features = neighbor_table.get("features", [])
	rows = []
	for idx, node_id in enumerate(neighbor_ids):
		if idx < len(features):
			feat_vec = features[idx]
			feat_text = "[" + ", ".join(f"{float(v):.4f}" for v in feat_vec) + "]"
		else:
			feat_text = "[]"
		rows.append(f"node {node_id}: {feat_text}")
	return "\n".join(rows) if rows else "(none)"


def _format_edge_list_text(subgraph):
	if not isinstance(subgraph, dict):
		return "(none)"
	edge_index = subgraph.get("edge_index")
	if edge_index is None:
		return "(none)"
	try:
		rows = []
		for src, dst in zip(edge_index[0], edge_index[1]):
			rows.append(f"{int(src)} -> {int(dst)}")
		return "\n".join(rows) if rows else "(none)"
	except Exception:
		return "(none)"


def _format_candidate_set_text(candidate_set):
	if not candidate_set:
		return "(none)"
	candidates = candidate_set.get("candidates", [])
	return ", ".join(str(int(v)) for v in candidates) if candidates else "(none)"


def _run_baseline_random(candidate_set, seed=None):
	import random
	rng = random.Random(seed)
	candidates = candidate_set.get("candidates", [])
	true_neighbors = candidate_set.get("true_neighbors", [])
	if not candidates:
		return []
	k = min(len(true_neighbors), len(candidates)) if true_neighbors else max(1, len(candidates) // 4)
	return rng.sample(list(candidates), k)


def _run_baseline_cosine(embedding, neighbor_table, candidate_set):
	try:
		import numpy as np
	except Exception:
		return []
	if embedding is None:
		return []
	candidates = candidate_set.get("candidates", [])
	neighbor_ids = neighbor_table.get("neighbor_ids", [])
	features = neighbor_table.get("features", [])
	if not candidates or not neighbor_ids or len(features) == 0:
		return []
	feature_map = {int(node_id): features[idx] for idx, node_id in enumerate(neighbor_ids)}
	vec = np.array(embedding, dtype=float)
	vec_norm = np.linalg.norm(vec) + 1e-8
	scores = []
	for node_id in candidates:
		feat = feature_map.get(int(node_id))
		if feat is None:
			continue
		feat = np.array(feat, dtype=float)
		# Skip if feature dimension doesn't match embedding dimension
		if feat.shape[0] != vec.shape[0]:
			continue
		score = float(np.dot(vec, feat) / (vec_norm * (np.linalg.norm(feat) + 1e-8)))
		scores.append((score, int(node_id)))
	if not scores:
		return []
	scores.sort(reverse=True)
	true_neighbors = candidate_set.get("true_neighbors", [])
	k = min(len(true_neighbors), len(scores)) if true_neighbors else max(1, len(scores) // 4)
	return [node_id for _, node_id in scores[:k]]


def _run_baseline_feature_distance(raw_features, neighbor_table, candidate_set):
	try:
		import numpy as np
	except Exception:
		return []
	if raw_features is None:
		return []
	candidates = candidate_set.get("candidates", [])
	neighbor_ids = neighbor_table.get("neighbor_ids", [])
	features = neighbor_table.get("features", [])
	if not candidates or not neighbor_ids or len(features) == 0:
		return []
	feature_map = {int(node_id): features[idx] for idx, node_id in enumerate(neighbor_ids)}
	vec = np.array(raw_features, dtype=float)
	scores = []
	for node_id in candidates:
		feat = feature_map.get(int(node_id))
		if feat is None:
			continue
		dist = float(np.linalg.norm(vec - np.array(feat, dtype=float)))
		scores.append((dist, int(node_id)))
	if not scores:
		return []
	scores.sort()
	true_neighbors = candidate_set.get("true_neighbors", [])
	k = min(len(true_neighbors), len(scores)) if true_neighbors else max(1, len(scores) // 4)
	return [node_id for _, node_id in scores[:k]]


def _extract_llm_prompt_and_raw_output(output, prompts, idx):
	"""Return the prompt sent to the LLM and the raw text returned by it."""
	prompt = prompts[idx] if idx < len(prompts) else None
	if isinstance(output, dict):
		prompt = output.get("prompt", output.get("llm_input_prompt", prompt))
		raw_output = output.get("raw_response")
		if raw_output is None:
			raw_output = output.get(
				"llm_output_raw",
				output.get("output", output.get("prediction", output.get("llm_pred"))),
			)
		return prompt, raw_output
	return prompt, output


def _extract_llm_error(output):
	if isinstance(output, dict):
		return output.get("error")
	return None


def _llm_output_to_text(raw_output):
	"""Convert a saved LLM output to text for parser-only code paths."""
	if raw_output is None or isinstance(raw_output, str):
		return raw_output
	try:
		return json.dumps(raw_output)
	except TypeError:
		return str(raw_output)


def _build_llm_io_records(experiment, extraction_records, prompts, predictions_by_llm):
	"""Build auditable rows containing each LLM input prompt and raw output."""
	rows = []
	for llm_name, llm_outputs in (predictions_by_llm or {}).items():
		for idx, output in enumerate(llm_outputs or []):
			record = extraction_records[idx] if idx < len(extraction_records) else {}
			prompt, raw_output = _extract_llm_prompt_and_raw_output(output, prompts, idx)
			row = {
				"experiment": experiment,
				"dataset": record.get("dataset"),
				"model": record.get("model"),
				"llm": llm_name,
				"target_node": record.get("target_node"),
				"prompt_index": idx,
				"llm_input_prompt": prompt,
				"llm_output_raw": raw_output,
			}
			if isinstance(output, dict) and output.get("parsed_prediction") is not None:
				row["parsed_prediction"] = output.get("parsed_prediction")
			error = _extract_llm_error(output)
			if error is not None:
				row["llm_error"] = error
			rows.append(row)
	return rows


def _save_llm_io_records(config, experiment, extraction_records, prompts, predictions_by_llm):
	"""Persist prompt/response audit records for LLM-backed experiments."""
	rows = _build_llm_io_records(experiment, extraction_records, prompts, predictions_by_llm)
	output_dir = Path(str(config.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)
	path = str(output_dir / f"llm_prompt_io_{experiment}.json")
	save_results(rows, path, fmt="json")
	return path


def _format_reconstruction_embedding_text(embedding, context_mode, pca_models):
	if context_mode == "pca":
		return _format_pca_projection(
			embedding,
			(pca_models or {}).get("embeddings", {}),
			"target_embedding_pca",
		)
	return format_embedding(embedding, max_length=None)


def _format_reconstruction_target_features_text(bundle, reconstruction_cfg, context_mode, pca_models):
	if not bool(reconstruction_cfg.get("include_node_features", True)):
		return "(omitted)"
	if context_mode == "pca":
		return _format_pca_projection(
			bundle.get("raw_features"),
			(pca_models or {}).get("features", {}),
			"target_raw_feature_pca",
		)
	return _format_raw_features_text(bundle.get("raw_features"))


def run_experiment_stage(config, extraction_records):
	"""Run experiment branches and return prompts/predictions by experiment."""
	experiment_names = config.get("experiments", []) or []
	if not experiment_names:
		raise ValueError("No experiments specified. Set config['experiments'] or pass a config file.")

	llm_names = config.get("llms", []) or []
	device = resolve_device(config.get("device"))
	generation_cfg = config.get("generation", {}) or {}
	reconstruction_cfg = config.get("reconstruction", {}) or {}
	reconstruction_context_mode = str(reconstruction_cfg.get("candidate_context_mode", "pca") or "pca").strip().lower()
	reconstruction_pca_models_by_group = None
	if reconstruction_context_mode == "pca":
		reconstruction_pca_models_by_group = _build_reconstruction_pca_models_by_group(
			extraction_records,
			reconstruction_cfg,
		)

	outputs = {}

	for experiment in experiment_names:
		if experiment == "embedding_classification":
			prompt_cfg = config.get("prompt", {})
			template = prompt_cfg.get("template", "{explanation}\n{embedding}\n{subgraph}")
			embedding_max_length = prompt_cfg.get("embedding_max_length")
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				edge_mask = bundle.get("explanation_mask", {}).get("edge_mask")
				embedding = bundle.get("embedding", {}).get("embedding")
				subgraph = bundle.get("k_hop_subgraph")
				if edge_mask is None:
					explanation_text = "No explanation edge mask available."
				else:
					explanation_text = format_explanation(torch.tensor(edge_mask))
				embedding_text = format_embedding(embedding, max_length=embedding_max_length)
				subgraph_text = _format_subgraph_text(subgraph)
				prompts.append(build_classification_prompt(explanation_text, embedding_text, subgraph_text, template))
			if not llm_names:
				raise ValueError("LLM list is empty for embedding_classification experiment.")
			predictions = run_inference_all(llm_names, prompts, device, return_raw=True, **generation_cfg)
			llm_io_path = _save_llm_io_records(config, experiment, extraction_records, prompts, predictions)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}
			continue

		if experiment == "embedding_classification_explainer_subgraph":
			base_prompt_cfg = config.get("prompt", {})
			prompt_cfg = config.get("prompt_explainer_subgraph", {}) or {}
			template = prompt_cfg.get("template") or base_prompt_cfg.get(
				"template",
				"{explanation}\n{embedding}\n{subgraph}",
			)
			embedding_max_length = prompt_cfg.get(
				"embedding_max_length",
				base_prompt_cfg.get("embedding_max_length"),
			)
			normalized_threshold = prompt_cfg.get("normalized_importance_threshold", 0.7)
			top_k = prompt_cfg.get("top_k", 5)
			fallback_top_k = prompt_cfg.get("fallback_top_k", 1)
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				edge_mask = bundle.get("explanation_mask", {}).get("edge_mask")
				embedding = bundle.get("embedding", {}).get("embedding")
				if edge_mask is None:
					explanation_text = "No explanation edge mask available."
				else:
					explanation_text = format_explanation(torch.tensor(edge_mask))
				embedding_text = format_embedding(embedding, max_length=embedding_max_length)
				subgraph_text = _format_explainer_subgraph_text(
					bundle,
					normalized_importance_threshold=normalized_threshold,
					top_k=top_k,
					fallback_top_k=fallback_top_k,
				)
				prompts.append(build_classification_prompt(explanation_text, embedding_text, subgraph_text, template))
			if not llm_names:
				raise ValueError("LLM list is empty for embedding_classification_explainer_subgraph experiment.")
			predictions = run_inference_all(llm_names, prompts, device, return_raw=True, **generation_cfg)
			llm_io_path = _save_llm_io_records(config, experiment, extraction_records, prompts, predictions)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}
			continue

		if experiment == "raw_graph_reasoning":
			prompt_cfg = config.get("prompt_raw_reasoning", {})
			template = prompt_cfg.get("template", "{raw_features}\n{neighbor_table}\n{edge_list}")
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				raw_features_text = _format_raw_features_text(bundle.get("raw_features"))
				neighbor_table_text = _format_neighbor_table_text(bundle.get("neighbor_feature_table"))
				edge_list_text = _format_edge_list_text(bundle.get("k_hop_subgraph"))
				prompts.append(build_raw_reasoning_prompt(raw_features_text, neighbor_table_text, edge_list_text, template))
			if not llm_names:
				raise ValueError("LLM list is empty for raw_graph_reasoning experiment.")
			predictions = run_inference_all(llm_names, prompts, device, return_raw=True, **generation_cfg)
			llm_io_path = _save_llm_io_records(config, experiment, extraction_records, prompts, predictions)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}
			continue

		if experiment == "reconstruction_1hop":
			prompt_cfg = config.get("prompt_reconstruction", {})
			template = prompt_cfg.get("template", "{embedding}\n{candidates}")
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				reconstruction_pca_models = _pca_models_for_record(reconstruction_pca_models_by_group, record)
				embedding = bundle.get("embedding", {}).get("embedding")
				embedding_text = _format_reconstruction_embedding_text(
					embedding,
					reconstruction_context_mode,
					reconstruction_pca_models,
				)
				candidate_text = _format_candidate_set_text(bundle.get("candidate_set"))
				target_features_text = _format_reconstruction_target_features_text(
					bundle,
					reconstruction_cfg,
					reconstruction_context_mode,
					reconstruction_pca_models,
				)
				candidate_context_text = _format_candidate_context_text(
					bundle,
					include_features=bool(reconstruction_cfg.get("include_candidate_features", True)),
					include_embeddings=bool(reconstruction_cfg.get("include_candidate_embeddings", True)),
					feature_max_length=reconstruction_cfg.get("candidate_feature_max_length"),
					embedding_max_length=reconstruction_cfg.get("candidate_embedding_max_length"),
					mode=reconstruction_context_mode,
					pca_models=reconstruction_pca_models,
				)
				prompts.append(
					build_neighbor_selection_prompt(
						embedding_text,
						candidate_text,
						template,
						target_features_text=target_features_text,
						candidate_context_text=candidate_context_text,
					)
				)
			if not llm_names:
				raise ValueError("LLM list is empty for reconstruction_1hop experiment.")
			predictions = run_inference_all(llm_names, prompts, device, parse_predictions=False, **generation_cfg)
			llm_io_path = _save_llm_io_records(config, experiment, extraction_records, prompts, predictions)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}
			continue

		if experiment == "reconstruction_1hop_embed_expl":
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				reconstruction_pca_models = _pca_models_for_record(reconstruction_pca_models_by_group, record)
				embedding = bundle.get("embedding", {}).get("embedding")
				embedding_text = _format_reconstruction_embedding_text(
					embedding,
					reconstruction_context_mode,
					reconstruction_pca_models,
				)
				candidate_text = _format_candidate_set_text(bundle.get("candidate_set"))
				target_features_text = _format_reconstruction_target_features_text(
					bundle,
					reconstruction_cfg,
					reconstruction_context_mode,
					reconstruction_pca_models,
				)
				candidate_context_text = _format_candidate_context_text(
					bundle,
					include_features=bool(reconstruction_cfg.get("include_candidate_features", True)),
					include_embeddings=bool(reconstruction_cfg.get("include_candidate_embeddings", True)),
					feature_max_length=reconstruction_cfg.get("candidate_feature_max_length"),
					embedding_max_length=reconstruction_cfg.get("candidate_embedding_max_length"),
					mode=reconstruction_context_mode,
					pca_models=reconstruction_pca_models,
				)

				feature_mask = bundle.get("explanation_mask", {}).get("feature_mask")
				if feature_mask is None:
					explanation_text = "No explanation feature mask available."
				else:
					try:
						mask_src = feature_mask
						# Prefer a per-target-node feature vector if provided as [num_nodes, num_features].
						if hasattr(mask_src, "ndim") and int(getattr(mask_src, "ndim", 1)) > 1:
							target_idx = int(record.get("target_node", 0) or 0)
							shape = getattr(mask_src, "shape", None)
							if shape is not None and 0 <= target_idx < int(shape[0]):
								mask_src = mask_src[target_idx]
							elif hasattr(mask_src, "reshape"):
								mask_src = mask_src.reshape(-1)
						mask = torch.tensor(mask_src)
						if mask.ndim > 1:
							mask = mask.reshape(-1)
						explanation_text = format_explanation(mask)
					except Exception:
						explanation_text = "No explanation feature mask available."

				prompt = (
					"You are a strict JSON generator for a graph reconstruction benchmark.\n"
					"Task: select the candidate node ids that are directly connected (1-hop neighbors) to the target node.\n"
					"Rules:\n"
					"- Output exactly one JSON object and nothing else.\n"
					"- Do not explain, do not write markdown, and do not write code.\n"
					"- selected_neighbors must contain only ids copied exactly from the candidate set.\n"
					"- The answer can be empty, contain one id, or contain multiple ids.\n"
					"- Do not copy the whole candidate set or a prefix of it unless every copied id is likely a direct neighbor.\n\n"
					f"Explanation (non-subgraph):\n{explanation_text}\n\n"
					f"Target embedding:\n{embedding_text}\n\n"
					f"Target raw features:\n{target_features_text}\n\n"
					f"Candidate node context:\n{candidate_context_text}\n\n"
					f"Candidate set (node ids):\n{candidate_text}\n\n"
					"Required output format, using actual integer ids:\n"
					'{"selected_neighbors": [], "confidence": 0.0}'
				)
				prompts.append(prompt)
			if not llm_names:
				raise ValueError("LLM list is empty for reconstruction_1hop_embed_expl experiment.")
			predictions = run_inference_all(llm_names, prompts, device, parse_predictions=False, **generation_cfg)
			llm_io_path = _save_llm_io_records(config, experiment, extraction_records, prompts, predictions)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}
			continue

		if experiment == "reconstruction_1hop_no_gnn":
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				reconstruction_pca_models = _pca_models_for_record(reconstruction_pca_models_by_group, record)
				candidate_text = _format_candidate_set_text(bundle.get("candidate_set"))
				target_features_text = _format_reconstruction_target_features_text(
					bundle,
					reconstruction_cfg,
					reconstruction_context_mode,
					reconstruction_pca_models,
				)
				candidate_context_text = _format_candidate_context_text(
					bundle,
					include_features=bool(reconstruction_cfg.get("include_candidate_features", True)),
					include_embeddings=False,
					feature_max_length=reconstruction_cfg.get("candidate_feature_max_length"),
					embedding_max_length=None,
					mode=reconstruction_context_mode,
					pca_models=reconstruction_pca_models,
				)
				prompt = (
					"You are a strict JSON generator for a graph reconstruction benchmark.\n"
					"Task: select the candidate node ids that are directly connected (1-hop neighbors) to the target node.\n"
					"Rules:\n"
					"- Output exactly one JSON object and nothing else.\n"
					"- Do not explain, do not write markdown, and do not write code.\n"
					"- selected_neighbors must contain only ids copied exactly from the candidate set.\n"
					"- The answer can be empty, contain one id, or contain multiple ids.\n"
					"- Do not copy the whole candidate set or a prefix of it unless every copied id is likely a direct neighbor.\n\n"
					f"Target node id: {int(record.get('target_node', 0) or 0)}\n\n"
					f"Target raw features:\n{target_features_text}\n\n"
					f"Candidate node context:\n{candidate_context_text}\n\n"
					f"Candidate set (node ids):\n{candidate_text}\n\n"
					"Required output format, using actual integer ids:\n"
					'{"selected_neighbors": [], "confidence": 0.0}'
				)
				prompts.append(prompt)
			if not llm_names:
				raise ValueError("LLM list is empty for reconstruction_1hop_no_gnn experiment.")
			predictions = run_inference_all(llm_names, prompts, device, parse_predictions=False, **generation_cfg)
			llm_io_path = _save_llm_io_records(config, experiment, extraction_records, prompts, predictions)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}
			continue

		if experiment == "baseline_random":
			rows = []
			seed = config.get("seed")
			for record in extraction_records:
				bundle = record["bundle"]
				candidate_set = bundle.get("candidate_set") or {}
				predicted = _run_baseline_random(candidate_set, seed=seed)
				rows.append(
					{
						"dataset": record["dataset"],
						"model": record["model"],
						"target_node": record["target_node"],
						"true_neighbors": candidate_set.get("true_neighbors", []),
						"explainer_neighbors": bundle.get("explainer_neighbors", []),
						"predicted_neighbors": predicted,
					}
				)
			outputs[experiment] = {"baseline": rows}
			continue

		if experiment == "baseline_cosine":
			rows = []
			for record in extraction_records:
				bundle = record["bundle"]
				candidate_set = bundle.get("candidate_set") or {}
				predicted = _run_baseline_cosine(
					bundle.get("embedding", {}).get("embedding"),
					bundle.get("neighbor_feature_table", {}),
					candidate_set,
				)
				rows.append(
					{
						"dataset": record["dataset"],
						"model": record["model"],
						"target_node": record["target_node"],
						"true_neighbors": candidate_set.get("true_neighbors", []),
						"explainer_neighbors": bundle.get("explainer_neighbors", []),
						"predicted_neighbors": predicted,
					}
				)
			outputs[experiment] = {"baseline": rows}
			continue

		if experiment == "baseline_feature":
			rows = []
			for record in extraction_records:
				bundle = record["bundle"]
				candidate_set = bundle.get("candidate_set") or {}
				predicted = _run_baseline_feature_distance(
					bundle.get("raw_features"),
					bundle.get("neighbor_feature_table", {}),
					candidate_set,
				)
				rows.append(
					{
						"dataset": record["dataset"],
						"model": record["model"],
						"target_node": record["target_node"],
						"true_neighbors": candidate_set.get("true_neighbors", []),
						"explainer_neighbors": bundle.get("explainer_neighbors", []),
						"predicted_neighbors": predicted,
					}
				)
			outputs[experiment] = {"baseline": rows}
			continue

		raise ValueError(f"Unknown experiment: {experiment}")

	return outputs


def run_llm_stage(config, extraction_records):
	"""Build prompts from extraction records and run LLM inference."""

	llm_names = config.get("llms", [])
	if not llm_names:
		raise ValueError("No LLMs specified. Set config['llms'] or pass --llms, or use --skip-llm.")

	prompt_cfg = config.get("prompt", {})
	template = prompt_cfg.get("template", "{explanation}\n{embedding}\n{subgraph}")
	embedding_max_length = prompt_cfg.get("embedding_max_length")

	prompts = []
	for record in extraction_records:
		bundle = record["bundle"]
		edge_mask = bundle.get("explanation", {}).get("edge_mask")
		embedding = bundle.get("embedding", {}).get("embedding")
		subgraph = bundle.get("subgraph")

		if edge_mask is None:
			explanation_text = "No explanation edge mask available."
		else:
			explanation_text = format_explanation(torch.tensor(edge_mask))

		embedding_text = format_embedding(embedding, max_length=embedding_max_length)
		subgraph_text = _format_subgraph_text(subgraph)

		prompt = build_prompt(explanation_text, embedding_text, subgraph_text, template)
		prompts.append(prompt)

	device = resolve_device(config.get("device"))

	generation_cfg = config.get("generation", {}) or {}
	if not isinstance(generation_cfg, dict):
		generation_cfg = {}

	predictions = run_inference_all(llm_names, prompts, device, return_raw=True, **generation_cfg)

	llm_io_path = _save_llm_io_records(config, "default", extraction_records, prompts, predictions)
	return {"prompts": prompts, "predictions": predictions, "paths": {"llm_prompt_io": llm_io_path}}


def _split_llm_prediction_output(output):
	"""Return (parsed_prediction, raw_response) from old or raw-preserving outputs."""
	if isinstance(output, dict):
		raw_response = output.get("raw_response")
		if raw_response is None:
			raw_response = output.get("llm_output_raw")
		parsed = output.get("parsed_prediction")
		if parsed is None:
			parsed = output.get("prediction", output.get("llm_pred", raw_response))
		return parsed, raw_response
	return output, None


def run_evaluation_stage(config, extraction_records, llm_outputs):
	"""Compare GNN predictions vs LLM predictions and save results."""

	prompts = llm_outputs.get("prompts", [])
	predictions_by_llm = llm_outputs.get("predictions", {})

	comparisons = []
	for llm_name, llm_preds in predictions_by_llm.items():
		if len(llm_preds) != len(extraction_records):
			raise ValueError(
				f"LLM '{llm_name}' returned {len(llm_preds)} predictions, "
				f"but there are {len(extraction_records)} extraction records."
			)

		for idx, record in enumerate(extraction_records):
			prediction_bundle = (record.get("bundle") or {}).get("prediction") or {}
			gnn_pred = prediction_bundle.get("predicted_class")
			if gnn_pred is None:
				raise KeyError(
					"Missing predicted_class in extraction record bundle: "
					"expected record['bundle']['prediction']['predicted_class']."
				)
			llm_output = llm_preds[idx]
			llm_pred, llm_raw_response = _split_llm_prediction_output(llm_output)
			llm_prompt, llm_raw_output = _extract_llm_prompt_and_raw_output(llm_output, prompts, idx)
			llm_error = _extract_llm_error(llm_output)
			row = {
				"dataset": record["dataset"],
				"model": record["model"],
				"llm": llm_name,
				"target_node": record["target_node"],
				"target_class": prediction_bundle.get("target_class"),
				"gnn_pred": gnn_pred,
				"llm_pred": llm_pred,
				"prompt": llm_prompt,
			}
			if llm_raw_response is not None or llm_raw_output is not None:
				row["llm_raw_response"] = llm_raw_response if llm_raw_response is not None else llm_raw_output
			if llm_error is not None:
				row["llm_error"] = llm_error
			comparisons.append(row)

	grouped = {}
	for row in comparisons:
		group = f"{row['model']}|{row['dataset']}|{row['llm']}"
		grouped.setdefault(group, []).append({"gnn_pred": row["gnn_pred"], "llm_pred": row["llm_pred"]})

	summary = aggregate_results(grouped)

	output_dir = Path(str(config.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)
	results_summary_path = str(output_dir / "results_summary.json")
	results_raw_path = str(output_dir / "results_raw.json")
	save_results(summary, results_summary_path, fmt="json")
	save_results(comparisons, results_raw_path, fmt="json")

	return {
		"summary": summary,
		"comparisons": comparisons,
		"paths": {
			"summary": results_summary_path,
			"raw": results_raw_path,
			"llm_prompt_io": (llm_outputs.get("paths") or {}).get("llm_prompt_io"),
		},
	}


def _build_reconstruction_group_summaries(rows):
	"""Return flat reconstruction metrics grouped for plotting."""
	group_specs = [
		("dataset", "model", "llm"),
		("dataset", "model"),
		("dataset", "llm"),
		("model",),
		("llm",),
	]
	summaries = []
	for group_fields in group_specs:
		groups = {}
		for row in rows:
			key = tuple(str(row.get(field, "unknown")) for field in group_fields)
			groups.setdefault(key, []).append(row)
		for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
			metrics = evaluate_reconstruction(group_rows)
			summary = {
				"group_by": "+".join(group_fields),
				"n": len(group_rows),
			}
			for field, value in zip(group_fields, key):
				summary[field] = value
			summary.update(metrics)
			summaries.append(summary)
	return summaries


def run_evaluation_stage_experiments(config, extraction_records, experiment_outputs):
	"""Evaluate each experiment branch and save results."""
	output_dir = Path(str(config.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)

	results = {}
	for experiment, payload in experiment_outputs.items():
		if "predictions" in payload:
			prompts = payload.get("prompts", [])
			predictions_by_llm = payload.get("predictions", {})
			comparisons = []
			for llm_name, llm_preds in predictions_by_llm.items():
				for idx, record in enumerate(extraction_records):
					bundle = record.get("bundle") or {}
					prediction_bundle = bundle.get("prediction") or {}
					gnn_pred = prediction_bundle.get("predicted_class")
					llm_output = llm_preds[idx] if idx < len(llm_preds) else None
					llm_pred, llm_raw_response = _split_llm_prediction_output(llm_output)
					llm_prompt, llm_raw_output = _extract_llm_prompt_and_raw_output(llm_output, prompts, idx)
					llm_error = _extract_llm_error(llm_output)
					row = {
						"experiment": experiment,
						"dataset": record["dataset"],
						"model": record["model"],
						"llm": llm_name,
						"target_node": record["target_node"],
						"target_class": prediction_bundle.get("target_class"),
						"gnn_pred": gnn_pred,
						"llm_pred": llm_pred,
						"prompt": llm_prompt,
					}
					row.update(_local_explainer_cap_fields(bundle))
					if llm_raw_response is not None or llm_raw_output is not None:
						row["llm_raw_response"] = llm_raw_response if llm_raw_response is not None else llm_raw_output
					if llm_error is not None:
						row["llm_error"] = llm_error
					comparisons.append(row)

			reconstruction_experiments = {
				"reconstruction_1hop",
				"reconstruction_1hop_embed_expl",
				"reconstruction_1hop_no_gnn",
			}
			if experiment in reconstruction_experiments:
				rows = []
				explainer_rows = []
				for llm_name, llm_preds in predictions_by_llm.items():
					for idx, record in enumerate(extraction_records):
						bundle = record.get("bundle") or {}
						candidate_set = bundle.get("candidate_set") or {}
						candidate_ids = [int(v) for v in (candidate_set.get("candidates", []) or [])]
						llm_output = llm_preds[idx] if idx < len(llm_preds) else None
						llm_prompt, llm_raw_output = _extract_llm_prompt_and_raw_output(llm_output, prompts, idx)
						llm_error = _extract_llm_error(llm_output)
						raw_predicted_neighbors = parse_neighbor_selection_response(_llm_output_to_text(llm_raw_output))
						predicted_neighbors = parse_neighbor_selection_response(
							_llm_output_to_text(llm_raw_output),
							allowed_ids=candidate_ids,
						)
						candidate_id_set = set(candidate_ids)
						invalid_predicted_neighbors = [
							int(value)
							for value in raw_predicted_neighbors
							if candidate_id_set and int(value) not in candidate_id_set
						]
						explainer_neighbors = bundle.get("explainer_neighbors", []) or []
						row = {
							"dataset": record["dataset"],
							"model": record["model"],
							"llm": llm_name,
							"target_node": record["target_node"],
							"true_neighbors": candidate_set.get("true_neighbors", []),
							"explainer_neighbors": explainer_neighbors,
							"predicted_neighbors": predicted_neighbors,
							"prompt": llm_prompt,
							"llm_raw_response": llm_raw_output,
						}
						row.update(_local_explainer_cap_fields(bundle))
						if raw_predicted_neighbors != predicted_neighbors:
							row["raw_predicted_neighbors"] = raw_predicted_neighbors
							row["invalid_predicted_neighbors"] = invalid_predicted_neighbors
						if llm_error is not None:
							row["llm_error"] = llm_error
						rows.append(row)
						explainer_row = {
							"dataset": record["dataset"],
							"model": record["model"],
							"llm": llm_name,
							"target_node": record["target_node"],
							"true_neighbors": explainer_neighbors,
							"predicted_neighbors": predicted_neighbors,
							"prompt": llm_prompt,
							"llm_raw_response": llm_raw_output,
						}
						explainer_row.update(_local_explainer_cap_fields(bundle))
						if raw_predicted_neighbors != predicted_neighbors:
							explainer_row["raw_predicted_neighbors"] = raw_predicted_neighbors
							explainer_row["invalid_predicted_neighbors"] = invalid_predicted_neighbors
						if llm_error is not None:
							explainer_row["llm_error"] = llm_error
						explainer_rows.append(explainer_row)

				metrics = evaluate_reconstruction(rows)
				explainer_metrics = evaluate_reconstruction(explainer_rows)
				grouped_metrics = _build_reconstruction_group_summaries(rows)
				explainer_grouped_metrics = _build_reconstruction_group_summaries(explainer_rows)
				summary_path = str(output_dir / f"results_summary_{experiment}.json")
				raw_path = str(output_dir / f"results_raw_{experiment}.json")
				explainer_summary_path = str(output_dir / f"results_summary_{experiment}_explainer.json")
				explainer_raw_path = str(output_dir / f"results_raw_{experiment}_explainer.json")
				grouped_summary_path = str(output_dir / f"results_summary_{experiment}_grouped.json")
				grouped_summary_csv_path = str(output_dir / f"results_summary_{experiment}_grouped.csv")
				explainer_grouped_summary_path = str(
					output_dir / f"results_summary_{experiment}_explainer_grouped.json"
				)
				explainer_grouped_summary_csv_path = str(
					output_dir / f"results_summary_{experiment}_explainer_grouped.csv"
				)
				save_results(metrics, summary_path, fmt="json")
				save_results(rows, raw_path, fmt="json")
				save_results(explainer_metrics, explainer_summary_path, fmt="json")
				save_results(explainer_rows, explainer_raw_path, fmt="json")
				save_results(grouped_metrics, grouped_summary_path, fmt="json")
				save_results(grouped_metrics, grouped_summary_csv_path, fmt="csv")
				save_results(explainer_grouped_metrics, explainer_grouped_summary_path, fmt="json")
				save_results(explainer_grouped_metrics, explainer_grouped_summary_csv_path, fmt="csv")
				results[experiment] = {
					"summary": metrics,
					"explainer_summary": explainer_metrics,
					"grouped_summary": grouped_metrics,
					"explainer_grouped_summary": explainer_grouped_metrics,
					"comparisons": rows,
					"explainer_comparisons": explainer_rows,
					"paths": {
						"summary": summary_path,
						"raw": raw_path,
						"explainer_summary": explainer_summary_path,
						"explainer_raw": explainer_raw_path,
						"grouped_summary": grouped_summary_path,
						"grouped_summary_csv": grouped_summary_csv_path,
						"explainer_grouped_summary": explainer_grouped_summary_path,
						"explainer_grouped_summary_csv": explainer_grouped_summary_csv_path,
						"llm_prompt_io": (payload.get("paths") or {}).get("llm_prompt_io"),
					},
				}
				continue

			grouped = {}
			for row in comparisons:
				group = f"{row['experiment']}|{row['model']}|{row['dataset']}|{row['llm']}"
				grouped.setdefault(group, []).append({"gnn_pred": row["gnn_pred"], "llm_pred": row["llm_pred"]})

			summary = {key: compute_classification_metrics(rows) for key, rows in grouped.items()}
			raw_path = str(output_dir / f"results_raw_{experiment}.json")
			summary_path = str(output_dir / f"results_summary_{experiment}.json")
			save_results(comparisons, raw_path, fmt="json")
			save_results(summary, summary_path, fmt="json")
			results[experiment] = {
				"summary": summary,
				"comparisons": comparisons,
				"paths": {
					"summary": summary_path,
					"raw": raw_path,
					"llm_prompt_io": (payload.get("paths") or {}).get("llm_prompt_io"),
				},
			}
			continue

		if "baseline" in payload:
			rows = payload.get("baseline", [])
			metrics = evaluate_reconstruction(rows)
			explainer_rows = [
				{
					"dataset": row.get("dataset"),
					"model": row.get("model"),
					"target_node": row.get("target_node"),
					"true_neighbors": row.get("explainer_neighbors", []) or [],
					"predicted_neighbors": row.get("predicted_neighbors", []) or [],
				}
				for row in rows
			]
			explainer_metrics = evaluate_reconstruction(explainer_rows)
			summary_path = str(output_dir / f"results_summary_{experiment}.json")
			raw_path = str(output_dir / f"results_raw_{experiment}.json")
			explainer_summary_path = str(output_dir / f"results_summary_{experiment}_explainer.json")
			explainer_raw_path = str(output_dir / f"results_raw_{experiment}_explainer.json")
			save_results(metrics, summary_path, fmt="json")
			save_results(rows, raw_path, fmt="json")
			save_results(explainer_metrics, explainer_summary_path, fmt="json")
			save_results(explainer_rows, explainer_raw_path, fmt="json")
			results[experiment] = {
				"summary": metrics,
				"explainer_summary": explainer_metrics,
				"comparisons": rows,
				"explainer_comparisons": explainer_rows,
				"paths": {
					"summary": summary_path,
					"raw": raw_path,
					"explainer_summary": explainer_summary_path,
					"explainer_raw": explainer_raw_path,
				},
			}
			continue

		raise ValueError(f"Unknown experiment payload format for {experiment}")

	return results


def run_pipeline(config, args):
	"""Execute the full pipeline end-to-end in order."""
	merged = default_config()
	merged.update(config)
	apply_cli_overrides(merged, args)

	device = resolve_device(merged.get("device"))
	merged["device"] = device

	stages = {
		"data": not args.skip_data,
		"train": not args.skip_train,
		"extract": not args.skip_extract,
		"llm": not args.skip_llm,
		"eval": not args.skip_eval,
	}

	if args.dry_run:
		print("Planned stages:")
		for name, enabled in stages.items():
			print(f"- {name}: {'ON' if enabled else 'OFF'}")
		print(f"Device: {device}")
		print(f"Output dir: {merged.get('output_dir')}")
		return {"config": merged, "stages": stages}

	output_dir = Path(str(merged.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)

	if merged.get("seed") is not None:
		from Train import set_seed
		set_seed(int(merged["seed"]))

	state = {"config": merged, "stages": stages}

	datasets = None
	model_bundle = None
	extraction_records = None
	llm_outputs = None

	if stages["data"]:
		datasets = run_data_stage(merged)
		state["datasets"] = datasets
	else:
		state["datasets"] = None

	if stages["train"]:
		if datasets is None:
			raise RuntimeError("Training requested but datasets are missing. Run data stage or implement dataset caching.")
		model_bundle = run_model_build_stage(merged, datasets)
		state["model_bundle"] = list(model_bundle.keys())

		training_artifacts = run_training_stage(merged, model_bundle, datasets)
		state["training"] = training_artifacts
	else:
		state["training"] = None

	if stages["extract"]:
		if datasets is None:
			raise RuntimeError("Extraction requested but datasets are missing.")
		if model_bundle is None:
			raise RuntimeError("Extraction requested but models are missing. Train or load models first.")
		extraction_records = run_extraction_stage(merged, model_bundle, datasets)
		state["extractions"] = extraction_records
	else:
		state["extractions"] = None

	if stages["llm"]:
		if extraction_records is None:
			raise RuntimeError("LLM inference requested but extraction records are missing.")
		if merged.get("experiments"):
			llm_outputs = run_experiment_stage(merged, extraction_records)
			state["llm"] = llm_outputs
		else:
			llm_outputs = run_llm_stage(merged, extraction_records)
			state["llm"] = llm_outputs
	else:
		state["llm"] = None

	if stages["eval"]:
		if extraction_records is None:
			raise RuntimeError("Evaluation requested but extraction records are missing.")
		if llm_outputs is None:
			raise RuntimeError("Evaluation requested but LLM outputs are missing.")
		if merged.get("experiments"):
			evaluation = run_evaluation_stage_experiments(merged, extraction_records, llm_outputs)
			state["evaluation"] = evaluation
		else:
			evaluation = run_evaluation_stage(merged, extraction_records, llm_outputs)
			state["evaluation"] = evaluation
	else:
		state["evaluation"] = None

	return state


def main(argv=None):
	"""CLI entry point."""
	args = parse_args(argv)
	config = load_config(args.config)

	num_runs = int(getattr(args, "num_runs", 1) or 1)
	if num_runs < 1:
		raise ValueError("--num-runs must be >= 1")

	for run_index in range(num_runs):
		run_id = _resolve_run_id(args, run_index)
		per_run_args = copy.deepcopy(args)
		per_run_args.seed = _resolve_run_seed(config, args, run_index)
		per_run_args.output_dir = _resolve_run_output_dir(config, args, run_id)

		if num_runs > 1 or run_id is not None:
			label = f"Run {run_index + 1}/{num_runs}"
			if run_id is not None:
				label += f" (id={run_id})"
			print(f"=== {label} ===")
			if per_run_args.seed is not None:
				print(f"Seed: {per_run_args.seed}")
			if per_run_args.output_dir is not None:
				print(f"Output dir: {per_run_args.output_dir}")

		state = run_pipeline(config, per_run_args)

		evaluation = state.get("evaluation")
		if isinstance(evaluation, dict):
			paths = evaluation.get("paths")
			if isinstance(paths, dict):
				summary_path = paths.get("summary")
				raw_path = paths.get("raw")
				if summary_path or raw_path:
					print("Saved results:")
					if summary_path:
						print(f"- summary: {summary_path}")
					if raw_path:
						print(f"- raw: {raw_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
