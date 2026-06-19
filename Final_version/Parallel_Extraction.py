import json
import torch

_DATA_CACHE = {}
_MODEL_CACHE = {}



def _stable_kwargs_key(kwargs):
	"""Create a stable cache key for dataset kwargs."""
	if not kwargs:
		return "{}"
	try:
		return json.dumps(kwargs, sort_keys=True, default=str)
	except Exception:
		return str(sorted((str(k), str(v)) for k, v in kwargs.items()))


def _load_data_cached(dataset_name, dataset_kwargs, device):
	"""Load and preprocess the dataset once per worker process."""
	from Data_File import load_dataset, preprocess

	device_key = str(device) if device is not None else "cpu"
	key = f"{dataset_name}|{_stable_kwargs_key(dataset_kwargs)}|{device_key}"
	cached = _DATA_CACHE.get(key)
	if cached is not None:
		return cached

	kwargs = dataset_kwargs or {}
	data = load_dataset(dataset_name, **kwargs)
	data = preprocess(data)
	if device is not None and hasattr(data, "to"):
		data = data.to(device)
	_DATA_CACHE[key] = data
	return data


def _load_model_cached(model_name, model_config, state_dict_path, device):
	"""Build the model architecture and load weights once per worker process."""
	from GNN_Definition import build_model_bundle

	device_key = str(device) if device is not None else "cpu"
	key = (model_name, state_dict_path, device_key)
	cached = _MODEL_CACHE.get(key)
	if cached is not None:
		return cached

	bundle = build_model_bundle(dict(model_config))
	if model_name not in bundle:
		available = ", ".join(bundle.keys())
		raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

	model = bundle[model_name]
	state = torch.load(state_dict_path, map_location="cpu")
	model.load_state_dict(state)
	if device is not None:
		model.to(device)
	model.eval()
	_MODEL_CACHE[key] = model
	return model


def extract_one(
	dataset_name,
	dataset_kwargs,
	model_name,
	model_config,
	state_dict_path,
	node_id,
	num_hops,
	device=None,
	torch_num_threads=None,
	seed=None,
	explainer_top_k=5,
	explainer_min_score=None,
	explanation_scope="full",
	explanation_num_hops=2,
	explanation_max_nodes=None,
	explanation_max_edges=None,
	include_subgraph=True,
	subgraph_max_nodes=None,
	subgraph_max_edges=None,
	subgraph_include_node_features=True,
	subgraph_include_node_labels=True,
):
	"""Run extraction for a single (dataset, model, node) tuple.

	Designed to be executed inside a worker process.
	"""
	if torch_num_threads is not None:
		torch.set_num_threads(int(torch_num_threads))
		try:
			torch.set_num_interop_threads(int(torch_num_threads))
		except Exception:
			pass

	if seed is not None:
		# Make per-node randomness stable but distinct across nodes.
		torch.manual_seed(int(seed) + int(node_id))

	torch_device = torch.device(device) if device is not None else torch.device("cpu")
	data = _load_data_cached(dataset_name, dataset_kwargs, torch_device)
	model = _load_model_cached(model_name, model_config, state_dict_path, torch_device)

	from Extracion import extract_all

	bundle = extract_all(
		model,
		data,
		int(node_id),
		num_hops=int(num_hops),
		include_candidate_set=False,
		explainer_top_k=explainer_top_k,
		explainer_min_score=explainer_min_score,
		explanation_scope=explanation_scope,
		explanation_num_hops=explanation_num_hops,
		explanation_max_nodes=explanation_max_nodes,
		explanation_max_edges=explanation_max_edges,
		include_subgraph=include_subgraph,
		subgraph_max_nodes=subgraph_max_nodes,
		subgraph_max_edges=subgraph_max_edges,
		subgraph_include_node_features=subgraph_include_node_features,
		subgraph_include_node_labels=subgraph_include_node_labels,
		seed=(int(seed) + int(node_id)) if seed is not None else None,
	)
	return {
		"dataset": dataset_name,
		"model": model_name,
		"target_node": int(node_id),
		"bundle": bundle,
	}
