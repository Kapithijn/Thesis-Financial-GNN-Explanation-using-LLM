from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset, DGraphFin
import torch

DATA_ROOT = Path(__file__).parent / "data"
ELLIPTIC_ROOT = DATA_ROOT / "elliptic_dataset"
ELLIPTIC_TEMPORAL_ROOT = DATA_ROOT / "elliptic_temporal_dataset"
DGRAPHFIN_ROOT = DATA_ROOT / "dgraphfin_dataset"
TFINANCE_ROOT = DATA_ROOT / "tfinance_dataset"
IBM_AML_ROOT = DATA_ROOT / "ibm_aml_dataset"


def _to_data(dataset):
	"""Return the first graph stored in a PyG dataset object."""
	if len(dataset) == 0:
		raise ValueError("The dataset did not contain any graph data.")
	return dataset[0]


def load_elliptic(root=ELLIPTIC_ROOT, force_reload=False):
	"""Load the static Elliptic Bitcoin dataset."""
	dataset = EllipticBitcoinDataset(root=str(root), force_reload=force_reload)
	return _to_data(dataset)


def load_elliptic_temporal(t=1, root=ELLIPTIC_TEMPORAL_ROOT, force_reload=False):
	"""Load the time-step aware Elliptic Bitcoin dataset.

	The PyG dataset requires an explicit timestep between 1 and 49.
	"""
	if t < 1 or t > 49:
		raise ValueError("EllipticBitcoinTemporalDataset expects t to be between 1 and 49.")

	dataset = EllipticBitcoinTemporalDataset(root=str(root), t=t, force_reload=force_reload)
	return _to_data(dataset)


def load_dgraphfin(root=DGRAPHFIN_ROOT, force_reload=False):
	"""Load the DGraphFin dynamic financial graph dataset."""
	root = Path(root)
	processed_path = root / "processed" / "data.pt"
	if processed_path.exists() and not force_reload:
		cached = torch.load(processed_path, map_location="cpu", weights_only=False)
		if isinstance(cached, Data):
			return cached
		if isinstance(cached, (tuple, list)) and cached and isinstance(cached[0], Data):
			return cached[0]
		return cached

	raw_zip = Path(root) / "raw" / "DGraphFin.zip"
	if not raw_zip.exists():
		raise FileNotFoundError(
			f"Missing DGraphFin dataset archive: {raw_zip}. "
			"Download 'DGraphFin.zip' from https://dgraph.xinye.com and place it in the raw directory."
		)
	dataset = DGraphFin(root=str(root), force_reload=force_reload)
	return _to_data(dataset)


def _as_tensor(value):
	"""Convert common graph-array values to torch tensors."""
	if isinstance(value, torch.Tensor):
		return value
	return torch.as_tensor(value)


def _get_node_data(graph, names):
	for name in names:
		if name in graph.ndata:
			return graph.ndata[name]
	available = ", ".join(sorted(str(k) for k in graph.ndata.keys()))
	raise KeyError(f"Missing node data key. Tried {names}; available keys: {available}")


def _select_mask(mask_value, split_id):
	mask = _as_tensor(mask_value)
	if mask.ndim > 1:
		if split_id < 0 or split_id >= int(mask.size(1)):
			raise ValueError(f"split_id={split_id} is outside mask columns 0..{int(mask.size(1)) - 1}")
		mask = mask[:, split_id]
	return mask.bool()


def _make_split_masks(y, train_ratio=0.6, val_ratio=0.2, seed=42):
	"""Create deterministic stratified masks when a graph ships without splits."""
	y = y.detach().cpu().long().view(-1)
	num_nodes = int(y.numel())
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)
	generator = torch.Generator(device="cpu")
	generator.manual_seed(int(seed))

	for label in torch.unique(y).tolist():
		idx = (y == int(label)).nonzero(as_tuple=False).view(-1)
		if idx.numel() == 0:
			continue
		idx = idx[torch.randperm(int(idx.numel()), generator=generator)]
		n_train = int(idx.numel() * float(train_ratio))
		n_val = int(idx.numel() * float(val_ratio))
		train_mask[idx[:n_train]] = True
		val_mask[idx[n_train:n_train + n_val]] = True
		test_mask[idx[n_train + n_val:]] = True

	return train_mask, val_mask, test_mask


def _resolve_tfinance_graph_path(root):
	root = Path(root)
	names = (
		"tfinance",
		"tfinance.bin",
		"t-finance",
		"t-finance.bin",
		"t_finance",
		"t_finance.bin",
		"T-Finance",
		"T-Finance.bin",
	)
	candidates = [
		*(root / "raw" / name for name in names),
		*(root / name for name in names),
	]
	for path in candidates:
		if path.exists():
			return path
	raise FileNotFoundError(
		"Missing T-Finance graph file. Expected one of: "
		+ ", ".join(str(path) for path in candidates)
		+ ". Place the Kaggle/GADBench T-Finance DGL graph there."
	)


def _resolve_tfinance_processed_path(root):
	root = Path(root)
	candidates = [
		root / "processed" / "tfinance_pyg.pt",
		root / "tfinance_pyg.pt",
	]
	for path in candidates:
		if path.exists():
			return path
	return root / "processed" / "tfinance_pyg.pt"


def load_tfinance(
	root=TFINANCE_ROOT,
	graph_path=None,
	split_id=0,
	semi_supervised=False,
	cache_processed=True,
	train_ratio=0.6,
	val_ratio=0.2,
	split_seed=42,
):
	"""Load the GADBench T-Finance graph and convert it to a PyG Data object."""
	if graph_path is None:
		processed_path = _resolve_tfinance_processed_path(root)
		if processed_path.exists():
			return torch.load(processed_path, map_location="cpu", weights_only=False)
	else:
		processed_path = None

	try:
		from dgl.data.utils import load_graphs
	except ImportError as exc:
		raise ImportError(
			"Loading T-Finance requires DGL because the raw GADBench/Kaggle file is a DGL graph. "
			"Install DGL once, or provide a converted PyG cache at "
			f"{_resolve_tfinance_processed_path(root)}."
		) from exc

	path = Path(graph_path) if graph_path is not None else _resolve_tfinance_graph_path(root)
	graphs, _ = load_graphs(str(path))
	if not graphs:
		raise ValueError(f"No graphs found in T-Finance file: {path}")
	graph = graphs[0]

	x = _as_tensor(_get_node_data(graph, ("feature", "features", "feat", "x"))).float()
	y = _as_tensor(_get_node_data(graph, ("label", "labels", "y"))).long()
	if y.ndim > 1:
		y = y.argmax(dim=1)
	else:
		y = y.view(-1)
	src, dst = graph.edges()
	edge_index = torch.stack([_as_tensor(src).long(), _as_tensor(dst).long()], dim=0)

	if "train_mask" in graph.ndata or "train_masks" in graph.ndata:
		effective_split = int(split_id) + (10 if semi_supervised else 0)
		train_mask = _select_mask(_get_node_data(graph, ("train_mask", "train_masks")), effective_split)
		val_mask = _select_mask(_get_node_data(graph, ("val_mask", "val_masks")), effective_split)
		test_mask = _select_mask(_get_node_data(graph, ("test_mask", "test_masks")), effective_split)
	else:
		train_mask, val_mask, test_mask = _make_split_masks(
			y,
			train_ratio=train_ratio,
			val_ratio=val_ratio,
			seed=split_seed,
		)

	data = Data(
		x=x,
		edge_index=edge_index,
		y=y,
		train_mask=train_mask,
		val_mask=val_mask,
		test_mask=test_mask,
	)
	if cache_processed and processed_path is not None:
		processed_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(data, processed_path)
	return data


def _resolve_ibm_aml_csv_path(root, csv_path=None):
	root = Path(root)
	if csv_path is not None:
		path = Path(csv_path)
		if path.exists():
			return path
		raise FileNotFoundError(f"IBM AML CSV not found: {path}")

	candidates = [
		root / "raw" / "HI-Small_Trans.csv",
		root / "raw" / "HI-Small_Trans.csv.zip",
		root / "HI-Small_Trans.csv",
		root / "HI-Small_Trans.csv.zip",
	]
	for path in candidates:
		if path.exists():
			return path
	raise FileNotFoundError(
		"Missing IBM AML HI-Small transaction file. Expected one of: "
		+ ", ".join(str(path) for path in candidates)
		+ ". Download HI-Small_Trans.csv.zip from the IBM/Kaggle AML dataset "
		"and place it in Final_version/data/ibm_aml_dataset/raw/."
	)


def _read_ibm_aml_csv(path, max_rows=None):
	try:
		import pandas as pd
	except ImportError as exc:
		raise ImportError("Loading IBM AML requires pandas.") from exc

	path = Path(path)
	read_kwargs = {}
	if max_rows is not None:
		read_kwargs["nrows"] = int(max_rows)
	if path.suffix.lower() == ".zip":
		return pd.read_csv(path, compression="zip", **read_kwargs)
	return pd.read_csv(path, **read_kwargs)


def _find_column(columns, candidates):
	normalized = {str(col).strip().lower(): col for col in columns}
	for candidate in candidates:
		key = candidate.strip().lower()
		if key in normalized:
			return normalized[key]
	raise KeyError(f"Could not find any of columns {candidates}; available columns: {list(columns)}")


def _make_account_ids(frame):
	from_bank_col = _find_column(frame.columns, ("From Bank", "FromBank", "from_bank"))
	from_account_col = _find_column(frame.columns, ("Account", "From Account", "FromAccount", "account"))
	to_bank_col = _find_column(frame.columns, ("To Bank", "ToBank", "to_bank"))
	to_account_col = _find_column(frame.columns, ("Account.1", "To Account", "ToAccount", "account.1"))

	from_accounts = (
		frame[from_bank_col].astype(str).str.strip()
		+ ":"
		+ frame[from_account_col].astype(str).str.strip()
	)
	to_accounts = (
		frame[to_bank_col].astype(str).str.strip()
		+ ":"
		+ frame[to_account_col].astype(str).str.strip()
	)
	return from_accounts, to_accounts


def _encode_ibm_aml_features(frame, from_accounts, to_accounts):
	try:
		import numpy as np
		import pandas as pd
	except ImportError as exc:
		raise ImportError("Loading IBM AML requires pandas.") from exc

	timestamp_col = _find_column(frame.columns, ("Timestamp", "Time", "timestamp"))
	amount_paid_col = _find_column(frame.columns, ("Amount Paid", "AmountPaid", "amount_paid"))
	amount_received_col = _find_column(frame.columns, ("Amount Received", "AmountReceived", "amount_received"))

	timestamp = pd.to_datetime(frame[timestamp_col], errors="coerce")
	if timestamp.notna().any():
		t0 = timestamp.min()
		timestamp_days = (timestamp - t0).dt.total_seconds().fillna(0.0) / 86400.0
		hour = timestamp.dt.hour.fillna(0).astype(float) / 23.0
	else:
		timestamp_days = pd.to_numeric(frame[timestamp_col], errors="coerce").fillna(0.0)
		hour = timestamp_days * 0.0

	numeric_features = pd.DataFrame(
		{
			"timestamp_days": timestamp_days.astype(float),
			"timestamp_hour": hour.astype(float),
			"amount_paid_log1p": np.log1p(
				pd.to_numeric(frame[amount_paid_col], errors="coerce").fillna(0.0).clip(lower=0.0)
			).astype(float),
			"amount_received_log1p": np.log1p(
				pd.to_numeric(frame[amount_received_col], errors="coerce").fillna(0.0).clip(lower=0.0)
			).astype(float),
			"same_account": (from_accounts == to_accounts).astype(float),
		}
	)

	categorical_candidates = [
		("Receiving Currency", "ReceivingCurrency", "receiving_currency"),
		("Payment Currency", "PaymentCurrency", "payment_currency"),
		("Payment Format", "PaymentFormat", "payment_format"),
	]
	categorical_cols = []
	for candidates in categorical_candidates:
		try:
			categorical_cols.append(_find_column(frame.columns, candidates))
		except KeyError:
			pass

	if categorical_cols:
		categorical_features = pd.get_dummies(
			frame[categorical_cols].astype(str),
			dummy_na=True,
			dtype=float,
		)
		features = pd.concat([numeric_features, categorical_features], axis=1)
	else:
		features = numeric_features

	return torch.tensor(features.to_numpy(dtype="float32"), dtype=torch.float32)


def _build_transaction_adjacency(from_accounts, to_accounts, max_edges_per_account=200):
	account_to_transactions = {}
	for tx_idx, (src_account, dst_account) in enumerate(zip(from_accounts, to_accounts)):
		account_to_transactions.setdefault(src_account, []).append(tx_idx)
		account_to_transactions.setdefault(dst_account, []).append(tx_idx)

	edges = set()
	limit = None if max_edges_per_account is None else max(0, int(max_edges_per_account))
	for tx_indices in account_to_transactions.values():
		if len(tx_indices) < 2:
			continue
		if limit is not None and len(tx_indices) > limit:
			tx_indices = tx_indices[:limit]
		for left, right in zip(tx_indices[:-1], tx_indices[1:]):
			if left == right:
				continue
			edges.add((int(left), int(right)))
			edges.add((int(right), int(left)))

	if not edges:
		return torch.empty((2, 0), dtype=torch.long)
	edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
	return edge_index


def _make_time_split_masks(num_nodes, train_ratio=0.6, val_ratio=0.2):
	train_end = int(num_nodes * float(train_ratio))
	val_end = train_end + int(num_nodes * float(val_ratio))
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)
	train_mask[:train_end] = True
	val_mask[train_end:val_end] = True
	test_mask[val_end:] = True
	return train_mask, val_mask, test_mask


def load_ibm_aml_hi_small(
	root=IBM_AML_ROOT,
	csv_path=None,
	max_rows=None,
	max_edges_per_account=200,
	cache_processed=True,
	processed_name=None,
	train_ratio=0.6,
	val_ratio=0.2,
):
	"""Load IBM AML HI-Small as a transaction-node PyG graph.

	Each CSV row becomes one transaction node. Consecutive transactions sharing
	an account are connected bidirectionally, and the transaction's
	``Is Laundering`` value is used as the node label.
	"""
	root = Path(root)
	if max_rows is not None:
		print(f"Ignoring IBM AML max_rows={max_rows}; loading full HI-Small rows.")
		max_rows = None
	if processed_name is None:
		row_suffix = "full" if max_rows is None else f"rows_{int(max_rows)}"
		edge_suffix = "all" if max_edges_per_account is None else f"acct_edges_{int(max_edges_per_account)}"
		processed_name = f"ibm_aml_hi_small_{row_suffix}_{edge_suffix}.pt"
	processed_path = root / "processed" / processed_name
	if cache_processed and processed_path.exists():
		return torch.load(processed_path, map_location="cpu", weights_only=False)

	path = _resolve_ibm_aml_csv_path(root, csv_path=csv_path)
	frame = _read_ibm_aml_csv(path, max_rows=max_rows)
	if frame.empty:
		raise ValueError(f"IBM AML file contains no rows: {path}")

	label_col = _find_column(frame.columns, ("Is Laundering", "IsLaundering", "is_laundering"))
	from_accounts, to_accounts = _make_account_ids(frame)
	x = _encode_ibm_aml_features(frame, from_accounts, to_accounts)
	y = torch.tensor(frame[label_col].astype(int).to_numpy(), dtype=torch.long)
	edge_index = _build_transaction_adjacency(
		from_accounts.tolist(),
		to_accounts.tolist(),
		max_edges_per_account=max_edges_per_account,
	)
	train_mask, val_mask, test_mask = _make_time_split_masks(
		int(frame.shape[0]),
		train_ratio=train_ratio,
		val_ratio=val_ratio,
	)

	data = Data(
		x=x,
		edge_index=edge_index,
		y=y,
		train_mask=train_mask,
		val_mask=val_mask,
		test_mask=test_mask,
	)
	data.dataset_name = "ibm_aml_hi_small"
	data.raw_path = str(path)
	data.node_semantics = "transaction"
	if cache_processed:
		processed_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(data, processed_path)
	return data


def load_dataset(name, **kwargs):
	"""Load one of the supported graph datasets by name."""
	normalized_name = name.strip().lower()

	if normalized_name in {"elliptic", "ellipticbitcoin", "ellipticbitcoindataset"}:
		return load_elliptic(**kwargs)
	if normalized_name in {"elliptictemp", "elliptic_temporal", "ellipticbitcointemporaldataset"}:
		return load_elliptic_temporal(**kwargs)
	if normalized_name in {"dgraph", "dgraphfin", "dgraphfindataset"}:
		return load_dgraphfin(**kwargs)
	if normalized_name in {"tfinance", "t-finance", "t_finance"}:
		return load_tfinance(**kwargs)
	if normalized_name in {"ibm_aml_hi_small", "ibm-aml-hi-small", "aml_hi_small", "hi-small", "small-hi", "small_hi"}:
		return load_ibm_aml_hi_small(**kwargs)

	raise ValueError(f"Unknown dataset name: {name}")


def preprocess(data):
	"""Validate a graph data object and return it unchanged.

	These datasets are already provided as graph objects, so loading is the
	main dataset-specific step. This hook keeps the pipeline interface stable.
	"""
	if not isinstance(data, Data):
		raise TypeError("preprocess expects a torch_geometric.data.Data object")

	if not hasattr(data, "x") or not hasattr(data, "edge_index"):
		raise ValueError("The data object must contain node features and an edge_index.")

	return data


def print_data_info(data):
	"""Print basic statistics for a graph dataset."""
	if not isinstance(data, Data):
		raise TypeError("print_data_info expects a torch_geometric.data.Data object")

	num_nodes = data.num_nodes
	num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
	num_features = data.num_node_features

	print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}")

	if getattr(data, "y", None) is not None:
		labels = data.y
		valid_labels = labels[labels >= 0] if labels.numel() > 0 else labels
		if valid_labels.numel() > 0:
			unique_labels, counts = torch.unique(valid_labels, return_counts=True)
			class_info = {int(label.item()): int(count.item()) for label, count in zip(unique_labels, counts)}
			print(f"Label counts: {class_info}")

	for mask_name in ("train_mask", "val_mask", "test_mask"):
		mask = getattr(data, mask_name, None)
		if mask is not None:
			print(f"{mask_name}: {int(mask.sum().item())}")

	print(f"Any NaN in X: {torch.isnan(data.x).any().item()}")
	print(f"Any Inf in X: {torch.isinf(data.x).any().item()}")
