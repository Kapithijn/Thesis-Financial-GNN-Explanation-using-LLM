import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import add_self_loops, k_hop_subgraph, remove_self_loops
import numpy as np


def _uses_gat_self_loops(model):
    """Return True when a model contains a GATConv that adds self-loops."""
    for module in model.modules():
        if module.__class__.__name__ == "GATConv" and hasattr(module, "add_self_loops"):
            if bool(module.add_self_loops):
                return True
    return False


def _edge_index_for_explanation(model, data):
    """Match GATConv's effective edge set so explainer masks align."""
    edge_index = data.edge_index
    if not _uses_gat_self_loops(model):
        return edge_index

    # GATConv removes existing self-loops and then appends one self-loop per
    # node internally. Passing that same edge set to GNNExplainer prevents its
    # trainable edge mask from being shorter than the messages GAT propagates.
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(
        edge_index,
        num_nodes=int(getattr(data, "num_nodes", data.x.size(0))),
    )
    return edge_index


def get_prediction(model, data, target_node):
    """
    Returns the GNN's class prediction for the target node.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
    
    Returns:
        Dictionary with 'node_id', 'logits', and 'predicted_class'
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        logits = out[target_node]
        predicted_class = logits.argmax(dim=0).item()

    target_class = None
    y = getattr(data, "y", None)
    if y is not None:
        try:
            target_class = int(y[target_node].item())
        except Exception:
            target_class = None
    
    return {
        "node_id": target_node,
        "target_class": target_class,
        "logits": logits.cpu().numpy(),
        "predicted_class": predicted_class,
    }


def get_explainer_neighbors(edge_index, edge_mask, target_node, top_k=5, min_score=None):
    """Return target-adjacent nodes selected by the highest explainer edge scores."""
    if edge_index is None or edge_mask is None:
        return []

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index)
    scores = torch.as_tensor(edge_mask, device=edge_index.device).reshape(-1)
    if edge_index.numel() == 0 or scores.numel() == 0:
        return []

    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            return []

    num_edges = min(int(edge_index.size(1)), int(scores.numel()))
    edge_index = edge_index[:, :num_edges]
    scores = scores[:num_edges]

    src, dst = edge_index
    target_node = int(target_node)
    incident_mask = (src == target_node) | (dst == target_node)
    incident_indices = incident_mask.nonzero(as_tuple=False).view(-1)
    if incident_indices.numel() == 0:
        return []

    incident_scores = scores[incident_indices]
    order = torch.argsort(incident_scores, descending=True)

    selected = []
    seen = set()
    threshold = float(min_score) if min_score is not None else None
    for pos in order.detach().cpu().tolist():
        edge_pos = int(incident_indices[pos].item())
        score = float(scores[edge_pos].detach().cpu().item())
        if threshold is not None and score < threshold:
            continue

        u = int(src[edge_pos].detach().cpu().item())
        v = int(dst[edge_pos].detach().cpu().item())
        neighbor = v if u == target_node else u
        if neighbor == target_node or neighbor in seen:
            continue

        selected.append(neighbor)
        seen.add(neighbor)
        if top_k is not None and len(selected) >= top_k:
            break

    return selected


def get_explainer_edges(edge_index, edge_mask, top_k=5, min_score=None):
    """Return the highest-scoring explainer edges with node ids and scores."""
    if edge_index is None or edge_mask is None:
        return []

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index)
    scores = torch.as_tensor(edge_mask, device=edge_index.device).reshape(-1)
    if edge_index.numel() == 0 or scores.numel() == 0:
        return []

    num_edges = min(int(edge_index.size(1)), int(scores.numel()))
    edge_index = edge_index[:, :num_edges]
    scores = scores[:num_edges]
    src, dst = edge_index

    valid_mask = src != dst
    if min_score is not None:
        valid_mask = valid_mask & (scores >= float(min_score))
    valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
    if valid_indices.numel() == 0:
        return []

    valid_scores = scores[valid_indices]
    max_score = float(valid_scores.max().detach().cpu().item())
    denom = max_score if max_score > 0.0 else 1.0

    if top_k is None:
        selected_count = int(valid_indices.numel())
    else:
        selected_count = min(int(top_k), int(valid_indices.numel()))
        if selected_count <= 0:
            return []

    top_scores, order = torch.topk(valid_scores, k=selected_count, largest=True, sorted=True)
    selected_indices = valid_indices[order]

    edges = []
    for edge_pos, score in zip(selected_indices.detach().cpu().tolist(), top_scores.detach().cpu().tolist()):
        edge_pos = int(edge_pos)
        source = int(src[edge_pos].detach().cpu().item())
        target = int(dst[edge_pos].detach().cpu().item())
        importance = float(score)
        edges.append(
            {
                "edge_index": edge_pos,
                "source": source,
                "target": target,
                "importance": importance,
                "normalized_importance": importance / denom if denom > 0.0 else 0.0,
            }
        )
    return edges


def _build_explainer(model):
    """Create the PyG explainer used by this pipeline."""
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def _sample_tensor(values, count, seed=None):
    """Sample tensor values deterministically on CPU, preserving value device."""
    count = int(count)
    if count <= 0 or values.numel() == 0:
        return values[:0]
    if count >= int(values.numel()):
        return values
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    perm = torch.randperm(int(values.numel()), generator=generator)[:count]
    return values[perm.to(values.device)]


def _maybe_cap_local_subgraph(subset, sub_edge_index, mapping, max_nodes=None, max_edges=None, seed=None):
    """Optionally cap a relabeled local subgraph while keeping the target node."""
    local_node_count = int(subset.numel())
    original_edge_count = int(sub_edge_index.size(1))
    mapping = int(mapping)
    stats = {
        "local_num_nodes_before_cap": local_node_count,
        "local_num_edges_before_cap": original_edge_count,
        "local_num_edges_before_edge_cap": original_edge_count,
        "hit_max_nodes": False,
        "hit_max_edges": False,
    }

    if max_nodes is not None:
        max_nodes = max(1, int(max_nodes))
        if local_node_count > max_nodes:
            stats["hit_max_nodes"] = True
            src, dst = sub_edge_index
            target = torch.tensor([mapping], dtype=torch.long, device=subset.device)
            one_hop = torch.cat([dst[src == mapping], src[dst == mapping]]).unique()
            one_hop = one_hop[one_hop != mapping]

            room_for_neighbors = max(0, max_nodes - 1)
            kept_neighbors = _sample_tensor(one_hop, room_for_neighbors, seed=seed)
            keep_parts = [target, kept_neighbors]

            current = int(1 + kept_neighbors.numel())
            if current < max_nodes:
                keep_mask = torch.zeros(local_node_count, dtype=torch.bool, device=subset.device)
                keep_mask[target] = True
                if kept_neighbors.numel() > 0:
                    keep_mask[kept_neighbors] = True
                remaining = (~keep_mask).nonzero(as_tuple=False).view(-1)
                sampled_remaining = _sample_tensor(remaining, max_nodes - current, seed=None if seed is None else int(seed) + 1)
                keep_parts.append(sampled_remaining)

            keep_local = torch.cat(keep_parts).unique()
            keep_local = torch.sort(keep_local).values
            keep_mask = torch.zeros(local_node_count, dtype=torch.bool, device=subset.device)
            keep_mask[keep_local] = True

            src, dst = sub_edge_index
            edge_keep = keep_mask[src] & keep_mask[dst]
            sub_edge_index = sub_edge_index[:, edge_keep]

            old_to_new = torch.full(
                (local_node_count,),
                -1,
                dtype=torch.long,
                device=subset.device,
            )
            old_to_new[keep_local] = torch.arange(int(keep_local.numel()), device=subset.device)
            sub_edge_index = old_to_new[sub_edge_index]
            subset = subset[keep_local]
            mapping = int(old_to_new[mapping].item())

    stats["local_num_edges_before_edge_cap"] = int(sub_edge_index.size(1))
    if max_edges is not None:
        max_edges = max(1, int(max_edges))
        edge_count = int(sub_edge_index.size(1))
        if edge_count > max_edges:
            stats["hit_max_edges"] = True
            src, dst = sub_edge_index
            target_edges = ((src == mapping) | (dst == mapping)).nonzero(as_tuple=False).view(-1)
            other_mask = torch.ones(edge_count, dtype=torch.bool, device=sub_edge_index.device)
            other_mask[target_edges] = False
            other_edges = other_mask.nonzero(as_tuple=False).view(-1)

            if int(target_edges.numel()) >= max_edges:
                keep_edges = _sample_tensor(target_edges, max_edges, seed=seed)
            else:
                rest = _sample_tensor(
                    other_edges,
                    max_edges - int(target_edges.numel()),
                    seed=None if seed is None else int(seed) + 2,
                )
                keep_edges = torch.cat([target_edges, rest])
            keep_edges = torch.sort(keep_edges).values
            sub_edge_index = sub_edge_index[:, keep_edges]

    stats["local_num_nodes_after_cap"] = int(subset.numel())
    stats["local_num_edges_after_cap"] = int(sub_edge_index.size(1))
    return subset, sub_edge_index, mapping, stats


def _local_explanation_data(data, target_node, num_hops=2, max_nodes=None, max_edges=None, seed=None):
    """Build the local graph used for scalable explanation."""
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=int(target_node),
        num_hops=int(num_hops),
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    subset, sub_edge_index, mapping, cap_stats = _maybe_cap_local_subgraph(
        subset,
        sub_edge_index,
        int(mapping.item()) if hasattr(mapping, "item") else int(mapping),
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed,
    )
    local_data = Data(
        x=data.x[subset],
        edge_index=sub_edge_index,
    )
    if hasattr(data, "y") and data.y is not None:
        local_data.y = data.y[subset]
    return local_data, subset, int(mapping), cap_stats


def _globalize_explainer_edges(edges, subset):
    """Map local explainer edge endpoints back to global node ids."""
    global_edges = []
    for edge in edges:
        local_source = int(edge.get("source"))
        local_target = int(edge.get("target"))
        item = dict(edge)
        item["local_source"] = local_source
        item["local_target"] = local_target
        item["source"] = int(subset[local_source].detach().cpu().item())
        item["target"] = int(subset[local_target].detach().cpu().item())
        global_edges.append(item)
    return global_edges


def get_local_explanation(
    model,
    data,
    target_node,
    num_hops=2,
    max_nodes=None,
    max_edges=None,
    seed=None,
    explainer_top_k=5,
    explainer_min_score=None,
):
    """Run GNNExplainer on a local/sampled subgraph and map ids globally."""
    model.eval()
    local_data, subset, local_target, cap_stats = _local_explanation_data(
        data,
        target_node,
        num_hops=num_hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed,
    )
    explainer = _build_explainer(model)
    explanation_edge_index = _edge_index_for_explanation(model, local_data)
    explanation = explainer(local_data.x, explanation_edge_index, index=local_target)
    edge_mask = explanation.edge_mask

    local_neighbors = get_explainer_neighbors(
        explanation_edge_index,
        edge_mask,
        local_target,
        top_k=explainer_top_k,
        min_score=explainer_min_score,
    )
    explainer_neighbors = [
        int(subset[int(node_id)].detach().cpu().item())
        for node_id in local_neighbors
        if 0 <= int(node_id) < int(subset.numel())
    ]
    explainer_edges = _globalize_explainer_edges(
        get_explainer_edges(
            explanation_edge_index,
            edge_mask,
            top_k=explainer_top_k,
            min_score=explainer_min_score,
        ),
        subset,
    )

    return {
        "node_id": target_node,
        "edge_mask": edge_mask.detach().cpu().numpy() if edge_mask is not None else None,
        "feature_mask": explanation.node_mask.detach().cpu().numpy() if explanation.node_mask is not None else None,
        "explainer_neighbors": explainer_neighbors,
        "explainer_edges": explainer_edges,
        "explainer_top_k": explainer_top_k,
        "explanation_scope": "local",
        "local_num_hops": int(num_hops),
        "local_num_nodes": int(subset.numel()),
        "local_num_edges": int(explanation_edge_index.size(1)),
        "local_message_edges": int(local_data.edge_index.size(1)),
        "local_explanation_edges": int(explanation_edge_index.size(1)),
        "local_num_nodes_before_cap": int(cap_stats.get("local_num_nodes_before_cap", subset.numel())),
        "local_num_edges_before_cap": int(cap_stats.get("local_num_edges_before_cap", local_data.edge_index.size(1))),
        "local_num_edges_before_edge_cap": int(cap_stats.get("local_num_edges_before_edge_cap", local_data.edge_index.size(1))),
        "local_num_nodes_after_cap": int(cap_stats.get("local_num_nodes_after_cap", subset.numel())),
        "local_num_edges_after_cap": int(cap_stats.get("local_num_edges_after_cap", local_data.edge_index.size(1))),
        "hit_max_nodes": bool(cap_stats.get("hit_max_nodes", False)),
        "hit_max_edges": bool(cap_stats.get("hit_max_edges", False)),
        "local_target_mapping": int(local_target),
        "local_subset_nodes": subset.detach().cpu().numpy(),
        "local_max_nodes": None if max_nodes is None else int(max_nodes),
        "local_max_edges": None if max_edges is None else int(max_edges),
    }


def get_explanation(
    model,
    data,
    target_node,
    explainer_top_k=5,
    explainer_min_score=None,
    explanation_scope="full",
    explanation_num_hops=2,
    explanation_max_nodes=None,
    explanation_max_edges=None,
    seed=None,
):
    """
    Runs GNNExplainer and returns edge/feature importance masks.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
    
    Returns:
        Dictionary with 'edge_mask' and 'feature_mask' (numpy arrays)
    """
    scope = str(explanation_scope or "full").strip().lower()
    if scope in {"local", "sampled", "subgraph"}:
        return get_local_explanation(
            model,
            data,
            target_node,
            num_hops=explanation_num_hops,
            max_nodes=explanation_max_nodes,
            max_edges=explanation_max_edges,
            seed=seed,
            explainer_top_k=explainer_top_k,
            explainer_min_score=explainer_min_score,
        )

    model.eval()
    explainer = _build_explainer(model)

    explanation_edge_index = _edge_index_for_explanation(model, data)
    explanation = explainer(data.x, explanation_edge_index, index=target_node)
    edge_mask = explanation.edge_mask
    explainer_neighbors = get_explainer_neighbors(
        explanation_edge_index,
        edge_mask,
        target_node,
        top_k=explainer_top_k,
        min_score=explainer_min_score,
    )
    explainer_edges = get_explainer_edges(
        explanation_edge_index,
        edge_mask,
        top_k=explainer_top_k,
        min_score=explainer_min_score,
    )
    
    return {
        "node_id": target_node,
        "edge_mask": edge_mask.detach().cpu().numpy() if edge_mask is not None else None,
        "feature_mask": explanation.node_mask.detach().cpu().numpy() if explanation.node_mask is not None else None,
        "explainer_neighbors": explainer_neighbors,
        "explainer_edges": explainer_edges,
        "explainer_top_k": explainer_top_k,
        "explanation_scope": "full",
    }


def compute_node_embedding_cache(model, data):
    """Compute first-layer node embeddings once for reuse during extraction."""
    if not hasattr(data, "x") or data.x is None or not hasattr(data, "edge_index") or data.edge_index is None:
        return None

    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = data.x.device

    with torch.no_grad():
        x = data.x.to(model_device)
        edge_index = data.edge_index.to(model_device)
        if hasattr(model, 'conv1'):
            x = model.conv1(x, edge_index)
            x = F.relu(x)
        else:
            x = model(x, edge_index)
        return x.detach().cpu()


def get_embedding(model, data, target_node, embedding_cache=None):
    """
    Extracts the latent embedding of the target node.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
    
    Returns:
        Dictionary with 'node_id' and 'embedding' (numpy array)
    """
    if embedding_cache is not None:
        embedding = embedding_cache[int(target_node)]
        return {
            "node_id": target_node,
            "embedding": embedding.detach().cpu().numpy(),
            "embedding_dim": len(embedding),
            "embedding_cached": True,
        }

    model.eval()
    
    # For GNN models with multiple layers, we extract from the second-to-last layer
    # This requires access to intermediate activations
    embedding = None
    
    with torch.no_grad():
        # Forward pass to get intermediate embeddings
        x = data.x
        edge_index = data.edge_index
        
        # For GCN, GAT, GIN, GraphSAGE, we extract after first conv layer
        # This is a general approach; adjust layer if needed
        if hasattr(model, 'conv1'):
            x = model.conv1(x, edge_index)
            x = F.relu(x)
            embedding = x[target_node]
        else:
            # Fallback: use final output if intermediate not available
            out = model(data.x, edge_index)
            embedding = out[target_node]
    
    return {
        "node_id": target_node,
        "embedding": embedding.cpu().numpy(),
        "embedding_dim": len(embedding),
        "embedding_cached": False,
    }


def get_subgraph(
    data,
    target_node,
    num_hops=2,
    max_nodes=None,
    max_edges=None,
    include_node_features=True,
    include_node_labels=True,
    seed=None,
):
    """
    Extracts the k-hop subgraph around the target node.
    
    Args:
        data: PyG Data object
        target_node: Index of the target node
        num_hops: Number of hops to extract (default: 2)
    
    Returns:
        Dictionary with subgraph structure, node indices, and edge indices
    """
    
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    original_num_nodes = int(subset.numel())
    original_num_edges = int(sub_edge_index.size(1))
    cap_stats = {
        "local_num_nodes_before_cap": original_num_nodes,
        "local_num_edges_before_cap": original_num_edges,
        "local_num_edges_before_edge_cap": original_num_edges,
        "hit_max_nodes": False,
        "hit_max_edges": False,
        "local_num_nodes_after_cap": original_num_nodes,
        "local_num_edges_after_cap": original_num_edges,
    }
    if max_nodes is not None or max_edges is not None:
        subset, sub_edge_index, mapping, cap_stats = _maybe_cap_local_subgraph(
            subset,
            sub_edge_index,
            int(mapping.item()) if hasattr(mapping, "item") else int(mapping),
            max_nodes=max_nodes,
            max_edges=max_edges,
            seed=seed,
        )
    
    # Extract subgraph node features
    sub_x = data.x[subset] if include_node_features and hasattr(data, 'x') else None
    
    # Extract subgraph labels if available
    sub_y = data.y[subset] if include_node_labels and hasattr(data, 'y') else None
    
    return {
        "node_id": target_node,
        "num_hops": num_hops,
        "subset_nodes": subset.cpu().numpy(),
        "edge_index": sub_edge_index.cpu().numpy(),
        "node_features": sub_x.cpu().numpy() if sub_x is not None else None,
        "node_labels": sub_y.cpu().numpy() if sub_y is not None else None,
        "target_mapping": mapping.item() if mapping.numel() == 1 else mapping.cpu().numpy(),
        "num_nodes": len(subset),
        "num_edges": sub_edge_index.shape[1],
        "num_nodes_before_cap": original_num_nodes,
        "num_edges_before_cap": original_num_edges,
        "num_edges_before_edge_cap": int(cap_stats.get("local_num_edges_before_edge_cap", original_num_edges)),
        "hit_max_nodes": bool(cap_stats.get("hit_max_nodes", False)),
        "hit_max_edges": bool(cap_stats.get("hit_max_edges", False)),
        "max_nodes": None if max_nodes is None else int(max_nodes),
        "max_edges": None if max_edges is None else int(max_edges),
        "include_node_features": bool(include_node_features),
        "include_node_labels": bool(include_node_labels),
    }


def get_raw_features(data, target_node):
    """Return the raw feature vector for the target node."""
    if not hasattr(data, "x") or data.x is None:
        return None
    return data.x[target_node].detach().cpu().numpy()


def get_one_hop_neighbors(data, target_node):
    """Return a sorted list of one-hop neighbor node ids (undirected view)."""
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return []
    src, dst = edge_index
    mask_src = src == int(target_node)
    mask_dst = dst == int(target_node)
    neighbors = torch.cat([dst[mask_src], src[mask_dst]]).unique()
    return sorted([int(v) for v in neighbors.detach().cpu().tolist()])


def get_neighbor_feature_table(data, neighbor_ids):
    """Return neighbor features aligned with neighbor ids for prompts."""
    table = get_node_feature_table(data, neighbor_ids)
    return {
        "neighbor_ids": table.get("node_ids", []),
        "features": table.get("features", []),
    }


def get_node_feature_table(data, node_ids):
    """Return raw node features aligned with node ids for prompts."""
    node_ids = [int(v) for v in (node_ids or [])]
    if not hasattr(data, "x") or data.x is None or not node_ids:
        return {"node_ids": [], "features": []}
    feats = data.x[torch.tensor(node_ids, device=data.x.device)]
    return {
        "node_ids": node_ids,
        "features": feats.detach().cpu().numpy(),
    }


def get_node_embedding_table(model, data, node_ids, embedding_cache=None):
    """Return first-layer GNN embeddings aligned with node ids for prompts."""
    node_ids = [int(v) for v in (node_ids or [])]
    if not node_ids:
        return {"node_ids": [], "embeddings": [], "embedding_dim": 0}
    if embedding_cache is not None:
        embeddings = embedding_cache[torch.tensor(node_ids, dtype=torch.long)]
        return {
            "node_ids": node_ids,
            "embeddings": embeddings.detach().cpu().numpy(),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else int(embeddings.numel()),
            "embedding_cached": True,
        }
    if not hasattr(data, "x") or data.x is None or not hasattr(data, "edge_index") or data.edge_index is None:
        return {"node_ids": [], "embeddings": [], "embedding_dim": 0}

    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = data.x.device if hasattr(data, "x") else torch.device("cpu")

    with torch.no_grad():
        x = data.x.to(model_device)
        edge_index = data.edge_index.to(model_device)
        if hasattr(model, 'conv1'):
            x = model.conv1(x, edge_index)
            x = F.relu(x)
        else:
            x = model(data.x.to(model_device), edge_index)
        node_tensor = torch.tensor(node_ids, dtype=torch.long, device=model_device)
        embeddings = x[node_tensor]

    return {
        "node_ids": node_ids,
        "embeddings": embeddings.detach().cpu().numpy(),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else int(embeddings.numel()),
        "embedding_cached": False,
    }


def build_candidate_set(data, target_node, true_neighbors, candidate_ratio=4, max_candidates=None, seed=None):
    """Build a candidate set with true neighbors plus sampled non-neighbors."""
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        return {
            "true_neighbors": [int(v) for v in true_neighbors],
            "candidates": [int(v) for v in true_neighbors],
        }

    rng = np.random.default_rng(seed)
    true_set = set(int(v) for v in true_neighbors)
    true_set.discard(int(target_node))
    excluded = set(true_set)
    excluded.add(int(target_node))

    neg_count = int(len(true_set) * candidate_ratio)
    if max_candidates is not None:
        max_candidates = int(max_candidates)
        neg_count = min(neg_count, max(0, max_candidates - len(true_set)))
    available_negatives = max(0, num_nodes - len(excluded))
    neg_count = min(neg_count, available_negatives)

    sampled_set = set()
    if neg_count > 0:
        max_attempts = max(1000, neg_count * 50)
        attempts = 0
        while len(sampled_set) < neg_count and attempts < max_attempts:
            candidate = int(rng.integers(0, num_nodes))
            attempts += 1
            if candidate not in excluded and candidate not in sampled_set:
                sampled_set.add(candidate)

        if len(sampled_set) < neg_count:
            start = int(rng.integers(0, num_nodes))
            for offset in range(num_nodes):
                candidate = (start + offset) % num_nodes
                if candidate not in excluded and candidate not in sampled_set:
                    sampled_set.add(candidate)
                    if len(sampled_set) >= neg_count:
                        break

    candidates = sorted(list(true_set) + [int(v) for v in sampled_set])
    return {
        "true_neighbors": sorted(list(true_set)),
        "candidates": candidates,
    }


def extract_all(
    model,
    data,
    target_node,
    num_hops=2,
    include_candidate_set=True,
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
    embedding_cache=None,
    seed=None,
):
    """
    Runs all extractions and returns a structured bundle containing:
    - GNN prediction
    - GNNExplainer explanation masks
    - Node embedding
    - k-hop subgraph
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
        num_hops: Number of hops for subgraph extraction (default: 2)
    
    Returns:
        Dictionary with all extracted components
    """
    prediction = get_prediction(model, data, target_node)
    explanation = get_explanation(
        model,
        data,
        target_node,
        explainer_top_k=explainer_top_k,
        explainer_min_score=explainer_min_score,
        explanation_scope=explanation_scope,
        explanation_num_hops=explanation_num_hops,
        explanation_max_nodes=explanation_max_nodes,
        explanation_max_edges=explanation_max_edges,
        seed=seed,
    )
    embedding = get_embedding(model, data, target_node, embedding_cache=embedding_cache)
    subgraph = None
    if include_subgraph:
        subgraph = get_subgraph(
            data,
            target_node,
            num_hops=num_hops,
            max_nodes=subgraph_max_nodes,
            max_edges=subgraph_max_edges,
            include_node_features=subgraph_include_node_features,
            include_node_labels=subgraph_include_node_labels,
            seed=seed,
        )
    raw_features = get_raw_features(data, target_node)
    one_hop_neighbors = get_one_hop_neighbors(data, target_node)
    neighbor_feature_table = get_neighbor_feature_table(data, one_hop_neighbors)
    candidate_set = None
    if include_candidate_set:
        candidate_set = build_candidate_set(data, target_node, one_hop_neighbors)
    
    return {
        "dataset": None,
        "model": None,
        "target_node": target_node,
        "ground_truth_label": prediction.get("target_class"),
        "prediction": prediction,
        "logits": prediction.get("logits"),
        "embedding": embedding,
        "embedding_dimension": embedding.get("embedding_dim"),
        "explanation": explanation,
        "explanation_mask": {
            "edge_mask": explanation.get("edge_mask"),
            "feature_mask": explanation.get("feature_mask"),
        },
        "explainer_neighbors": explanation.get("explainer_neighbors", []),
        "explainer_edges": explanation.get("explainer_edges", []),
        "subgraph": subgraph,
        "k_hop_subgraph": subgraph,
        "one_hop_neighbors": one_hop_neighbors,
        "raw_features": raw_features,
        "neighbor_feature_table": neighbor_feature_table,
        "candidate_set": candidate_set,
        "metadata": {},
    }
