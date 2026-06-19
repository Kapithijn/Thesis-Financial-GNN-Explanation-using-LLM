import json
import csv
import os
from LLM_Module import parse_prediction

"""Evaluation helpers for comparing GNN outputs with LLM predictions.
Functions are intentionally small stubs; fill in implementation as needed.
"""


def compare_predictions(gnn_pred, llm_pred):
    """Compare a single GNN prediction to a single LLM prediction.
    - If labels are ints, compare ints; if strings, normalize case.
    - If LLM output is a sentence, parse it first (use parse_prediction()).
    - Implement custom matching rules here.
    """
    if isinstance(gnn_pred, int):
        try:
            llm_label = llm_pred if isinstance(llm_pred, int) else parse_prediction(str(llm_pred))
            return gnn_pred == int(llm_label)
        except (TypeError, ValueError):
            return False
    elif isinstance(gnn_pred, str):
        gnn_label = gnn_pred.strip().lower()
        llm_label = parse_prediction(str(llm_pred))
        return gnn_label == str(llm_label).strip().lower()
    else:
        raise TypeError("Unsupported prediction types for comparison.")


def compute_accuracy(results):
    """Compute accuracy over a set of result records.
    - results: list of dicts containing 'gnn_pred' and 'llm_pred'.
    - Use compare_predictions() for per-instance matching.
    - Return 0.0 for empty input to avoid division-by-zero.
    """
    if not results:
        return 0.0

    points = 0
    for record in results:
        if compare_predictions(record['gnn_pred'], record['llm_pred']):
            points += 1
    return points / len(results)


def compute_precision_recall_f1(y_true, y_pred):
    """Compute precision/recall/F1 for binary labels without sklearn."""
    tp = fp = tn = fn = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_classification_metrics(results):
    """Compute accuracy, precision, recall, and F1 for results records."""
    if not results:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "parse_rate": 0.0,
            "valid_n": 0,
            "unknown_n": 0,
        }

    y_true = [int(r["gnn_pred"]) for r in results]
    y_pred = []
    valid_n = 0
    unknown_n = 0
    for r in results:
        try:
            parsed = parse_prediction(str(r["llm_pred"]))
            if parsed not in (0, 1):
                raise ValueError(f"Unparseable LLM prediction: {r['llm_pred']}")
            r["llm_pred"] = int(parsed)
            valid_n += 1
        except (ValueError, TypeError):
            r["llm_pred"] = 2  # Assign a default label for unparseable predictions
            unknown_n += 1
        y_pred.append(int(r["llm_pred"]))

    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred)
    accuracy = compute_accuracy(results)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_rate": valid_n / len(results),
        "valid_n": valid_n,
        "unknown_n": unknown_n,
    }


def evaluate_reconstruction(records):
    """Evaluate neighbor reconstruction outputs using set-based metrics."""
    if not records:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "overlap": 0.0,
            "edit_distance": 0.0,
        }

    precisions = []
    recalls = []
    f1s = []
    jaccards = []
    overlaps = []
    edit_distances = []

    for record in records:
        true_neighbors = set(record.get("true_neighbors", []) or [])
        predicted = set(record.get("predicted_neighbors", []) or [])

        intersection = true_neighbors.intersection(predicted)
        union = true_neighbors.union(predicted)

        precision = len(intersection) / len(predicted) if predicted else 0.0
        recall = len(intersection) / len(true_neighbors) if true_neighbors else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / max(1, len(true_neighbors))
        edit_distance = len(union) - len(intersection)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        jaccards.append(jaccard)
        overlaps.append(overlap)
        edit_distances.append(edit_distance)

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "jaccard": sum(jaccards) / len(jaccards),
        "overlap": sum(overlaps) / len(overlaps),
        "edit_distance": sum(edit_distances) / len(edit_distances),
    }


def aggregate_results(all_results):
    """Aggregate results grouped by model/dataset/LLM.
    - all_results: {'group_name': [record, ...], ...}
    - For each group compute accuracy and sample counts (use compute_accuracy()).
    """
    summary = {}
    for group, records in all_results.items():
        accuracy = compute_accuracy(records)
        count = len(records)
        summary[group] = {'accuracy': accuracy, 'count': count}
    return summary


def save_results(results, path, fmt="json"):
    """Persist results to disk in 'json' or 'csv' format.
    - For csv: results must be a list of flat dicts (use csv.DictWriter).
    - For json: use json.dump(results, f, indent=2).
    - Ensure parent directory exists before writing.
    """
    # FIX: Handle case where path is just a filename (no directory component)
    # os.path.dirname('results.json') returns '', so we use '.' as fallback
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if fmt == "json":
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    elif fmt == "csv":
        if not results:
            raise ValueError("No results to save.")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_results(path):
    """Load persisted results from disk (json or csv inferred from extension).
    - Return structure written by save_results().
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, 'r') as f:
            return json.load(f)
    elif ext == ".csv":
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def plot_results(results, out_path=None):
    """Produce simple plots comparing GNN vs LLM predictions.
    - Optional: use matplotlib/seaborn. Skip plotting if libraries unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Plotting libraries not available. Skipping plot generation.")
        return

    # Example: Accuracy by model/dataset group
    groups = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=groups, y=accuracies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('GNN vs LLM Prediction Accuracy by Group')
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
    else:
        plt.show()


def compute_confusion_matrix(results):
    """Compute confusion matrix between GNN and LLM predictions.
    - Use sklearn.metrics.confusion_matrix if available.
    - Otherwise, implement manual counting.
    """
    try:
        from sklearn.metrics import confusion_matrix
        y_true = [record['gnn_pred'] for record in results]
        y_pred = [parse_prediction(record['llm_pred']) for record in results]
        return confusion_matrix(y_true, y_pred)
    except ImportError:
        print("sklearn not available. Implementing manual confusion matrix.")
        # Manual implementation (simplified, assumes binary classification)
        tp = fp = tn = fn = 0
        for record in results:
            gnn_label = record['gnn_pred']
            llm_label = parse_prediction(record['llm_pred'])
            if gnn_label == 1 and llm_label == 1:
                tp += 1
            elif gnn_label == 0 and llm_label == 1:
                fp += 1
            elif gnn_label == 0 and llm_label == 0:
                tn += 1
            elif gnn_label == 1 and llm_label == 0:
                fn += 1
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def summarize_by_group(results, group_key):
    """Summarize results grouped by `group_key` (e.g., 'model', 'dataset', 'llm').
    - Collect records per group value and compute basic stats (accuracy, count).
    """
    summary = {}
    for record in results:
        group_value = record.get(group_key, 'Unknown')
        if group_value not in summary:
            summary[group_value] = []
        summary[group_value].append(record)
    
    # Compute accuracy for each group
    for group_value, group_records in summary.items():
        summary[group_value] = {
            'accuracy': compute_accuracy(group_records),
            'count': len(group_records)
        }
    
    return summary
