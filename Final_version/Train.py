import random
import numpy as np
import torch

ENABLE_EVALUATION = True  # Set to False to skip evaluation after training


def _move_data_to_device(data, device):
    if hasattr(data, "to"):
        return data.to(device)
    return data


def _is_cuda_oom(error):
	message = str(error).lower()
	return isinstance(error, torch.OutOfMemoryError) or "out of memory" in message


def _build_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, data, optimizer, criterion):
    """Single training epoch for one model."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    logits = out[data.train_mask]
    target = data.y[data.train_mask]
    loss = criterion(logits, target)

    if not torch.isfinite(loss):
        raise RuntimeError("Loss became non-finite. Check feature preprocessing and labels.")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    return float(loss.item())

def evaluate(model, data):
    """Evaluate model on test data."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        if hasattr(data, 'test_mask'):
            logits = out[data.test_mask]
            target = data.y[data.test_mask]
        else:
            logits = out
            target = data.y
        
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == target).float().mean().item()
    
    return accuracy

def train_model(model, data, optimizer, criterion, config):
    """Full training loop for one model with optional early stopping."""
    epochs = config.get("epochs", 200)
    print_every = config.get("print_every", 20)
    patience = config.get("patience", None)
    
    history = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        loss = train_epoch(model, data, optimizer, criterion)
        history.append(loss)
        
        if patience is not None:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    return history

def train_all(model_bundle, datasets, config, device="cpu"):
    """Train all models on all datasets."""
    histories = {}
    lr = config.get("lr", 0.01)
    enable_large_graph_cpu_fallback = bool(config.get("large_graph_cpu_fallback", True))

    torch_device = torch.device(device) if device is not None else torch.device("cpu")

    for dataset_name, data in datasets.items():
        for model_name, model in model_bundle.items():
            initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            criterion = torch.nn.CrossEntropyLoss()
            
            key = f"{model_name}_{dataset_name}"
            try:
                optimizer = _build_optimizer(model, lr)
                model.to(torch_device)
                data_device = _move_data_to_device(data, torch_device)
                histories[key] = train_model(model, data_device, optimizer, criterion, config)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if (
                    not enable_large_graph_cpu_fallback
                    or not _is_cuda_oom(exc)
                    or torch_device.type != "cuda"
                ):
                    raise

                print(f"CUDA OOM while training {model_name} on {dataset_name}; retrying on CPU.")
                torch.cuda.empty_cache()
                model.load_state_dict(initial_state)
                model.to("cpu")
                optimizer = _build_optimizer(model, lr)
                cpu_data = _move_data_to_device(data, torch.device("cpu"))
                histories[key] = train_model(model, cpu_data, optimizer, criterion, config)

    print("Training complete for all models and datasets.")
    if ENABLE_EVALUATION:
        print("Evaluating models...")
        for dataset_name, data in datasets.items():
            for model_name, model in model_bundle.items():
                try:
                    model.to(torch_device)
                    data_device = _move_data_to_device(data, torch_device)
                    accuracy = evaluate(model, data_device)
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if (
                        not enable_large_graph_cpu_fallback
                        or not _is_cuda_oom(exc)
                        or torch_device.type != "cuda"
                    ):
                        raise

                    torch.cuda.empty_cache()
                    model.to("cpu")
                    accuracy = evaluate(model, _move_data_to_device(data, torch.device("cpu")))
                print(f"{model_name} on {dataset_name}: Accuracy = {accuracy:.4f}")
    return histories

def save_model(model, path):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load model weights from disk."""
    model.load_state_dict(torch.load(path))
    return model
