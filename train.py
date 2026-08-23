"""
Training & Evaluation Utilities
================================
Training loop, evaluation, and helper functions for all experiments.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

from config import TRAIN_CONFIG, RESULTS_DIR


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_model(model, train_dataset, test_dataset, config=None,
                model_name="model", dataset_name="dataset",
                device=None, verbose=True):
    """
    Full training pipeline with early stopping and learning rate scheduling.

    Args:
        model: The PyTorch model.
        train_dataset: TensorDataset for training.
        test_dataset: TensorDataset for testing.
        config: TrainingConfig (uses default if None).
        model_name: Name for logging.
        dataset_name: Dataset name for logging.
        device: Device to train on.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training history and best test results.
    """
    if config is None:
        config = TRAIN_CONFIG
    if device is None:
        device = get_device()

    model = model.to(device)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True,
    )

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=1e-4,
    )

    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.step_gamma
        )

    # Training loop
    best_test_acc = 0.0
    best_test_results = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    start_time = time.time()

    iterator = range(1, config.epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc=f"{model_name}/{dataset_name}")

    for epoch in iterator:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_results = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_results["loss"])
        history["test_acc"].append(test_results["accuracy"])

        if verbose:
            iterator.set_postfix({
                "tr_loss": f"{train_loss:.4f}",
                "tr_acc": f"{train_acc:.4f}",
                "te_acc": f"{test_results['accuracy']:.4f}",
                "te_f1": f"{test_results['f1_weighted']:.4f}",
            })

        # Early stopping check
        if test_results["accuracy"] > best_test_acc:
            best_test_acc = test_results["accuracy"]
            best_test_results = test_results.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time

    result = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "best_accuracy": best_test_results["accuracy"],
        "best_f1_weighted": best_test_results["f1_weighted"],
        "best_f1_macro": best_test_results["f1_macro"],
        "total_epochs": epoch,
        "training_time_sec": elapsed,
        "num_parameters": count_parameters(model),
        "history": history,
    }

    if verbose:
        print(f"\n  Best: Acc={result['best_accuracy']:.4f}, "
              f"F1w={result['best_f1_weighted']:.4f}, "
              f"F1m={result['best_f1_macro']:.4f}, "
              f"Params={result['num_parameters']:,}, "
              f"Time={elapsed:.1f}s")

    return result


def save_results(results: dict, filename: str):
    """Save results dictionary to JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filename: str) -> dict:
    """Load results from JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "r") as f:
        return json.load(f)


def print_results_table(results_list: list, title: str = "Results"):
    """Pretty-print results as a table."""
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback
        print(f"\n{'='*70}")
        print(title)
        print(f"{'='*70}")
        for r in results_list:
            print(f"  {r['model_name']:25s} | "
                  f"Acc: {r['best_accuracy']:.4f} | "
                  f"F1w: {r['best_f1_weighted']:.4f} | "
                  f"F1m: {r['best_f1_macro']:.4f} | "
                  f"Params: {r['num_parameters']:>10,}")
        return

    headers = ["Model", "Accuracy", "F1 (Weighted)", "F1 (Macro)", "Params",
               "Time (s)"]
    rows = []
    for r in results_list:
        rows.append([
            r["model_name"],
            f"{r['best_accuracy']:.4f}",
            f"{r['best_f1_weighted']:.4f}",
            f"{r['best_f1_macro']:.4f}",
            f"{r['num_parameters']:,}",
            f"{r.get('training_time_sec', 0):.1f}",
        ])

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
