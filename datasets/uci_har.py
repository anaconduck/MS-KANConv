import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from config import DATA_DIR, UCI_HAR_CONFIG


def _load_signals(base_path: str, split: str):
    signal_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]
    signals = []
    signal_dir = os.path.join(base_path, split, "Inertial Signals")
    for sig in signal_names:
        filename = f"{sig}_{split}.txt"
        filepath = os.path.join(signal_dir, filename)
        data = np.loadtxt(filepath)
        signals.append(data)
    return np.stack(signals, axis=1).astype(np.float32)


def _load_labels(base_path: str, split: str):
    filepath = os.path.join(base_path, split, f"y_{split}.txt")
    labels = np.loadtxt(filepath, dtype=int) - 1
    return labels


def load_uci_har():
    dataset_dir = os.path.join(DATA_DIR, "uci_har")
    possible_paths = [
        os.path.join(dataset_dir, "UCI HAR Dataset"),
        os.path.join(dataset_dir, "UCI HAR Dataset", "UCI HAR Dataset"),
        dataset_dir,
    ]
    base_path = None
    for p in possible_paths:
        if os.path.isdir(os.path.join(p, "train")):
            base_path = p
            break
    if base_path is None:
        raise FileNotFoundError(
            f"UCI-HAR dataset not found. Expected directory structure at "
            f"{dataset_dir}. Run datasets/download.py first."
        )
    print(f"Loading UCI-HAR from {base_path}...")
    X_train = _load_signals(base_path, "train")
    y_train = _load_labels(base_path, "train")
    X_test = _load_signals(base_path, "test")
    y_test = _load_labels(base_path, "test")
    for c in range(X_train.shape[1]):
        mean = X_train[:, c, :].mean()
        std = X_train[:, c, :].std() + 1e-8
        X_train[:, c, :] = (X_train[:, c, :] - mean) / std
        X_test[:, c, :] = (X_test[:, c, :] - mean) / std
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}, " f"Labels: {np.unique(y_train)}")
    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).long()
    )
    return train_dataset, test_dataset, UCI_HAR_CONFIG
