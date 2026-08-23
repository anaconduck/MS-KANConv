"""
PAMAP2 Dataset Loader
=====================
Loads the PAMAP2 Physical Activity Monitoring dataset.
Uses acc_16g + gyroscope from 3 IMUs (hand, chest, ankle) = 18 channels.
Sliding window segmentation with cross-validation splits.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold

from config import DATA_DIR, PAMAP2_CONFIG

# PAMAP2 activity ID mapping (original ID -> 0-indexed label)
ACTIVITY_MAP = {
    1: 0,   # lying
    2: 1,   # sitting
    3: 2,   # standing
    4: 3,   # walking
    5: 4,   # running
    6: 5,   # cycling
    7: 6,   # Nordic walking
    12: 7,  # ascending stairs
    13: 8,  # descending stairs
    16: 9,  # vacuum cleaning
    17: 10, # ironing
    24: 11, # rope jumping
}

# Column indices for the sensor channels we use:
# 3 IMUs (hand=cols 4-20, chest=cols 21-37, ankle=cols 38-54)
# From each IMU: acc_16g (3 cols) + gyro (3 cols) = 6 cols
# Hand: acc_16g -> cols 4,5,6; gyro -> cols 10,11,12
# Chest: acc_16g -> cols 21,22,23; gyro -> cols 27,28,29
# Ankle: acc_16g -> cols 38,39,40; gyro -> cols 44,45,46
SENSOR_COLS = [
    4, 5, 6, 10, 11, 12,    # hand (acc_16g + gyro)
    21, 22, 23, 27, 28, 29,  # chest (acc_16g + gyro)
    38, 39, 40, 44, 45, 46,  # ankle (acc_16g + gyro)
]

SUBJECT_FILES = [
    "subject101.dat", "subject102.dat", "subject103.dat",
    "subject104.dat", "subject105.dat", "subject106.dat",
    "subject107.dat", "subject108.dat", "subject109.dat",
]


def _sliding_window(data, labels, window_size, overlap):
    """Create sliding windows from continuous data."""
    step = int(window_size * (1 - overlap))
    windows = []
    window_labels = []

    for start in range(0, len(data) - window_size + 1, step):
        end = start + window_size
        window = data[start:end]
        window_lab = labels[start:end]

        # Only keep windows where all samples have the same label
        unique_labels = np.unique(window_lab)
        if len(unique_labels) == 1 and unique_labels[0] >= 0:
            windows.append(window)
            window_labels.append(unique_labels[0])

    if len(windows) == 0:
        return np.empty((0, window_size, data.shape[1])), np.empty(0, dtype=int)

    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=int)


def _load_subject(filepath):
    """Load data for a single subject."""
    data = np.loadtxt(filepath)

    # Extract activity labels (column 1)
    raw_labels = data[:, 1].astype(int)

    # Map activity IDs to 0-indexed labels; -1 for unlabeled/other
    labels = np.full(len(raw_labels), -1, dtype=int)
    for orig_id, new_id in ACTIVITY_MAP.items():
        labels[raw_labels == orig_id] = new_id

    # Extract sensor channels
    sensor_data = data[:, SENSOR_COLS]

    # Handle NaN: forward-fill then backfill, then zero
    for col in range(sensor_data.shape[1]):
        col_data = sensor_data[:, col]
        mask = np.isnan(col_data)
        if mask.all():
            sensor_data[:, col] = 0.0
            continue
        # Forward fill
        valid = np.where(~mask)[0]
        if len(valid) > 0:
            for i in range(len(col_data)):
                if mask[i]:
                    prev = valid[valid <= i]
                    if len(prev) > 0:
                        sensor_data[i, col] = sensor_data[prev[-1], col]
                    else:
                        nxt = valid[valid > i]
                        if len(nxt) > 0:
                            sensor_data[i, col] = sensor_data[nxt[0], col]
                        else:
                            sensor_data[i, col] = 0.0

    return sensor_data, labels


def load_pamap2():
    """
    Load PAMAP2 dataset with sliding window segmentation.

    Returns:
        X: np.ndarray of shape (N, C, T) — all windowed samples
        y: np.ndarray of shape (N,) — labels
        subject_ids: np.ndarray of shape (N,) — subject ID per sample
        config: DatasetConfig
    """
    dataset_dir = os.path.join(DATA_DIR, "pamap2")

    # Handle nested extraction
    possible_paths = [
        os.path.join(dataset_dir, "PAMAP2_Dataset", "Protocol"),
        os.path.join(dataset_dir, "PAMAP2_Dataset", "PAMAP2_Dataset", "Protocol"),
        os.path.join(dataset_dir, "Protocol"),
    ]

    base_path = None
    for p in possible_paths:
        if os.path.isdir(p):
            base_path = p
            break

    if base_path is None:
        raise FileNotFoundError(
            f"PAMAP2 dataset not found at {dataset_dir}. "
            f"Run datasets/download.py first."
        )

    print(f"Loading PAMAP2 from {base_path}...")

    all_X, all_y, all_subj = [], [], []
    cfg = PAMAP2_CONFIG

    for subj_idx, fname in enumerate(SUBJECT_FILES):
        fpath = os.path.join(base_path, fname)
        if not os.path.isfile(fpath):
            print(f"  Warning: {fname} not found, skipping.")
            continue

        sensor_data, labels = _load_subject(fpath)
        X_windows, y_windows = _sliding_window(
            sensor_data, labels, cfg.window_size, cfg.overlap
        )

        if len(X_windows) > 0:
            # Transpose to (N, C, T)
            X_windows = X_windows.transpose(0, 2, 1)
            all_X.append(X_windows)
            all_y.append(y_windows)
            all_subj.append(np.full(len(y_windows), subj_idx, dtype=int))
            print(f"  {fname}: {len(y_windows)} windows")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids = np.concatenate(all_subj, axis=0)

    # Normalize per channel (z-score across all data)
    for c in range(X.shape[1]):
        mean = X[:, c, :].mean()
        std = X[:, c, :].std() + 1e-8
        X[:, c, :] = (X[:, c, :] - mean) / std

    print(f"  Total: {X.shape}, Classes: {len(np.unique(y))}")

    return X, y, subject_ids, cfg


def get_pamap2_fold(X, y, subject_ids, fold: int, n_folds: int = 5,
                    seed: int = 42):
    """
    Get train/test split for a given fold using stratified k-fold.

    Returns:
        train_dataset, test_dataset
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    train_idx, test_idx = splits[fold]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx]).long()
    X_test = torch.from_numpy(X[test_idx])
    y_test = torch.from_numpy(y[test_idx]).long()

    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
