"""
mHealth Dataset Loader
======================
Loads the MHEALTH (Mobile Health) dataset.
Uses 21 sensor channels (chest acc + ankle IMU + wrist IMU), excluding ECG.
Sliding window segmentation with cross-validation splits.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold

from config import DATA_DIR, MHEALTH_CONFIG

# mHealth data columns (0-indexed):
# 0-2: chest accelerometer (x,y,z)
# 3-4: ECG (lead 1, lead 2) -- EXCLUDED
# 5-7: left-ankle accelerometer (x,y,z)
# 8-10: left-ankle gyroscope (x,y,z)
# 11-13: left-ankle magnetometer (x,y,z)
# 14-16: right-lower-arm accelerometer (x,y,z)
# 17-19: right-lower-arm gyroscope (x,y,z)
# 20-22: right-lower-arm magnetometer (x,y,z)
# 23: label

# We use all sensor channels except ECG (columns 3,4)
SENSOR_COLS = list(range(0, 3)) + list(range(5, 23))  # 21 channels

NUM_SUBJECTS = 10


def _sliding_window(data, labels, window_size, overlap):
    """Create sliding windows from continuous data."""
    step = int(window_size * (1 - overlap))
    windows = []
    window_labels = []

    for start in range(0, len(data) - window_size + 1, step):
        end = start + window_size
        window = data[start:end]
        window_lab = labels[start:end]

        # Only keep windows with single consistent label (non-null)
        unique_labels = np.unique(window_lab)
        if len(unique_labels) == 1 and unique_labels[0] > 0:
            # Map label to 0-indexed (original: 1-12 -> 0-11)
            windows.append(window)
            window_labels.append(unique_labels[0] - 1)

    if len(windows) == 0:
        return np.empty((0, window_size, data.shape[1])), np.empty(0, dtype=int)

    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=int)


def load_mhealth():
    """
    Load mHealth dataset with sliding window segmentation.

    Returns:
        X: np.ndarray of shape (N, C, T) — all windowed samples
        y: np.ndarray of shape (N,) — labels (0-indexed)
        subject_ids: np.ndarray of shape (N,) — subject ID per sample
        config: DatasetConfig
    """
    dataset_dir = os.path.join(DATA_DIR, "mhealth")

    # Handle nested extraction
    possible_paths = [
        os.path.join(dataset_dir, "MHEALTHDATASET"),
        os.path.join(dataset_dir, "MHEALTHDATASET", "MHEALTHDATASET"),
        dataset_dir,
    ]

    base_path = None
    for p in possible_paths:
        test_file = os.path.join(p, "mHealth_subject1.log")
        if os.path.isfile(test_file):
            base_path = p
            break

    if base_path is None:
        raise FileNotFoundError(
            f"mHealth dataset not found at {dataset_dir}. "
            f"Run datasets/download.py first."
        )

    print(f"Loading mHealth from {base_path}...")

    all_X, all_y, all_subj = [], [], []
    cfg = MHEALTH_CONFIG

    for subj_idx in range(1, NUM_SUBJECTS + 1):
        fname = f"mHealth_subject{subj_idx}.log"
        fpath = os.path.join(base_path, fname)

        if not os.path.isfile(fpath):
            print(f"  Warning: {fname} not found, skipping.")
            continue

        data = np.loadtxt(fpath)
        sensor_data = data[:, SENSOR_COLS]
        labels = data[:, 23].astype(int)  # last column is label

        # Replace NaN with 0
        sensor_data = np.nan_to_num(sensor_data, nan=0.0)

        X_windows, y_windows = _sliding_window(
            sensor_data, labels, cfg.window_size, cfg.overlap
        )

        if len(X_windows) > 0:
            # Transpose to (N, C, T)
            X_windows = X_windows.transpose(0, 2, 1)
            all_X.append(X_windows)
            all_y.append(y_windows)
            all_subj.append(np.full(len(y_windows), subj_idx - 1, dtype=int))
            print(f"  Subject {subj_idx}: {len(y_windows)} windows")

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


def get_mhealth_fold(X, y, subject_ids, fold: int, n_folds: int = 5,
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
