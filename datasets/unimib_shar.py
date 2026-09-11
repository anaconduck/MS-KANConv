import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
from config import DATA_DIR, UNIMIB_SHAR_CONFIG


def load_unimib_shar():
    """Memuat dataset UniMiB-SHAR dari berkas acc_data.mat dan acc_labels.mat.
    
    Returns:
        X (np.ndarray): Sinyal akselerometer bentuk (N, 3, 151)
        y (np.ndarray): Label aktivitas bentuk (N,) dengan rentang 0-16
        subject_ids (np.ndarray): ID subjek bentuk (N,) dengan rentang 0-29
        cfg (DatasetConfig): Konfigurasi dataset UniMiB-SHAR
    """
    dataset_dir = os.path.join(DATA_DIR, "unimib_shar")
    possible_paths = [
        os.path.join(dataset_dir, "UniMiB-SHAR", "data"),
        os.path.join(dataset_dir, "data"),
        os.path.join(dataset_dir, "UniMiB-SHAR"),
        dataset_dir,
    ]
    
    data_file = None
    labels_file = None
    
    for p in possible_paths:
        f_data = os.path.join(p, "acc_data.mat")
        f_labels = os.path.join(p, "acc_labels.mat")
        if os.path.isfile(f_data) and os.path.isfile(f_labels):
            data_file = f_data
            labels_file = f_labels
            break
            
    # Jika belum ditemukan, lakukan pencarian rekursif di folder dataset_dir
    if data_file is None and os.path.isdir(dataset_dir):
        for root, _, files in os.walk(dataset_dir):
            if "acc_data.mat" in files and "acc_labels.mat" in files:
                data_file = os.path.join(root, "acc_data.mat")
                labels_file = os.path.join(root, "acc_labels.mat")
                break
                
    if data_file is None:
        raise FileNotFoundError(
            f"UniMiB-SHAR dataset tidak ditemukan di {dataset_dir}. "
            f"Pastikan berkas acc_data.mat dan acc_labels.mat ada di folder data/unimib_shar/. "
            f"Jalankan 'python datasets/download.py' untuk mengunduh otomatis."
        )
        
    print(f"Loading UniMiB-SHAR from {data_file}...")
    cfg = UNIMIB_SHAR_CONFIG
    
    # Membaca berkas MATLAB .mat
    mat_data = sio.loadmat(data_file)
    mat_labels = sio.loadmat(labels_file)
    
    raw_data = mat_data["acc_data"]  # Bentuk: (11771, 453)
    raw_labels = mat_labels["acc_labels"]  # Bentuk: (11771, 3)
    
    N = raw_data.shape[0]
    T = cfg.window_size  # 151
    C = cfg.input_channels  # 3
    
    # 453 kolom terdiri dari 151 sampel sumbu X, 151 sumbu Y, dan 151 sumbu Z
    X = np.zeros((N, C, T), dtype=np.float32)
    X[:, 0, :] = raw_data[:, 0:T]
    X[:, 1, :] = raw_data[:, T:2 * T]
    X[:, 2, :] = raw_data[:, 2 * T:3 * T]
    
    # Kolom 0: Label aktivitas 1-17 -> jadikan 0-16
    y = (raw_labels[:, 0] - 1).astype(int)
    
    # Kolom 1: ID subjek 1-30 -> jadikan 0-29
    subject_ids = (raw_labels[:, 1] - 1).astype(int)
    
    # Standarisasi sinyal per kanal sensor
    for c in range(X.shape[1]):
        mean = X[:, c, :].mean()
        std = X[:, c, :].std() + 1e-8
        X[:, c, :] = (X[:, c, :] - mean) / std
        
    print(f"  UniMiB-SHAR dimuat: {X.shape}, Jumlah Kelas: {len(np.unique(y))}, Jumlah Subjek: {len(np.unique(subject_ids))}")
    return X, y, subject_ids, cfg


def get_unimib_shar_fold(X, y, subject_ids, fold: int, n_folds: int = 5, seed: int = 42):
    """Membagi data UniMiB-SHAR ke dalam lipatan validasi silang (StratifiedKFold)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    train_idx, test_idx = splits[fold]
    
    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx]).long()
    X_test = torch.from_numpy(X[test_idx])
    y_test = torch.from_numpy(y[test_idx]).long()
    
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
