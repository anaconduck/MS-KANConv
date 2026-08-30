import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from config import DATASET_CONFIGS, DATA_DIR
from sklearn.preprocessing import StandardScaler

def load_wisdm():
    cfg = DATASET_CONFIGS["wisdm"]
    filepath = os.path.join(DATA_DIR, "csv", "wisdm_raw.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File {filepath} tidak ditemukan. Silakan jalankan generate_csv.py terlebih dahulu dengan data mentah WISDM di folder data/wisdm/."
        )
        
    df = pd.read_csv(filepath)
    
    # Mapping label to integer 0-5
    # WISDM labels: "Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"
    label_mapping = {
        "Walking": 0,
        "Jogging": 1,
        "Upstairs": 2,
        "Downstairs": 3,
        "Sitting": 4,
        "Standing": 5
    }
    
    df['label'] = df['activity'].map(label_mapping)
    
    # Drop rows with unmapped labels if any
    df = df.dropna(subset=['label'])
    
    # Sort by user and then keep order
    df = df.sort_values(by=['user'])
    
    # Create windows
    # Since WISDM is user-based, we window per user to avoid mixing
    window_size = cfg.window_size
    step = int(window_size * (1 - cfg.overlap))
    
    X_windows = []
    y_windows = []
    user_windows = []
    
    sensor_cols = ['x', 'y', 'z']
    
    for user_id, group in df.groupby("user"):
        data_values = group[sensor_cols].values
        labels = group['label'].values
        
        n_samples = len(data_values)
        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            window_data = data_values[start:end]
            window_labels = labels[start:end]
            
            # Use mode for window label
            unique, counts = np.unique(window_labels, return_counts=True)
            label = unique[np.argmax(counts)]
            
            X_windows.append(window_data)
            y_windows.append(label)
            user_windows.append(user_id)
            
    X_windows = np.array(X_windows) # Shape (N, 128, 3)
    # Permute to (N, C, T) -> (N, 3, 128)
    X_windows = X_windows.transpose(0, 2, 1)
    
    y_windows = np.array(y_windows, dtype=np.int64)
    user_windows = np.array(user_windows)
    
    return X_windows, y_windows, user_windows, cfg

def get_wisdm_fold(X, y, user, fold_idx, n_folds, seed=42):
    unique_users = np.unique(user)
    np.random.seed(seed)
    np.random.shuffle(unique_users)
    
    # Split users into n_folds
    users_per_fold = len(unique_users) // n_folds
    
    start_idx = fold_idx * users_per_fold
    if fold_idx == n_folds - 1:
        end_idx = len(unique_users)
    else:
        end_idx = start_idx + users_per_fold
        
    test_users = unique_users[start_idx:end_idx]
    
    test_mask = np.isin(user, test_users)
    train_mask = ~test_mask
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Standarisasi (fit di train, transform di train & test)
    N_train, C, T = X_train.shape
    N_test = X_test.shape[0]
    
    # Reshape untuk standard scaler (N*T, C)
    X_train_flat = X_train.transpose(0, 2, 1).reshape(N_train * T, C)
    X_test_flat = X_test.transpose(0, 2, 1).reshape(N_test * T, C)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Kembalikan ke shape semula
    X_train = X_train_flat.reshape(N_train, T, C).transpose(0, 2, 1)
    X_test = X_test_flat.reshape(N_test, T, C).transpose(0, 2, 1)
    
    # Konversi ke Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_ds, test_ds
