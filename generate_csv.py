import os
import pandas as pd
import numpy as np
from config import DATA_DIR

def convert_uci_har():
    print("Memproses UCI-HAR...")
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
        print("UCI HAR tidak ditemukan!")
        return
        
    signal_names = ["body_acc_x", "body_acc_y", "body_acc_z", 
                    "body_gyro_x", "body_gyro_y", "body_gyro_z", 
                    "total_acc_x", "total_acc_y", "total_acc_z"]
    
    signals = []
    for sig in signal_names:
        filepath = os.path.join(base_path, "train", "Inertial Signals", f"{sig}_train.txt")
        data = np.loadtxt(filepath) 
        signals.append(data)
        
    signals = np.stack(signals, axis=1) # (N, 9, 128)
    
    y_path = os.path.join(base_path, "train", "y_train.txt")
    labels = np.loadtxt(y_path, dtype=int)
    
    # Meratakan data agar mudah dibaca di Excel
    N, C, T = signals.shape
    window_ids = np.repeat(np.arange(N), T)
    labels_rep = np.repeat(labels, T)
    
    X_reshaped = signals.transpose(0, 2, 1).reshape(N * T, C)
    
    df = pd.DataFrame(X_reshaped, columns=signal_names)
    df.insert(0, 'label', labels_rep)
    df.insert(0, 'window_id', window_ids)
    
    out_path = os.path.join(DATA_DIR, "csv", "uci_har_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"Berhasil menyimpan {out_path}")

def convert_pamap2():
    print("Memproses PAMAP2...")
    dataset_dir = os.path.join(DATA_DIR, "pamap2")
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
        print("PAMAP2 tidak ditemukan!")
        return
    
    dfs = []
    # Memberi nama pada 54 kolom asli dari dataset PAMAP2
    cols = ["timestamp", "activity_id", "heart_rate"]
    for pos in ["hand", "chest", "ankle"]:
        cols += [f"{pos}_temperature"]
        cols += [f"{pos}_acc16g_{axis}" for axis in ['x','y','z']]
        cols += [f"{pos}_acc6g_{axis}" for axis in ['x','y','z']]
        cols += [f"{pos}_gyro_{axis}" for axis in ['x','y','z']]
        cols += [f"{pos}_mag_{axis}" for axis in ['x','y','z']]
        cols += [f"{pos}_orientation_{i}" for i in range(4)]
        
    for i in range(101, 110):
        filepath = os.path.join(base_path, f"subject{i}.dat")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols)
            df.insert(0, 'subject_id', i)
            dfs.append(df)
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "csv", "pamap2_raw.csv")
        full_df.to_csv(out_path, index=False)
        print(f"Berhasil menyimpan {out_path}")

def convert_mhealth():
    print("Memproses mHealth...")
    dataset_dir = os.path.join(DATA_DIR, "mhealth")
    possible_paths = [
        os.path.join(dataset_dir, "MHEALTHDATASET"),
        os.path.join(dataset_dir, "MHEALTHDATASET", "MHEALTHDATASET"),
        dataset_dir,
    ]
    base_path = None
    for p in possible_paths:
        if os.path.isfile(os.path.join(p, "mHealth_subject1.log")):
            base_path = p
            break
            
    if base_path is None:
        print("mHealth tidak ditemukan!")
        return
        
    cols = []
    cols += [f"acc_chest_{axis}" for axis in ['x','y','z']]
    cols += ["ecg_lead_1", "ecg_lead_2"]
    cols += [f"acc_ankle_{axis}" for axis in ['x','y','z']]
    cols += [f"gyro_ankle_{axis}" for axis in ['x','y','z']]
    cols += [f"mag_ankle_{axis}" for axis in ['x','y','z']]
    cols += [f"acc_arm_{axis}" for axis in ['x','y','z']]
    cols += [f"gyro_arm_{axis}" for axis in ['x','y','z']]
    cols += [f"mag_arm_{axis}" for axis in ['x','y','z']]
    cols += ["activity_id"]
    
    dfs = []
    for i in range(1, 11):
        filepath = os.path.join(base_path, f"mHealth_subject{i}.log")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols)
            df.insert(0, 'subject_id', i)
            dfs.append(df)
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "csv", "mhealth_raw.csv")
        full_df.to_csv(out_path, index=False)
        print(f"Berhasil menyimpan {out_path}")

def convert_wisdm():
    print("Memproses WISDM...")
    dataset_dir = os.path.join(DATA_DIR, "wisdm")
    possible_paths = [
        os.path.join(dataset_dir, "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt"),
        os.path.join(dataset_dir, "WISDM_ar_v1.1_raw.txt"),
    ]
    filepath = None
    for p in possible_paths:
        if os.path.exists(p):
            filepath = p
            break
            
    if filepath is None:
        print("WISDM tidak ditemukan! Pastikan file WISDM_ar_v1.1_raw.txt ada di data/wisdm/")
        return
        
    cols = ["user", "activity", "timestamp", "x", "y", "z"]
    
    # Membaca data WISDM, mengabaikan baris error dan menghapus semicolon di akhir baris Z
    try:
        df = pd.read_csv(filepath, header=None, names=cols, on_bad_lines='skip', lineterminator='\n')
    except TypeError:
        # For older pandas versions
        df = pd.read_csv(filepath, header=None, names=cols, error_bad_lines=False, lineterminator='\n')
        
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float, errors='ignore')
    
    # Hapus baris dengan nilai NaN
    df = df.dropna()
    
    # Ubah z menjadi numerik setelah menghapus ';'
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df = df.dropna()
    
    out_path = os.path.join(DATA_DIR, "csv", "wisdm_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"Berhasil menyimpan {out_path}")

if __name__ == "__main__":
    os.makedirs(os.path.join(DATA_DIR, "csv"), exist_ok=True)
    convert_uci_har()
    convert_pamap2()
    convert_mhealth()
    convert_wisdm()
