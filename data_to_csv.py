import os
import pandas as pd
import numpy as np

from datasets.uci_har import load_uci_har
from datasets.pamap2 import load_pamap2
from datasets.mhealth import load_mhealth
from config import DATA_DIR

def flatten_to_csv(X, y, output_path, dataset_name):
    """
    Mengubah tensor 3D (N, C, T) menjadi format tabel 2D untuk CSV.
    Format yang digunakan adalah long-format: 
    Setiap baris adalah satu time-step, dikelompokkan berdasarkan window_id.
    """
    N, C, T = X.shape
    
    # Buat array window_id (untuk menandai potongan sinyal mana)
    window_ids = np.repeat(np.arange(N), T)
    
    # Buat array label
    labels = np.repeat(y, T)
    
    # Transpose X dari (N, C, T) menjadi (N, T, C)
    # Kemudian reshape (ratakan) menjadi (N * T, C)
    X_reshaped = X.transpose(0, 2, 1).reshape(N * T, C)
    
    # Tentukan nama kolom
    cols = ['window_id', 'label'] + [f'channel_{i+1}' for i in range(C)]
    
    # Gabungkan semua kolom
    data = np.column_stack((window_ids, labels, X_reshaped))
    
    df = pd.DataFrame(data, columns=cols)
    
    # Pastikan tipe data id dan label adalah integer
    df['window_id'] = df['window_id'].astype(int)
    df['label'] = df['label'].astype(int)
    
    print(f"Menyimpan {dataset_name} ke {output_path} (Ukuran: {df.shape})...")
    df.to_csv(output_path, index=False)
    print(f"Berhasil menyimpan {dataset_name}.")

def main():
    # Buat folder csv di dalam folder data
    csv_dir = os.path.join(DATA_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    # 1. UCI HAR
    try:
        print("\n=== Memproses UCI HAR ===")
        train_ds, test_ds, _ = load_uci_har()
        X_train, y_train = train_ds.tensors[0].numpy(), train_ds.tensors[1].numpy()
        X_test, y_test = test_ds.tensors[0].numpy(), test_ds.tensors[1].numpy()
        
        # Gabungkan data train dan test untuk export
        X_uci = np.concatenate([X_train, X_test], axis=0)
        y_uci = np.concatenate([y_train, y_test], axis=0)
        
        flatten_to_csv(X_uci, y_uci, os.path.join(csv_dir, "uci_har.csv"), "UCI-HAR")
    except Exception as e:
        print(f"Gagal memproses UCI-HAR: {e}")

    # 2. PAMAP2
    try:
        print("\n=== Memproses PAMAP2 ===")
        X_pamap, y_pamap, _, _ = load_pamap2()
        flatten_to_csv(X_pamap, y_pamap, os.path.join(csv_dir, "pamap2.csv"), "PAMAP2")
    except Exception as e:
        print(f"Gagal memproses PAMAP2: {e}")
        
    # 3. mHealth
    try:
        print("\n=== Memproses mHealth ===")
        X_mhealth, y_mhealth, _, _ = load_mhealth()
        flatten_to_csv(X_mhealth, y_mhealth, os.path.join(csv_dir, "mhealth.csv"), "mHealth")
    except Exception as e:
        print(f"Gagal memproses mHealth: {e}")

if __name__ == "__main__":
    main()
