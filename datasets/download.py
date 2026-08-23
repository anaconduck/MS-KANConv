"""
Dataset Download Script
=======================
Downloads UCI-HAR, PAMAP2, and mHealth datasets from UCI ML Repository.
"""

import os
import zipfile
import requests
from tqdm import tqdm

DATASETS = {
    "uci_har": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
        "extract_dir": "UCI HAR Dataset",
    },
    "pamap2": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        "extract_dir": "PAMAP2_Dataset",
    },
    "mhealth": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
        "extract_dir": "MHEALTHDATASET",
    },
}


def download_file(url: str, dest_path: str):
    """Download a file with progress bar."""
    print(f"  Downloading from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="  Progress"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_dataset(name: str, data_dir: str):
    """
    Download and extract a single dataset.

    Args:
        name: Dataset name ("uci_har", "pamap2", "mhealth").
        data_dir: Root data directory.
    """
    info = DATASETS[name]
    dataset_dir = os.path.join(data_dir, name)
    zip_path = os.path.join(data_dir, f"{name}.zip")

    # Check if already downloaded
    if os.path.isdir(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        print(f"[{name}] Already exists at {dataset_dir}, skipping.")
        return

    os.makedirs(dataset_dir, exist_ok=True)

    # Download
    print(f"\n[{name}] Downloading...")
    try:
        download_file(info["url"], zip_path)
    except Exception as e:
        print(f"  ⚠ Auto-download failed: {e}")
        print(f"  Please manually download from: {info['url']}")
        print(f"  Extract to: {dataset_dir}")
        return

    # Extract
    print(f"[{name}] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    print(f"[{name}] Done! Extracted to {dataset_dir}")

    # Clean up zip
    os.remove(zip_path)


def download_all(data_dir: str):
    """Download all three datasets."""
    print("=" * 60)
    print("Downloading HAR Datasets")
    print("=" * 60)

    for name in DATASETS:
        download_dataset(name, data_dir)

    print("\n✓ All datasets ready.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_DIR
    download_all(DATA_DIR)
