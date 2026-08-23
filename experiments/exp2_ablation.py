import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import TRAIN_CONFIG, ABLATION_VARIANTS
from models.ms_kanconv import build_ms_kanconv
from models.baselines import get_baseline_model
from train import (
    set_seed,
    get_device,
    train_model,
    save_results,
    print_results_table,
    count_parameters,
)
from datasets.uci_har import load_uci_har
from datasets.pamap2 import load_pamap2, get_pamap2_fold
from datasets.mhealth import load_mhealth, get_mhealth_fold


def run_ablation_single_dataset(dataset_name: str, device=None):
    if device is None:
        device = get_device()
    set_seed(TRAIN_CONFIG.seed)
    if dataset_name == "uci_har":
        train_ds, test_ds, cfg = load_uci_har()
        folds = [(train_ds, test_ds)]
    elif dataset_name == "pamap2":
        X, y, subj, cfg = load_pamap2()
        folds = [get_pamap2_fold(X, y, subj, 0, cfg.n_folds, TRAIN_CONFIG.seed)]
    elif dataset_name == "mhealth":
        X, y, subj, cfg = load_mhealth()
        folds = [get_mhealth_fold(X, y, subj, 0, cfg.n_folds, TRAIN_CONFIG.seed)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    results = []
    for variant in ABLATION_VARIANTS:
        print(f"\n{'='*60}")
        print(f"Ablation: {variant} | Dataset: {dataset_name}")
        print(f"{'='*60}")
        fold_results = []
        for fold_idx, (train_ds, test_ds) in enumerate(folds):
            set_seed(TRAIN_CONFIG.seed + fold_idx)
            if variant == "tcn_vanilla":
                model = get_baseline_model(
                    "tcn_vanilla",
                    input_channels=cfg.input_channels,
                    num_classes=cfg.num_classes,
                )
            else:
                model = build_ms_kanconv(
                    input_channels=cfg.input_channels,
                    num_classes=cfg.num_classes,
                    variant=variant,
                )
            print(f"  Params: {count_parameters(model):,}")
            result = train_model(
                model,
                train_ds,
                test_ds,
                model_name=variant,
                dataset_name=dataset_name,
                device=device,
                verbose=True,
            )
            fold_results.append(result)
        avg_result = {
            "model_name": variant,
            "dataset_name": dataset_name,
            "best_accuracy": sum(r["best_accuracy"] for r in fold_results)
            / len(fold_results),
            "best_f1_weighted": sum(r["best_f1_weighted"] for r in fold_results)
            / len(fold_results),
            "best_f1_macro": sum(r["best_f1_macro"] for r in fold_results)
            / len(fold_results),
            "training_time_sec": sum(r["training_time_sec"] for r in fold_results)
            / len(fold_results),
            "num_parameters": fold_results[0]["num_parameters"],
        }
        results.append(avg_result)
    return results


def run_ablation():
    device = get_device()
    all_results = {}
    for dataset_name in ["uci_har", "pamap2", "mhealth"]:
        print(f"\n{'#'*70}")
        print(f"# ABLATION: {dataset_name}")
        print(f"{'#'*70}")
        results = run_ablation_single_dataset(dataset_name, device)
        all_results[dataset_name] = results
        print_results_table(results, title=f"Ablation — {dataset_name}")
    save_results(all_results, "exp2_ablation_results.json")
    return all_results


if __name__ == "__main__":
    run_ablation()
