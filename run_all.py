import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR


def download_datasets():
    print("\n" + "=" * 70)
    print("DATASET PREPARATION: DOWNLOADING & EXTRACTING")
    print("=" * 70)
    from datasets.download import download_all

    download_all(DATA_DIR)


def execute_benchmark():
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("MS-KANConv vs 6 baselines on UCI-HAR, PAMAP2, mHealth, WISDM")
    print("=" * 70)
    from experiments.benchmark import run_benchmark

    return run_benchmark()


def execute_ablation():
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("Testing each component's contribution")
    print("=" * 70)
    from experiments.ablation import run_ablation

    return run_ablation()


def execute_efficiency():
    print("\n" + "=" * 70)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("Parameters, FLOPs, inference latency")
    print("=" * 70)
    from experiments.efficiency import run_efficiency

    return run_efficiency()


def execute_interpret():
    print("\n" + "=" * 70)
    print("INTERPRETABILITY ANALYSIS")
    print("KAN activation visualization + channel attention")
    print("=" * 70)
    from experiments.interpret import run_interpretability

    return run_interpretability()


def main():
    parser = argparse.ArgumentParser(description="MS-KANConv HAR Experiments Pipeline")
    parser.add_argument(
        "--run",
        "--exp",
        nargs="*",
        default=None,
        help="Experiments to run (1/benchmark, 2/ablation, 3/efficiency, 4/interpret). Default: run all.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Only download datasets, don't run experiments.",
    )
    args = parser.parse_args()
    total_start = time.time()
    print("+" + "=" * 68 + "+")
    print(
        "|"
        + " MS-KANConv: Multi-Scale KAN-augmented Convolution for HAR ".center(68)
        + "|"
    )
    print("|" + " Complete Experimental Pipeline ".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print(f"\n  Results directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")

    download_datasets()
    if args.download:
        print("\nDatasets downloaded. Exiting (--download flag).")
        return

    exp_map = {
        "1": ("Benchmark Comparison", execute_benchmark),
        "benchmark": ("Benchmark Comparison", execute_benchmark),
        "2": ("Ablation Study", execute_ablation),
        "ablation": ("Ablation Study", execute_ablation),
        "3": ("Efficiency Analysis", execute_efficiency),
        "efficiency": ("Efficiency Analysis", execute_efficiency),
        "4": ("Interpretability", execute_interpret),
        "interpret": ("Interpretability", execute_interpret),
    }

    selected = args.run if args.run else ["benchmark", "ablation", "efficiency", "interpret"]
    
    # Filter unique tasks maintaining order
    tasks_to_run = []
    seen = set()
    for item in selected:
        key = str(item).lower()
        if key in exp_map:
            name, func = exp_map[key]
            if func not in seen:
                seen.add(func)
                tasks_to_run.append((name, func))
        else:
            print(f"\n⚠ Unknown option: {item}, skipping.")

    for idx, (task_name, task_func) in enumerate(tasks_to_run, start=1):
        print(f"\n[{idx}/{len(tasks_to_run)}] Starting: {task_name}")
        try:
            start_t = time.time()
            task_func()
            elapsed_t = time.time() - start_t
            print(f"\n  ✓ {task_name} completed in {elapsed_t/60:.1f} minutes")
        except Exception as e:
            print(f"\n  ✗ {task_name} failed: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"ALL PIPELINE TASKS COMPLETED! Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
