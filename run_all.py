import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR


def download_datasets():
    print("\n" + "=" * 70)
    print("STEP 0: DOWNLOADING DATASETS")
    print("=" * 70)
    from datasets.download import download_all

    download_all(DATA_DIR)


def run_exp1():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BENCHMARK COMPARISON")
    print("MS-KANConv vs 6 baselines on UCI-HAR, PAMAP2, mHealth")
    print("=" * 70)
    from experiments.exp1_benchmark import run_benchmark

    return run_benchmark()


def run_exp2():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ABLATION STUDY")
    print("Testing each component's contribution")
    print("=" * 70)
    from experiments.exp2_ablation import run_ablation

    return run_ablation()


def run_exp3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: EFFICIENCY ANALYSIS")
    print("Parameters, FLOPs, inference latency")
    print("=" * 70)
    from experiments.exp3_efficiency import run_efficiency

    return run_efficiency()


def run_exp4():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: INTERPRETABILITY ANALYSIS")
    print("KAN activation visualization + channel attention")
    print("=" * 70)
    from experiments.exp4_interpret import run_interpretability

    return run_interpretability()


def main():
    parser = argparse.ArgumentParser(description="MS-KANConv HAR Experiments")
    parser.add_argument(
        "--exp",
        nargs="*",
        type=int,
        default=None,
        help="Experiment numbers to run (1-4). Default: run all.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Only download datasets, don't run experiments.",
    )
    args = parser.parse_args()
    total_start = time.time()
    print("╔" + "═" * 68 + "╗")
    print(
        "║"
        + " MS-KANConv: Multi-Scale KAN-augmented Convolution for HAR ".center(68)
        + "║"
    )
    print("║" + " Complete Experimental Pipeline ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Results directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")
    download_datasets()
    if args.download:
        print("\nDatasets downloaded. Exiting (--download flag).")
        return
    experiments = args.exp if args.exp else [1, 2, 3, 4]
    exp_functions = {
        1: ("Benchmark Comparison", run_exp1),
        2: ("Ablation Study", run_exp2),
        3: ("Efficiency Analysis", run_exp3),
        4: ("Interpretability", run_exp4),
    }
    for exp_num in experiments:
        if exp_num not in exp_functions:
            print(f"\n⚠ Unknown experiment number: {exp_num}, skipping.")
            continue
        exp_name, exp_func = exp_functions[exp_num]
        print(f"\n{'▶':>3} Starting Experiment {exp_num}: {exp_name}")
        try:
            exp_start = time.time()
            exp_func()
            exp_elapsed = time.time() - exp_start
            print(
                f"\n  ✓ Experiment {exp_num} completed in "
                f"{exp_elapsed/60:.1f} minutes"
            )
        except Exception as e:
            print(f"\n  ✗ Experiment {exp_num} failed: {e}")
            import traceback

            traceback.print_exc()
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"ALL DONE! Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
