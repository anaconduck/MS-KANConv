import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
from tabulate import tabulate
from config import TRAIN_CONFIG, BASELINE_MODELS, DATASET_CONFIGS
from models.ms_kanconv import MSKANConv
from models.baselines import get_baseline_model
from train import set_seed, get_device, count_parameters, save_results
def measure_inference_time(model, input_shape, device, num_runs=100,
                           warmup=10):
    model.eval()
    model.to(device)
    dummy = torch.randn(1, *input_shape).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  
    return np.mean(times), np.std(times)
def estimate_flops(model, input_shape, device):
    try:
        from thop import profile
        dummy = torch.randn(1, *input_shape).to(device)
        model = model.to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except Exception as e:
        print(f"  FLOPs estimation failed: {e}")
        return None
def run_efficiency():
    device = get_device()
    set_seed(TRAIN_CONFIG.seed)
    all_models = BASELINE_MODELS + ["ms_kanconv"]
    datasets_to_test = ["uci_har", "pamap2", "mhealth"]
    all_results = {}
    for dataset_name in datasets_to_test:
        cfg = DATASET_CONFIGS[dataset_name]
        input_shape = (cfg.input_channels, cfg.window_size)
        print(f"\n{'='*70}")
        print(f"Efficiency Analysis — {dataset_name}")
        print(f"Input shape: {input_shape}")
        print(f"{'='*70}")
        results = []
        for model_name in all_models:
            if model_name == "ms_kanconv":
                model = MSKANConv(
                    input_channels=cfg.input_channels,
                    num_classes=cfg.num_classes,
                )
            else:
                model = get_baseline_model(
                    model_name,
                    input_channels=cfg.input_channels,
                    num_classes=cfg.num_classes,
                )
            params = count_parameters(model)
            flops = estimate_flops(model, input_shape, device)
            mean_time_gpu, std_time_gpu = measure_inference_time(
                model, input_shape, device
            )
            mean_time_cpu, std_time_cpu = measure_inference_time(
                model, input_shape, torch.device("cpu")
            )
            result = {
                "model_name": model_name,
                "num_parameters": params,
                "flops": flops,
                "inference_gpu_ms": mean_time_gpu,
                "inference_gpu_std": std_time_gpu,
                "inference_cpu_ms": mean_time_cpu,
                "inference_cpu_std": std_time_cpu,
            }
            results.append(result)
            flops_str = f"{flops/1e6:.2f}M" if flops else "N/A"
            print(f"  {model_name:25s} | "
                  f"Params: {params:>10,} | "
                  f"FLOPs: {flops_str:>10s} | "
                  f"GPU: {mean_time_gpu:.2f}±{std_time_gpu:.2f}ms | "
                  f"CPU: {mean_time_cpu:.2f}±{std_time_cpu:.2f}ms")
        all_results[dataset_name] = results
        headers = ["Model", "Parameters", "FLOPs", "GPU (ms)", "CPU (ms)"]
        rows = []
        for r in results:
            flops_str = f"{r['flops']/1e6:.2f}M" if r["flops"] else "N/A"
            rows.append([
                r["model_name"],
                f"{r['num_parameters']:,}",
                flops_str,
                f"{r['inference_gpu_ms']:.2f}±{r['inference_gpu_std']:.2f}",
                f"{r['inference_cpu_ms']:.2f}±{r['inference_cpu_std']:.2f}",
            ])
        print(f"\n{tabulate(rows, headers=headers, tablefmt='grid')}")
    save_results(all_results, "exp3_efficiency_results.json")
    return all_results
if __name__ == "__main__":
    run_efficiency()
