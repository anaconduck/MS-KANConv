# MS-KANConv: Multi-Scale KAN-augmented Convolution for Human Activity Recognition

This repository contains the official PyTorch implementation of **MS-KANConv** for wearable sensor-based Human Activity Recognition (HAR). The proposed model replaces static activation functions (such as ReLU) with learnable B-spline Kolmogorov-Arnold Network (KAN) activations embedded directly inside a multi-scale dilated temporal convolution backbone.

---

## 📁 Project Structure

```text
Adapative_HAR/
├── config.py                 # Central hyperparameters, paths, and dataset configs
├── train.py                  # Training pipeline with early stopping & metrics
├── run_all.py                # Master execution pipeline
├── smoke_test.py             # Integrity & gradient flow verification script
├── requirements.txt          # Python dependencies
├── models/
│   ├── kan_modules.py        # B-spline basis, KAN-Activation, KAN-Linear, SE-Attention
│   ├── ms_kanconv.py         # MS-KANConv architecture & ablation factory
│   └── baselines.py          # 6 Baselines (CNN-1D, DeepConvLSTM, TCN, TCN-Attn, Transformer, KAN-HAR)
├── datasets/
│   ├── download.py           # Automated dataset downloader (UCI ML repository)
│   ├── uci_har.py            # UCI-HAR loader (9 inertial channels)
│   ├── pamap2.py             # PAMAP2 loader (18 IMU channels, sliding window, 5-fold CV)
│   └── mhealth.py            # mHealth loader (21 IMU channels, sliding window, 5-fold CV)
├── experiments/
│   ├── benchmark.py          # Exp 1: Benchmark comparison against 6 baselines
│   ├── ablation.py           # Exp 2: Ablation study on individual components
│   ├── efficiency.py         # Exp 3: Parameters, FLOPs, GPU/CPU latency
│   └── interpret.py          # Exp 4: KAN activation curves & channel attention maps
└── notebooks/
    └── eda_and_preprocessing.ipynb # Interactive EDA, signal waveforms, and data pipeline
```

---

## 🚀 Step-by-Step Execution Guide (Server / GPU Setup)

Follow these detailed steps to set up the environment and run the complete experimental pipeline on your machine or server.

### Step 1: Setup Environment (Miniconda)

Open your terminal or Anaconda Prompt, then create and activate a clean Python 3.10 environment:

```bash
# 1. Create a dedicated conda environment
conda create -n har_env python=3.10 -y

# 2. Activate the environment
conda activate har_env
```

---

### Step 2: Install PyTorch with CUDA & Dependencies

Install PyTorch with CUDA acceleration (for NVIDIA GPUs such as RTX 5070 / 4090 / 3090) followed by the required packages:

```bash
# 1. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install all required dependencies
pip install -r requirements.txt
```

---

### Step 3: Verify GPU & Run Smoke Test

Before running long training jobs, verify that PyTorch detects your GPU and all model components pass shape and gradient checks:

```bash
# Check GPU recognition
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Run component & gradient flow smoke test
python smoke_test.py
```
*(Ensure that `✓ ALL TESTS PASSED!` is displayed).*

---

### Step 4: Download Datasets Automatically

Download and extract the three benchmark datasets (UCI-HAR, PAMAP2, and mHealth) directly from the official UCI Machine Learning Repository:

```bash
python run_all.py --download
```
*This downloads the raw archives into the `data/` directory and unzips them automatically.*

---

### Step 5: Execute Experiments

You can either run the entire experimental pipeline automatically or run specific stages individually.

#### Option A: Run Full Pipeline (All 4 Experiments)
```bash
python run_all.py
```
*This will sequentially execute Benchmark $\to$ Ablation $\to$ Efficiency $\to$ Interpretability across all datasets.*

#### Option B: Run Specific Experiments
```bash
# 1. Run Benchmark Comparison (MS-KANConv vs 6 Baselines across 3 Datasets)
python run_all.py --run benchmark

# 2. Run Ablation Study (Testing KAN-Act, Multi-Scale, SE-Attention, KAN-Head)
python run_all.py --run ablation

# 3. Run Computational Efficiency Analysis (Params, FLOPs, Latency)
python run_all.py --run efficiency

# 4. Generate Interpretability Figures (B-Spline curves & Attention heatmaps)
python run_all.py --run interpret
```

---

### Step 6: Inspect Results & Figures

After completion, all evaluation metrics and publication-ready figures are stored in the `results/` folder:

* **Quantitative Tables (JSON):**
  * `results/benchmark_results.json` — Accuracy, Weighted F1, Macro F1 across models.
  * `results/ablation_results.json` — Component contribution results.
  * `results/efficiency_results.json` — Parameter counts, FLOPs, and GPU/CPU inference latency.
* **Publication Figures (PNG, 300 DPI):**
  * `results/figures/architecture_diagram.png` — Architectural schematic.
  * `results/figures/kan_activations_*.png` — Learned B-spline non-linear activation curves per channel.
  * `results/figures/channel_attention_*.png` — Sensor channel attention weights per activity class.
