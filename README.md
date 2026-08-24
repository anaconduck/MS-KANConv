# MS-KANConv: Multi-Scale KAN-augmented Convolution for Human Activity Recognition

This repository contains the official PyTorch implementation of **MS-KANConv** for wearable sensor-based Human Activity Recognition (HAR). The proposed model replaces static activation functions (such as ReLU) with learnable B-spline Kolmogorov-Arnold Network (KAN) activations embedded directly inside a multi-scale dilated temporal convolution backbone.

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

### Step 3: Download Datasets Automatically

Download and extract the three benchmark datasets (UCI-HAR, PAMAP2, and mHealth) directly from the official UCI Machine Learning Repository:

```bash
python run_all.py --download
```
*This downloads the raw archives into the `data/` directory and unzips them automatically.*

---

### Step 4: Execute Experiments

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
