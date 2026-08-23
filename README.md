# MS-KANConv: Multi-Scale KAN-augmented Convolution for Human Activity Recognition

This repository contains the implementation of the **MS-KANConv** architecture for Human Activity Recognition (HAR) using wearable inertial sensors. The proposed model improves upon standard Temporal Convolutional Networks (TCN) by introducing learnable B-spline activation functions (KAN-Act) and parallel multi-scale convolution branches.

## Project Structure
- `models/`: Contains the KAN modules and MS-KANConv model.
- `datasets/`: Data loading and sliding window segmentation for UCI-HAR, PAMAP2, and mHealth.
- `experiments/`: Scripts for benchmark comparison, ablation studies, efficiency analysis, and interpretability.
- `config.py`: Hyperparameters and dataset configurations.
- `train.py`: Training loop with early stopping.

## Installation

Create a virtual environment (e.g., using conda) and install the dependencies:

```bash
conda create -n har_env python=3.10
conda activate har_env

# Install PyTorch with CUDA support (adjust for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

## Running the Experiments

You can use the master script `run_all.py` to handle the entire pipeline:

1. **Download Datasets**
   ```bash
   python run_all.py --download
   ```

2. **Run All Experiments**
   ```bash
   python run_all.py
   ```

3. **Run Specific Experiments**
   ```bash
   # Run only Experiment 1 (Benchmark) and Experiment 2 (Ablation)
   python run_all.py --exp 1 2
   ```

## Smoke Testing
To verify that all components (model shape, gradient flow, modules) are working correctly before starting long experiments, run:
```bash
python smoke_test.py
```
