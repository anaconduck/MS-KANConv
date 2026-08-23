"""
MS-KANConv Configuration
========================
All hyperparameters and paths for the MS-KANConv HAR project.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Dataset Configurations
# ============================================================
@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    input_channels: int
    sampling_rate: int  # Hz
    window_size: int = 128  # samples
    overlap: float = 0.5
    # For cross-validation
    n_folds: int = 5
    # Activity labels
    activity_labels: List[str] = field(default_factory=list)


UCI_HAR_CONFIG = DatasetConfig(
    name="UCI-HAR",
    num_classes=6,
    input_channels=9,  # body_acc(3) + body_gyro(3) + total_acc(3)
    sampling_rate=50,
    window_size=128,
    overlap=0.5,
    n_folds=1,  # predefined train/test split
    activity_labels=[
        "Walking", "Walking Upstairs", "Walking Downstairs",
        "Sitting", "Standing", "Laying"
    ],
)

PAMAP2_CONFIG = DatasetConfig(
    name="PAMAP2",
    num_classes=12,
    input_channels=18,  # 3 IMUs × (acc_xyz + gyro_xyz) = 3 × 6
    sampling_rate=100,
    window_size=128,
    overlap=0.5,
    n_folds=5,
    activity_labels=[
        "Lying", "Sitting", "Standing", "Walking", "Running",
        "Cycling", "Nordic Walking", "Ascending Stairs",
        "Descending Stairs", "Vacuum Cleaning", "Ironing",
        "Rope Jumping"
    ],
)

MHEALTH_CONFIG = DatasetConfig(
    name="mHealth",
    num_classes=12,
    input_channels=21,  # chest_acc(3) + ankle(acc3+gyro3+mag3) + wrist(acc3+gyro3+mag3)
    sampling_rate=50,
    window_size=128,
    overlap=0.5,
    n_folds=5,
    activity_labels=[
        "Standing Still", "Sitting", "Lying Down", "Walking",
        "Climbing Stairs", "Waist Bends Forward", "Frontal Elevation of Arms",
        "Knees Bending", "Cycling", "Jogging", "Running",
        "Jump Front & Back"
    ],
)

DATASET_CONFIGS = {
    "uci_har": UCI_HAR_CONFIG,
    "pamap2": PAMAP2_CONFIG,
    "mhealth": MHEALTH_CONFIG,
}


# ============================================================
# Model Configurations
# ============================================================
@dataclass
class MSKANConvConfig:
    """Configuration for the proposed MS-KANConv model."""
    # Multi-scale branch parameters
    branch_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    branch_dilations: Tuple[int, ...] = (1, 2, 4)
    branch_out_channels: int = 64  # per branch

    # Block configurations
    block1_out_channels: int = 192   # 64 * 3 branches
    block2_out_channels: int = 384   # 128 * 3 branches
    block2_branch_out: int = 128

    # KAN-Activation parameters
    kan_grid_size: int = 5
    kan_spline_order: int = 3

    # SE Attention
    se_reduction: int = 4

    # KAN Classification Head
    head_hidden_dim: int = 128

    # Training
    dropout: float = 0.2

    # Regularization
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    patience: int = 15  # early stopping
    scheduler: str = "cosine"  # "cosine" or "step"
    step_size: int = 30
    step_gamma: float = 0.5
    seed: int = 42
    num_workers: int = 2


MODEL_CONFIG = MSKANConvConfig()
TRAIN_CONFIG = TrainingConfig()


# ============================================================
# Baseline Model Names
# ============================================================
BASELINE_MODELS = [
    "cnn_1d",
    "deep_conv_lstm",
    "tcn_vanilla",
    "tcn_attention",
    "transformer_har",
    "kan_har",
]

# All models including proposed
ALL_MODELS = BASELINE_MODELS + ["ms_kanconv"]

# ============================================================
# Ablation Variants
# ============================================================
ABLATION_VARIANTS = [
    "ms_kanconv_full",           # Full model
    "ms_kanconv_no_multiscale",  # Single branch (d=1, k=3)
    "ms_kanconv_no_kanact",      # ReLU instead of KAN-Act
    "ms_kanconv_no_kanhead",     # MLP instead of KAN head
    "ms_kanconv_no_se",          # Without SE attention
    "tcn_vanilla",               # Vanilla TCN (all off)
]
