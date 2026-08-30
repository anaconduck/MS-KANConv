import os
from dataclasses import dataclass, field
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    input_channels: int
    sampling_rate: int
    window_size: int = 128
    overlap: float = 0.5
    n_folds: int = 5
    activity_labels: List[str] = field(default_factory=list)


UCI_HAR_CONFIG = DatasetConfig(
    name="UCI-HAR",
    num_classes=6,
    input_channels=9,
    sampling_rate=50,
    window_size=128,
    overlap=0.5,
    n_folds=1,
    activity_labels=[
        "Walking",
        "Walking Upstairs",
        "Walking Downstairs",
        "Sitting",
        "Standing",
        "Laying",
    ],
)
PAMAP2_CONFIG = DatasetConfig(
    name="PAMAP2",
    num_classes=12,
    input_channels=18,
    sampling_rate=100,
    window_size=128,
    overlap=0.5,
    n_folds=5,
    activity_labels=[
        "Lying",
        "Sitting",
        "Standing",
        "Walking",
        "Running",
        "Cycling",
        "Nordic Walking",
        "Ascending Stairs",
        "Descending Stairs",
        "Vacuum Cleaning",
        "Ironing",
        "Rope Jumping",
    ],
)
MHEALTH_CONFIG = DatasetConfig(
    name="mHealth",
    num_classes=12,
    input_channels=21,
    sampling_rate=50,
    window_size=128,
    overlap=0.5,
    n_folds=5,
    activity_labels=[
        "Standing Still",
        "Sitting",
        "Lying Down",
        "Walking",
        "Climbing Stairs",
        "Waist Bends Forward",
        "Frontal Elevation of Arms",
        "Knees Bending",
        "Cycling",
        "Jogging",
        "Running",
        "Jump Front & Back",
    ],
)
WISDM_CONFIG = DatasetConfig(
    name="WISDM",
    num_classes=6,
    input_channels=3,
    sampling_rate=20,
    window_size=128,
    overlap=0.5,
    n_folds=5,
    activity_labels=[
        "Walking",
        "Jogging",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing",
    ],
)
DATASET_CONFIGS = {
    "uci_har": UCI_HAR_CONFIG,
    "pamap2": PAMAP2_CONFIG,
    "mhealth": MHEALTH_CONFIG,
    "wisdm": WISDM_CONFIG,
}


@dataclass
class MSKANConvConfig:
    branch_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    branch_dilations: Tuple[int, ...] = (1, 2, 4)
    branch_out_channels: int = 64
    block1_out_channels: int = 192
    block2_out_channels: int = 384
    block2_branch_out: int = 128
    kan_grid_size: int = 8
    kan_spline_order: int = 3
    se_reduction: int = 4
    head_hidden_dim: int = 256
    dropout: float = 0.2
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 150
    patience: int = 25
    scheduler: str = "cosine"
    step_size: int = 30
    step_gamma: float = 0.5
    seed: int = 42
    num_workers: int = 2


MODEL_CONFIG = MSKANConvConfig()
TRAIN_CONFIG = TrainingConfig()
BASELINE_MODELS = [
    "cnn_1d",
    "deep_conv_lstm",
    "tcn_vanilla",
    "tcn_attention",
    "transformer_har",
    "kan_har",
]
ALL_MODELS = BASELINE_MODELS + ["ms_kanconv"]
ABLATION_VARIANTS = [
    "ms_kanconv_full",
    "ms_kanconv_no_multiscale",
    "ms_kanconv_no_kanact",
    "ms_kanconv_no_kanhead",
    "ms_kanconv_no_se",
    "tcn_vanilla",
]
