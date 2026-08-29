import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
from config import TRAIN_CONFIG, FIGURES_DIR, DATASET_CONFIGS
from models.ms_kanconv import MSKANConv
from models.kan_modules import KANActivation
from train import set_seed, get_device, train_model
from datasets.uci_har import load_uci_har
from datasets.pamap2 import load_pamap2, get_pamap2_fold
from datasets.mhealth import load_mhealth, get_mhealth_fold


def plot_kan_activations(model, save_path, dataset_name=""):
    activations = model.get_kan_activations()
    if not activations:
        print("No KAN activations found in model.")
        return
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Learned KAN B-Spline Activation Functions — {dataset_name}",
        fontsize=16,
        fontweight="bold",
    )
    x_ref = torch.linspace(-2, 2, 200)
    relu_ref = torch.relu(x_ref).numpy()
    silu_ref = torch.nn.functional.silu(x_ref).numpy()
    plot_idx = 0
    for name, kan_act in list(activations.items())[:8]:
        if plot_idx >= 8:
            break
        ax = axes[plot_idx // 4, plot_idx % 4]
        x_vals, y_vals = kan_act.get_spline_curves()
        num_ch_to_show = min(6, y_vals.shape[0])
        colors = plt.cm.viridis(np.linspace(0, 1, num_ch_to_show))
        for ch in range(num_ch_to_show):
            ax.plot(
                x_vals.numpy(),
                y_vals[ch].numpy(),
                color=colors[ch],
                alpha=0.7,
                linewidth=1.5,
                label=f"ch{ch}",
            )
        ax.plot(x_ref.numpy(), relu_ref, "k--", alpha=0.3, linewidth=1, label="ReLU")
        ax.plot(x_ref.numpy(), silu_ref, "r--", alpha=0.3, linewidth=1, label="SiLU")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)
        if plot_idx == 0:
            ax.legend(fontsize=7, loc="upper left")
        plot_idx += 1
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Activation plot saved to {save_path}")


def plot_channel_attention(
    model, test_dataset, device, save_path, dataset_name="", activity_labels=None
):
    model.eval()
    model.to(device)
    se_blocks = []
    for name, module in model.named_modules():
        if hasattr(module, "se") and hasattr(module.se, "fc1"):
            se_blocks.append((name, module.se))
    if not se_blocks:
        print("No SE blocks found.")
        return
    attention_weights = {}
    hooks = []

    def make_hook(block_name):
        def hook_fn(module, input, output):
            x = input[0]
            w = x.mean(dim=-1)
            w = torch.relu(module.fc1(w))
            w = torch.sigmoid(module.fc2(w))
            if block_name not in attention_weights:
                attention_weights[block_name] = []
            attention_weights[block_name].append(w.detach().cpu())

        return hook_fn

    for block_name, se in se_blocks:
        h = se.register_forward_hook(make_hook(block_name))
        hooks.append(h)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            _ = model(X_batch)
            all_labels.extend(y_batch.numpy())
    for h in hooks:
        h.remove()
    all_labels = np.array(all_labels)
    for block_name in attention_weights:
        weights = torch.cat(attention_weights[block_name], dim=0).numpy()
        unique_labels = sorted(np.unique(all_labels))
        class_weights = []
        class_names = []
        for label in unique_labels:
            mask = all_labels == label
            if mask.sum() > 0:
                class_weights.append(weights[mask].mean(axis=0))
                if activity_labels and label < len(activity_labels):
                    class_names.append(activity_labels[label])
                else:
                    class_names.append(f"Class {label}")
        class_weights = np.array(class_weights)
        if class_weights.shape[1] > 32:
            step = class_weights.shape[1] // 32
            class_weights = class_weights[:, ::step]
            ch_labels = [f"ch{i*step}" for i in range(class_weights.shape[1])]
        else:
            ch_labels = [f"ch{i}" for i in range(class_weights.shape[1])]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            class_weights,
            annot=False,
            cmap="YlOrRd",
            xticklabels=ch_labels,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"Channel Attention Weights — {dataset_name}\n({block_name})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Activity Class")
        block_save = block_name.replace(".", "_").replace("/", "_")
        fpath = save_path.replace(".png", f"_{block_save}.png")
        plt.tight_layout()
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Attention heatmap saved to {fpath}")


def run_interpretability():
    device = get_device()
    set_seed(TRAIN_CONFIG.seed)
    datasets = {
        "uci_har": {
            "load": lambda: (load_uci_har()[:2], load_uci_har()[2]),
        },
        "mhealth": {
            "load": lambda: (
                get_mhealth_fold(*load_mhealth()[:3], 0),
                load_mhealth()[3],
            ),
        },
    }
    for dataset_name in ["uci_har", "mhealth"]:
        print(f"\n{'='*60}")
        print(f"Interpretability: {dataset_name}")
        print(f"{'='*60}")
        cfg = DATASET_CONFIGS[dataset_name]
        if dataset_name == "uci_har":
            train_ds, test_ds, cfg = load_uci_har()
        else:
            X, y, subj, cfg = load_mhealth()
            train_ds, test_ds = get_mhealth_fold(
                X, y, subj, 0, cfg.n_folds, TRAIN_CONFIG.seed
            )
        model = MSKANConv(
            input_channels=cfg.input_channels,
            num_classes=cfg.num_classes,
        )
        print("  Training model for interpretability analysis...")
        train_model(
            model,
            train_ds,
            test_ds,
            model_name="ms_kanconv",
            dataset_name=dataset_name,
            device=device,
            verbose=True,
        )
        act_path = os.path.join(FIGURES_DIR, f"kan_activations_{dataset_name}.png")
        plot_kan_activations(model, act_path, dataset_name)
        attn_path = os.path.join(FIGURES_DIR, f"channel_attention_{dataset_name}.png")
        plot_channel_attention(
            model,
            test_ds,
            device,
            attn_path,
            dataset_name=dataset_name,
            activity_labels=cfg.activity_labels,
        )
    print(f"\n✓ All interpretability figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_interpretability()
