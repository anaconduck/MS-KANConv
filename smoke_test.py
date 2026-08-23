"""
Smoke Test: Verify all components work correctly.
Run this before starting experiments.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def test_kan_modules():
    """Test KAN activation and KAN linear."""
    print("Testing KAN modules...")
    from models.kan_modules import KANActivation, KANLinear, SqueezeExcitation

    # KAN Activation
    kan_act = KANActivation(num_channels=64, grid_size=5, spline_order=3)
    x = torch.randn(2, 64, 128)
    out = kan_act(x)
    assert out.shape == x.shape, f"KAN-Act shape mismatch: {out.shape}"
    print(f"  ✓ KAN-Activation: {x.shape} -> {out.shape}")

    # Spline curve extraction
    x_vals, y_vals = kan_act.get_spline_curves()
    assert y_vals.shape == (64, 200), f"Spline curve shape: {y_vals.shape}"
    print(f"  ✓ Spline curves: {y_vals.shape}")

    # KAN Linear
    kan_lin = KANLinear(256, 64, grid_size=5, spline_order=3)
    x = torch.randn(2, 256)
    out = kan_lin(x)
    assert out.shape == (2, 64), f"KAN-Linear shape mismatch: {out.shape}"
    print(f"  ✓ KAN-Linear: (2, 256) -> {out.shape}")

    # SE block
    se = SqueezeExcitation(128, reduction=4)
    x = torch.randn(2, 128, 64)
    out = se(x)
    assert out.shape == x.shape, f"SE shape mismatch: {out.shape}"
    print(f"  ✓ SE-Attention: {x.shape} -> {out.shape}")


def test_ms_kanconv():
    """Test the proposed MS-KANConv model."""
    print("\nTesting MS-KANConv model...")
    from models.ms_kanconv import MSKANConv, build_ms_kanconv

    configs = [
        ("UCI-HAR", 9, 6),
        ("PAMAP2", 18, 12),
        ("mHealth", 21, 12),
    ]

    for name, in_ch, n_cls in configs:
        model = MSKANConv(input_channels=in_ch, num_classes=n_cls)
        x = torch.randn(2, in_ch, 128)
        out = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == (2, n_cls), f"Shape mismatch for {name}"
        print(f"  ✓ {name}: ({in_ch}, 128) -> {out.shape}, "
              f"params={params:,}")

    # Test ablation variants
    print("\n  Testing ablation variants...")
    variants = ["ms_kanconv_full", "ms_kanconv_no_multiscale",
                "ms_kanconv_no_kanact", "ms_kanconv_no_kanhead",
                "ms_kanconv_no_se"]
    for v in variants:
        model = build_ms_kanconv(9, 6, variant=v)
        x = torch.randn(2, 9, 128)
        out = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == (2, 6)
        print(f"  ✓ {v}: params={params:,}")


def test_baselines():
    """Test all baseline models."""
    print("\nTesting baseline models...")
    from models.baselines import get_baseline_model

    baselines = ["cnn_1d", "deep_conv_lstm", "tcn_vanilla",
                 "tcn_attention", "transformer_har", "kan_har"]

    for name in baselines:
        model = get_baseline_model(name, input_channels=9, num_classes=6)
        x = torch.randn(2, 9, 128)
        out = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == (2, 6), f"{name} shape mismatch: {out.shape}"
        print(f"  ✓ {name:25s}: params={params:,}")


def test_gradient_flow():
    """Verify gradients flow through KAN components."""
    print("\nTesting gradient flow...")
    from models.ms_kanconv import MSKANConv

    model = MSKANConv(input_channels=9, num_classes=6)
    x = torch.randn(4, 9, 128)
    y = torch.randint(0, 6, (4,))

    criterion = torch.nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    # Check that KAN parameters have gradients
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad += 1
            else:
                no_grad += 1

    print(f"  ✓ Gradient flow: {has_grad} params with grad, "
          f"{no_grad} without grad (expected 0)")
    assert no_grad == 0 or no_grad < has_grad * 0.1, \
        f"Too many params without gradient: {no_grad}"


def main():
    print("=" * 60)
    print("MS-KANConv Smoke Test")
    print("=" * 60)

    test_kan_modules()
    test_ms_kanconv()
    test_baselines()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Download datasets: python run_all.py --download")
    print("  2. Run all experiments: python run_all.py")
    print("  3. Run specific experiment: python run_all.py --exp 1")


if __name__ == "__main__":
    main()
