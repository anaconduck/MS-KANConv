"""
KAN Modules: KAN-Activation and KAN-Linear
===========================================
Implements the learnable B-spline activation function (KAN-Act) and
KAN-Linear layer used in MS-KANConv.

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BSplineBasis(nn.Module):
    """
    Computes B-spline basis functions using the Cox-de Boor recursion.
    Efficient implementation that avoids Python-level recursion.
    """

    def __init__(self, grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Number of basis functions
        self.num_basis = grid_size + spline_order

        # Create extended grid (knot vector)
        # Interior grid points
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values.

        Args:
            x: Input tensor of any shape (will be flattened then reshaped).

        Returns:
            Basis values of shape (*x.shape, num_basis).
        """
        x_shape = x.shape
        x = x.reshape(-1)  # flatten

        grid = self.grid  # (num_knots,)

        # Order 0: piecewise constant
        # bases[i] = 1 if grid[i] <= x < grid[i+1], else 0
        bases = ((x.unsqueeze(-1) >= grid[:-1].unsqueeze(0)) &
                 (x.unsqueeze(-1) < grid[1:].unsqueeze(0))).float()

        # Cox-de Boor recursion for orders 1..spline_order
        for k in range(1, self.spline_order + 1):
            left_num = x.unsqueeze(-1) - grid[:-(k + 1)].unsqueeze(0)
            left_den = grid[k:-1] - grid[:-(k + 1)]
            left_den = left_den.unsqueeze(0).clamp(min=1e-8)

            right_num = grid[k + 1:].unsqueeze(0) - x.unsqueeze(-1)
            right_den = grid[k + 1:] - grid[1:-k]
            right_den = right_den.unsqueeze(0).clamp(min=1e-8)

            bases = (left_num / left_den * bases[:, :-1] +
                     right_num / right_den * bases[:, 1:])

        return bases.reshape(*x_shape, -1)


class KANActivation(nn.Module):
    """
    Learnable KAN-based activation function.

    Replaces fixed activation functions (ReLU, GELU) with learnable
    B-spline activation per channel, enabling the network to learn
    optimal nonlinearities for each feature map.

    Architecture:
        f(x) = w_base * SiLU(x) + w_spline * Σ_i c_i * B_i(x)

    where B_i are B-spline basis functions and c_i are learned coefficients.
    """

    def __init__(self, num_channels: int, grid_size: int = 5,
                 spline_order: int = 3):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order

        # B-spline basis computer
        self.bspline = BSplineBasis(grid_size, spline_order)

        # Learnable spline coefficients: one set per channel
        # Shape: (num_channels, num_basis)
        self.spline_weight = nn.Parameter(
            torch.randn(num_channels, self.num_basis) * 0.1
        )

        # Scaling factors for base and spline components
        self.base_weight = nn.Parameter(torch.ones(num_channels) * 1.0)
        self.spline_scale = nn.Parameter(torch.ones(num_channels) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T) where C == num_channels.

        Returns:
            Activated tensor of same shape (B, C, T).
        """
        B, C, T = x.shape
        assert C == self.num_channels, (
            f"Expected {self.num_channels} channels, got {C}"
        )

        # Base activation: SiLU (smooth, proven baseline)
        base = F.silu(x) * self.base_weight.view(1, C, 1)

        # Spline activation: learned per-channel nonlinearity
        # Normalize x to [-1, 1] range for stable B-spline computation
        x_norm = torch.tanh(x)  # smooth normalization to [-1, 1]

        # Compute B-spline basis: (B, C, T) -> (B, C, T, num_basis)
        basis = self.bspline(x_norm)

        # Apply per-channel spline weights
        # spline_weight: (C, num_basis) -> (1, C, 1, num_basis)
        # basis: (B, C, T, num_basis)
        spline_out = (basis * self.spline_weight.unsqueeze(0).unsqueeze(2)).sum(-1)
        spline_out = spline_out * self.spline_scale.view(1, C, 1)

        return base + spline_out

    def get_spline_curves(self, x_range=(-2.0, 2.0), num_points=200):
        """
        Get the learned activation curves for visualization.

        Returns:
            x_vals: (num_points,) tensor
            y_vals: (num_channels, num_points) tensor — learned activations
        """
        x_vals = torch.linspace(x_range[0], x_range[1], num_points)
        x_vals = x_vals.to(self.spline_weight.device)

        # Shape: (1, 1, num_points)
        x_input = x_vals.unsqueeze(0).unsqueeze(0).expand(
            1, self.num_channels, num_points
        )

        with torch.no_grad():
            y_vals = self.forward(x_input).squeeze(0)  # (C, num_points)

        return x_vals.cpu(), y_vals.cpu()


class KANLinear(nn.Module):
    """
    KAN-Linear layer: replaces standard nn.Linear with KAN-style
    learnable edge functions.

    Each connection (input_i -> output_j) has its own learnable
    B-spline function instead of a fixed weight multiplication.
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order

        # B-spline basis
        self.bspline = BSplineBasis(grid_size, spline_order)

        # Spline weights: each (in, out) pair has num_basis coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_basis) *
            (1.0 / math.sqrt(in_features * self.num_basis))
        )

        # Base linear transformation (SiLU path)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) *
            (1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Scale factor
        self.spline_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)
        Returns:
            (B, out_features)
        """
        # Base path: standard linear with SiLU
        base_out = F.linear(F.silu(x), self.base_weight)

        # Spline path
        x_norm = torch.tanh(x)  # normalize to [-1, 1]
        basis = self.bspline(x_norm)  # (B, in_features, num_basis)

        # Compute spline output
        # basis: (B, in_features, num_basis)
        # spline_weight: (out_features, in_features, num_basis)
        # result: (B, out_features)
        spline_out = torch.einsum(
            "bin,oin->bo", basis, self.spline_weight
        )

        return base_out + self.spline_scale * spline_out + self.bias


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Re-weights channel features based on global context.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) — channel-reweighted
        """
        # Global Average Pooling
        w = x.mean(dim=-1)  # (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)
