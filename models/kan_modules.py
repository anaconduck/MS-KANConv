import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BSplineBasis(nn.Module):
    def __init__(
        self, grid_size: int = 5, spline_order: int = 3, grid_range: tuple = (-1.0, 1.0)
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.num_basis = grid_size + spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x_expanded = x.unsqueeze(-1)
        bases = (
            (x_expanded >= grid[:-1]) & (x_expanded < grid[1:])
        ).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_num = x_expanded - grid[: -(k + 1)]
            left_den = (grid[k:-1] - grid[: -(k + 1)]).clamp(min=1e-8)
            right_num = grid[k + 1 :] - x_expanded
            right_den = (grid[k + 1 :] - grid[1:-k]).clamp(min=1e-8)
            bases = (
                (left_num / left_den) * bases[..., :-1]
                + (right_num / right_den) * bases[..., 1:]
            )
        return bases


class KANActivation(nn.Module):
    def __init__(self, num_channels: int, grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order
        self.bspline = BSplineBasis(grid_size, spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(num_channels, self.num_basis) * 0.1
        )
        self.base_weight = nn.Parameter(torch.ones(num_channels) * 1.0)
        self.spline_scale = nn.Parameter(torch.ones(num_channels) * 0.2)
        self.input_scale = nn.Parameter(torch.ones(num_channels) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"
        base = F.silu(x) * self.base_weight.view(1, C, 1)
        x_norm = torch.tanh(x * self.input_scale.view(1, C, 1))
        basis = self.bspline(x_norm)
        spline_out = (basis * self.spline_weight.unsqueeze(0).unsqueeze(2)).sum(-1)
        spline_out = spline_out * self.spline_scale.view(1, C, 1)
        return base + spline_out

    def get_spline_curves(self, x_range=(-2.0, 2.0), num_points=200):
        x_vals = torch.linspace(x_range[0], x_range[1], num_points)
        x_vals = x_vals.to(self.spline_weight.device)
        x_input = (
            x_vals.unsqueeze(0).unsqueeze(0).expand(1, self.num_channels, num_points)
        )
        with torch.no_grad():
            y_vals = self.forward(x_input).squeeze(0)
        return x_vals.cpu(), y_vals.cpu()


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order
        self.bspline = BSplineBasis(grid_size, spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_basis)
            * (1.0 / math.sqrt(in_features * self.num_basis))
        )
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.spline_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(F.silu(x), self.base_weight)
        x_norm = torch.tanh(x)
        basis = self.bspline(x_norm)
        spline_out = torch.einsum("bin,oin->bo", basis, self.spline_weight)
        return base_out + self.spline_scale * spline_out + self.bias


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=-1)
        max_out = x.amax(dim=-1)
        avg_out = self.fc2(F.relu(self.fc1(avg_out)))
        max_out = self.fc2(F.relu(self.fc1(max_out)))
        return torch.sigmoid(avg_out + max_out).unsqueeze(-1)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(out))


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
