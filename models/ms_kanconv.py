"""
MS-KANConv: Multi-Scale KAN-augmented Convolution for HAR
==========================================================
The proposed model. Improves TCN by:
1. Replacing fixed activations (ReLU) with learnable KAN B-spline activations
2. Parallel multi-scale dilated convolution branches
3. KAN-based classification head for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_modules import KANActivation, KANLinear, SqueezeExcitation


class MSKANConvBranch(nn.Module):
    """
    Single branch of the multi-scale KAN-Conv block.
    DilatedConv1D -> KAN-Act -> DilatedConv1D -> KAN-Act
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2,
                 use_kan_act: bool = True, kan_grid_size: int = 5,
                 kan_spline_order: int = 3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # causal-ish padding

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Activation: KAN-Act (learnable) or ReLU (fixed)
        self.use_kan_act = use_kan_act
        if use_kan_act:
            self.act1 = KANActivation(out_channels, kan_grid_size,
                                       kan_spline_order)
            self.act2 = KANActivation(out_channels, kan_grid_size,
                                       kan_spline_order)
        else:
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T)
        Returns:
            (B, C_out, T)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)

        return out


class MSKANConvBlock(nn.Module):
    """
    Multi-Scale KAN-Conv Block: parallel branches with different
    kernel sizes and dilation rates, followed by SE attention.

    Structure:
        Input -> [Branch_1(d=1) || Branch_2(d=2) || Branch_3(d=4)]
              -> Concatenate -> SE-Attention -> LayerNorm -> + Residual
    """

    def __init__(self, in_channels: int, branch_out_channels: int,
                 kernel_sizes=(3, 5, 7), dilations=(1, 2, 4),
                 dropout: float = 0.2, se_reduction: int = 4,
                 use_kan_act: bool = True, use_se: bool = True,
                 kan_grid_size: int = 5, kan_spline_order: int = 3):
        super().__init__()

        self.num_branches = len(kernel_sizes)
        total_out = branch_out_channels * self.num_branches

        # Create parallel branches
        self.branches = nn.ModuleList([
            MSKANConvBranch(
                in_channels, branch_out_channels,
                kernel_size=ks, dilation=d, dropout=dropout,
                use_kan_act=use_kan_act,
                kan_grid_size=kan_grid_size,
                kan_spline_order=kan_spline_order,
            )
            for ks, d in zip(kernel_sizes, dilations)
        ])

        # SE Attention
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(total_out, se_reduction)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(total_out)

        # Residual connection (1x1 conv for channel dim matching)
        self.residual = nn.Conv1d(in_channels, total_out, 1) \
            if in_channels != total_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T)
        Returns:
            (B, total_out, T)
        """
        # Parallel multi-scale branches
        branch_outputs = [branch(x) for branch in self.branches]

        # Truncate to minimum temporal length (in case of padding differences)
        min_t = min(o.size(-1) for o in branch_outputs)
        branch_outputs = [o[:, :, :min_t] for o in branch_outputs]

        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)  # (B, total_out, T)

        # SE Attention
        if self.use_se:
            out = self.se(out)

        # Layer normalization (applied over channels)
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # (B, C, T)

        # Residual connection
        residual = self.residual(x)
        residual = residual[:, :, :min_t]  # match temporal dim
        out = out + residual

        return out


class MSKANConv(nn.Module):
    """
    MS-KANConv: Multi-Scale KAN-augmented Convolution Network for HAR.

    Architecture:
        Input (B, C, T)
        -> MS-KANConv Block 1 (multi-scale + KAN-Act + SE)
        -> MS-KANConv Block 2 (multi-scale + KAN-Act + SE)
        -> Global Average Pooling
        -> KAN Classification Head
        -> Output (B, num_classes)
    """

    def __init__(self, input_channels: int, num_classes: int,
                 branch_out_channels_1: int = 64,
                 branch_out_channels_2: int = 128,
                 kernel_sizes=(3, 5, 7), dilations=(1, 2, 4),
                 dropout: float = 0.2, se_reduction: int = 4,
                 kan_grid_size: int = 5, kan_spline_order: int = 3,
                 head_hidden_dim: int = 128,
                 use_kan_act: bool = True,
                 use_multiscale: bool = True,
                 use_kan_head: bool = True,
                 use_se: bool = True):
        super().__init__()

        self.use_kan_head = use_kan_head

        # Determine branch configuration
        if use_multiscale:
            ks = kernel_sizes
            ds = dilations
        else:
            # Single branch: only first kernel/dilation
            ks = (kernel_sizes[0],)
            ds = (dilations[0],)

        num_branches = len(ks)
        block1_out = branch_out_channels_1 * num_branches
        block2_out = branch_out_channels_2 * num_branches

        # Block 1
        self.block1 = MSKANConvBlock(
            in_channels=input_channels,
            branch_out_channels=branch_out_channels_1,
            kernel_sizes=ks, dilations=ds,
            dropout=dropout, se_reduction=se_reduction,
            use_kan_act=use_kan_act, use_se=use_se,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
        )

        # Block 2
        self.block2 = MSKANConvBlock(
            in_channels=block1_out,
            branch_out_channels=branch_out_channels_2,
            kernel_sizes=ks, dilations=ds,
            dropout=dropout, se_reduction=se_reduction,
            use_kan_act=use_kan_act, use_se=use_se,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
        )

        # Classification Head
        if use_kan_head:
            self.head = nn.Sequential(
                KANLinear(block2_out, head_hidden_dim,
                          kan_grid_size, kan_spline_order),
                nn.Dropout(dropout),
                KANLinear(head_hidden_dim, num_classes,
                          kan_grid_size, kan_spline_order),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(block2_out, head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — sensor data window
        Returns:
            (B, num_classes) — class logits
        """
        # Feature extraction
        out = self.block1(x)     # (B, block1_out, T')
        out = self.block2(out)   # (B, block2_out, T'')

        # Global Average Pooling
        out = out.mean(dim=-1)   # (B, block2_out)

        # Classification
        out = self.head(out)     # (B, num_classes)
        return out

    def get_kan_activations(self):
        """
        Extract all KAN activation modules for visualization.

        Returns:
            Dict mapping names to KANActivation modules.
        """
        activations = {}
        for block_name, block in [("block1", self.block1),
                                   ("block2", self.block2)]:
            for i, branch in enumerate(block.branches):
                for j, act_name in enumerate(["act1", "act2"]):
                    act = getattr(branch, act_name)
                    if isinstance(act, KANActivation):
                        key = f"{block_name}/branch{i}/{act_name}"
                        activations[key] = act
        return activations


def build_ms_kanconv(input_channels: int, num_classes: int,
                     variant: str = "full", **kwargs):
    """
    Factory function to build MS-KANConv variants for ablation study.

    Args:
        input_channels: Number of input sensor channels.
        num_classes: Number of activity classes.
        variant: Model variant name.
    """
    defaults = dict(
        branch_out_channels_1=64,
        branch_out_channels_2=128,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout=0.2,
        se_reduction=4,
        kan_grid_size=5,
        kan_spline_order=3,
        head_hidden_dim=128,
        use_kan_act=True,
        use_multiscale=True,
        use_kan_head=True,
        use_se=True,
    )
    defaults.update(kwargs)

    if variant == "ms_kanconv_full" or variant == "ms_kanconv":
        pass  # use all defaults
    elif variant == "ms_kanconv_no_multiscale":
        defaults["use_multiscale"] = False
    elif variant == "ms_kanconv_no_kanact":
        defaults["use_kan_act"] = False
    elif variant == "ms_kanconv_no_kanhead":
        defaults["use_kan_head"] = False
    elif variant == "ms_kanconv_no_se":
        defaults["use_se"] = False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return MSKANConv(input_channels, num_classes, **defaults)
