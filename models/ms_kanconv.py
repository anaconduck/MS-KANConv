import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan_modules import KANActivation, KANLinear, SqueezeExcitation


class MSKANConvBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_kan_act: bool = True,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.use_kan_act = use_kan_act
        if use_kan_act:
            self.act1 = KANActivation(out_channels, kan_grid_size, kan_spline_order)
            self.act2 = KANActivation(out_channels, kan_grid_size, kan_spline_order)
        else:
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self,
        in_channels: int,
        branch_out_channels: int,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout: float = 0.2,
        se_reduction: int = 4,
        use_kan_act: bool = True,
        use_se: bool = True,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
    ):
        super().__init__()
        self.num_branches = len(kernel_sizes)
        total_out = branch_out_channels * self.num_branches
        self.branches = nn.ModuleList(
            [
                MSKANConvBranch(
                    in_channels,
                    branch_out_channels,
                    kernel_size=ks,
                    dilation=d,
                    dropout=dropout,
                    use_kan_act=use_kan_act,
                    kan_grid_size=kan_grid_size,
                    kan_spline_order=kan_spline_order,
                )
                for ks, d in zip(kernel_sizes, dilations)
            ]
        )
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(total_out, se_reduction)
        self.layer_norm = nn.LayerNorm(total_out)
        self.residual = (
            nn.Conv1d(in_channels, total_out, 1)
            if in_channels != total_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        min_t = min(o.size(-1) for o in branch_outputs)
        branch_outputs = [o[:, :, :min_t] for o in branch_outputs]
        out = torch.cat(branch_outputs, dim=1)
        if self.use_se:
            out = self.se(out)
        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)
        residual = self.residual(x)
        residual = residual[:, :, :min_t]
        out = out + residual
        return out


class MSKANConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        branch_out_channels_1: int = 32,
        branch_out_channels_2: int = 64,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout: float = 0.2,
        se_reduction: int = 4,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        head_hidden_dim: int = 64,
        use_kan_act: bool = True,
        use_multiscale: bool = True,
        use_kan_head: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        self.use_kan_head = use_kan_head
        if use_multiscale:
            ks = kernel_sizes
            ds = dilations
        else:
            ks = (kernel_sizes[0],)
            ds = (dilations[0],)
        num_branches = len(ks)
        block1_out = branch_out_channels_1 * num_branches
        block2_out = branch_out_channels_2 * num_branches
        self.block1 = MSKANConvBlock(
            in_channels=input_channels,
            branch_out_channels=branch_out_channels_1,
            kernel_sizes=ks,
            dilations=ds,
            dropout=dropout,
            se_reduction=se_reduction,
            use_kan_act=use_kan_act,
            use_se=use_se,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
        )
        self.block2 = MSKANConvBlock(
            in_channels=block1_out,
            branch_out_channels=branch_out_channels_2,
            kernel_sizes=ks,
            dilations=ds,
            dropout=dropout,
            se_reduction=se_reduction,
            use_kan_act=use_kan_act,
            use_se=use_se,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
        )
        if use_kan_head:
            self.head = nn.Sequential(
                KANLinear(block2_out, head_hidden_dim, kan_grid_size, kan_spline_order),
                nn.Dropout(dropout),
                KANLinear(
                    head_hidden_dim, num_classes, kan_grid_size, kan_spline_order
                ),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(block2_out, head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = out.mean(dim=-1)
        out = self.head(out)
        return out

    def get_kan_activations(self):
        activations = {}
        for block_name, block in [("block1", self.block1), ("block2", self.block2)]:
            for i, branch in enumerate(block.branches):
                for j, act_name in enumerate(["act1", "act2"]):
                    act = getattr(branch, act_name)
                    if isinstance(act, KANActivation):
                        key = f"{block_name}/branch{i}/{act_name}"
                        activations[key] = act
        return activations


def build_ms_kanconv(
    input_channels: int, num_classes: int, variant: str = "full", **kwargs
):
    defaults = dict(
        branch_out_channels_1=32,
        branch_out_channels_2=64,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout=0.2,
        se_reduction=4,
        kan_grid_size=5,
        kan_spline_order=3,
        head_hidden_dim=64,
        use_kan_act=True,
        use_multiscale=True,
        use_kan_head=True,
        use_se=True,
    )
    defaults.update(kwargs)
    if variant == "ms_kanconv_full" or variant == "ms_kanconv" or variant == "full":
        pass
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
