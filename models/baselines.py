import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .kan_modules import KANLinear


class CNN1D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.mean(dim=-1)
        return self.classifier(out)


class DeepConvLSTM(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.transpose(1, 2)
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        return self.classifier(out)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        min_t = min(out.size(-1), res.size(-1))
        return self.relu(out[:, :, :min_t] + res[:, :, :min_t])


class TCNVanilla(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_channels, 64, 3, dilation=1, dropout=dropout),
            TCNBlock(64, 128, 3, dilation=2, dropout=dropout),
            TCNBlock(128, 256, 3, dilation=4, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.tcn(x)
        out = out.mean(dim=-1)
        return self.classifier(out)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        weights = self.attention(x_t).squeeze(-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.bmm(weights.unsqueeze(1), x_t).squeeze(1)
        return out


class TCNAttention(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_channels, 64, 3, dilation=1, dropout=dropout),
            TCNBlock(64, 128, 3, dilation=2, dropout=dropout),
            TCNBlock(128, 256, 3, dilation=4, dropout=dropout),
        )
        self.attention = TemporalAttention(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.tcn(x)
        out = self.attention(out)
        return self.classifier(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerHAR(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class KANHAR(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            KANLinear(256, 128, grid_size=5, spline_order=3),
            nn.Dropout(dropout),
            KANLinear(128, num_classes, grid_size=5, spline_order=3),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.mean(dim=-1)
        return self.head(out)


def get_baseline_model(
    name: str, input_channels: int, num_classes: int, dropout: float = 0.2
):
    models = {
        "cnn_1d": CNN1D,
        "deep_conv_lstm": DeepConvLSTM,
        "tcn_vanilla": TCNVanilla,
        "tcn_attention": TCNAttention,
        "transformer_har": TransformerHAR,
        "kan_har": KANHAR,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](input_channels, num_classes, dropout=dropout)
