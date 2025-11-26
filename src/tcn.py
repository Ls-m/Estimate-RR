import torch
import torch. nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, stride: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
    
    def forward(self, x: torch. Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future positions
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn. BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self. bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn. Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch. Tensor:
        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self. dropout(F.gelu(self.bn2(self.conv2(out))))
        
        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially growing dilation."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 6,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        channels = [input_size] + [hidden_size] * num_layers
        
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32... 
            layers.append(
                TemporalBlock(
                    channels[i], channels[i + 1],
                    kernel_size, dilation, dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.ln_out = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, T, C) -> need (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = self. network(x)
        x = x.transpose(1, 2)  # Back to (B, T, C)
        return self.ln_out(x)


class TCNScalogramModel(nn. Module):
    """TCN-based model for scalogram data."""
    
    def __init__(self, hidden_size: int = 256, num_layers: int = 6,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn. GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn. BatchNorm2d(64),
            nn.GELU(),
            nn. Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        self.bridge = nn. Linear(32 * 64, hidden_size)
        
        # TCN
        self.tcn = TCN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn. GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch. Tensor, return_embedding: bool = False) -> torch. Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = self.feature_extractor(x)
        features = features.permute(0, 3, 1, 2)
        
        B, T, C, F = features.shape
        features = features.reshape(B, T, C * F)
        features = self.bridge(features)
        
        seq_features = self.tcn(features)
        
        if return_embedding:
            return torch. mean(seq_features, dim=1)
        
        out = self.head(seq_features)
        return out.squeeze(-1)