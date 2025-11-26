import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class S6Layer(nn.Module):
    """Selective State Space (S6) layer - core of Mamba."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self. d_conv = d_conv
        self. expand = expand
        self. d_inner = d_model * expand
        
        # Input projection (creates x and z)
        self. in_proj = nn. Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM parameters (input-dependent)
        self. x_proj = nn. Linear(self. d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        
        # Initialize A matrix (diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log-space for stability
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Skip connection
        
        # dt projection
        self.dt_proj = nn. Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias for reasonable timescales
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = self.dt_proj. weight @ torch.ones(1) + self.dt_proj.bias
            self.dt_proj.bias. copy_(dt_init - inv_dt. squeeze())
        
        # Output projection
        self. out_proj = nn.Linear(self. d_inner, d_model, bias=False)
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan (simplified version)."""
        B, L, D = x.shape
        
        # Project to get B, C, dt
        x_proj = self.x_proj(x)  # (B, L, d_state*2 + 1)
        delta = F.softplus(self.dt_proj(x_proj[:, :, -1:]))  # (B, L, D)
        B_input = x_proj[:, :, :self.d_state]  # (B, L, d_state)
        C = x_proj[:, :, self.d_state:self. d_state*2]  # (B, L, d_state)
        
        # Get A from log space
        A = -torch.exp(self.A_log)  # (D, d_state)
        
        # Discretize A and B
        # dA = exp(delta * A)
        dA = torch.exp(delta. unsqueeze(-1) * A)  # (B, L, D, d_state)
        dB = delta.unsqueeze(-1) * B_input. unsqueeze(2)  # (B, L, D, d_state)
        
        # Selective scan
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :]. transpose(1, 2)
            y = (h * C[:, t]. unsqueeze(1)).sum(-1)  # (B, D)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (B, L, D)
        
        # Add skip connection
        y = y + self.D * x
        
        return y
    
    def forward(self, x: torch. Tensor) -> torch. Tensor:
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz. chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Conv1d (for local context)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Causal: truncate
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        return y


class MambaBlock(nn.Module):
    """Mamba block with residual connection."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self. ln = nn.LayerNorm(d_model)
        self. mamba = S6Layer(d_model, d_state, d_conv, expand)
        self. dropout = nn. Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.ln(x)))


class Mamba(nn. Module):
    """Mamba model for sequence processing."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_size, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)


class MambaScalogramModel(nn.Module):
    """Mamba-based model for scalogram data."""
    
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn. Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn. GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn. GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn. BatchNorm2d(64),
            nn.GELU(),
        )
        
        self.bridge = nn. Linear(32 * 64, hidden_size)
        
        # Mamba
        self.mamba = Mamba(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            d_state=d_state,
            dropout=dropout
        )
        
        # Head
        self. head = nn.Sequential(
            nn. Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn. Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = self.feature_extractor(x)
        features = features.permute(0, 3, 1, 2)
        
        B, T, C, F = features.shape
        features = features.reshape(B, T, C * F)
        features = self.bridge(features)
        
        seq_features = self. mamba(features)
        
        if return_embedding:
            return torch.mean(seq_features, dim=1)
        
        out = self.head(seq_features)
        return out.squeeze(-1)