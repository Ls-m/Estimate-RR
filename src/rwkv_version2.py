import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RWKVBlock(nn.Module):
    """Single RWKV block with time mixing and channel mixing."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Time mixing parameters
        self.time_decay = nn.Parameter(torch.randn(d_model))
        self.time_first = nn.Parameter(torch.randn(d_model))
        
        # Time mixing layers
        self.time_mix_k = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_v = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_r = nn.Linear(d_model, d_model, bias=False)
        
        # Channel mixing layers  
        self.channel_mix_k = nn.Linear(d_model, d_model * 4, bias=False)
        self.channel_mix_v = nn.Linear(d_model * 4, d_model, bias=False)
        self.channel_mix_r = nn.Linear(d_model, d_model, bias=False)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        # self.dropout = nn.Dropout(dropout)
        
        # Time shift mixing ratios
        self.time_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        
        self.channel_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        self.channel_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        
    def time_shift(self, x: torch.Tensor, mix_ratio: torch.Tensor) -> torch.Tensor:
        """Apply time shifting with learnable mixing ratio."""
        B, T, C = x.size()
        
        if T == 1:
            return x
        
        # Shift: [x0, x1, x2, ...] -> [0, x0, x1, ...]
        x_prev = torch.cat([torch.zeros(B, 1, C, device=x.device, dtype=x.dtype), 
                           x[:, :-1, :]], dim=1)
        
        # Mix current and previous
        return x * mix_ratio + x_prev * (1 - mix_ratio)
    
    def wkv_computation(self, w: torch.Tensor, u: torch.Tensor, 
                       k: torch.Tensor, v: torch.Tensor, 
                       state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified but correct WKV operation."""
        B, T, C = k.size()
        device = k.device
        dtype = k.dtype
        
        # Initialize output
        output = torch.zeros_like(v)
        
        # Initialize or use provided state [aa, bb, pp]
        if state is None:
            aa = torch.zeros(B, C, device=device, dtype=dtype)
            bb = torch.zeros(B, C, device=device, dtype=dtype) - float('inf')
            pp = torch.zeros(B, C, device=device, dtype=dtype) - float('inf')
        else:
            aa, bb, pp = state[:, :, 0], state[:, :, 1], state[:, :, 2]
        
        # Process each time step
        for t in range(T):
            kk = k[:, t, :]  # (B, C)
            vv = v[:, t, :]  # (B, C)
            
            # Compute WKV for this timestep
            ww = u + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            
            # Output for this timestep
            output[:, t, :] = (e1 * aa + e2 * vv) / (e1 + e2 + 1e-8)  # Add small epsilon for stability
            
            # Update state for next timestep
            ww = w + pp
            p = torch.maximum(pp, kk)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(kk - p)
            
            aa = e1 * aa + e2 * vv
            pp = p + torch.log(e1 + e2 + 1e-8)  # Add small epsilon for stability
            bb = ww  # Not used in this simplified version but kept for completeness
        
        # Return output and final state
        final_state = torch.stack([aa, bb, pp], dim=2)  # (B, C, 3)
        return output, final_state
    
    def time_mixing(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Time mixing operation."""
        B, T, C = x.size()
        
        # Apply time shift mixing
        xk = self.time_shift(x, self.time_mix_k_ratio)
        xv = self.time_shift(x, self.time_mix_v_ratio)
        xr = self.time_shift(x, self.time_mix_r_ratio)
        
        # Compute key, value, receptance
        k = self.time_mix_k(xk)
        v = self.time_mix_v(xv)
        r = self.time_mix_r(xr)
        
        # Apply sigmoid to receptance
        r = torch.sigmoid(r)
        
        # WKV operation
        w = -torch.exp(self.time_decay)  # Decay weights (negative)
        u = self.time_first                # First step weights
        
        wkv_out, new_state = self.wkv_computation(w, u, k, v, state)
        
        return r * wkv_out, new_state
    
    def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Channel mixing operation."""
        # Apply time shift mixing for channel mixing
        xk = self.time_shift(x, self.channel_mix_k_ratio)
        xr = self.time_shift(x, self.channel_mix_r_ratio)
        
        k = self.channel_mix_k(xk)
        r = self.channel_mix_r(xr)
        
        # Apply squared ReLU activation
        vv = self.channel_mix_v(F.relu(k) ** 2)
        
        return torch.sigmoid(r) * vv
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through RWKV block."""
        # Time mixing with residual connection
        tm_out, new_state = self.time_mixing(self.ln1(x), state)
        # x = x + self.dropout(tm_out)
        x = x + self.tm_out
        
        # Channel mixing with residual connection
        cm_out = self.channel_mixing(self.ln2(x))
        # x = x + self.dropout(cm_out)
        x = x + self.cm_out
        
        return x, new_state


class RWKV(nn.Module):
    """Multi-layer RWKV model for time series."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # RWKV blocks
        self.blocks = nn.ModuleList([
            RWKVBlock(hidden_size, dropout) for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.ln_out = nn.LayerNorm(hidden_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Parameter):
            if 'time_decay' in str(module):
                # Initialize time decay to reasonable values
                torch.nn.init.uniform_(module, -4.0, -1.0)
            elif 'time_first' in str(module):
                # Initialize time first to small positive values
                torch.nn.init.uniform_(module, 0.0, 1.0)
            elif 'mix_ratio' in str(module):
                # Initialize mixing ratios to 0.5 (equal mix)
                torch.nn.init.constant_(module, 0.5)
    
    def forward(self, x: torch.Tensor, state: Optional[list] = None) -> torch.Tensor:
        """Forward pass through RWKV."""
        # Input projection
        x = self.input_proj(x)
        
        # Process through RWKV blocks
        states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state is not None else None
            x, new_state = block(x, block_state)
            states.append(new_state)
        
        # Output normalization
        x = self.ln_out(x)
        
        # Return final representation (last time step) - SAME AS YOUR ORIGINAL
        # return x[:, -1, :]  # (batch_size, hidden_size)
        return x  # (batch_size, hidden_size)
    


# class RWKVRRModel(nn.Module):
#     """RWKV model for respiratory rate estimation - SAME OUTPUT SHAPE AS ORIGINAL."""
    
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=64, dropout=0.2):
#         super().__init__()
#         self.rwkv = RWKV(input_size, hidden_size, num_layers, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         """
#         Forward pass - EXACTLY SAME AS YOUR ORIGINAL MODEL.
        
#         Args:
#             x: Input PPG signal (B, T, 1)
            
#         Returns:
#             Output tensor (B, output_size) - SAME SHAPE AS ORIGINAL
#         """
#         # x: (B, T, 1)
#         rwkv_out = self.rwkv(x)        # returns (B, hidden_size)
#         out = self.fc(rwkv_out)        # map to embedding dimension (B, output_size)
#         return out

class RWKVRRModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k // 2)
            for k in [3, 7, 15, 31]
        ])
        self.conv_bn = nn.BatchNorm1d(hidden_size)  # normalize concatenated conv features
        self.conv_act = nn.ReLU()
        self.rwkv = RWKV(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.temporal_pool = nn.AdaptiveAvgPool1d(32)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 1, 4000) for Conv1d

        # Apply multi-scale convolutions
        conv_outs = [conv(x) for conv in self.input_convs]   # list of (B, hidden_size//4, 4000)
        x = torch.cat(conv_outs, dim=1)                      # (B, hidden_size, 4000)
        x = self.conv_bn(x)
        x = self.conv_act(x)

        # Prepare for RWKV input
        x = x.transpose(1, 2)  # (B, 4000, hidden_size)
        x = self.rwkv(x)                  # (B, 4000, hidden_size)
        x = x.transpose(1, 2)             # (B, hidden_size, 4000)
        x = self.temporal_pool(x)         # (B, hidden_size, 32)
        x = x.transpose(1, 2)             # (B, 32, hidden_size)
        out = self.fc(x)                  # (B, 32, 1)
        return out.squeeze(-1)            # (B, 32)