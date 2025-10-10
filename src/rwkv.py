import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class WKV(torch.autograd.Function):
    """Weighted Key-Value operation for RWKV."""
    
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v, last_state):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.save_for_backward(w, u, k, v, last_state)
        
        # Initialize output and state
        y = torch.empty((B, T, C), device=w.device, dtype=w.dtype)
        new_state = torch.empty((B, C, 3), device=w.device, dtype=w.dtype)
        
        # Process each batch
        for b in range(B):
            # Get initial state
            if last_state is not None:
                state = last_state[b].clone()
            else:
                state = torch.zeros(C, 3, device=w.device, dtype=w.dtype)
            
            # Process each time step
            for t in range(T):
                kk = k[b, t]
                vv = v[b, t]
                ww = w
                uu = u
                
                # RWKV computation
                wkv = (state[:, 0] + uu * kk) / (state[:, 1] + uu)
                y[b, t] = wkv * vv
                
                # Update state
                new_kk = torch.exp(ww) * kk + state[:, 0]
                new_vv = torch.exp(ww) * vv + state[:, 2]
                new_aa = torch.exp(ww) + state[:, 1]
                
                state[:, 0] = new_kk
                state[:, 1] = new_aa
                state[:, 2] = new_vv
            
            new_state[b] = state
        
        return y, new_state
    
    @staticmethod
    def backward(ctx, grad_y, grad_new_state):
        # Simplified backward pass
        w, u, k, v, last_state = ctx.saved_tensors
        return None, None, None, None, None, None, None, None


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
        self.dropout = nn.Dropout(dropout)
        
        # Time shift parameters
        self.time_shift = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def time_mixing(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Time mixing operation."""
        B, T, C = x.size()
        
        # Time shift
        if T > 1:
            x_shifted = torch.cat([self.time_shift.expand(B, 1, C), x[:, :-1, :]], dim=1)
        else:
            x_shifted = self.time_shift.expand(B, T, C)
        
        # Mix current and previous states
        xx = x - x_shifted
        xxx = x + xx * self.time_shift
        
        # Compute key, value, receptance
        k = self.time_mix_k(xxx)
        v = self.time_mix_v(xxx)
        r = self.time_mix_r(xxx)
        
        # Apply sigmoid to receptance
        r = torch.sigmoid(r)
        
        # WKV operation (simplified)
        w = -torch.exp(self.time_decay)
        u = self.time_first
        
        # Simple implementation without custom CUDA kernel
        wkv_out = self.simple_wkv(w, u, k, v, state)
        
        return r * wkv_out, None
    
    def simple_wkv(self, w: torch.Tensor, u: torch.Tensor, 
                   k: torch.Tensor, v: torch.Tensor, 
                   state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simplified WKV operation for CPU/GPU compatibility."""
        B, T, C = k.size()
        
        # Initialize output
        output = torch.zeros_like(v)
        
        # Initialize state
        if state is None:
            aa = torch.zeros(B, C, device=k.device, dtype=k.dtype)
            bb = torch.zeros(B, C, device=k.device, dtype=k.dtype)
        else:
            aa, bb = state[:, :, 0], state[:, :, 1]
        
        # Process each time step
        for t in range(T):
            kk = k[:, t, :]
            vv = v[:, t, :]
            
            ww = torch.exp(w + kk)
            p = torch.maximum(bb, u + kk)
            e1 = torch.exp(bb - p)
            e2 = torch.exp(u + kk - p)
            
            output[:, t, :] = (e1 * aa + e2 * vv) / (e1 + e2)
            
            # Update state
            ww = torch.exp(w + bb)
            p = torch.maximum(bb, kk)
            e1 = torch.exp(bb - p)
            e2 = torch.exp(kk - p)
            
            aa = e1 * aa + e2 * vv
            bb = p + torch.log(e1 + e2)
        
        return output
    
    def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Channel mixing operation."""
        # Time shift for channel mixing
        xx = torch.cat([x[:, -1:, :], x[:, :-1, :]], dim=1)
        xxx = x + (xx - x) * self.time_shift
        
        k = self.channel_mix_k(xxx)
        r = self.channel_mix_r(xxx)
        
        # Apply activation and channel mixing
        vv = self.channel_mix_v(F.relu(k) ** 2)
        
        return torch.sigmoid(r) * vv
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through RWKV block."""
        # Time mixing
        tm_out, new_state = self.time_mixing(self.ln1(x), state)
        x = x + self.dropout(tm_out)
        
        # Channel mixing
        cm_out = self.channel_mixing(self.ln2(x))
        x = x + self.dropout(cm_out)
        
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
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # Return final representation (last time step)
        return x[:, -1, :]  # (batch_size, hidden_size)
    

class RWKVRRModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=64, dropout=0.2):
        super().__init__()
        self.rwkv = RWKV(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, 1)
        rwkv_out = self.rwkv(x)        # returns (B, hidden_size)
        out = self.fc(rwkv_out)        # map to embedding dimension
        return out


