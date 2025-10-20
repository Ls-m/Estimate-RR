# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple, List
# import math


# class RWKVBlock(nn.Module):
#     """Fixed RWKV block that can actually learn."""
    
#     def __init__(self, d_model: int, layer_id: int = 0, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.layer_id = layer_id
        
#         # Time mixing parameters (learnable)
#         self.time_decay = nn.Parameter(torch.ones(d_model))
#         self.time_first = nn.Parameter(torch.ones(d_model))
        
#         # Time mixing layers - ADD BIAS for better learning
#         self.time_mix_k = nn.Linear(d_model, d_model, bias=True)
#         self.time_mix_v = nn.Linear(d_model, d_model, bias=True)
#         self.time_mix_r = nn.Linear(d_model, d_model, bias=True)
        
#         # Output projection for time mixing
#         self.time_output = nn.Linear(d_model, d_model, bias=True)
        
#         # Channel mixing layers  
#         self.channel_mix_k = nn.Linear(d_model, d_model * 4, bias=True)
#         self.channel_mix_v = nn.Linear(d_model * 4, d_model, bias=True)
#         self.channel_mix_r = nn.Linear(d_model, d_model, bias=True)
        
#         # Layer normalization
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
        
#         # Time shift mixing ratios - Initialize to allow gradient flow
#         self.time_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
#         self.time_mix_v_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
#         self.time_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
#         self.channel_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
#         self.channel_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
#         # Initialize parameters properly
#         self._init_weights()
        
#     def _init_weights(self):
#         """Initialize weights with LESS aggressive scaling."""
#         with torch.no_grad():
#             # FIXED: Less extreme decay initialization
#             # Start with milder decay values
#             ratio_0_to_1 = self.layer_id / max(1, self.layer_id + 1)
            
#             # Initialize decay values (LESS NEGATIVE = easier to learn initially)
#             decay_speed = torch.ones(self.d_model)
#             for h in range(self.d_model):
#                 # Changed from -5 to -2, and reduced range
#                 decay_speed[h] = -2 + 3 * (h / max(1, self.d_model - 1)) ** 0.5
#             self.time_decay.data = decay_speed
            
#             # FIXED: More reasonable time_first initialization
#             self.time_first.data = torch.ones(self.d_model) * 0.5
        
#         # Initialize linear layers with HIGHER gain (was 0.5, now 1.0)
#         for name, module in self.named_modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight, gain=1.0)  # Changed from 0.5
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
    
#     def time_shift(self, x: torch.Tensor, mix_ratio: torch.Tensor, 
#                    state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Apply time shifting with learnable mixing ratio."""
#         B, T, C = x.size()
        
#         if T == 1 and state is not None:
#             x_prev = state.unsqueeze(1)
#         else:
#             if state is not None:
#                 x_prev = torch.cat([state.unsqueeze(1), x[:, :-1, :]], dim=1)
#             else:
#                 x_prev = torch.cat([torch.zeros(B, 1, C, device=x.device, dtype=x.dtype), 
#                                    x[:, :-1, :]], dim=1)
        
#         # Clamp mix_ratio to prevent gradient issues
#         mix_ratio = torch.clamp(mix_ratio, 0.0, 1.0)
#         mixed = x * mix_ratio + x_prev * (1 - mix_ratio)
        
#         return mixed, x[:, -1, :].clone()
    
#     def wkv_computation(self, w: torch.Tensor, u: torch.Tensor, 
#                        k: torch.Tensor, v: torch.Tensor, 
#                        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
#                        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#         """FIXED WKV computation - corrected state update logic."""
#         B, T, C = k.size()
#         device = k.device
#         dtype = k.dtype
        
#         output = torch.zeros_like(v)
        
#         if state is None:
#             aa = torch.zeros(B, C, device=device, dtype=dtype)
#             bb = torch.zeros(B, C, device=device, dtype=dtype)
#             pp = torch.zeros(B, C, device=device, dtype=dtype) - 1e30
#         else:
#             aa, bb, pp = state
        
#         # Process each time step
#         for t in range(T):
#             kk = k[:, t, :]
#             vv = v[:, t, :]
            
#             # COMPUTE OUTPUT
#             ww = u + kk
#             qq = torch.maximum(pp, ww)
#             e1 = torch.exp(pp - qq)
#             e2 = torch.exp(ww - qq)
            
#             numerator = e1 * aa + e2 * vv
#             denominator = e1 * bb + e2
            
#             output[:, t, :] = numerator / (denominator + 1e-8)
            
#             # FIXED: Update state for NEXT timestep
#             # The key bug was here - we need to use w (decay) + pp, not overwrite ww
#             ww_next = w + pp  # Decay the previous max
#             qq_next = torch.maximum(ww_next, kk)
#             e1_next = torch.exp(ww_next - qq_next)
#             e2_next = torch.exp(kk - qq_next)
            
#             # Update accumulators
#             aa = e1_next * aa + e2_next * vv
#             bb = e1_next * bb + e2_next
#             pp = qq_next
        
#         return output, (aa, bb, pp)
    
#     def time_mixing(self, x: torch.Tensor, 
#                    state: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
#         """Time mixing operation with proper state handling."""
#         B, T, C = x.size()
        
#         shift_states = state.get('shift', {}) if state else {}
#         wkv_state = state.get('wkv', None) if state else None
        
#         xk, k_state = self.time_shift(x, self.time_mix_k_ratio, 
#                                       shift_states.get('k', None))
#         xv, v_state = self.time_shift(x, self.time_mix_v_ratio,
#                                       shift_states.get('v', None))
#         xr, r_state = self.time_shift(x, self.time_mix_r_ratio,
#                                       shift_states.get('r', None))
        
#         k = self.time_mix_k(xk)
#         v = self.time_mix_v(xv)
#         r = self.time_mix_r(xr)
        
#         sr = torch.sigmoid(r)
        
#         # FIXED: Less aggressive decay clamping
#         w = -F.softplus(self.time_decay)  # Removed the -0.5 offset
#         u = self.time_first
        
#         wkv_out, new_wkv_state = self.wkv_computation(w, u, k, v, wkv_state)
        
#         output = self.time_output(sr * wkv_out)
        
#         new_state = {
#             'shift': {'k': k_state, 'v': v_state, 'r': r_state},
#             'wkv': new_wkv_state
#         }
        
#         return output, new_state
    
#     def channel_mixing(self, x: torch.Tensor,
#                       state: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
#         """Channel mixing operation (FFN-like)."""
#         shift_states = state if state else {}
        
#         xk, k_state = self.time_shift(x, self.channel_mix_k_ratio,
#                                       shift_states.get('k', None))
#         xr, r_state = self.time_shift(x, self.channel_mix_r_ratio,
#                                       shift_states.get('r', None))
        
#         k = self.channel_mix_k(xk)
#         r = self.channel_mix_r(xr)
        
#         vv = self.channel_mix_v(torch.relu(k) ** 2)
        
#         output = torch.sigmoid(r) * vv
        
#         new_state = {'k': k_state, 'r': r_state}
        
#         return output, new_state
    
#     def forward(self, x: torch.Tensor, 
#                state: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
#         """Forward pass through RWKV block."""
#         time_state = state.get('time', None) if state else None
#         channel_state = state.get('channel', None) if state else None
        
#         tm_out, new_time_state = self.time_mixing(self.ln1(x), time_state)
#         x = x + self.dropout(tm_out)
        
#         cm_out, new_channel_state = self.channel_mixing(self.ln2(x), channel_state)
#         x = x + self.dropout(cm_out)
        
#         new_state = {
#             'time': new_time_state,
#             'channel': new_channel_state
#         }
        
#         return x, new_state


# class RWKV(nn.Module):
#     """Multi-layer RWKV model for time series."""

#     def __init__(self, input_size: int, hidden_size: int = 64, 
#                  num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Input projection with layer norm
#         self.input_ln = nn.LayerNorm(input_size)
#         self.input_proj = nn.Linear(input_size, hidden_size, bias=True)  # Added bias
        
#         # RWKV blocks with layer IDs for proper initialization
#         self.blocks = nn.ModuleList([
#             RWKVBlock(hidden_size, layer_id=i, dropout=dropout) 
#             for i in range(num_layers)
#         ])
        
#         # Output layer norm
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#         # Initialize input/output layers
#         self._init_io_weights()
    
#     def _init_io_weights(self):
#         """Initialize input and output layers with BETTER scaling."""
#         # Changed gain from 1.0 to higher
#         nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
#         if self.input_proj.bias is not None:
#             nn.init.zeros_(self.input_proj.bias)
    
#     def forward(self, x: torch.Tensor, 
#                state: Optional[List[dict]] = None,
#                return_sequence: bool = False) -> torch.Tensor:
#         """Forward pass through RWKV."""
#         # Input normalization and projection
#         x = self.input_ln(x)
#         x = self.input_proj(x)
        
#         # Process through RWKV blocks
#         states = []
#         for i, block in enumerate(self.blocks):
#             block_state = state[i] if state is not None else None
#             x, new_state = block(x, block_state)
#             states.append(new_state)
        
#         # Output normalization
#         x = self.ln_out(x)
        
#         return x[:, -1, :]  # (B, hidden_size)


# class RWKVRRModel(nn.Module):
#     """Complete model for respiratory rate estimation from PPG signals."""
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=64, dropout=0.2):
#         super().__init__()
#         self.rwkv = RWKV(input_size, hidden_size, num_layers, dropout)
#         self.fc = nn.Linear(hidden_size, output_size, bias=True)
        
#         # Initialize FC layer properly
#         nn.init.xavier_uniform_(self.fc.weight)
#         nn.init.zeros_(self.fc.bias)

#     def forward(self, x: torch.Tensor):
#         if x.dim() == 2:
#             x = x.unsqueeze(-1)  # (B, T, 1)
#         rwkv_out = self.rwkv(x)  # (B, hidden_size)
#         return self.fc(rwkv_out)  # (B, output_size)
    









# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple, List
# import math


# class RWKVBlock(nn.Module):
#     """RWKV block with CORRECTED WKV computation that can learn."""
    
#     def __init__(self, d_model: int, layer_id: int = 0, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.layer_id = layer_id
        
#         # Time mixing layers
#         self.time_mix_k = nn.Linear(d_model, d_model)
#         self.time_mix_v = nn.Linear(d_model, d_model)
#         self.time_mix_r = nn.Linear(d_model, d_model)
#         self.time_output = nn.Linear(d_model, d_model)
        
#         # Channel mixing layers  
#         self.channel_mix_k = nn.Linear(d_model, d_model * 4)
#         self.channel_mix_v = nn.Linear(d_model * 4, d_model)
#         self.channel_mix_r = nn.Linear(d_model, d_model)
        
#         # Layer normalization
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
        
#         # CRITICAL FIX: Initialize these as learnable but with reasonable ranges
#         # Time decay: controls how much past information decays
#         self.time_decay = nn.Parameter(torch.ones(d_model))
#         # Time first: bonus for current timestep
#         self.time_first = nn.Parameter(torch.ones(d_model))
        
#         # Time shift mixing ratios
#         self.time_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model))
#         self.time_mix_v_ratio = nn.Parameter(torch.ones(1, 1, d_model))
#         self.time_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        
#         self.channel_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model))
#         self.channel_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        
#         self._init_weights()
        
#     def _init_weights(self):
#         """Initialize with values that allow gradient flow."""
#         with torch.no_grad():
#             # CRITICAL: Initialize decay to reasonable values
#             # Use -1 to -0.1 range instead of -5 to 3
#             decay_init = torch.linspace(-1.0, -0.1, self.d_model)
#             self.time_decay.data = decay_init
            
#             # Initialize time_first to small positive values
#             self.time_first.data = torch.ones(self.d_model) * 0.5
            
#             # Initialize mixing ratios to 0.5
#             self.time_mix_k_ratio.data.fill_(0.5)
#             self.time_mix_v_ratio.data.fill_(0.5)
#             self.time_mix_r_ratio.data.fill_(0.5)
#             self.channel_mix_k_ratio.data.fill_(0.5)
#             self.channel_mix_r_ratio.data.fill_(0.5)
        
#         # Initialize linear layers with normal initialization
#         for module in [self.time_mix_k, self.time_mix_v, self.time_mix_r, 
#                        self.time_output, self.channel_mix_k, self.channel_mix_v, 
#                        self.channel_mix_r]:
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
    
#     def time_shift(self, x: torch.Tensor, mix_ratio: torch.Tensor) -> torch.Tensor:
#         """Simplified time shift without state management (for training)."""
#         B, T, C = x.size()
        
#         # Clamp ratio to valid range
#         mix_ratio = torch.clamp(mix_ratio, 0.0, 1.0)
        
#         # Simple padding approach
#         x_shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0.0)
        
#         return x * mix_ratio + x_shifted * (1 - mix_ratio)
    
#     def wkv_computation_v2(self, w: torch.Tensor, u: torch.Tensor, 
#                           k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         """
#         SIMPLIFIED WKV that actually allows gradients to flow.
        
#         The original implementation has numerical stability issues and
#         complex state tracking that causes vanishing gradients.
        
#         This version uses a more stable cumulative approach.
#         """
#         B, T, C = k.size()
        
#         # Convert w to actual decay values in (0, 1)
#         # Use sigmoid to ensure stability
#         decay = torch.sigmoid(w)  # Now in range (0, 1)
#         bonus = torch.tanh(u)      # Bonus in range (-1, 1)
        
#         # Initialize accumulators
#         num = torch.zeros(B, C, device=k.device, dtype=k.dtype)
#         den = torch.zeros(B, C, device=k.device, dtype=k.dtype)
        
#         outputs = []
        
#         for t in range(T):
#             kt = k[:, t, :]  # (B, C)
#             vt = v[:, t, :]  # (B, C)
            
#             # Compute attention weight for current timestep
#             # bonus gives extra weight to current input
#             attn_current = torch.sigmoid(bonus + kt)
            
#             # Compute output: blend of accumulated state and current value
#             output = (attn_current * vt + (1 - attn_current) * num) / (den + 1e-8)
#             outputs.append(output)
            
#             # Update accumulators with decay
#             num = decay * num + vt
#             den = decay * den + 1.0
        
#         return torch.stack(outputs, dim=1)
    
#     def time_mixing(self, x: torch.Tensor) -> torch.Tensor:
#         """Time mixing operation."""
#         # Apply time shift mixing
#         xk = self.time_shift(x, self.time_mix_k_ratio)
#         xv = self.time_shift(x, self.time_mix_v_ratio)
#         xr = self.time_shift(x, self.time_mix_r_ratio)
        
#         # Compute key, value, receptance
#         k = self.time_mix_k(xk)
#         v = self.time_mix_v(xv)
#         r = torch.sigmoid(self.time_mix_r(xr))
        
#         # WKV operation with simplified computation
#         wkv_out = self.wkv_computation_v2(self.time_decay, self.time_first, k, v)
        
#         # Output projection
#         return self.time_output(r * wkv_out)
    
#     def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
#         """Channel mixing operation (FFN-like)."""
#         # Apply time shift mixing for channel mixing
#         xk = self.time_shift(x, self.channel_mix_k_ratio)
#         xr = self.time_shift(x, self.channel_mix_r_ratio)
        
#         k = self.channel_mix_k(xk)
#         r = self.channel_mix_r(xr)
        
#         # Apply squared ReLU activation
#         vv = self.channel_mix_v(torch.relu(k) ** 2)
        
#         return torch.sigmoid(r) * vv
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through RWKV block."""
#         # Time mixing with residual connection (pre-norm)
#         tm_out = self.time_mixing(self.ln1(x))
#         x = x + self.dropout(tm_out)
        
#         # Channel mixing with residual connection (pre-norm)
#         cm_out = self.channel_mixing(self.ln2(x))
#         x = x + self.dropout(cm_out)
        
#         return x


# class RWKV(nn.Module):
#     """Multi-layer RWKV model."""

#     def __init__(self, input_size: int, hidden_size: int = 64, 
#                  num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Input projection
#         self.input_ln = nn.LayerNorm(input_size)
#         self.input_proj = nn.Linear(input_size, hidden_size)
        
#         # RWKV blocks
#         self.blocks = nn.ModuleList([
#             RWKVBlock(hidden_size, layer_id=i, dropout=dropout) 
#             for i in range(num_layers)
#         ])
        
#         # Output layer norm
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#         # Initialize
#         self._init_io_weights()
    
#     def _init_io_weights(self):
#         """Initialize input/output layers."""
#         nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
#         if self.input_proj.bias is not None:
#             nn.init.zeros_(self.input_proj.bias)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through RWKV."""
#         # Input normalization and projection
#         x = self.input_ln(x)
#         x = self.input_proj(x)
        
#         # Process through RWKV blocks
#         for block in self.blocks:
#             x = block(x)
        
#         # Output normalization
#         x = self.ln_out(x)
        
#         return x[:, -1, :]  # (B, hidden_size)


# class RWKVRRModel(nn.Module):
#     """Complete model for respiratory rate estimation."""
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
#                  output_size=64, dropout=0.2):
#         super().__init__()
#         self.rwkv = RWKV(input_size, hidden_size, num_layers, dropout)
#         self.fc = nn.Linear(hidden_size, output_size)
        
#         # Initialize FC
#         nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
#         nn.init.zeros_(self.fc.bias)

#     def forward(self, x: torch.Tensor):
#         if x.dim() == 2:
#             x = x.unsqueeze(-1)
#         rwkv_out = self.rwkv(x)
#         return self.fc(rwkv_out)
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional
# import math


# class RWKVBlockWithAttention(nn.Module):
#     """RWKV block that replaces WKV with standard attention for debugging."""
    
#     def __init__(self, d_model: int, layer_id: int = 0, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.layer_id = layer_id
        
#         # Time mixing layers (same as RWKV)
#         self.time_mix_k = nn.Linear(d_model, d_model)
#         self.time_mix_v = nn.Linear(d_model, d_model)
#         self.time_mix_r = nn.Linear(d_model, d_model)
#         self.time_output = nn.Linear(d_model, d_model)
        
#         # Channel mixing layers  
#         self.channel_mix_k = nn.Linear(d_model, d_model * 4)
#         self.channel_mix_v = nn.Linear(d_model * 4, d_model)
#         self.channel_mix_r = nn.Linear(d_model, d_model)
        
#         # Layer normalization
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
        
#         # Time shift mixing ratios
#         self.time_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
#         self.time_mix_v_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
#         self.time_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
#         self.channel_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
#         self.channel_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
#         # Scaling factor for attention
#         self.scale = 1.0 / math.sqrt(d_model)
        
#         self._init_weights()
        
#     def _init_weights(self):
#         """Initialize weights."""
#         for module in [self.time_mix_k, self.time_mix_v, self.time_mix_r, 
#                        self.time_output, self.channel_mix_k, self.channel_mix_v, 
#                        self.channel_mix_r]:
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
    
#     def time_shift(self, x: torch.Tensor, mix_ratio: torch.Tensor) -> torch.Tensor:
#         """Simple time shift."""
#         B, T, C = x.size()
#         mix_ratio = torch.clamp(mix_ratio, 0.0, 1.0)
#         x_shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0.0)
#         return x * mix_ratio + x_shifted * (1 - mix_ratio)
    
#     def standard_attention(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         """
#         Replace WKV with standard causal self-attention.
#         This should definitely work if the architecture is sound.
#         """
#         B, T, C = k.shape
        
#         # Compute attention scores: Q @ K^T (using k as both Q and K for simplicity)
#         attn_scores = torch.matmul(k, k.transpose(-2, -1)) * self.scale  # (B, T, T)
        
#         # Apply causal mask (attend only to past and current)
#         causal_mask = torch.tril(torch.ones(T, T, device=k.device, dtype=torch.bool))
#         attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        
#         # Softmax
#         attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)
        
#         # Apply attention to values
#         output = torch.matmul(attn_weights, v)  # (B, T, C)
        
#         return output
    
#     def time_mixing(self, x: torch.Tensor) -> torch.Tensor:
#         """Time mixing with STANDARD ATTENTION instead of WKV."""
#         # Apply time shift mixing
#         xk = self.time_shift(x, self.time_mix_k_ratio)
#         xv = self.time_shift(x, self.time_mix_v_ratio)
#         xr = self.time_shift(x, self.time_mix_r_ratio)
        
#         # Compute key, value, receptance
#         k = self.time_mix_k(xk)
#         v = self.time_mix_v(xv)
#         r = torch.sigmoid(self.time_mix_r(xr))
        
#         # Use standard attention instead of WKV
#         attn_out = self.standard_attention(k, v)
        
#         # Output projection
#         return self.time_output(r * attn_out)
    
#     def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
#         """Channel mixing operation."""
#         xk = self.time_shift(x, self.channel_mix_k_ratio)
#         xr = self.time_shift(x, self.channel_mix_r_ratio)
        
#         k = self.channel_mix_k(xk)
#         r = self.channel_mix_r(xr)
        
#         vv = self.channel_mix_v(torch.relu(k) ** 2)
        
#         return torch.sigmoid(r) * vv
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass."""
#         tm_out = self.time_mixing(self.ln1(x))
#         x = x + self.dropout(tm_out)
        
#         cm_out = self.channel_mixing(self.ln2(x))
#         x = x + self.dropout(cm_out)
        
#         return x


# class RWKVWithAttention(nn.Module):
#     """RWKV architecture but with standard attention instead of WKV."""

#     def __init__(self, input_size: int, hidden_size: int = 64, 
#                  num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.input_ln = nn.LayerNorm(input_size)
#         self.input_proj = nn.Linear(input_size, hidden_size)
        
#         self.blocks = nn.ModuleList([
#             RWKVBlockWithAttention(hidden_size, layer_id=i, dropout=dropout) 
#             for i in range(num_layers)
#         ])
        
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#         self._init_io_weights()
    
#     def _init_io_weights(self):
#         nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
#         if self.input_proj.bias is not None:
#             nn.init.zeros_(self.input_proj.bias)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.input_ln(x)
#         x = self.input_proj(x)
        
#         for block in self.blocks:
#             x = block(x)
        
#         x = self.ln_out(x)
#         return x[:, -1, :]


# class RWKVRRModelWithAttention(nn.Module):
#     """Model using attention instead of WKV."""
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
#                  output_size=1, dropout=0.0):
#         super().__init__()
#         self.rwkv = RWKVWithAttention(input_size, hidden_size, num_layers, dropout)
#         self.fc = nn.Linear(hidden_size, output_size)
        
#         nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
#         nn.init.zeros_(self.fc.bias)

#     def forward(self, x: torch.Tensor):
#         if x.dim() == 2:
#             x = x.unsqueeze(-1)
#         rwkv_out = self.rwkv(x)
#         return self.fc(rwkv_out)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


# class RWKVBlockFixed(nn.Module):
#     """RWKV block with PROPER initialization and NO saturating activations."""
    
#     def __init__(self, d_model: int, layer_id: int = 0, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.layer_id = layer_id
        
#         # Time mixing layers - REMOVE sigmoid from receptance!
#         self.time_mix_k = nn.Linear(d_model, d_model)
#         self.time_mix_v = nn.Linear(d_model, d_model)
#         self.time_mix_r = nn.Linear(d_model, d_model)
#         self.time_output = nn.Linear(d_model, d_model)
        
#         # Channel mixing layers  
#         self.channel_mix_k = nn.Linear(d_model, d_model * 4)
#         self.channel_mix_v = nn.Linear(d_model * 4, d_model)
#         self.channel_mix_r = nn.Linear(d_model, d_model)
        
#         # Layer normalization
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
        
#         # Learnable parameters
#         self.time_decay = nn.Parameter(torch.ones(d_model))
#         self.time_first = nn.Parameter(torch.ones(d_model))
        
#         # Time shift mixing ratios
#         self.time_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model))
#         self.time_mix_v_ratio = nn.Parameter(torch.ones(1, 1, d_model))
#         self.time_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        
#         self.channel_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model))
#         self.channel_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model))
        
#         self._init_weights()
        
#     def _init_weights(self):
#         """Better initialization."""
#         with torch.no_grad():
#             # Initialize decay to reasonable values
#             self.time_decay.data = torch.linspace(-0.5, -0.1, self.d_model)
#             self.time_first.data.fill_(0.5)
            
#             # Initialize mixing ratios
#             self.time_mix_k_ratio.data.fill_(0.5)
#             self.time_mix_v_ratio.data.fill_(0.5)
#             self.time_mix_r_ratio.data.fill_(0.5)
#             self.channel_mix_k_ratio.data.fill_(0.5)
#             self.channel_mix_r_ratio.data.fill_(0.5)
        
#         # IMPORTANT: Use Xavier initialization, not too small!
#         for module in [self.time_mix_k, self.time_mix_v, self.time_mix_r, 
#                        self.time_output, self.channel_mix_k, self.channel_mix_v, 
#                        self.channel_mix_r]:
#             nn.init.xavier_uniform_(module.weight, gain=1.0)  # Use gain=1.0
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
    
#     def time_shift(self, x: torch.Tensor, mix_ratio: torch.Tensor) -> torch.Tensor:
#         """Simple time shift."""
#         B, T, C = x.size()
#         mix_ratio = torch.clamp(mix_ratio, 0.0, 1.0)
#         x_shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0.0)
#         return x * mix_ratio + x_shifted * (1 - mix_ratio)
    
#     def simple_recurrence(self, w: torch.Tensor, u: torch.Tensor, 
#                          k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         """Simplified recurrence mechanism."""
#         B, T, C = k.size()
        
#         # Use tanh to keep values bounded but allow gradients
#         decay = torch.sigmoid(w)  # (0, 1)
#         bonus = torch.tanh(u)      # (-1, 1)
        
#         # Simple RNN-like update
#         state = torch.zeros(B, C, device=k.device, dtype=k.dtype)
#         outputs = []
        
#         for t in range(T):
#             kt = k[:, t, :]
#             vt = v[:, t, :]
            
#             # Compute gate
#             gate = torch.sigmoid(bonus + kt)
            
#             # Output is blend of state and current value
#             out = gate * vt + (1 - gate) * state
#             outputs.append(out)
            
#             # Update state
#             state = decay * state + (1 - decay) * vt
        
#         return torch.stack(outputs, dim=1)
    
#     def time_mixing(self, x: torch.Tensor) -> torch.Tensor:
#         """Time mixing with less saturation."""
#         xk = self.time_shift(x, self.time_mix_k_ratio)
#         xv = self.time_shift(x, self.time_mix_v_ratio)
#         xr = self.time_shift(x, self.time_mix_r_ratio)
        
#         k = self.time_mix_k(xk)
#         v = self.time_mix_v(xv)
#         r = self.time_mix_r(xr)  # NO SIGMOID HERE!
        
#         # Recurrence
#         rec_out = self.simple_recurrence(self.time_decay, self.time_first, k, v)
        
#         # Use tanh instead of sigmoid for receptance (less saturating)
#         return self.time_output(torch.tanh(r) * rec_out)
    
#     def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
#         """Channel mixing with less saturation."""
#         xk = self.time_shift(x, self.channel_mix_k_ratio)
#         xr = self.time_shift(x, self.channel_mix_r_ratio)
        
#         k = self.channel_mix_k(xk)
#         r = self.channel_mix_r(xr)
        
#         # Use GELU instead of squared ReLU (better gradients)
#         vv = self.channel_mix_v(F.gelu(k))
        
#         # Use tanh instead of sigmoid
#         return torch.tanh(r) * vv
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass."""
#         # CRITICAL FIX: Scale residual connections
#         tm_out = self.time_mixing(self.ln1(x))
#         x = x + 0.5 * self.dropout(tm_out)  # Scale down residual
        
#         cm_out = self.channel_mixing(self.ln2(x))
#         x = x + 0.5 * self.dropout(cm_out)  # Scale down residual
        
#         return x


# class RWKVFinal(nn.Module):
#     """Final fixed RWKV."""

#     def __init__(self, input_size: int, hidden_size: int = 64, 
#                  num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # REMOVE input LayerNorm - it normalizes away signal!
#         self.input_proj = nn.Linear(input_size, hidden_size)
        
#         self.blocks = nn.ModuleList([
#             RWKVBlockFixed(hidden_size, layer_id=i, dropout=dropout) 
#             for i in range(num_layers)
#         ])
        
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#         self._init_io_weights()
    
#     def _init_io_weights(self):
#         nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
#         if self.input_proj.bias is not None:
#             nn.init.zeros_(self.input_proj.bias)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # NO input LayerNorm!
#         x = self.input_proj(x)
        
#         for block in self.blocks:
#             x = block(x)
        
#         x = self.ln_out(x)
#         return x[:, -1, :]


# class RWKVRRModelFinal(nn.Module):
#     """Final working model."""
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
#                  output_size=1, dropout=0.0):
#         super().__init__()
#         self.rwkv = RWKVFinal(input_size, hidden_size, num_layers, dropout)
#         self.fc = nn.Linear(hidden_size, output_size)
        
#         nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
#         nn.init.zeros_(self.fc.bias)

#     def forward(self, x: torch.Tensor):
#         if x.dim() == 2:
#             x = x.unsqueeze(-1)
#         rwkv_out = self.rwkv(x)
#         return self.fc(rwkv_out)
    








class RWKVBlock(nn.Module):
    def __init__(self, d_model: int, layer_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Time mixing
        self.time_mix_k = nn.Linear(d_model, d_model)
        self.time_mix_v = nn.Linear(d_model, d_model)
        self.time_mix_r = nn.Linear(d_model, d_model)
        self.time_output = nn.Linear(d_model, d_model)
        
        # Channel mixing  
        self.channel_mix_k = nn.Linear(d_model, d_model * 4)
        self.channel_mix_v = nn.Linear(d_model * 4, d_model)
        self.channel_mix_r = nn.Linear(d_model, d_model)
        
        # Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable parameters
        self.time_decay = nn.Parameter(torch.linspace(-0.5, -0.1, d_model))
        self.time_first = nn.Parameter(torch.ones(d_model) * 0.5)
        
        # Time shift ratios
        self.time_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.channel_mix_k_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.channel_mix_r_ratio = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.time_mix_k, self.time_mix_v, self.time_mix_r, 
                  self.time_output, self.channel_mix_k, self.channel_mix_v, 
                  self.channel_mix_r]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)  # Key: gain=1.0!
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def time_shift(self, x, ratio):
        B, T, C = x.shape
        ratio = torch.clamp(ratio, 0.0, 1.0)
        x_shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0.0)
        return x * ratio + x_shifted * (1 - ratio)
    
    def simple_recurrence(self, w, u, k, v):
        B, T, C = k.size()
        decay = torch.sigmoid(w)
        bonus = torch.tanh(u)
        
        state = torch.zeros(B, C, device=k.device, dtype=k.dtype)
        outputs = []
        
        for t in range(T):
            kt, vt = k[:, t, :], v[:, t, :]
            gate = torch.sigmoid(bonus + kt)
            out = gate * vt + (1 - gate) * state
            outputs.append(out)
            state = decay * state + (1 - decay) * vt
        
        return torch.stack(outputs, dim=1)
    
    def time_mixing(self, x):
        xk = self.time_shift(x, self.time_mix_k_ratio)
        xv = self.time_shift(x, self.time_mix_v_ratio)
        xr = self.time_shift(x, self.time_mix_r_ratio)
        
        k = self.time_mix_k(xk)
        v = self.time_mix_v(xv)
        r = self.time_mix_r(xr)
        
        rec_out = self.simple_recurrence(self.time_decay, self.time_first, k, v)
        return self.time_output(torch.tanh(r) * rec_out)  # tanh not sigmoid!
    
    def channel_mixing(self, x):
        xk = self.time_shift(x, self.channel_mix_k_ratio)
        xr = self.time_shift(x, self.channel_mix_r_ratio)
        
        k = self.channel_mix_k(xk)
        r = self.channel_mix_r(xr)
        
        vv = self.channel_mix_v(F.gelu(k))  # GELU not squared ReLU!
        return torch.tanh(r) * vv  # tanh not sigmoid!
    
    def forward(self, x):
        # Key: Scale residual by 0.5!
        x = x + 0.5 * self.dropout(self.time_mixing(self.ln1(x)))
        x = x + 0.5 * self.dropout(self.channel_mixing(self.ln2(x)))
        return x


class RWKV(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)  # No LayerNorm!
        
        self.blocks = nn.ModuleList([
            RWKVBlock(hidden_size, layer_id=i, dropout=dropout) 
            for i in range(num_layers)
        ])
        
        self.ln_out = nn.LayerNorm(hidden_size)
        
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return x[:, -1, :]


class RWKVRRModel(nn.Module):
    """Complete model for PPG → Respiratory Rate."""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, 
                 output_size=64, dropout=0.2):
        super().__init__()
        self.rwkv = RWKV(input_size, hidden_size, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) → (B, T, 1)
        return self.fc(self.rwkv(x))