import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

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
        x = x + tm_out
        
        # Channel mixing with residual connection
        cm_out = self.channel_mixing(self.ln2(x))
        # x = x + self.dropout(cm_out)
        x = x + cm_out
        
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
        return x  # Shape: (Batch, Time, Hidden_Size)
        # return x  # (batch_size, hidden_size)
class BaseSequenceModel(nn.Module):
    """Base class for all sequence models.  All models must implement forward()."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self. dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Time, Features)
        Returns:
            Output tensor of shape (Batch, Time, Hidden_Size)
        """
        raise NotImplementedError
class TransformerModel(BaseSequenceModel):
    """Transformer encoder for time series."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, nhead: int = 8):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        
        self.nhead = nhead
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer norm
        self.ln_out = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Transformer."""
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Output normalization
        x = self.ln_out(x)
        
        return x  # (Batch, Time, Hidden_Size)


class PositionalEncoding(nn. Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch. exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same interface as in many Transformer implementations)."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix, shape (1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)   # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) where D == d_model
        Returns:
            x + positional_encoding, same shape
        """
        B, T, D = x.shape
        if D != self.pe.size(2):
            raise ValueError(f"PosEnc d_model ({self.pe.size(2)}) != input dim ({D})")
        x = x + self.pe[:, :T, :].to(x.device)
        return self.dropout(x)
    
class TemporalAttentionPooling(nn.Module):
    """
    Learns to weight different time steps based on their importance.
    Instead of simple averaging, this allows the model to focus on 
    the most informative parts of the 60-second window.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Hidden) - RWKV output
        Returns:
            pooled: (Batch, Hidden) - Weighted representation
            attention_weights: (Batch, Time, 1) - Optional, for visualization
        """
        # Compute attention scores for each time step
        attn_scores = self.attention(x)  # (B, T, 1)
        
        # Normalize to get attention weights (sum to 1 across time)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        
        # Apply attention weights
        weighted = x * attn_weights  # (B, T, Hidden)
        
        # Sum across time dimension
        pooled = weighted.sum(dim=1)  # (B, Hidden)
        
        return pooled, attn_weights  # Return weights for visualization
    
class GRUModel(BaseSequenceModel):
    """Bidirectional GRU for time series."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2 if bidirectional else hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer norm
        self.ln_out = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch. Tensor:
        """Forward pass through GRU."""
        # Input projection
        x = self.input_proj(x)
        
        # GRU processing
        x, _ = self.gru(x)
        
        # Output normalization
        x = self.ln_out(x)
        
        return x  # (Batch, Time, Hidden_Size)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixerTokenizer(nn.Module):
    """
    Acts as the 'Eye'. 
    Processes the Scalogram but preserves the TIME dimension for RWKV.
    """
    def __init__(self, dim, depth, kernel_size=7, patch_size=1):
        super().__init__()
        
        # 1. Patch Embedding 
        # We use patch_size=1 or (Freq_Patch, 1) to keep Time resolution high
        # Input: (B, 1, 128, 60)
        # We want to compress 128 Freq bins -> 1 "Token" vector, but keep 60 Time steps.
        
        # Strategy: Use a stride on Frequency axis, but stride=1 on Time axis
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=(4, 1), stride=(4, 1)), # Compresses Freq only
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        # After this, Freq dim is 128/4 = 32. Time dim is 60.

        # 2. ConvMixer Blocks
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # Spatial Mixing (Freq and Time neighbors)
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1), # Channel Mixing
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)]
        )
        
        # 3. Final Projection to clean up Frequency dimension
        # We pool the remaining Frequency bins into a single vector per time step
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) # Output (B, dim, 1, T)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        
        x = self.patch_embed(x) # (B, dim, 32, 60)
        x = self.blocks(x)      # (B, dim, 32, 60)
        x = self.freq_pool(x)   # (B, dim, 1, 60)
        
        # Reshape for RWKV: (Batch, Time, Dim)
        x = x.squeeze(2).permute(0, 2, 1) 
        return x
    
class CNNRWKV(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),  # Stride here
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),  # Another stride
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )
        # self.bridge = nn.Linear(32 * 128, hidden_size)
        self.tokenizer = ConvMixerTokenizer(
            dim=hidden_size, 
            depth=8, 
            kernel_size=7
        )
        # --- 2. The "Brain" (RWKV Seq2Seq) ---
        self.rwkv = RWKV(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        )
    def forward(self, x, return_embedding=False):
        # Input x: (Batch, Freq=128, Time=60)
        tokens = self.tokenizer(x) # -> (B, 60, Hidden)
        # 1. Add Channel Dimension for CNN -> (B, 1, 128, 60)
        # if x.dim() == 3:
        #     x = x.unsqueeze(1)
            
        # # 2. Extract Spectral Features
        # # Output: (B, 64, 32, 60) -> (Batch, Channels, New_Freq, Time)
        # features = self.feature_extractor(x)
        
        # # 3. Prepare for RWKV
        # # We need (Batch, Time, Features)
        # # Permute to (B, Time, Channels, New_Freq) -> (B, 60, 64, 32)
        # features = features.permute(0, 3, 1, 2)
        
        # # Flatten the feature vector for each time step
        # B, T, C, F = features.shape
        # features = features.reshape(B, T, C * F) # (B, 60, 2048)
        
        # # Project to hidden size
        # features = self.bridge(features) # (B, 60, Hidden)
        

        # 4. Run RWKV
        seq_features = self.rwkv(tokens) 
        
        window_embedding = seq_features.mean(dim=1)
        return window_embedding


class RWKVScalogramModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.1, output_size=1, mode='freq_only'):
        super().__init__()
        
        # --- 1. The "Eye" (Feature Extractor) ---
        # Input: (Batch, 1, Freq=128, Time=60)
        # We use kernels like (kernel_freq, 1) to process Frequency ONLY, preserving Time.
        
        # self.feature_extractor = nn.Sequential(
        #     # Layer 1: Square Kernel (3x3) to capture 2D texture (artifacts vs breath)
        #     # We pad (1, 1) to keep dimensions consistent before stride
        #     nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)), 
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
            
        #     # Layer 2: Vertical Kernel (3, 1) to compress Frequency axis further
        #     nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 32, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),

        #     nn.Conv2d(64, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # After 2 strides of 2, Freq 128 becomes 32.
        # Channels are 64.
        # Total feature dimension = 32 * 64 = 2048.
        # We project this down to RWKV hidden size.
        self.encoder = CNNRWKV(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # self.rwkv = GRUModel(hidden_size, hidden_size, num_layers, dropout, bidirectional=True)
        # --- 3. NEW: Temporal Attention Pooling ---
        # self.temporal_attention = TemporalAttentionPooling(hidden_size, dropout=dropout)
        # self.rwkv = TransformerModel(hidden_size, hidden_size, num_layers, dropout, nhead=8)
        if mode == "freq_only":
            # --- 3. The Head ---
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, output_size),
            )


    def forward(self, x):
        window_embedding = self.encoder(x)
        
        # window_embedding = seq_features.mean(dim=1)  # (B, Hidden)
        # 5. Apply Temporal Attention Pooling (REPLACED global average pooling)
        # window_embedding, attn_weights = self.temporal_attention(seq_features)  # (B, Hidden)
        # Predict RR for the window
        out = self.head(window_embedding)
        # 5. Apply Head
        # out = self.head(seq_features)
        
        # return out.squeeze(-1)
        return out