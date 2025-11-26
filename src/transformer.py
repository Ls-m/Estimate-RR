import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Standard Transformer Positional Encoding."""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Time, Batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CNNTransformerRegressor(nn.Module):
    def __init__(self, 
                 hidden_size=256, 
                 num_layers=4, 
                 nhead=4, 
                 dropout=0.1):
        super().__init__()
        
        # --- 1. CNN Feature Extractor ---
        # GOAL: Reduce Freq (128) to 1, but keep Time (60) EXACTLY 60.
        # We use MaxPool2d with kernel (Freq_Pool, 1) to only pool frequency.
        
        self.cnn = nn.Sequential(
            # Stage 1
            # In: (B, 1, 128, 60)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Freq 128 -> 64, Time stays 60
            
            # Stage 2
            # In: (B, 32, 64, 60)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Freq 64 -> 32
            
            # Stage 3
            # In: (B, 64, 32, 60)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(4, 1)), # Freq 32 -> 8
            
            # Stage 4
            # In: (B, 128, 8, 60)
            nn.Conv2d(128, hidden_size, kernel_size=(8, 1)), # Kernel covers remaining freq 8
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),
            # Out: (B, Hidden, 1, 60)
        )
        
        # --- 2. Transformer Encoder ---
        self.d_model = hidden_size
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=60, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size*4, 
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. Regression Head ---
        # Projects the transformer output back to a single value per time step
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1) # Output 1 value (the rate)
        )

    def forward(self, x):
        # Input x: (Batch, Freq=128, Time=60) or (Batch, 1, 128, 60)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # 1. Extract Features (CNN)
        # Output: (B, Hidden, 1, 60)
        features = self.cnn(x)
        
        # Remove the singleton Freq dimension -> (B, Hidden, 60)
        features = features.squeeze(2)
        
        # Permute for Transformer: (Time, Batch, Hidden)
        # This is critical: Transformer expects [Seq_Len, Batch, Dim]
        features = features.permute(2, 0, 1) 
        
        # 2. Apply Transformer
        src = self.pos_encoder(features)
        # output: (Time=60, Batch, Hidden)
        memory = self.transformer(src)
        
        # 3. Regression Head
        # We apply the head to every time step individually
        # memory: (60, B, Hidden) -> permute back to (B, 60, Hidden)
        memory = memory.permute(1, 0, 2)
        
        # out: (B, 60, 1)
        out = self.head(memory)
        
        # Final Squeeze: (B, 60)
        return out.squeeze(-1)