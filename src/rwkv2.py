import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKV_TimeMix(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.time_decay = nn.Parameter(torch.zeros(embed_size))
        self.time_first = nn.Parameter(torch.zeros(embed_size))
        
        # Linear layers for key, value, and receptance
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.receptance = nn.Linear(embed_size, embed_size, bias=False)
        
    def forward(self, x):
        # Get input shape: (Batch, Sequence Length, Embedding Size)
        B, T, C = x.shape
        
        # Calculate key, value, and receptance
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        
        # Time-decay and first-token bonus
        w = -torch.exp(self.time_decay) # Decay factor, ensures it's negative
        u = self.time_first             # Bonus for the first token

        # Initialize hidden state
        state = torch.zeros(B, C, device=x.device)
        outputs = []
        
        # Iterate over each time step (like an RNN)
        for t in range(T):
            kt, vt, rt = k[:, t, :], v[:, t, :], r[:, t, :]
            
            # The core RWKV state update logic
            wkv = (state * u) + kt if t == 0 else (state * w) + kt
            rwkv = rt * wkv
            
            # Update the state for the next time step
            state = (state * w) + kt
            
            outputs.append(self.value(rwkv))
            
        # Stack the outputs for all time steps
        return torch.stack(outputs, dim=1)
    
class RWKV_ChannelMix(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        hidden_size = 4 * embed_size
        
        # Linear layers for key and receptance
        self.key = nn.Linear(embed_size, hidden_size, bias=False)
        self.receptance = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(hidden_size, embed_size, bias=False)
        
    def forward(self, x):
        # Key is processed with a non-linearity (Mish is a good choice)
        k = torch.square(F.relu(self.key(x)))
        kv = self.value(k)
        
        # Mix with receptance
        r = torch.sigmoid(self.receptance(x))
        
        return r * kv
    
class RWKV_Block(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = RWKV_TimeMix(embed_size)
        self.ffn = RWKV_ChannelMix(embed_size)
        
    def forward(self, x):
        # Time Mixing with residual connection
        x = x + self.attn(self.ln1(x))
        # Channel Mixing with residual connection
        x = x + self.ffn(self.ln2(x))
        return x
    
class RWKVTimeModel(nn.Module):
    def __init__(self, input_size=1, embed_size=64, output_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Project the single PPG feature to the model's embedding dimension
        self.embed = nn.Linear(input_size, embed_size)
        
        # Stack of RWKV blocks
        self.blocks = nn.Sequential(*[RWKV_Block(embed_size) for _ in range(num_layers)])
        
        # Final normalization and output head
        self.ln_out = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Embed the input
        x = self.embed(x)
        
        # 2. Pass through RWKV blocks
        x = self.blocks(x)
        
        # 3. Final normalization
        x = self.ln_out(x)
        
        # 4. Take the output of the LAST time step (same as your LSTM)
        x = x[:, -1, :]
        
        # 5. Apply dropout and final linear head
        x = self.dropout(x)
        x = self.head(x)
        
        return x