# import torch
# import torch.nn as nn
# from torch.utils.cpp_extension import load
# import os


# conda_env_path = os.getenv("CONDA_PREFIX")

# # Check if CUDA_HOME is already set, if not, set it to the conda env path
# if "CUDA_HOME" not in os.environ and conda_env_path:
#     print(f"Setting CUDA_HOME to: {conda_env_path}")
#     os.environ["CUDA_HOME"] = conda_env_path
# # --- Step 1: Compile the CUDA Kernel On-the-Fly ---
# # This will compile the kernel the first time it's run and cache it.
# # Make sure the paths to your .cpp and .cu files are correct.
# wkv_cuda = load(
#     name="wkv",
#     sources=["src/wkv_op.cpp", "src/wkv_cuda.cu"],
#     verbose=True,
#     extra_cuda_cflags=[
#         "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
#         "--extra-device-vectorization"
#     ]
# )

# # --- Step 2: Create a PyTorch Function to Call the Kernel ---
# class WKV(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, B, T, C, w, u, k, v):
#         ctx.B, ctx.T, ctx.C = B, T, C
#         ctx.save_for_backward(w, u, k, v)
#         y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
#         wkv_cuda.forward(B, T, C, w, u, k, v, y)
#         return y

#     @staticmethod
#     def backward(ctx, gy):
#         B, T, C = ctx.B, ctx.T, ctx.C
#         w, u, k, v = ctx.saved_tensors
#         gw = torch.zeros((B, C), device=gy.device)
#         gu = torch.zeros((B, C), device=gy.device)
#         gk = torch.zeros((B, T, C), device=gy.device)
#         gv = torch.zeros((B, T, C), device=gy.device)
#         wkv_cuda.backward(B, T, C, w, u, k, v, gy, gw, gu, gk, gv)
#         gw = torch.sum(gw, dim=0)
#         gu = torch.sum(gu, dim=0)
#         return (None, None, None, gw, gu, gk, gv)

# # --- Step 3: Create the Fast nn.Module ---
# # This is your new, fast, drop-in replacement module.
# class RWKV_TimeMix_CUDA(nn.Module):
#     def __init__(self, embed_size):
#         super().__init__()
#         self.time_decay = nn.Parameter(torch.zeros(embed_size))
#         self.time_first = nn.Parameter(torch.zeros(embed_size))
        
#         self.key = nn.Linear(embed_size, embed_size, bias=False)
#         self.value = nn.Linear(embed_size, embed_size, bias=False)
#         self.receptance = nn.Linear(embed_size, embed_size, bias=False)

#     def forward(self, x):
#         B, T, C = x.shape
        
#         # Calculate time-decay and bonus
#         w = -torch.exp(self.time_decay)
#         u = self.time_first

#         # Calculate k, v, r
#         k = self.key(x)
#         v = self.value(x)
#         r = torch.sigmoid(self.receptance(x))
        
#         # Call the CUDA kernel
#         wkv = WKV.apply(B, T, C, w, u, k, v)
        
#         return r * wkv
    
# class RWKV_ChannelMix(nn.Module):
#     def __init__(self, embed_size):
#         super().__init__()
#         hidden_size = 4 * embed_size
        
#         # Linear layers for key and receptance
#         self.key = nn.Linear(embed_size, hidden_size, bias=False)
#         self.receptance = nn.Linear(embed_size, embed_size, bias=False)
#         self.value = nn.Linear(hidden_size, embed_size, bias=False)
        
#     def forward(self, x):
#         # Key is processed with a non-linearity (Mish is a good choice)
#         k = torch.square(F.relu(self.key(x)))
#         kv = self.value(k)
        
#         # Mix with receptance
#         r = torch.sigmoid(self.receptance(x))
        
#         return r * kv
    
# class RWKV_Block(nn.Module):
#     def __init__(self, embed_size):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(embed_size)
#         self.ln2 = nn.LayerNorm(embed_size)
#         self.attn = RWKV_TimeMix_CUDA(embed_size)
#         self.ffn = RWKV_ChannelMix(embed_size)
        
#     def forward(self, x):
#         # Time Mixing with residual connection
#         x = x + self.attn(self.ln1(x))
#         # Channel Mixing with residual connection
#         x = x + self.ffn(self.ln2(x))
#         return x
    
# class RWKVTimeModelOPT(nn.Module):
#     def __init__(self, input_size=1, embed_size=64, output_size=64, num_layers=2, dropout=0.2):
#         super().__init__()
        
#         # Project the single PPG feature to the model's embedding dimension
#         self.embed = nn.Linear(input_size, embed_size)
        
#         # Stack of RWKV blocks
#         self.blocks = nn.Sequential(*[RWKV_Block(embed_size) for _ in range(num_layers)])
        
#         # Final normalization and output head
#         self.ln_out = nn.LayerNorm(embed_size)
#         self.head = nn.Linear(embed_size, output_size, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # 1. Embed the input
#         x = self.embed(x)
        
#         # 2. Pass through RWKV blocks
#         x = self.blocks(x)
        
#         # 3. Final normalization
#         x = self.ln_out(x)
        
#         # 4. Take the output of the LAST time step (same as your LSTM)
#         x = x[:, -1, :]
        
#         # 5. Apply dropout and final linear head
#         x = self.dropout(x)
#         x = self.head(x)
        
#         return x