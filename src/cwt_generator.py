import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

class PyTorchCWT(nn.Module):
    def __init__(self, fs=125, num_scales=128, fmin=0.1, fmax=0.8, wavelet='morl'):
        super().__init__()
        
        # 1. Calculate Scales (Same logic as your CPU function)
        dt = 1.0 / fs
        fc = pywt.central_frequency(wavelet)
        scale_min = fc / (fmax * dt)
        scale_max = fc / (fmin * dt)
        scales = np.linspace(scale_min, scale_max, num_scales)
        
        # 2. Generate Wavelet Filters (Kernels)
        # We find the widest wavelet (lowest freq) to determine kernel size
        max_len = int(10 * scale_max) # Heuristic for Morlet support
        if max_len % 2 == 0: max_len += 1 # Make odd for centering
        
        filters = []
        for scale in scales:
            # Generate discrete wavelet function
            # Range: [-5*scale, 5*scale] covers 99.9% of energy
            t = np.arange(-max_len//2, max_len//2 + 1) * dt
            
            # Morlet formula (Simplified real component for energy detection)
            # Psi(t) = exp(-t^2/2) * cos(5t) (approx)
            # We use pywt's internal generation to be safe, or analytical formula:
            norm_factor = 1 / (np.sqrt(scale) * np.pi**0.25)
            scaled_t = t / scale
            wavelet_data = norm_factor * np.exp(-0.5 * scaled_t**2) * np.cos(2 * np.pi * fc * scaled_t)
            
            filters.append(wavelet_data)
            
        # 3. Stack into Weights for Conv1d
        # Shape: (Out_Channels, In_Channels, Kernel_Size)
        filters = np.stack(filters).astype(np.float32)
        filters = torch.from_numpy(filters).unsqueeze(1)
        
        # 4. Create Conv1d Layer
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_scales,
            kernel_size=max_len,
            padding=max_len//2, # 'Same' padding
            bias=False
        )
        
        # Load weights and freeze
        self.conv.weight.data = filters
        self.conv.weight.requires_grad = False
        
    def forward(self, x, target_time=60):
        """
        x: Input tensor (Batch, Time_Samples) -> e.g., (B, 7500)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dim -> (B, 1, 7500)
            
        # 1. Convolve (Compute CWT)
        # Output: (B, Freq_Bins, Time_Samples) -> (B, 128, 7500)
        cwt_out = self.conv(x)
        
        # 2. Magnitude
        scalogram = torch.abs(cwt_out)
        
        # 3. Resize (Downsample Time)
        # --- FIX IS HERE ---
        # We use 'linear' because we are treating Freq_Bins as channels 
        # and only resizing the Time Length.
        scalogram = F.interpolate(
            scalogram, 
            size=target_time,   # Integer (60)
            mode='linear',      # <--- CHANGED FROM 'bilinear'
            align_corners=False
        )
        
        # 4. Fixed Scaling
        FIXED_MAX_VAL = 6.0
        scalogram = scalogram / FIXED_MAX_VAL
        scalogram = torch.clamp(scalogram, 0.0, 1.0)
        
        return scalogram