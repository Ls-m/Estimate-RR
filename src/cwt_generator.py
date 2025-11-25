import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

class PyTorchCWT(nn.Module):
    def __init__(self, fs=125, num_scales=128, fmin=0.1, fmax=0.8, wavelet='morl'):
        super().__init__()
        
        dt = 1.0 / fs
        fc = pywt.central_frequency(wavelet)
        
        scale_min = fc / (fmax * dt)
        scale_max = fc / (fmin * dt)
        scales = np.linspace(scale_min, scale_max, num_scales)
        
        # Kernel size (in Samples)
        max_len = int(10 * scale_max) 
        if max_len % 2 == 0: max_len += 1 
        self.padding_size = max_len // 2
        
        filters = []
        for scale in scales:
            # --- THE FIX: Work in Sample Space (Indices), NOT Seconds ---
            # t goes from -Window/2 to +Window/2 in integer steps.
            # We do NOT multiply by dt here.
            t = np.arange(-max_len//2, max_len//2 + 1)
            
            # Definition: exp(-x^2/2) * cos(5x)
            # x is dimensionless: samples / scale
            x_val = t / scale
            
            wavelet_data = np.exp(-0.5 * x_val**2) * np.cos(5 * x_val)
            
            # L1 Normalization (1/scale)
            # This ensures amplitude consistency across frequencies
            norm = 1.0 / scale
            wavelet_data = norm * wavelet_data
            
            # Zero Mean
            wavelet_data = wavelet_data - np.mean(wavelet_data)
            
            filters.append(wavelet_data)
            
        filters = np.stack(filters).astype(np.float32)
        filters = torch.from_numpy(filters).unsqueeze(1)
        
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_scales,
            kernel_size=max_len,
            padding=0, 
            bias=False
        )
        self.conv.weight.data = filters
        self.conv.weight.requires_grad = False
        
    def forward(self, x, target_time=60):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 1. Reflection Padding
        x_padded = F.pad(x, (self.padding_size, self.padding_size), mode='reflect')
        
        # 2. Convolve
        cwt_out = self.conv(x_padded)
        scalogram = torch.abs(cwt_out)
        
        # 3. Resize / Pooling
        original_device = scalogram.device
        if x.device.type == 'mps': scalogram = scalogram.cpu()
        
        scalogram = F.adaptive_avg_pool1d(scalogram, target_time)
        
        if x.device.type == 'mps': scalogram = scalogram.to(original_device)
        
        return scalogram