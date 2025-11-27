import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

# class PyTorchCWT(nn.Module):
#     def __init__(self, fs=125, num_scales=128, fmin=0.1, fmax=0.8, wavelet='morl'):
#         super().__init__()
        
#         dt = 1.0 / fs
#         fc = pywt.central_frequency(wavelet)
        
#         scale_min = fc / (fmax * dt)
#         scale_max = fc / (fmin * dt)
#         scales = np.linspace(scale_min, scale_max, num_scales)
        
#         # Kernel size (in Samples)
#         max_len = int(10 * scale_max) 
#         if max_len % 2 == 0: max_len += 1 
#         self.padding_size = max_len // 2
        
#         filters = []
#         for scale in scales:
#             # --- THE FIX: Work in Sample Space (Indices), NOT Seconds ---
#             # t goes from -Window/2 to +Window/2 in integer steps.
#             # We do NOT multiply by dt here.
#             t = np.arange(-max_len//2, max_len//2 + 1)
            
#             # Definition: exp(-x^2/2) * cos(5x)
#             # x is dimensionless: samples / scale
#             x_val = t / scale
            
#             wavelet_data = np.exp(-0.5 * x_val**2) * np.cos(5 * x_val)
            
#             # L1 Normalization (1/scale)
#             # This ensures amplitude consistency across frequencies
#             norm = 1.0 / scale
#             wavelet_data = norm * wavelet_data
            
#             # Zero Mean
#             wavelet_data = wavelet_data - np.mean(wavelet_data)
            
#             filters.append(wavelet_data)
            
#         filters = np.stack(filters).astype(np.float32)
#         filters = torch.from_numpy(filters).unsqueeze(1)
        
#         self.conv = nn.Conv1d(
#             in_channels=1,
#             out_channels=num_scales,
#             kernel_size=max_len,
#             padding=0, 
#             bias=False
#         )
#         self.conv.weight.data = filters
#         self.conv.weight.requires_grad = False
        
#     def forward(self, x, target_time=60):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
            
#         # 1. Reflection Padding
#         x_padded = F.pad(x, (self.padding_size, self.padding_size), mode='reflect')
        
#         # 2. Convolve
#         cwt_out = self.conv(x_padded)
#         scalogram = torch.abs(cwt_out)
        
#         # 3. Resize / Pooling
#         original_device = scalogram.device
#         if x.device.type == 'mps': scalogram = scalogram.cpu()
        
#         scalogram = F.adaptive_avg_pool1d(scalogram, target_time)
        
#         if x.device.type == 'mps': scalogram = scalogram.to(original_device)
        
#         return scalogram



class PyTorchCWT(nn.Module):
    def __init__(self, fs=125, num_scales=128, fmin=0.1, fmax=0.5, wavelet='morl'):
        super().__init__()
        
        self.fmin = fmin
        self.fmax = fmax
        
        dt = 1.0 / fs
        fc = pywt.central_frequency(wavelet)
        
        scale_min = fc / (fmax * dt)
        scale_max = fc / (fmin * dt)
        scales = np.linspace(scale_min, scale_max, num_scales)
        
        # Kernel size (in samples)
        max_len = int(10 * scale_max)
        if max_len % 2 == 0:
            max_len += 1
        self.padding_size = max_len // 2
        
        filters = []
        for scale in scales:
            t = np.arange(-max_len // 2, max_len // 2 + 1)
            x_val = t / scale
            
            wavelet_data = np.exp(-0.5 * x_val**2) * np.cos(5 * x_val)
            norm = 1.0 / scale
            wavelet_data = norm * wavelet_data
            wavelet_data = wavelet_data - np.mean(wavelet_data)
            
            filters. append(wavelet_data)
        
        filters = np.stack(filters). astype(np.float32)
        filters = torch.from_numpy(filters). unsqueeze(1)
        
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_scales,
            kernel_size=max_len,
            padding=0,
            bias=False
        )
        self.conv. weight.data = filters
        self.conv.weight. requires_grad = False
    
    def forward(self, x, target_time=60):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 1. Reflection padding
        x_padded = F.pad(x, (self.padding_size, self.padding_size), mode='reflect')
        
        # 2.  Convolve
        cwt_out = self.conv(x_padded)
        scalogram = torch.abs(cwt_out)
        
        # 3. Temporal pooling
        original_device = scalogram.device
        if x.device. type == 'mps':
            scalogram = scalogram. cpu()
        
        scalogram = F.adaptive_avg_pool1d(scalogram, target_time)
        
        if x.device.type == 'mps':
            scalogram = scalogram.to(original_device)
        
        return scalogram
    
# def compute_freq_features_gpu(ppg_segments, fs=125, batch_size=500, device='cuda',
#                                fmin=0.1, fmax=0.5, normalization='per_column'):
#     """
#     Compute CWT scalograms on GPU with improved normalization.
    
#     CHANGES:
#     1. fmax: 0.8 -> 0. 5 (respiratory only)
#     2.  normalization: 'per_column' for peak detection, 'quantile' for global patterns
    
#     Args:
#         ppg_segments: List of 1D PPG arrays
#         fs: Sampling frequency
#         batch_size: GPU batch size
#         device: 'cuda' or 'cpu'
#         fmin: Minimum frequency (Hz)
#         fmax: Maximum frequency (Hz) - CHANGED to 0.5
#         normalization: 'per_column', 'quantile', or 'global'
    
#     Returns:
#         np.ndarray of shape (N, 128, 60)
#     """
#     if not ppg_segments:
#         return np.array([])
    
#     # Initialize model with CORRECT frequency range
#     cwt_model = PyTorchCWT(
#         fs=fs, 
#         num_scales=128, 
#         fmin=fmin, 
#         fmax=fmax  # Now 0.5 by default
#     ).to(device)
#     cwt_model.eval()
    
#     # Input Z-Score normalization
#     input_tensor = torch.tensor(np.stack(ppg_segments), dtype=torch.float32)
#     mu = input_tensor.mean(dim=1, keepdim=True)
#     std = input_tensor.std(dim=1, keepdim=True)
#     input_tensor = (input_tensor - mu) / torch.clamp(std, min=0.001)
#     input_tensor = input_tensor.to(device)
    
#     all_scalograms = []
    
#     with torch.no_grad():
#         for i in range(0, len(input_tensor), batch_size):
#             batch = input_tensor[i:i + batch_size]
            
#             # Compute raw CWT
#             raw_scalograms = cwt_model(batch, target_time=60)  # (B, 128, 60)
            
#             # Apply normalization based on method
#             if normalization == 'per_column':
#                 # Per-column (per-timestep) normalization
#                 # This is BEST for peak detection at each timestep
#                 scalograms = normalize_per_column(raw_scalograms)
                
#             elif normalization == 'quantile':
#                 # Quantile normalization per image (your original method)
#                 scalograms = normalize_quantile(raw_scalograms)
                
#             elif normalization == 'global':
#                 # Simple min-max per image
#                 scalograms = normalize_global(raw_scalograms)
                
#             else:
#                 raise ValueError(f"Unknown normalization: {normalization}")
            
#             all_scalograms.append(scalograms.cpu().numpy())
    
#     return np.concatenate(all_scalograms, axis=0)


def normalize_per_column(scalograms):
    """
    Normalize each time column independently.
    
    This ensures that peak detection works at each timestep,
    regardless of overall signal amplitude.
    
    Args:
        scalograms: (B, F, T) tensor
        
    Returns:
        (B, F, T) normalized tensor
    """
    B, F, T = scalograms.shape
    
    # Compute min/max along frequency axis for each timestep
    col_min = scalograms.min(dim=1, keepdim=True)[0]  # (B, 1, T)
    col_max = scalograms.max(dim=1, keepdim=True)[0]  # (B, 1, T)
    
    # Avoid division by zero
    col_range = col_max - col_min
    col_range = torch.clamp(col_range, min=1e-8)
    
    # Normalize
    normalized = (scalograms - col_min) / col_range
    
    return normalized


def normalize_quantile(scalograms):
    """
    Quantile normalization per image (original method).
    
    Good for preserving relative patterns across time,
    but can make peak detection unreliable.
    
    Args:
        scalograms: (B, F, T) tensor
        
    Returns:
        (B, F, T) normalized tensor
    """
    B = scalograms.shape[0]
    flat = scalograms.view(B, -1)
    
    q_min = torch.quantile(flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
    q_max = torch.quantile(flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)
    
    normalized = (scalograms - q_min) / (q_max - q_min + 1e-8)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    return normalized


def normalize_global(scalograms):
    """
    Simple min-max normalization per image.
    
    Args:
        scalograms: (B, F, T) tensor
        
    Returns:
        (B, F, T) normalized tensor
    """
    B = scalograms.shape[0]
    flat = scalograms.view(B, -1)
    
    s_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
    s_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
    
    normalized = (scalograms - s_min) / (s_max - s_min + 1e-8)
    
    return normalized


# =============================================================================
# VALIDATION FUNCTION - Test that peak detection works
# =============================================================================

def validate_cwt_peak_detection(ppg_segments, rr_labels, fs=125, device='cuda',
                                 fmin=0.1, fmax=0.5, n_samples=50):
    """
    Validate that CWT peak detection matches RR labels.
    
    This should be run after fixing the CWT to confirm the fix works.
    """
    import random
    
    # Generate scalograms with the IMPROVED settings
    scalograms = compute_freq_features_gpu(
        ppg_segments, 
        fs=fs, 
        device=device,
        fmin=fmin,
        fmax=fmax,
        normalization='per_column'  # Use per-column for peak detection
    )
    
    # Sample random indices
    n = min(n_samples, len(scalograms))
    indices = random.sample(range(len(scalograms)), n)
    
    # Frequency axis
    freqs_bpm = np.linspace(fmin * 60, fmax * 60, scalograms.shape[1])
    
    all_maes = []
    all_corrs = []
    
    for idx in indices:
        scalogram = scalograms[idx]  # (128, 60)
        rr_label = np.array(rr_labels[idx])  # (60,)
        
        # Peak detection at each timestep
        detected_rr = []
        for t in range(scalogram.shape[1]):
            peak_idx = np.argmax(scalogram[:, t])
            detected_rr.append(freqs_bpm[peak_idx])
        
        detected_rr = np.array(detected_rr)
        
        # Metrics
        mae = np.mean(np.abs(detected_rr - rr_label))
        
        if np.std(detected_rr) > 0 and np.std(rr_label) > 0:
            corr = np.corrcoef(detected_rr, rr_label)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        all_maes.append(mae)
        all_corrs.append(corr)
    
    # Summary
    print("=" * 60)
    print("IMPROVED CWT VALIDATION")
    print("=" * 60)
    print(f"Frequency range: {fmin}-{fmax} Hz ({fmin*60:.0f}-{fmax*60:.0f} BPM)")
    print(f"Normalization: per_column")
    print(f"Samples tested: {n}")
    print(f"Mean MAE: {np.mean(all_maes):.2f} ± {np.std(all_maes):.2f} BPM")
    print(f"Mean Correlation: {np.mean(all_corrs):.3f} ± {np.std(all_corrs):.3f}")
    print("=" * 60)
    
    if np.mean(all_maes) < 3.0:
        print("✅ EXCELLENT: Peak detection MAE < 3 BPM")
    elif np.mean(all_maes) < 5.0:
        print("✅ GOOD: Peak detection MAE < 5 BPM")
    else:
        print("⚠️ Still needs improvement")
    
    return {
        'maes': all_maes,
        'correlations': all_corrs,
        'mean_mae': np.mean(all_maes),
        'mean_corr': np.mean(all_corrs)
    }