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
        
        self.fs = fs
        self.num_scales = num_scales
        self.fmin = fmin
        self.fmax = fmax
        
        dt = 1.0 / fs
        fc = pywt.central_frequency(wavelet)
        
        # Scale is INVERSELY related to frequency
        # High frequency -> Low scale
        # Low frequency -> High scale
        scale_at_fmax = fc / (fmax * dt)  # Small scale for high freq
        scale_at_fmin = fc / (fmin * dt)  # Large scale for low freq
        
        # Create scales from HIGH freq to LOW freq (small to large scale)
        # This means output index 0 = fmax, index N-1 = fmin
        scales = np.linspace(scale_at_fmax, scale_at_fmin, num_scales)
        
        # Store the corresponding frequencies for each scale
        # freqs[i] = fc / (scales[i] * dt)
        self.register_buffer(
            'frequencies',
            torch.from_numpy(fc / (scales * dt)).float()
        )
        
        # Kernel size
        max_len = int(10 * scale_at_fmin)  # Based on largest scale
        if max_len % 2 == 0:
            max_len += 1
        self.padding_size = max_len // 2
        self.kernel_size = max_len
        
        # Build wavelet filters
        filters = []
        for scale in scales:
            t = np.arange(-max_len // 2, max_len // 2 + 1)
            x_val = t / scale
            
            # Morlet wavelet: exp(-x^2/2) * cos(5x)
            wavelet_data = np.exp(-0.5 * x_val**2) * np.cos(5 * x_val)
            
            # Normalize by 1/sqrt(scale) for energy preservation
            # (This is more correct than 1/scale)
            norm = 1.0 / np.sqrt(scale)
            wavelet_data = norm * wavelet_data
            
            # Zero mean
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
        """
        Args:
            x: (B, L) or (B, 1, L) - PPG signals
            target_time: Output time resolution
            
        Returns:
            (B, num_scales, target_time) - Scalogram
            Index 0 = highest frequency (fmax)
            Index N-1 = lowest frequency (fmin)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Reflection padding
        x_padded = F.pad(x, (self.padding_size, self.padding_size), mode='reflect')
        
        # Convolve
        cwt_out = self.conv(x_padded)
        scalogram = torch.abs(cwt_out)
        
        # IMPORTANT: Flip the frequency axis so that:
        # Index 0 = fmin (low frequency)
        # Index N-1 = fmax (high frequency)
        # This matches the intuitive ordering for visualization
        scalogram = torch.flip(scalogram, dims=[1])
        
        # Temporal pooling
        original_device = scalogram.device
        if x.device.type == 'mps':
            scalogram = scalogram.cpu()
        
        scalogram = F.adaptive_avg_pool1d(scalogram, target_time)
        
        if x.device.type == 'mps':
            scalogram = scalogram.to(original_device)
        
        return scalogram
    
    def get_frequency_axis(self):
        """
        Returns the frequency axis in Hz.
        After flipping, index 0 = fmin, index N-1 = fmax. 
        """
        return torch.flip(self.frequencies, dims=[0])


def compute_freq_features_gpu(ppg_segments, fs=125, batch_size=500, device='cuda',
                               fmin=0.1, fmax=0.5, normalization='per_column'):
    """
    Compute CWT scalograms on GPU with FIXED implementation.
    """
    if not ppg_segments:
        return np.array([])

    # Initialize FIXED model
    cwt_model = PyTorchCWT(
        fs=fs,
        num_scales=128,
        fmin=fmin,
        fmax=fmax
    ).to(device)
    cwt_model.eval()

    # Input Z-Score normalization
    input_tensor = torch.tensor(np.stack(ppg_segments), dtype=torch.float32)
    mu = input_tensor.mean(dim=1, keepdim=True)
    std = input_tensor.std(dim=1, keepdim=True)
    input_tensor = (input_tensor - mu) / torch.clamp(std, min=0.001)
    input_tensor = input_tensor.to(device)
    
    all_scalograms = []

    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i + batch_size]
            
            # Compute CWT
            raw_scalograms = cwt_model(batch, target_time=60)
            
            # Normalization
            if normalization == 'per_column':
                B, F, T = raw_scalograms.shape
                col_min = raw_scalograms.min(dim=1, keepdim=True)[0]
                col_max = raw_scalograms.max(dim=1, keepdim=True)[0]
                col_range = torch.clamp(col_max - col_min, min=1e-8)
                scalograms = (raw_scalograms - col_min) / col_range
                
            elif normalization == 'quantile':
                B_curr = raw_scalograms.shape[0]
                flat = raw_scalograms.view(B_curr, -1)
                q_min = torch.quantile(flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
                q_max = torch.quantile(flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)
                scalograms = (raw_scalograms - q_min) / (q_max - q_min + 1e-8)
                scalograms = torch.clamp(scalograms, 0.0, 1.0)
                
            elif normalization == 'global':
                B_curr = raw_scalograms.shape[0]
                flat = raw_scalograms.view(B_curr, -1)
                s_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
                s_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
                scalograms = (raw_scalograms - s_min) / (s_max - s_min + 1e-8)
            else:
                raise ValueError(f"Unknown normalization: {normalization}")
            
            all_scalograms.append(scalograms.cpu().numpy())

    return np.concatenate(all_scalograms, axis=0)


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

def quick_test_fixed_cwt():
    """
    Quick test to verify the fix works. 
    """
    import matplotlib.pyplot as plt
    
    # Create a synthetic signal with known frequency
    fs = 125
    duration = 60
    t = np.arange(0, duration, 1/fs)
    
    # Respiratory rate = 15 BPM = 0.25 Hz
    true_freq = 0.25
    true_bpm = 15
    
    # Simulate PPG with respiratory modulation
    ppg = np.sin(2 * np.pi * true_freq * t)  # Pure respiratory
    ppg += 0.1 * np.random.randn(len(ppg))  # Add noise
    
    # Z-score normalize
    ppg_norm = (ppg - np.mean(ppg)) / np.std(ppg)
    
    # Test PyTorch CWT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cwt_model = PyTorchCWT(fs=fs, num_scales=128, fmin=0.1, fmax=0.5).to(device)

    ppg_tensor = torch.tensor(ppg_norm, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        scalogram = cwt_model(ppg_tensor, target_time=60)[0].cpu().numpy()

    # Get frequency axis
    freq_axis = cwt_model.get_frequency_axis().cpu().numpy()
    freq_bpm = freq_axis * 60
    
    # Find peak
    mean_power = scalogram.mean(axis=1)
    peak_idx = np.argmax(mean_power)
    detected_bpm = freq_bpm[peak_idx]
    
    print(f"True RR: {true_bpm} BPM")
    print(f"Detected RR: {detected_bpm:.1f} BPM")
    print(f"Error: {abs(detected_bpm - true_bpm):.1f} BPM")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].imshow(scalogram, aspect='auto', origin='lower',
                   extent=[0, 60, freq_bpm[0], freq_bpm[-1]], cmap='viridis')
    axes[0].axhline(true_bpm, color='r', linestyle='--', label=f'True: {true_bpm} BPM')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('RR (BPM)')
    axes[0].set_title('Fixed PyTorch CWT Scalogram')
    axes[0].legend()
    
    axes[1].plot(freq_bpm, mean_power)
    axes[1].axvline(true_bpm, color='r', linestyle='--', label=f'True: {true_bpm} BPM')
    axes[1].axvline(detected_bpm, color='g', linestyle='--', label=f'Detected: {detected_bpm:.1f} BPM')
    axes[1].set_xlabel('RR (BPM)')
    axes[1].set_ylabel('Mean Power')
    axes[1].set_title('Frequency Profile')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('validation_plots/fixed_cwt_test.png', dpi=150)
    plt.close()
    
    print(f"Saved test plot to: validation_plots/fixed_cwt_test.png")
    
    return abs(detected_bpm - true_bpm) < 1.0  # Should be very close


if __name__ == "__main__":
    import os
    os.makedirs("validation_plots", exist_ok=True)
    
    success = quick_test_fixed_cwt()
    if success:
        print("\n✅ Fixed CWT implementation works correctly!")
    else:
        print("\n❌ Still needs debugging")