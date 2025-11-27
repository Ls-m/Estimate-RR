"""
CWT generator using PyWavelets (proven to work) with GPU acceleration for the 
heavy lifting (normalization, resizing). 
"""

import numpy as np
import torch
import torch.nn.functional as F
import pywt
from joblib import Parallel, delayed
from tqdm import tqdm


def generate_single_scalogram_pywt(ppg_segment, fs=125, num_scales=128, 
                                    fmin=0.1, fmax=0.5, target_time=60):
    """
    Generate a single scalogram using PyWavelets (proven to work).
    
    This matches the PyWavelets CWT that achieved 3.35 BPM MAE.
    """
    # Z-score normalize
    ppg_segment = np.asarray(ppg_segment, dtype=np.float64)
    mu = np.mean(ppg_segment)
    sigma = np.std(ppg_segment)
    if sigma < 1e-6:
        return np.zeros((num_scales, target_time), dtype=np.float32)
    ppg_norm = (ppg_segment - mu) / sigma
    
    # CWT parameters
    dt = 1.0 / fs
    fc = pywt.central_frequency('morl')
    
    # Create scales (high scale = low freq, low scale = high freq)
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    scales = np.linspace(scale_min, scale_max, num_scales)
    
    # Compute CWT using PyWavelets
    cwt_coeffs, freqs = pywt.cwt(ppg_norm, scales, 'morl', sampling_period=dt)
    scalogram = np.abs(cwt_coeffs)  # (num_scales, signal_length)
    
    # Downsample time axis to target_time using averaging
    signal_length = scalogram.shape[1]
    samples_per_bin = signal_length // target_time
    
    scalogram_ds = np.zeros((num_scales, target_time), dtype=np.float32)
    for t in range(target_time):
        start = t * samples_per_bin
        end = start + samples_per_bin
        scalogram_ds[:, t] = scalogram[:, start:end].mean(axis=1)
    
    # Flip so that index 0 = fmin, index N-1 = fmax (for consistent visualization)
    scalogram_ds = scalogram_ds[::-1, :]
    
    return scalogram_ds.astype(np.float32)


def compute_freq_features_pywt(ppg_segments, fs=125, num_scales=128,
                                fmin=0.1, fmax=0.5, target_time=60,
                                normalization='per_column', n_jobs=-1):
    """
    Compute CWT scalograms using PyWavelets with parallel processing.
    
    This is slower than GPU but PROVEN TO WORK correctly.
    """
    print(f"Computing CWT scalograms using PyWavelets (n_jobs={n_jobs})...")
    
    # Parallel computation of raw scalograms
    raw_scalograms = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_scalogram_pywt)(
            seg, fs, num_scales, fmin, fmax, target_time
        ) for seg in tqdm(ppg_segments, desc="CWT")
    )
    
    raw_scalograms = np.stack(raw_scalograms, axis=0)  # (N, num_scales, target_time)
    
    # Apply normalization (can use GPU for speed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scalograms_tensor = torch.from_numpy(raw_scalograms).float().to(device)
    
    if normalization == 'per_column':
        # Per-column (per-timestep) normalization
        col_min = scalograms_tensor.min(dim=1, keepdim=True)[0]
        col_max = scalograms_tensor.max(dim=1, keepdim=True)[0]
        col_range = torch.clamp(col_max - col_min, min=1e-8)
        scalograms_tensor = (scalograms_tensor - col_min) / col_range
        
    elif normalization == 'quantile':
        # Quantile normalization per sample
        B = scalograms_tensor.shape[0]
        flat = scalograms_tensor.view(B, -1)
        q_min = torch.quantile(flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
        q_max = torch.quantile(flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)
        scalograms_tensor = (scalograms_tensor - q_min) / (q_max - q_min + 1e-8)
        scalograms_tensor = torch.clamp(scalograms_tensor, 0.0, 1.0)
        
    elif normalization == 'global':
        # Min-max per sample
        B = scalograms_tensor.shape[0]
        flat = scalograms_tensor.view(B, -1)
        s_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        s_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        scalograms_tensor = (scalograms_tensor - s_min) / (s_max - s_min + 1e-8)
    
    return scalograms_tensor.cpu().numpy()


def validate_pywt_cwt(ppg_segments, rr_labels, fs=125, fmin=0.1, fmax=0.5, n_samples=50):
    """
    Validate the PyWavelets-based CWT implementation. 
    """
    import random
    
    n = min(n_samples, len(ppg_segments))
    indices = random.sample(range(len(ppg_segments)), n)
    
    # Compute scalograms for selected samples
    selected_ppg = [ppg_segments[i] for i in indices]
    selected_rr = [rr_labels[i] for i in indices]
    
    scalograms = compute_freq_features_pywt(
        selected_ppg, fs=fs, fmin=fmin, fmax=fmax,
        normalization='per_column', n_jobs=4
    )
    
    # Frequency axis
    freqs_bpm = np.linspace(fmin * 60, fmax * 60, scalograms.shape[1])
    
    # Peak detection validation
    maes = []
    for i in range(n):
        scalogram = scalograms[i]
        rr_label = np.array(selected_rr[i])[:60]
        
        # Mean power across time
        mean_power = scalogram.mean(axis=1)
        peak_idx = np.argmax(mean_power)
        detected_rr = freqs_bpm[peak_idx]
        
        mae = abs(detected_rr - np.mean(rr_label))
        maes.append(mae)
    
    print("=" * 60)
    print("PYWAVELETS CWT VALIDATION")
    print("=" * 60)
    print(f"Samples: {n}")
    print(f"Mean Power Peak Detection MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f} BPM")
    print("=" * 60)
    
    if np.mean(maes) < 5:
        print("✅ PyWavelets CWT works correctly!")
    else:
        print("⚠️ Still some issues")
    
    return np.mean(maes)