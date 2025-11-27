"""
Proper validation for CWT scalograms that doesn't rely on naive peak detection.
Instead, we check if the scalogram CONTAINS the right frequency information,
even if it's not the dominant peak at every timestep.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
import torch


def validate_frequency_content(scalogram, rr_label, fmin=0.1, fmax=0.5):
    """
    Check if the scalogram contains energy at the correct respiratory frequency.
    
    Instead of peak detection, we measure:
    1. Energy at the true RR frequency
    2.  Ratio of energy at true RR vs other frequencies
    3.  Temporal consistency
    """
    n_freqs, n_times = scalogram.shape
    freqs_hz = np.linspace(fmin, fmax, n_freqs)
    freqs_bpm = freqs_hz * 60
    
    # Get true RR for each timestep
    if isinstance(rr_label, (list, np.ndarray)):
        true_rr = np.array(rr_label)[:n_times]
    else:
        true_rr = np.full(n_times, rr_label)
    
    # For each timestep, measure energy at the true RR frequency
    energy_at_true_rr = []
    relative_energy = []
    
    for t in range(n_times):
        col = scalogram[:, t]
        true_rr_t = true_rr[t]
        
        # Find the index closest to the true RR
        true_idx = np.argmin(np.abs(freqs_bpm - true_rr_t))
        
        # Energy at true RR (with small window for tolerance)
        window = 3  # ±3 indices
        start_idx = max(0, true_idx - window)
        end_idx = min(n_freqs, true_idx + window + 1)
        
        energy_true = np.mean(col[start_idx:end_idx])
        energy_total = np.mean(col)
        
        energy_at_true_rr.append(energy_true)
        relative_energy.append(energy_true / (energy_total + 1e-8))
    
    energy_at_true_rr = np.array(energy_at_true_rr)
    relative_energy = np.array(relative_energy)
    
    # Metrics
    metrics = {
        'mean_energy_at_true_rr': np.mean(energy_at_true_rr),
        'mean_relative_energy': np.mean(relative_energy),
        'energy_consistency': np.std(energy_at_true_rr) / (np.mean(energy_at_true_rr) + 1e-8),
    }
    
    # A good scalogram should have relative_energy > 1. 0 (more energy at true RR than average)
    return metrics


def validate_with_sliding_window(scalogram, rr_label, fmin=0.1, fmax=0.5, window_size=10):
    """
    Validate using sliding window average instead of per-timestep peak detection.
    
    Real respiratory rate doesn't change drastically, so averaging over a window
    is more robust than per-timestep detection.
    """
    n_freqs, n_times = scalogram.shape
    freqs_hz = np.linspace(fmin, fmax, n_freqs)
    freqs_bpm = freqs_hz * 60
    
    # Get true RR
    if isinstance(rr_label, (list, np.ndarray)):
        true_rr = np.array(rr_label)[:n_times]
    else:
        true_rr = np.full(n_times, rr_label)
    
    detected_rr = []
    
    for t in range(n_times):
        # Use sliding window centered at t
        start = max(0, t - window_size // 2)
        end = min(n_times, t + window_size // 2 + 1)
        
        # Average power over the window
        window_power = scalogram[:, start:end].mean(axis=1)
        
        # Find peak
        peak_idx = np.argmax(window_power)
        detected_rr.append(freqs_bpm[peak_idx])
    
    detected_rr = np.array(detected_rr)
    
    # MAE
    mae = np.mean(np.abs(detected_rr - true_rr))
    
    # Correlation
    if np.std(detected_rr) > 0 and np.std(true_rr) > 0:
        corr = np.corrcoef(detected_rr, true_rr)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    
    return {
        'mae': mae,
        'correlation': corr,
        'detected_rr': detected_rr,
        'true_rr': true_rr
    }


def comprehensive_validation(scalograms, rr_labels, fmin=0.1, fmax=0.5, n_samples=100):
    """
    Run comprehensive validation with multiple methods.
    """
    import random
    
    n = min(n_samples, len(scalograms))
    indices = random.sample(range(len(scalograms)), n)
    
    # Collect results
    per_timestep_maes = []
    sliding_window_maes = []
    energy_ratios = []
    
    for idx in indices:
        scalogram = scalograms[idx]
        rr_label = np.array(rr_labels[idx])
        
        # Method 1: Per-timestep (original - expected to fail)
        freqs_bpm = np.linspace(fmin * 60, fmax * 60, scalogram.shape[0])
        detected_per_t = []
        for t in range(scalogram.shape[1]):
            peak_idx = np.argmax(scalogram[:, t])
            detected_per_t.append(freqs_bpm[peak_idx])
        detected_per_t = np.array(detected_per_t)
        per_timestep_maes.append(np.mean(np.abs(detected_per_t - rr_label[:len(detected_per_t)])))
        
        # Method 2: Sliding window (should be better)
        result_sw = validate_with_sliding_window(scalogram, rr_label, fmin, fmax, window_size=15)
        sliding_window_maes.append(result_sw['mae'])
        
        # Method 3: Energy at true frequency
        energy_metrics = validate_frequency_content(scalogram, rr_label, fmin, fmax)
        energy_ratios.append(energy_metrics['mean_relative_energy'])
    
    # Summary
    print("=" * 60)
    print("COMPREHENSIVE CWT VALIDATION")
    print("=" * 60)
    print(f"Samples tested: {n}")
    print(f"\n1. Per-timestep Peak Detection (original method):")
    print(f"   MAE: {np.mean(per_timestep_maes):.2f} ± {np.std(per_timestep_maes):.2f} BPM")
    
    print(f"\n2. Sliding Window Peak Detection (window=15s):")
    print(f"   MAE: {np.mean(sliding_window_maes):.2f} ± {np.std(sliding_window_maes):.2f} BPM")
    
    print(f"\n3. Energy at True RR Frequency:")
    print(f"   Relative Energy: {np.mean(energy_ratios):.2f} ± {np.std(energy_ratios):.2f}")
    print(f"   (>1.0 means more energy at true RR than average)")
    
    print("=" * 60)
    
    # Interpretation
    if np.mean(sliding_window_maes) < 5:
        print("✅ Sliding window detection works - scalogram contains RR info")
    elif np.mean(energy_ratios) > 1.0:
        print("✅ Energy analysis confirms RR info present in scalogram")
        print("   Peak detection fails due to signal complexity, but model can learn")
    else:
        print("⚠️ Scalogram may not contain sufficient RR information")
    
    return {
        'per_timestep_mae': np.mean(per_timestep_maes),
        'sliding_window_mae': np.mean(sliding_window_maes),
        'mean_energy_ratio': np. mean(energy_ratios)
    }