"""
Compare CWT peak detection vs PSD peak detection vs direct peak counting. 
This will help identify exactly where the CWT is failing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch, find_peaks
import pywt
import torch
import torch.nn.functional as F


def compare_frequency_methods(ppg_segment, rr_label, fs=125, subject_id="test"):
    """
    Compare different methods of extracting respiratory frequency.
    """
    print(f"\n{'='*70}")
    print(f"COMPARING METHODS - Subject: {subject_id}")
    print(f"{'='*70}")
    
    # Ground truth
    rr_mean = np.mean(rr_label)
    expected_freq = rr_mean / 60.0
    print(f"\nGround Truth: {rr_mean:.2f} BPM = {expected_freq:.3f} Hz")
    
    # =====================================================================
    # METHOD 1: Direct peak counting (gold standard)
    # =====================================================================
    sos = signal.butter(4, [0.1, 0.5], btype='band', fs=fs, output='sos')
    ppg_filtered = signal.sosfiltfilt(sos, ppg_segment)
    
    min_distance = int(2 * fs)  # Minimum 2 seconds between breaths
    peaks, _ = find_peaks(ppg_filtered, distance=min_distance, 
                          prominence=0.1 * np.std(ppg_filtered))
    peak_count_rr = len(peaks)  # Peaks in 60 seconds = BPM
    
    print(f"\n1.  PEAK COUNTING (filtered PPG):")
    print(f"   Detected peaks: {len(peaks)}")
    print(f"   Estimated RR: {peak_count_rr:.1f} BPM")
    print(f"   Error: {abs(peak_count_rr - rr_mean):.1f} BPM")
    
    # =====================================================================
    # METHOD 2: PSD (Welch) - find peak in respiratory band
    # =====================================================================
    freqs_psd, psd = welch(ppg_segment, fs=fs, nperseg=min(2048, len(ppg_segment)//2))
    
    # Find peak in respiratory band
    resp_mask = (freqs_psd >= 0.1) & (freqs_psd <= 0.5)
    resp_freqs = freqs_psd[resp_mask]
    resp_psd = psd[resp_mask]
    
    peak_idx = np.argmax(resp_psd)
    psd_peak_freq = resp_freqs[peak_idx]
    psd_rr = psd_peak_freq * 60
    
    print(f"\n2.  PSD PEAK (Welch):")
    print(f"   Peak frequency: {psd_peak_freq:.3f} Hz")
    print(f"   Estimated RR: {psd_rr:.1f} BPM")
    print(f"   Error: {abs(psd_rr - rr_mean):.1f} BPM")
    
    # =====================================================================
    # METHOD 3: PyWavelets CWT (reference implementation)
    # =====================================================================
    dt = 1.0 / fs
    fc = pywt.central_frequency('morl')
    
    fmin, fmax = 0.1, 0.5
    num_scales = 128
    
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    scales = np.linspace(scale_min, scale_max, num_scales)
    
    # Z-score normalize input
    ppg_norm = (ppg_segment - np.mean(ppg_segment)) / (np.std(ppg_segment) + 1e-8)
    
    cwt_coeffs, cwt_freqs = pywt.cwt(ppg_norm, scales, 'morl', sampling_period=dt)
    cwt_power = np.abs(cwt_coeffs)  # (num_scales, signal_length)
    
    # Average power across time
    mean_power = cwt_power.mean(axis=1)  # (num_scales,)
    
    # Find peak
    cwt_peak_idx = np.argmax(mean_power)
    cwt_peak_freq = cwt_freqs[cwt_peak_idx]
    cwt_rr = cwt_peak_freq * 60
    
    print(f"\n3.  PYWAVELETS CWT (mean power):")
    print(f"   Peak frequency: {cwt_peak_freq:.3f} Hz")
    print(f"   Estimated RR: {cwt_rr:.1f} BPM")
    print(f"   Error: {abs(cwt_rr - rr_mean):.1f} BPM")
    
    # =====================================================================
    # METHOD 4: Your PyTorch CWT implementation
    # =====================================================================
    # Simulate your implementation
    from cwt_generator import PyTorchCWT
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cwt_model = PyTorchCWT(fs=fs, num_scales=128, fmin=0.1, fmax=0.5). to(device)
    cwt_model.eval()
    
    # Prepare input
    ppg_tensor = torch.tensor(ppg_norm, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pytorch_scalogram = cwt_model(ppg_tensor, target_time=60)  # (1, 128, 60)
    
    pytorch_scalogram = pytorch_scalogram.cpu().numpy()[0]  # (128, 60)
    
    # Average power across time
    pytorch_mean_power = pytorch_scalogram.mean(axis=1)  # (128,)
    
    # Create frequency axis for PyTorch CWT
    pytorch_freqs = np.linspace(0.1, 0.5, 128)
    
    pytorch_peak_idx = np.argmax(pytorch_mean_power)
    pytorch_peak_freq = pytorch_freqs[pytorch_peak_idx]
    pytorch_rr = pytorch_peak_freq * 60
    
    print(f"\n4. YOUR PYTORCH CWT (mean power):")
    print(f"   Peak frequency: {pytorch_peak_freq:.3f} Hz")
    print(f"   Estimated RR: {pytorch_rr:.1f} BPM")
    print(f"   Error: {abs(pytorch_rr - rr_mean):.1f} BPM")
    
    # =====================================================================
    # METHOD 5: Per-timestep CWT peak detection (your current approach)
    # =====================================================================
    detected_rr_per_t = []
    for t in range(pytorch_scalogram.shape[1]):
        col = pytorch_scalogram[:, t]
        # Per-column normalization
        col_norm = (col - col.min()) / (col.max() - col.min() + 1e-8)
        peak_idx = np.argmax(col_norm)
        detected_rr_per_t.append(pytorch_freqs[peak_idx] * 60)
    
    detected_rr_per_t = np.array(detected_rr_per_t)
    per_t_mae = np.mean(np.abs(detected_rr_per_t - np.array(rr_label)[:60]))
    
    print(f"\n5. YOUR PER-TIMESTEP CWT:")
    print(f"   Mean detected RR: {np.mean(detected_rr_per_t):.1f} BPM")
    print(f"   Range: {detected_rr_per_t.min():.1f} - {detected_rr_per_t.max():.1f} BPM")
    print(f"   MAE vs labels: {per_t_mae:.1f} BPM")
    
    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Filtered PPG with peaks
    ax = axes[0, 0]
    t = np.arange(len(ppg_filtered)) / fs
    ax.plot(t, ppg_filtered, 'b-', linewidth=0.8)
    if len(peaks) > 0:
        ax.plot(peaks / fs, ppg_filtered[peaks], 'ro', markersize=8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered PPG')
    ax.set_title(f'Peak Counting: {peak_count_rr} peaks = {peak_count_rr:.0f} BPM (Error: {abs(peak_count_rr - rr_mean):.1f})')
    
    # 2.  PSD
    ax = axes[0, 1]
    ax.semilogy(freqs_psd, psd)
    ax.axvline(expected_freq, color='r', linestyle='--', linewidth=2, label=f'True: {rr_mean:.1f} BPM')
    ax.axvline(psd_peak_freq, color='g', linestyle='--', linewidth=2, label=f'Detected: {psd_rr:.1f} BPM')
    ax.axvspan(0.1, 0.5, alpha=0.2, color='green')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(f'PSD: {psd_rr:.1f} BPM (Error: {abs(psd_rr - rr_mean):.1f})')
    ax.set_xlim(0, 1.0)
    ax.legend()
    
    # 3.  PyWavelets CWT frequency profile
    ax = axes[1, 0]
    ax.plot(cwt_freqs * 60, mean_power)
    ax.axvline(rr_mean, color='r', linestyle='--', linewidth=2, label=f'True: {rr_mean:.1f} BPM')
    ax.axvline(cwt_rr, color='g', linestyle='--', linewidth=2, label=f'Detected: {cwt_rr:.1f} BPM')
    ax.set_xlabel('RR (BPM)')
    ax.set_ylabel('Mean CWT Power')
    ax.set_title(f'PyWavelets CWT: {cwt_rr:.1f} BPM (Error: {abs(cwt_rr - rr_mean):.1f})')
    ax.legend()
    
    # 4. Your PyTorch CWT frequency profile
    ax = axes[1, 1]
    ax.plot(pytorch_freqs * 60, pytorch_mean_power)
    ax.axvline(rr_mean, color='r', linestyle='--', linewidth=2, label=f'True: {rr_mean:.1f} BPM')
    ax.axvline(pytorch_rr, color='g', linestyle='--', linewidth=2, label=f'Detected: {pytorch_rr:.1f} BPM')
    ax.set_xlabel('RR (BPM)')
    ax.set_ylabel('Mean CWT Power')
    ax.set_title(f'PyTorch CWT: {pytorch_rr:.1f} BPM (Error: {abs(pytorch_rr - rr_mean):.1f})')
    ax.legend()
    
    # 5. PyTorch scalogram
    ax = axes[2, 0]
    im = ax.imshow(pytorch_scalogram, aspect='auto', origin='lower',
                   extent=[0, 60, 6, 30], cmap='viridis')
    ax.plot(np.arange(len(rr_label)), rr_label, 'r-', linewidth=2, label='True RR')
    ax.plot(np.arange(len(detected_rr_per_t)), detected_rr_per_t, 'w--', linewidth=1.5, label='Detected')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RR (BPM)')
    ax.set_title(f'PyTorch Scalogram - Per-timestep MAE: {per_t_mae:.1f} BPM')
    ax.legend()
    plt.colorbar(im, ax=ax)
    
    # 6. Per-timestep detection vs labels
    ax = axes[2, 1]
    ax.plot(rr_label, 'r-', linewidth=2, label='True RR')
    ax.plot(detected_rr_per_t, 'b--', linewidth=1.5, label='Detected RR')
    ax.fill_between(range(len(rr_label)),
                    np.array(rr_label) - 3, np.array(rr_label) + 3,
                    alpha=0.2, color='red', label='±3 BPM')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RR (BPM)')
    ax.set_title(f'Per-timestep Detection - MAE: {per_t_mae:.1f} BPM')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'validation_plots/method_comparison_{subject_id}.png', dpi=150)
    plt.close()
    
    print(f"\nSaved comparison plot to: validation_plots/method_comparison_{subject_id}.png")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Error vs Ground Truth:")
    print(f"{'='*70}")
    print(f"  1. Peak Counting:     {abs(peak_count_rr - rr_mean):.1f} BPM")
    print(f"  2.  PSD Peak:          {abs(psd_rr - rr_mean):.1f} BPM")
    print(f"  3. PyWavelets CWT:    {abs(cwt_rr - rr_mean):.1f} BPM")
    print(f"  4. PyTorch CWT:       {abs(pytorch_rr - rr_mean):.1f} BPM")
    print(f"  5. Per-timestep CWT:  {per_t_mae:.1f} BPM (MAE)")
    
    return {
        'peak_count_error': abs(peak_count_rr - rr_mean),
        'psd_error': abs(psd_rr - rr_mean),
        'pywavelets_error': abs(cwt_rr - rr_mean),
        'pytorch_error': abs(pytorch_rr - rr_mean),
        'per_timestep_mae': per_t_mae
    }


def run_comparison(processed_data, n_samples=5):
    """
    Run comparison across multiple samples. 
    """
    import os
    os.makedirs("validation_plots", exist_ok=True)
    
    all_results = []
    subjects = list(processed_data.keys())
    
    for subject_id in subjects[:n_samples]:
        ppg_segments, rr_segments, _, _ = processed_data[subject_id]
        
        if len(ppg_segments) > 0:
            ppg = np.array(ppg_segments[0])
            rr = np.array(rr_segments[0])
            
            result = compare_frequency_methods(ppg, rr, fs=125, subject_id=subject_id)
            all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    methods = ['peak_count_error', 'psd_error', 'pywavelets_error', 'pytorch_error', 'per_timestep_mae']
    names = ['Peak Counting', 'PSD Peak', 'PyWavelets CWT', 'PyTorch CWT', 'Per-timestep CWT']
    
    for method, name in zip(methods, names):
        errors = [r[method] for r in all_results]
        print(f"  {name:20s}: {np.mean(errors):.2f} ± {np.std(errors):.2f} BPM")
    
    return all_results