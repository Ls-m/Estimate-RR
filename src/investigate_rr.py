"""
Investigate the relationship between PPG respiratory content and RR labels. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch, find_peaks
import pywt


def investigate_rr_labels(ppg_segment, rr_labels, fs=125, subject_id="unknown"):
    """
    Deep investigation of why CWT peaks don't match RR labels.
    """
    print(f"\n{'='*70}")
    print(f"INVESTIGATING RR LABELS - Subject: {subject_id}")
    print(f"{'='*70}")
    
    # 1. Understand the labels
    rr_mean = np.mean(rr_labels)
    rr_std = np.std(rr_labels)
    expected_freq = rr_mean / 60.0  # Hz
    
    print(f"\n1. RR LABEL ANALYSIS:")
    print(f"   Mean RR: {rr_mean:.2f} BPM = {expected_freq:.3f} Hz")
    print(f"   Expected breath period: {1/expected_freq:.2f} seconds")
    print(f"   In 60 seconds, expected ~{60 * expected_freq:.1f} breaths")
    
    # 2. Find actual periodicity in PPG using autocorrelation
    print(f"\n2.  AUTOCORRELATION ANALYSIS:")
    
    # Compute autocorrelation
    ppg_centered = ppg_segment - np.mean(ppg_segment)
    autocorr = np.correlate(ppg_centered, ppg_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find peaks in autocorrelation (excluding lag 0)
    # Look for peaks between 2-10 seconds (6-30 BPM respiratory range)
    min_lag = int(2 * fs)  # 2 seconds = 30 BPM
    max_lag = int(10 * fs)  # 10 seconds = 6 BPM
    
    peaks, properties = find_peaks(autocorr[min_lag:max_lag], height=0.1, distance=fs)
    peaks = peaks + min_lag  # Adjust for offset
    
    if len(peaks) > 0:
        # First significant peak is the dominant period
        dominant_lag = peaks[0]
        dominant_period = dominant_lag / fs
        dominant_freq = 1 / dominant_period
        dominant_bpm = dominant_freq * 60
        
        print(f"   Dominant period from autocorr: {dominant_period:.2f} seconds")
        print(f"   Corresponding frequency: {dominant_freq:.3f} Hz = {dominant_bpm:.1f} BPM")
        print(f"   Difference from label: {abs(dominant_bpm - rr_mean):.1f} BPM")
    else:
        print("   No clear periodic pattern found in autocorrelation")
        dominant_bpm = None
    
    # 3. Bandpass filter and count zero crossings
    print(f"\n3. ZERO-CROSSING ANALYSIS:")
    
    # Bandpass filter for respiratory band
    sos = signal.butter(4, [0.1, 0.5], btype='band', fs=fs, output='sos')
    ppg_filtered = signal.sosfiltfilt(sos, ppg_segment)
    
    # Count zero crossings
    zero_crossings = np.where(np.diff(np.sign(ppg_filtered)))[0]
    n_crossings = len(zero_crossings)
    
    # Each breath cycle has 2 zero crossings
    estimated_breaths = n_crossings / 2
    estimated_rr = estimated_breaths  # breaths per 60 seconds = BPM
    
    print(f"   Zero crossings in 60s: {n_crossings}")
    print(f"   Estimated breaths: {estimated_breaths:.1f}")
    print(f"   Estimated RR: {estimated_rr:.1f} BPM")
    print(f"   Difference from label: {abs(estimated_rr - rr_mean):.1f} BPM")
    
    # 4. Peak counting in filtered signal
    print(f"\n4. PEAK COUNTING ANALYSIS:")
    
    # Find peaks in filtered signal
    min_distance = int(2 * fs)  # Minimum 2 seconds between breaths
    peaks_resp, _ = find_peaks(ppg_filtered, distance=min_distance, prominence=0.1*np.std(ppg_filtered))
    
    n_peaks = len(peaks_resp)
    peak_based_rr = n_peaks  # peaks in 60 seconds = BPM
    
    print(f"   Detected peaks in filtered PPG: {n_peaks}")
    print(f"   Peak-based RR: {peak_based_rr:.1f} BPM")
    print(f"   Difference from label: {abs(peak_based_rr - rr_mean):.1f} BPM")
    
    # 5.  Compare all estimates
    print(f"\n5.  COMPARISON SUMMARY:")
    print(f"   Method                  | Estimated RR | Diff from Label")
    print(f"   -----------------------|--------------|----------------")
    print(f"   RR Label (ground truth) | {rr_mean:12.1f} | -")
    if dominant_bpm:
        print(f"   Autocorrelation         | {dominant_bpm:12.1f} | {abs(dominant_bpm - rr_mean):15.1f}")
    print(f"   Zero-crossing           | {estimated_rr:12.1f} | {abs(estimated_rr - rr_mean):15.1f}")
    print(f"   Peak counting           | {peak_based_rr:12.1f} | {abs(peak_based_rr - rr_mean):15.1f}")
    
    # 6.  Visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    
    # Raw PPG
    t = np.arange(len(ppg_segment)) / fs
    axes[0].plot(t, ppg_segment, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('PPG')
    axes[0].set_title('Raw PPG Signal')
    
    # Filtered PPG with peaks
    axes[1].plot(t, ppg_filtered, 'b-', linewidth=0.8)
    if len(peaks_resp) > 0:
        axes[1].plot(peaks_resp/fs, ppg_filtered[peaks_resp], 'ro', markersize=8, label=f'{n_peaks} peaks')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Filtered PPG')
    axes[1].set_title(f'Bandpass Filtered (0.1-0.5 Hz) - Detected {n_peaks} breaths = {peak_based_rr:.0f} BPM')
    axes[1].legend()
    
    # Autocorrelation
    lags = np.arange(len(autocorr)) / fs
    axes[2].plot(lags[:max_lag], autocorr[:max_lag])
    if len(peaks) > 0:
        axes[2].axvline(dominant_lag/fs, color='r', linestyle='--', 
                       label=f'Dominant period: {dominant_period:.2f}s = {dominant_bpm:.1f} BPM')
    # Mark expected period from label
    expected_period = 60 / rr_mean
    axes[2].axvline(expected_period, color='g', linestyle='--', 
                   label=f'Expected from label: {expected_period:.2f}s = {rr_mean:.1f} BPM')
    axes[2].set_xlabel('Lag (seconds)')
    axes[2].set_ylabel('Autocorrelation')
    axes[2].set_title('Autocorrelation of PPG')
    axes[2].legend()
    axes[2].set_xlim(0, 15)
    
    # PSD with multiple annotations
    freqs, psd = welch(ppg_segment, fs=fs, nperseg=min(2048, len(ppg_segment)//2))
    axes[3].semilogy(freqs, psd)
    axes[3].axvline(expected_freq, color='r', linestyle='--', linewidth=2,
                   label=f'Label RR: {rr_mean:.1f} BPM')
    if dominant_bpm:
        axes[3].axvline(dominant_bpm/60, color='g', linestyle='--', linewidth=2,
                       label=f'Autocorr: {dominant_bpm:.1f} BPM')
    axes[3].axvline(peak_based_rr/60, color='orange', linestyle='--', linewidth=2,
                   label=f'Peak count: {peak_based_rr:.1f} BPM')
    axes[3].axvspan(0.1, 0.5, alpha=0.2, color='green', label='Resp band')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('PSD')
    axes[3].set_title('Power Spectral Density')
    axes[3].set_xlim(0, 1.0)
    axes[3].legend()
    
    # RR labels over time
    axes[4].plot(rr_labels, 'r-', linewidth=2, label='RR Labels')
    axes[4].axhline(peak_based_rr, color='orange', linestyle='--', label=f'Peak-based: {peak_based_rr:.0f}')
    if dominant_bpm:
        axes[4].axhline(dominant_bpm, color='g', linestyle='--', label=f'Autocorr: {dominant_bpm:.1f}')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('RR (BPM)')
    axes[4].set_title('RR Labels vs Estimated')
    axes[4].legend()
    axes[4].set_ylim(0, 35)
    
    plt.tight_layout()
    plt.savefig(f'validation_plots/label_investigation_{subject_id}.png', dpi=150)
    plt.close()
    
    print(f"\n   Saved plot to: validation_plots/label_investigation_{subject_id}.png")
    
    # 7.  Conclusion
    print(f"\n6. CONCLUSION:")
    
    # Check if signal-based estimates are consistent with each other
    signal_estimates = [peak_based_rr]
    if dominant_bpm:
        signal_estimates.append(dominant_bpm)
    
    signal_mean = np.mean(signal_estimates)
    label_diff = abs(signal_mean - rr_mean)
    
    if label_diff < 3:
        print(f"   ✅ Signal content matches labels well (diff: {label_diff:.1f} BPM)")
    elif label_diff < 6:
        print(f"   ⚠️ Moderate mismatch between signal and labels (diff: {label_diff:.1f} BPM)")
        print(f"   → The CWT might need temporal smoothing to better track the labels")
    else:
        print(f"   ❌ Large mismatch between signal content and labels (diff: {label_diff:.1f} BPM)")
        print(f"   → Possible causes:")
        print(f"      1. Labels are derived from a different signal (e.g., chest belt, not PPG)")
        print(f"      2. PPG respiratory modulation is weak for this subject")
        print(f"      3.  Preprocessing is removing respiratory information")
    
    return {
        'label_mean': rr_mean,
        'autocorr_rr': dominant_bpm,
        'zerocross_rr': estimated_rr,
        'peak_rr': peak_based_rr,
        'signal_mean': signal_mean,
        'label_diff': label_diff
    }


def batch_label_investigation(processed_data, n_samples=10):
    """
    Investigate labels across multiple samples. 
    """
    import os
    os.makedirs("validation_plots", exist_ok=True)
    
    all_results = []
    
    subjects = list(processed_data.keys())
    sample_count = 0
    
    for subject_id in subjects:
        ppg_segments, rr_segments, freq_segments, _ = processed_data[subject_id]
        
        if len(ppg_segments) > 0:
            # Take first segment from each subject
            ppg = np.array(ppg_segments[0])
            rr = np.array(rr_segments[0])
            
            result = investigate_rr_labels(ppg, rr, fs=125, subject_id=f"{subject_id}")
            all_results.append(result)
            
            sample_count += 1
            if sample_count >= n_samples:
                break
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH LABEL INVESTIGATION SUMMARY")
    print(f"{'='*70}")
    
    label_means = [r['label_mean'] for r in all_results]
    peak_rrs = [r['peak_rr'] for r in all_results]
    diffs = [r['label_diff'] for r in all_results]
    
    print(f"\nAcross {len(all_results)} samples:")
    print(f"   Label RR range: {min(label_means):.1f} - {max(label_means):.1f} BPM")
    print(f"   Peak-detected RR range: {min(peak_rrs):.1f} - {max(peak_rrs):.1f} BPM")
    print(f"   Average label-signal difference: {np.mean(diffs):.1f} ± {np.std(diffs):.1f} BPM")
    
    # Correlation between label and signal-based estimates
    corr = np.corrcoef(label_means, peak_rrs)[0, 1]
    print(f"   Correlation (label vs peak-detected): {corr:.3f}")
    
    if corr > 0.7:
        print(f"\n   ✅ Good correlation - labels and signal are related")
        print(f"   → A learned model should be able to bridge the gap")
    elif corr > 0.3:
        print(f"\n   ⚠️ Moderate correlation - some relationship exists")
        print(f"   → Model may need to learn complex transformations")
    else:
        print(f"\n   ❌ Poor correlation - labels may come from different source")
        print(f"   → Consider using signal-derived pseudo-labels for pre-training")
    
    return all_results