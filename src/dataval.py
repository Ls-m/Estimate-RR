import numpy as np
import matplotlib.pyplot as plt
from scipy. signal import find_peaks


def analyze_scalogram_quality(scalogram, rr_label, fmin=0.1, fmax=0.8):
    """
    Check if peak frequency in scalogram matches the RR label.
    """
    n_freqs, n_times = scalogram.shape
    
    # Create frequency axis
    freqs = np.linspace(fmin, fmax, n_freqs)
    
    # For each timestep, find peak frequency
    detected_rr = []
    for t in range(n_times):
        freq_profile = scalogram[:, t]
        peak_idx = np. argmax(freq_profile)
        peak_freq_hz = freqs[peak_idx]
        peak_rr_bpm = peak_freq_hz * 60
        detected_rr.append(peak_rr_bpm)
    
    detected_rr = np.array(detected_rr)
    
    # Compare to label
    if isinstance(rr_label, (list, np.ndarray)):
        true_rr = np.array(rr_label)
    else:
        true_rr = np.full(n_times, rr_label)
    
    # Metrics
    mae = np.mean(np.abs(detected_rr - true_rr))
    correlation = np.corrcoef(detected_rr, true_rr)[0, 1]
    
    return {
        'detected_rr': detected_rr,
        'true_rr': true_rr,
        'mae': mae,
        'correlation': correlation
    }


def visualize_scalogram_with_rr(scalogram, rr_label, fmin=0.1, fmax=0.8, 
                                  title="Scalogram Analysis"):
    """
    Visualize scalogram with overlaid RR annotations. 
    """
    fig, axes = plt. subplots(3, 1, figsize=(14, 10))
    
    n_freqs, n_times = scalogram.shape
    
    # Frequency axis
    freqs = np.linspace(fmin, fmax, n_freqs)
    freqs_bpm = freqs * 60  # Convert to BPM for display
    
    # 1.  Scalogram
    ax1 = axes[0]
    im = ax1.imshow(scalogram, aspect='auto', origin='lower',
                    extent=[0, n_times, freqs_bpm[0], freqs_bpm[-1]],
                    cmap='viridis')
    ax1.set_ylabel('Respiratory Rate (BPM)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title(f'{title} - Scalogram')
    plt.colorbar(im, ax=ax1, label='Magnitude')
    
    # Overlay true RR as a line
    if isinstance(rr_label, (list, np.ndarray)):
        ax1.plot(np. arange(len(rr_label)), rr_label, 'r-', linewidth=2, 
                 label='True RR')
        ax1.legend()
    
    # 2. Mean frequency profile
    ax2 = axes[1]
    mean_profile = scalogram.mean(axis=1)
    ax2.plot(freqs_bpm, mean_profile)
    ax2. set_xlabel('Respiratory Rate (BPM)')
    ax2.set_ylabel('Mean Magnitude')
    ax2.set_title('Average Frequency Profile')
    ax2.axvline(x=np.mean(rr_label) if isinstance(rr_label, (list, np.ndarray)) 
                else rr_label, color='r', linestyle='--', label='Mean True RR')
    ax2.legend()
    
    # 3.  Detected vs True RR over time
    ax3 = axes[2]
    results = analyze_scalogram_quality(scalogram, rr_label, fmin, fmax)
    ax3.plot(results['detected_rr'], 'b-', label='Detected RR (peak freq)', alpha=0.7)
    ax3. plot(results['true_rr'], 'r-', label='True RR', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Respiratory Rate (BPM)')
    ax3.set_title(f"Peak Detection - MAE: {results['mae']:.2f}, Corr: {results['correlation']:. 3f}")
    ax3.legend()
    
    plt.tight_layout()
    return fig, results


def batch_quality_check(scalograms, rr_labels, n_samples=10):
    """
    Check quality across multiple samples.
    """
    n = min(n_samples, len(scalograms))
    indices = np.random. choice(len(scalograms), n, replace=False)
    
    all_maes = []
    all_corrs = []
    
    for idx in indices:
        results = analyze_scalogram_quality(scalograms[idx], rr_labels[idx])
        all_maes.append(results['mae'])
        all_corrs.append(results['correlation'])
    
    print("="*50)
    print("SCALOGRAM QUALITY CHECK (Peak Detection Baseline)")
    print("="*50)
    print(f"Samples checked: {n}")
    print(f"Mean MAE: {np.mean(all_maes):. 2f} ± {np. std(all_maes):.2f} BPM")
    print(f"Mean Correlation: {np.mean(all_corrs):.3f} ± {np.std(all_corrs):. 3f}")
    print("="*50)
    
    if np.mean(all_maes) > 5:
        print("⚠️ WARNING: High MAE suggests scalograms may not contain clear RR info!")
        print("   Check: fmin/fmax range, CWT normalization, PPG preprocessing")
    elif np.mean(all_corrs) < 0.5:
        print("⚠️ WARNING: Low correlation - temporal patterns may be noisy")
    else:
        print("✓ Scalograms appear to contain RR information")
    
    return {
        'maes': all_maes,
        'correlations': all_corrs,
        'mean_mae': np. mean(all_maes),
        'mean_corr': np.mean(all_corrs)
    }