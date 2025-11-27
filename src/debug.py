import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy. signal import welch
import pywt


def diagnose_single_sample(ppg_segment, rr_label, fs=125, subject_id="unknown"):
    """
    Complete diagnostic for a single PPG segment.
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSING SAMPLE - Subject: {subject_id}")
    print(f"{'='*70}")
    
    # Basic stats
    print(f"\n1. INPUT DATA:")
    print(f"   PPG length: {len(ppg_segment)} samples ({len(ppg_segment)/fs:.1f} seconds)")
    print(f"   PPG range: [{ppg_segment.min():.4f}, {ppg_segment.max():.4f}]")
    print(f"   PPG mean: {ppg_segment.mean():.4f}, std: {ppg_segment.std():.4f}")
    
    if isinstance(rr_label, (list, np.ndarray)):
        rr_mean = np.mean(rr_label)
        rr_std = np.std(rr_label)
        print(f"   RR label: {rr_mean:.2f} ± {rr_std:.2f} BPM (range: {np.min(rr_label):.1f}-{np.max(rr_label):.1f})")
    else:
        rr_mean = rr_label
        print(f"   RR label: {rr_mean:.2f} BPM")
    
    # Expected respiratory frequency
    expected_resp_freq = rr_mean / 60.0  # Convert BPM to Hz
    print(f"   Expected respiratory freq: {expected_resp_freq:.3f} Hz")
    
    # 2. Check PSD of raw PPG
    print(f"\n2. POWER SPECTRAL DENSITY:")
    freqs_psd, psd = welch(ppg_segment, fs=fs, nperseg=min(1024, len(ppg_segment)//2))
    
    # Find peaks in different frequency bands
    resp_band = (freqs_psd >= 0.1) & (freqs_psd <= 0.5)  # Respiratory: 6-30 BPM
    cardiac_band = (freqs_psd >= 0.8) & (freqs_psd <= 2.0)  # Cardiac: 48-120 BPM
    
    resp_power = psd[resp_band]. sum() if resp_band.any() else 0
    cardiac_power = psd[cardiac_band].sum() if cardiac_band.any() else 0
    total_power = psd. sum()
    
    print(f"   Respiratory band (0.1-0. 5 Hz) power: {resp_power/total_power*100:.1f}%")
    print(f"   Cardiac band (0. 8-2.0 Hz) power: {cardiac_power/total_power*100:.1f}%")
    
    # Find dominant frequency in respiratory band
    if resp_band.any():
        resp_freqs = freqs_psd[resp_band]
        resp_psd = psd[resp_band]
        dominant_resp_freq = resp_freqs[np.argmax(resp_psd)]
        dominant_resp_bpm = dominant_resp_freq * 60
        print(f"   Dominant freq in resp band: {dominant_resp_freq:.3f} Hz ({dominant_resp_bpm:.1f} BPM)")
        print(f"   Difference from label: {abs(dominant_resp_bpm - rr_mean):.1f} BPM")
    
    # 3. Generate CWT with different parameters
    print(f"\n3. CWT ANALYSIS:")
    
    # Test different frequency ranges
    test_configs = [
        {"fmin": 0.1, "fmax": 0.5, "name": "Respiratory only (0.1-0.5 Hz)"},
        {"fmin": 0.1, "fmax": 0.8, "name": "Extended (0.1-0.8 Hz)"},
        {"fmin": 0.05, "fmax": 0.6, "name": "Wide respiratory (0.05-0.6 Hz)"},
    ]
    
    results = {}
    
    for config in test_configs:
        fmin, fmax = config["fmin"], config["fmax"]
        
        # Generate CWT
        dt = 1.0 / fs
        num_scales = 128
        fc = pywt.central_frequency('morl')
        
        scale_min = fc / (fmax * dt)
        scale_max = fc / (fmin * dt)
        scales = np.linspace(scale_min, scale_max, num_scales)
        
        cwt_coeffs, freqs = pywt.cwt(ppg_segment, scales, 'morl', sampling_period=dt)
        scalogram = np.abs(cwt_coeffs)
        
        # Downsample time axis to 60 (1 per second)
        n_times = 60
        time_indices = np.linspace(0, scalogram.shape[1]-1, n_times). astype(int)
        scalogram_ds = scalogram[:, time_indices]
        
        # Find peak frequency at each timestep
        freqs_hz = np.linspace(fmin, fmax, num_scales)
        detected_rr = []
        for t in range(n_times):
            peak_idx = np. argmax(scalogram_ds[:, t])
            peak_freq = freqs_hz[peak_idx]
            detected_rr.append(peak_freq * 60)
        
        detected_rr = np.array(detected_rr)
        
        # Compare to label
        if isinstance(rr_label, (list, np.ndarray)):
            true_rr = np.array(rr_label)[:n_times]
        else:
            true_rr = np.full(n_times, rr_label)
        
        mae = np.mean(np.abs(detected_rr - true_rr))
        
        # Check if detected is always at boundary (indicates wrong range)
        at_fmin = np.mean(detected_rr == fmin * 60) * 100
        at_fmax = np.mean(detected_rr == fmax * 60) * 100
        
        print(f"\n   {config['name']}:")
        print(f"      Peak detection MAE: {mae:.2f} BPM")
        print(f"      Detected RR range: {detected_rr.min():.1f} - {detected_rr.max():.1f} BPM")
        print(f"      True RR range: {true_rr.min():.1f} - {true_rr.max():.1f} BPM")
        print(f"      % at fmin boundary: {at_fmin:.1f}%")
        print(f"      % at fmax boundary: {at_fmax:.1f}%")
        
        results[config['name']] = {
            'scalogram': scalogram_ds,
            'freqs': freqs_hz,
            'detected_rr': detected_rr,
            'true_rr': true_rr,
            'mae': mae
        }
    
    # 4.  Visualization
    fig, axes = plt. subplots(4, 2, figsize=(16, 14))
    
    # Raw PPG (first 10 seconds)
    ax = axes[0, 0]
    t_ppg = np.arange(len(ppg_segment)) / fs
    show_samples = min(len(ppg_segment), 10 * fs)
    ax.plot(t_ppg[:show_samples], ppg_segment[:show_samples])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PPG Amplitude')
    ax. set_title('Raw PPG Signal (first 10s)')
    
    # PSD
    ax = axes[0, 1]
    ax.semilogy(freqs_psd, psd)
    ax.axvline(expected_resp_freq, color='r', linestyle='--', label=f'Expected RR: {expected_resp_freq:.2f} Hz')
    ax.axvspan(0.1, 0.5, alpha=0.2, color='green', label='Resp band')
    ax. axvspan(0.8, 2.0, alpha=0.2, color='orange', label='Cardiac band')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Power Spectral Density')
    ax.set_xlim(0, 2.5)
    ax.legend()
    
    # Scalograms for each config
    for i, (name, res) in enumerate(results.items()):
        row = 1 + i // 2
        col = i % 2
        
        if row < 4:
            ax = axes[row, col]
            
            freqs_bpm = res['freqs'] * 60
            im = ax.imshow(res['scalogram'], aspect='auto', origin='lower',
                          extent=[0, 60, freqs_bpm[0], freqs_bpm[-1]],
                          cmap='viridis')
            
            # Overlay true RR
            ax.plot(np.arange(len(res['true_rr'])), res['true_rr'], 'r-', 
                   linewidth=2, label='True RR')
            ax.plot(np.arange(len(res['detected_rr'])), res['detected_rr'], 'w--',
                   linewidth=1.5, label='Detected RR')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('RR (BPM)')
            ax.set_title(f"{name}\nMAE: {res['mae']:.2f} BPM")
            ax.legend(loc='upper right')
            plt.colorbar(im, ax=ax)
    
    # Detected vs True comparison
    ax = axes[3, 1]
    best_config = min(results.keys(), key=lambda k: results[k]['mae'])
    best_res = results[best_config]
    ax.plot(best_res['true_rr'], 'r-', label='True RR', linewidth=2)
    ax.plot(best_res['detected_rr'], 'b--', label='Detected RR', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RR (BPM)')
    ax.set_title(f"Best Config: {best_config}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'validation_plots/diagnosis_{subject_id}.png', dpi=150)
    plt.close()

    print(f"\n   Saved diagnostic plot to: validation_plots/diagnosis_{subject_id}.png")

    # 5. Recommendations
    print(f"\n4. RECOMMENDATIONS:")
    
    best_mae = min(r['mae'] for r in results.values())
    
    if best_mae > 5:
        print("   ❌ Even with optimal frequency range, MAE is high.")
        print("   Possible issues:")
        print("      - PPG signal quality is poor")
        print("      - Respiratory modulation is very weak in this signal")
        print("      - RR labels might be incorrect")
        print("      - Need different preprocessing (try without bandpass filter)")
    elif best_mae < 3:
        print(f"   ✅ Good MAE achievable with config: {best_config}")
        print("   → Update your CWT generation to use these parameters")
    else:
        print(f"   ⚠️ Moderate MAE.  Best config: {best_config}")
        print("   → May need preprocessing improvements")
    
    return results


def run_batch_diagnosis(processed_data, n_samples=5):
    """
    Run diagnosis on multiple samples from different subjects.
    """
    import os
    os.makedirs("validation_plots", exist_ok=True)
    
    all_results = []
    
    # Get samples from different subjects
    subjects = list(processed_data. keys())
    samples_per_subject = max(1, n_samples // len(subjects))
    
    sample_count = 0
    for subject_id in subjects:
        ppg_segments, rr_segments, freq_segments, _ = processed_data[subject_id]
        
        for i in range(min(samples_per_subject, len(ppg_segments))):
            if sample_count >= n_samples:
                break
                
            # Get raw PPG (before any scalogram processing)
            ppg = np.array(ppg_segments[i])
            rr = np.array(rr_segments[i])
            
            results = diagnose_single_sample(
                ppg, rr, fs=125, 
                subject_id=f"{subject_id}_seg{i}"
            )
            all_results.append({
                'subject': subject_id,
                'segment': i,
                'results': results
            })
            
            sample_count += 1
        
        if sample_count >= n_samples:
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    
    # Find best config across all samples
    config_maes = {}
    for sample in all_results:
        for config_name, res in sample['results']. items():
            if config_name not in config_maes:
                config_maes[config_name] = []
            config_maes[config_name].append(res['mae'])
    
    print("\nAverage MAE by configuration:")
    for config_name, maes in config_maes.items():
        print(f"   {config_name}: {np.mean(maes):.2f} ± {np.std(maes):.2f} BPM")
    
    best_config = min(config_maes.keys(), key=lambda k: np.mean(config_maes[k]))
    print(f"\n   → Best config: {best_config}")
    
    return all_results