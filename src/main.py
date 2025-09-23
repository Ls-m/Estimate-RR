import os
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from scipy import signal
import logging
import matplotlib.pyplot as plt
from scipy.signal import welch,spectrogram, find_peaks
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ReadData")
def load_subjects_bidmc(path):
    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        return None
    subjects = []
    for file_name in os.listdir(path):
        if not file_name.endswith(".csv"):
            continue
        subject_id = file_name.split("_")[1]
        if subject_id not in subjects:
            subjects.append(subject_id)
    return subjects

def load_files_bidmc(path,subjects):
    raw_data = []
    min_RR = float("inf")
    min_subject = []
    for subject in subjects:
        
        signal_file = path+"/bidmc_"+subject+"_Signals.csv"
        numeric_file = path+"/bidmc_"+subject+"_Numerics.csv"

        signal_df = pd.read_csv(signal_file) if os.path.exists(signal_file) else None
        numeric_df = pd.read_csv(numeric_file) if os.path.exists(numeric_file) else None

        signal_df.columns = signal_df.columns.str.strip()
        numeric_df.columns = numeric_df.columns.str.strip()

        for col in signal_df.columns:
            signal_df[col] = pd.to_numeric(signal_df[col], errors='coerce')
        
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

        ppg = signal_df["PLETH"].values
        rr = numeric_df["RESP"].values
        min_RR = min(min_RR, np.nanmin(rr))
        if np.nanmin(rr) == 0:
            min_subject.append(subject)
        if np.any(np.isnan(ppg)) or np.any(np.isinf(ppg)):
            logger.info(f"PPG data contains NaN or Inf values for subject: {subject}")

        if np.any(np.isnan(rr)) or np.any(np.isinf(rr)):
            logger.info(f"RR data contains NaN or Inf values for subject: {subject}")

        subject_dict = {
            "subject_id": subject,
            "PPG": ppg,
            "RR": rr
        }
        raw_data.append(subject_dict)

    logger.info(f"Minimum RR subjects for bidmc dataset: {min_subject}")
    logger.info(f"Minimum RR for bidmc dataset: {min_RR}")
    for data in raw_data:
        logger.debug(f"for subject {data['subject_id']} PPG length: {len(data['PPG'])}, RR length: {len(data['RR'])}")

    return raw_data

def read_data(path):
    # Code to read data goes here
    for dataset_name in os.listdir(path):
        dataset_path = os.path.join(path,dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if dataset_name == "bidmc":
            subjects = load_subjects_bidmc(dataset_path)
            raw_data = load_files_bidmc(dataset_path, subjects)
            print(len(subjects))
            # print(raw_data)
    return raw_data

def apply_bandpass_filter(cfg, signal_data, original_rate):
    # Design a Butterworth bandpass filter
    low_freq = cfg.preprocessing.bandpass_filter.low_freq
    high_freq = cfg.preprocessing.bandpass_filter.high_freq
    order = cfg.preprocessing.bandpass_filter.order
    nyquist = 0.5 * original_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, signal_data)
    return filtered_signal

def check_bandpass_filter_effect(original_signal, filtered_signal, original_rate, lowcut, highcut, nperseg=2048, plot_seconds=10):
    # Check if the bandpass filter has removed low-frequency noise
    low_freq_content = np.abs(np.fft.fft(original_signal))[:len(original_signal) // 2]
    filtered_low_freq_content = np.abs(np.fft.fft(filtered_signal))[:len(filtered_signal) // 2]
    if np.mean(filtered_low_freq_content) < np.mean(low_freq_content):
        logger.info("Bandpass filter effectively removed low-frequency noise.")
    else:
        logger.info("Bandpass filter did not remove low-frequency noise.")
    assert original_signal.shape == filtered_signal.shape

    for name, sig in (("original", original_signal),("band pass filtered", filtered_signal)):
        if np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
            logger.warning(f"{name} signal contains NaN or Inf values.")
    N = len(original_signal)
    t = np.arange(N) / original_rate
    show_samples = min(N,int(plot_seconds * original_rate))
    plt.figure(figsize=(10,3))
    plt.plot(t[:show_samples], original_signal[:show_samples], label="Original Signal", alpha=0.6)
    plt.plot(t[:show_samples],filtered_signal[:show_samples], label="Filtered Signal", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Time Domain (first {plot_seconds}s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # PSD via Welch
    f_orig,Pxx_orig = welch(original_signal, fs=original_rate, nperseg=1024)
    f_filt,Pxx_filt = welch(filtered_signal, fs=original_rate, nperseg=1024)

    plt.figure(figsize=(10,3))
    plt.plot(f_orig,Pxx_orig, label="Original Signal")
    plt.plot(f_filt, Pxx_filt, label="Filtered Signal")
    plt.axvline(lowcut, color='k', linestyle="--", alpha=0.6)
    plt.axvline(highcut, color='k', linestyle="--", alpha=0.6)
    plt.xlim(0,min(original_rate/2, highcut*4))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (PSD)")
    plt.title("Welch PSD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    f_original, t_original, Sxx_original = spectrogram(original_signal,original_rate, nperseg=nperseg, noverlap=nperseg/2)
    f_filtered, t_filtered, Sxx_filtered = spectrogram(filtered_signal,original_rate, nperseg=nperseg, noverlap=nperseg/2)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.pcolormesh(t_original, f_original, 10*np.log10(Sxx_original+1e-16),shading="gouraud")
    plt.ylim(0,min(original_rate/2, highcut*3))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Original Spectrogram (dB)")
    plt.colorbar(label='dB')

    plt.subplot(1,2,2)
    plt.pcolormesh(t_filtered, f_filtered, 10*np.log10(Sxx_filtered+1e-16), shading='gouraud')
    plt.ylim(0, min(original_rate/2, highcut*3))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Filtered Spectrogram (dB)")
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.show()
    # Numeric metrics: power in passband vs out-of-band
    # Compute PSD from Welch arrays
    # Note: f_orig == f_filt
    df = f_orig[1] - f_orig[0]
    passband_mask = (f_orig >= lowcut) & (f_orig <= highcut)
    out_mask = ~passband_mask & (f_orig > 0)
    passband_power_orig = np.sum(Pxx_orig[passband_mask]) * df
    passband_power_filt = np.sum(Pxx_filt[passband_mask]) * df
    outpower_orig = np.sum(Pxx_orig[out_mask]) * df
    outpower_filt = np.sum(Pxx_filt[out_mask]) * df
    # attenuation of out-of-band energy in dB
    # avoid div by zero
    if outpower_filt <= 0:
        attenuation_db = np.inf
    else:
        attenuation_db = 10 * np.log10((outpower_orig + 1e-20)/(outpower_filt + 1e-20))
    # passband preservation ratio (increase/decrease)
    passband_preservation_db = 10 * np.log10((passband_power_filt + 1e-20)/(passband_power_orig + 1e-20))

     # Lag estimation via cross-correlation (should be ~0 for filtfilt)
    # Use normalized cross-correlation on short segment to avoid boundary effects
    seg_len = min(N, int(original_rate*30))  # use up to 30s for stable estimate
    x = original_signal[:seg_len] - np.mean(original_signal[:seg_len])
    y = filtered_signal[:seg_len] - np.mean(filtered_signal[:seg_len])
    corr = np.correlate(y, x, mode='full')
    lag_idx = np.argmax(corr) - (len(x) - 1)
    lag_seconds = lag_idx / original_rate

    metrics = {
        "passband_power_orig": passband_power_orig,
        "passband_power_filt": passband_power_filt,
        "outband_power_orig": outpower_orig,
        "outband_power_filt": outpower_filt,
        "attenuation_db_outband": float(attenuation_db),
        "passband_preservation_db": float(passband_preservation_db),
        "lag_samples": int(lag_idx),
        "lag_seconds": float(lag_seconds),
        "n_samples": N
    }

    # print summary
    print("=== Band-pass Filter check summary ===")
    print(f"Samples: {N}, fs={original_rate} Hz")
    print(f"Passband: {lowcut} - {highcut} Hz")
    print(f"Passband power (orig -> filt): {passband_power_orig:.4e} -> {passband_power_filt:.4e} ({passband_preservation_db:.2f} dB)")
    print(f"Out-of-band power (orig -> filt): {outpower_orig:.4e} -> {outpower_filt:.4e} (attenuation {attenuation_db:.2f} dB)")
    print(f"Estimated lag: {lag_idx} samples -> {lag_seconds:.4f} s")
    print("===========================")

    return metrics

def remove_outliers(signal):
    Q1 = np.percentile(signal,25)
    Q3 = np.percentile(signal,75)
    IQR = Q3 - Q1
    lower_band = Q1 - 1.5*IQR
    higher_band = Q3 + 1.5*IQR
    signal = np.clip(signal,lower_band,higher_band)
    return signal, lower_band, higher_band

def check_outliers_removal_effect(original_signal,cleaned_signal,original_rate, lower, higher, plot_seconds=10):

    N = len(original_signal)
    show_samples = min(N,int(plot_seconds * original_rate))
    plt.figure(figsize=(12,4))
    plt.plot(original_signal[:show_samples], label="Original Signal")
    plt.plot(cleaned_signal[:show_samples],label="Cleaned Signal")
    plt.hlines([lower, higher], 0, len(original_signal[:show_samples]), colors='r', linestyles='--', label="IQR bounds")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.legend()
    plt.show()

def normalize_signal(signal,method="zscore"):
    
    if method == "zscore":
        signal = np.array(signal, dtype=float).reshape(-1, 1)  # make it 2D for sklearn
        scaler = StandardScaler()
        normalized = scaler.fit_transform(signal).flatten()  # back to 1D
        return normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def check_normalization_effect(original_signal, normalized_signal, original_rate, plot_seconds=10):
    N = len(original_signal)
    show_samples = original_rate * plot_seconds 
    plt.figure(figsize=(12,4))
    plt.plot(original_signal[:show_samples], label="Original Signal")
    plt.plot(normalized_signal[:show_samples], label="Normalized Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def edpa_denoiser(ppg_signal, fs, bandpass_freqs = [0.5,12.0], threshold_multiplier = 1.5, window_duration_sec = 1.5):
    if ppg_signal.ndim != 1:
        raise ValueError("Input PPG signal must be a 1D array.")
    
    nyquist = 0.5 * fs
    low = bandpass_freqs[0]/nyquist
    high = bandpass_freqs[1]/nyquist
    sos = signal.butter(4, [low, high], btype='band', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, ppg_signal)

    distance_in_seconds = 0.2 * window_duration_sec
    distance_in_samples = int(distance_in_seconds * fs)

    peaks, _ = find_peaks(filtered_signal, distance=distance_in_samples)
    logger.info(f"Detected peaks at: {peaks}")
    throughs, _ = find_peaks(-filtered_signal, distance=distance_in_samples)
    logger.info(f"Detected throughs at: {throughs}")

    if len(peaks) < 2 or len(throughs) < 2:
        logger.error("Insufficient peaks or troughs detected.")
        return ppg_signal, []
    
    upper_envelope = np.interp(np.arange(len(filtered_signal)), peaks, filtered_signal[peaks])
    lower_envelope = np.interp(np.arange(len(filtered_signal)), throughs, filtered_signal[throughs])

    envelope_diff = np.abs(upper_envelope - lower_envelope)

    Q1, Q3 = np.percentile(envelope_diff,[25,75])
    IQR = Q3 - Q1
    upper_thresh = Q3 + threshold_multiplier * IQR
    lower_thresh = Q1 - threshold_multiplier * IQR

    anomalous_indices = np.where((envelope_diff > upper_thresh) | (envelope_diff < lower_thresh))[0]
    
    if len(anomalous_indices) == 0:
        logger.error("No anomalous segments detected.")
        return ppg_signal, []
    
    logger.info(f"Detected anomalous indices at: {anomalous_indices}")

    median_env_diff = np.median(envelope_diff)
    logger.info(f"Median envelope difference is: {median_env_diff}")
    anomalous_segments = []
    
    diffs = np.diff(anomalous_indices)
    logger.info(f"Diffs are: {diffs}")
    split_points = np.where(diffs > fs)[0] + 1 # Split if gap > 1 second
    logger.info(f"Split points are: {split_points}")
    anomaly_groups = np.split(anomalous_indices, split_points)
    logger.info(f"Anomaly groups are: {anomaly_groups}")

    for group in anomaly_groups:
        if len(group) == 0:
            continue

        start_anomaly, end_anomaly = group[0], group[-1]
        search_start_idx = max(0, start_anomaly - fs * 2)
        crossings_before = np.where(envelope_diff[search_start_idx:start_anomaly] < median_env_diff)[0]
        start_segment = search_start_idx + crossings_before[-1] if len(crossings_before) > 0 else search_start_idx

        search_end_idx = min(len(envelope_diff), end_anomaly + fs * 2)
        crossings_after = np.where(envelope_diff[end_anomaly:search_end_idx] < median_env_diff)[0]
        end_segment = end_anomaly + crossings_after[0] if len(crossings_after) > 0 else search_end_idx
        
        anomalous_segments.append((start_segment, end_segment))

        # --- Step 6: Merge and Remove ---
    if not anomalous_segments:
        return ppg_signal, []
    
    anomalous_segments.sort()
    merged_segments = [anomalous_segments[0]]
    for current_start, current_end in anomalous_segments[1:]:
        last_start, last_end = merged_segments[-1]
        if current_start <= last_end:
            merged_segments[-1] = (last_start, max(last_end, current_end))
        else:
            merged_segments.append((current_start, current_end))

    keep_mask = np.ones(len(ppg_signal), dtype=bool)
    for start, end in merged_segments:
        keep_mask[start:end] = False
    

    t = np.arange(len(ppg_signal)) / fs  
    start_sec = 0
    plot_seconds = 15
    start_idx = int(start_sec * fs)
    end_idx = min(len(ppg_signal), start_idx + int(plot_seconds * fs))

    t_win = t[start_idx:end_idx]
    raw_win = ppg_signal[start_idx:end_idx]
    filt_win = filtered_signal[start_idx:end_idx]
    upper_win = upper_envelope[start_idx:end_idx]
    lower_win = lower_envelope[start_idx:end_idx]
    diff_win = envelope_diff[start_idx:end_idx]
    median_win = np.median(diff_win)

    peaks_win = peaks[(peaks >= start_idx) & (peaks < end_idx)]
    throughs_win = throughs[(throughs >= start_idx) & (throughs < end_idx)]

    plt.figure(figsize=(14, 8))

    # Raw + filtered
    plt.subplot(3,1,1)
    plt.plot(t_win, raw_win, label="Raw PPG", alpha=0.6)
    plt.plot(t_win, filt_win, label="Filtered", linewidth=1.5)
    plt.scatter(peaks_win/fs, filtered_signal[peaks_win], color="red", marker="x", label="Peaks")
    plt.scatter(throughs_win/fs, filtered_signal[throughs_win], color="blue", marker="o", label="Troughs")
    plt.legend()
    plt.title(f"PPG Signal ({plot_seconds}s window)")

    # Envelopes + anomaly thresholds
    plt.subplot(3,1,2)
    plt.plot(t_win, upper_win, label="Upper Envelope", color="red")
    plt.plot(t_win, lower_win, label="Lower Envelope", color="blue")
    plt.plot(t_win, diff_win, label="Envelope Diff", color="purple", alpha=0.7)
    plt.axhline(median_win, color="gray", linestyle="--", label="Median Diff")
    plt.axhline(upper_thresh, color="green", linestyle="--", label="Upper Threshold")
    plt.axhline(lower_thresh, color="orange", linestyle="--", label="Lower Threshold")
    plt.legend()
    plt.title("Envelopes and Anomaly Thresholds")

    # Anomalous segments with refinement visualization
    plt.subplot(3,1,3)
    plt.plot(t_win, raw_win, label="Raw PPG", alpha=0.5)

    # Plot refined segments (before merge) in orange
    raw_anomaly_segments = [(group[0], group[-1]) for group in anomaly_groups if len(group) > 0]
    for start, end in raw_anomaly_segments:
        if start < end_idx and end > start_idx:
            plt.axvspan(max(start,start_idx)/fs, min(end,end_idx)/fs, color="blue", alpha=0.6, label="Raw Segment" if 'Raw Segment' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot final merged segments in red
    for start, end in anomalous_segments:
        if start < end_idx and end > start_idx:
            plt.axvspan(max(start,start_idx)/fs, min(end,end_idx)/fs, color="red", alpha=0.3, label="Anomalous Segment" if 'Anomalous Segment' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.legend()
    plt.title("Detected Anomalous Segments")
    plt.tight_layout()
    plt.show()

    return ppg_signal[keep_mask], merged_segments



def process_data(cfg, raw_data, dataset_name='bidmc'):
    # Code to process data goes here
    processed_data = []
    for subject_data in raw_data:
        # Example processing: normalize PPG and RR signals
        ppg = subject_data["PPG"]
        rr = subject_data["RR"]

        if dataset_name == "bidmc":
            original_rate = 125

        
        ppg_denoised, merged_segments = edpa_denoiser(ppg, original_rate)
        logger.info(f"type raw ppg: {type(ppg)}")
        logger.info(f"type ppg_denoised: {type(ppg_denoised)}")
        logger.info(f"raw_ppg[615:700]: {ppg[615:700]}")
        logger.info(f"ppg_denoised[615:700]: {ppg_denoised[615:700]}")
        ppg_filtered = apply_bandpass_filter(cfg, ppg, original_rate)
        if np.any(np.isnan(ppg_filtered)):
            logger.info(f"after bandpass filter NaNs in PPG: {np.isnan(ppg_filtered).sum()} ")

        check_bandpass_filter_effect(ppg, ppg_filtered, original_rate, cfg.preprocessing.bandpass_filter.low_freq, cfg.preprocessing.bandpass_filter.high_freq)

        ppg_cliped,lower_band, higher_band = remove_outliers(ppg_filtered)
        check_outliers_removal_effect(ppg_filtered, ppg_cliped, original_rate, lower_band, higher_band)

        ppg_normalized = normalize_signal(ppg_cliped)
        check_normalization_effect(ppg_cliped, ppg_normalized, original_rate)

        processed_data.append({
            "subject_id": subject_data["subject_id"],
            "PPG": ppg_normalized,
            "RR": rr
        })
        exit()
    return processed_data

@hydra.main(config_path="../", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    raw_data = read_data("data/")
    processed_data = process_data(cfg, raw_data)
    print("Hello, World!")


if __name__ == "__main__":
    main()