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
from scipy.interpolate import interp1d
from dataset import PPGRRDataset, PPGRRDataModule
from model import RRLightningModule, SSLPretrainModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import time
from pytorch_lightning.profilers import SimpleProfiler
from tqdm import tqdm
import pywt
from joblib import Parallel, delayed, cpu_count
import multiprocessing
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import optuna
from my_optuna import objective
import json
from pytorch_lightning.strategies import DDPStrategy
from skimage.transform import resize
import torch.distributed as dist
from sklearn.model_selection import KFold
import random
from skimage.transform import resize
from model import FreqSSLPretrainModule
from dataset import FrequencySSLDataset
import copy
from torch.utils.data import Dataset

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
    print(f"subjects are {subjects}")
    raw_data = {}
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

        # subject_dict = {
        #     "subject_id": subject,
        #     "PPG": ppg,
        #     "RR": rr
        # }
        raw_data[subject] = (ppg, rr)

    logger.info(f"Minimum RR subjects for bidmc dataset: {min_subject}")
    logger.info(f"Minimum RR for bidmc dataset: {min_RR}")
    for subject, (ppg, rr) in raw_data.items():
        logger.debug(f"for subject {subject} PPG length: {len(ppg)}, RR length: {len(rr)}")

    return raw_data


def load_files_capnobase(path,subjects):
    print(f"subjects are {subjects}")
    raw_data = {}

    for subject in subjects:
        signal_file = path+"/bidmc_"+subject+"_Signals.csv"
        signal_df = pd.read_csv(signal_file) if os.path.exists(signal_file) else None
        signal_df.columns = signal_df.columns.str.strip()
        for col in signal_df.columns:
            signal_df[col] = pd.to_numeric(signal_df[col], errors='coerce')
        ppg = signal_df["PLETH"].values
        if np.any(np.isnan(ppg)) or np.any(np.isinf(ppg)):
            logger.info(f"PPG data contains NaN or Inf values for subject: {subject}")
        raw_data[subject] = (ppg,)
        for subject, (ppg,) in raw_data.items():
            logger.debug(f"for subject {subject} PPG length: {len(ppg)}")

    return raw_data


def read_data(path):
    # Code to read data goes here
    for dataset_name in os.listdir(path):
        dataset_path = os.path.join(path,dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if dataset_name == "bidmc":
            subjects = load_subjects_bidmc(dataset_path)
            # print(subjects)
            
            raw_data = load_files_bidmc(dataset_path, subjects)
            # print(raw_data)
            print(len(subjects))
            
            break
            # print(raw_data)
    return raw_data


def read_capnobase_data(path):
    raw_data = {}
    for dataset_name in os.listdir(path):
        dataset_path = os.path.join(path,dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if dataset_name == "capnobase":
            subjects = load_subjects_bidmc(dataset_path)
            raw_data = load_files_capnobase(dataset_path, subjects)
            print(len(subjects))
            break
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
    if cfg.preprocessing.bandpass_filter.type == "butter":
        sos = signal.butter(order, [low, high], btype='band', output='sos')
    if cfg.preprocessing.bandpass_filter.type == "cheby1":
        ripple = getattr(cfg.preprocessing.bandpass_filter, 'ripple', 1)  # dB ripple in passband
        sos = signal.cheby1(order, ripple, [low, high], btype='band', output='sos')
    if cfg.preprocessing.bandpass_filter.type == "cheby2":
        # Design a Chebyshev Type II bandpass filter
        stopband_atten = getattr(cfg.preprocessing.bandpass_filter, 'stopband_atten', 40)
        sos = signal.cheby2(order, stopband_atten, [low, high], btype='band', output='sos')
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


def edpa_denoiser(ppg_signal, fs, bandpass_freqs = [0.5,12.0], threshold_multiplier = 2, window_duration_sec = 1.5, check_effect = False):
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
    # logger.info(f"Detected peaks at: {peaks}")
    throughs, _ = find_peaks(-filtered_signal, distance=distance_in_samples)
    # logger.info(f"Detected throughs at: {throughs}")

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
    
    # logger.info(f"Detected anomalous indices at: {anomalous_indices}")

    median_env_diff = np.median(envelope_diff)
    # logger.info(f"Median envelope difference is: {median_env_diff}")
    anomalous_segments = []
    
    diffs = np.diff(anomalous_indices)
    # logger.info(f"Diffs are: {diffs}")
    split_points = np.where(diffs > fs)[0] + 1 # Split if gap > 1 second
    # logger.info(f"Split points are: {split_points}")
    anomaly_groups = np.split(anomalous_indices, split_points)
    # logger.info(f"Anomaly groups are: {anomaly_groups}")

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
    
    if not check_effect:
        return ppg_signal[keep_mask], merged_segments
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

def expand_removals_to_second_blocks(removed_segments, fs):
    if not removed_segments:
        logger.warning("No removed segments to expand.")
        return []

    seconds_to_remove = set()

    for start_idx, end_idx in removed_segments:
        start_second = start_idx // fs
        end_second = (end_idx - 1) // fs
        for sec in range(start_second,end_second+1):
            seconds_to_remove.add(sec)

    if not seconds_to_remove:
        logger.warning("No seconds to remove.")
        return []
    
    expanded_segments = []

    for sec in sorted(list(seconds_to_remove)):
        expanded_segments.append((sec * fs, (sec + 1) * fs))

    expanded_segments.sort()
    merged_segments = [expanded_segments[0]]
    for current_start, current_end in expanded_segments:
        last_start, last_end = merged_segments[-1]
        if current_start <= last_end:
            merged_segments[-1] = (last_start, max(last_end, current_end))
        else:
            merged_segments.append((current_start, current_end))

    return merged_segments

def apply_removals(signal, removed_segments):
    keep_mask = np.ones(len(signal), dtype=bool)
    for start,end in removed_segments:
        start = max(0, start)
        end = min(len(signal), end)
        if start < end:
                keep_mask[start:end] = False

    return signal[keep_mask]

def remove_corresponding_labels(rr_labels, removed_ppg_segments, ppg_fs, rr_fs):
    if not removed_ppg_segments:
        logger.warning("No removed ppg segments")
        return rr_labels
    fs_ratio = ppg_fs/rr_fs
    keep_mask = np.ones(len(rr_labels), dtype=bool)

    for start_ppg, end_ppg in removed_ppg_segments:
        start_rr = int(np.floor(start_ppg/fs_ratio))
        end_rr = int(np.floor(end_ppg/fs_ratio))

        start_rr = max(0, start_rr)
        end_rr = min(len(rr_labels), end_rr)

        if start_rr < end_rr:
            keep_mask[start_rr:end_rr] = False
    return rr_labels[keep_mask]

def reconstruct_short_gaps(ppg_signal, segments_to_reconstruct, fs):

    reconstructed_signal = ppg_signal.copy()

    for start_idx, end_idx in segments_to_reconstruct:

        anchor_len = int(fs * 0.5)

        pre_gap_start = max(0, start_idx - anchor_len)
        post_gap_end = min(len(ppg_signal), end_idx + anchor_len)

        anchor_indices = np.concatenate([np.arange(pre_gap_start, start_idx), np.arange(end_idx, post_gap_end)])

        anchor_values = reconstructed_signal[anchor_indices]

        if len(anchor_indices) < 2:
            logger.warning("anchor indices is less than 2")
            continue

        try:
            interp_func = interp1d(anchor_indices, anchor_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
        except ValueError:
            logger.warning("raised error in interpolation")
            interp_func = interp1d(anchor_indices, anchor_values, kind='linear', bounds_error=False, fill_value="extrapolate")

        gap_indices = np.arange(start_idx, end_idx)
        reconstructed_signal[gap_indices] = interp_func(gap_indices)

    return reconstructed_signal

def reconstruct_noise(raw_ppg, all_noisy_segments, fs):
    RECONSTRUCTION_THRESHOLD_SEC = 1.0

    segments_to_remove = []
    segments_to_reconstruct = []
    threshold_samples = RECONSTRUCTION_THRESHOLD_SEC * fs

    for start, end in all_noisy_segments:
        if (end - start) <= threshold_samples:
            segments_to_reconstruct.append((start, end))
        else:
            segments_to_remove.append((start, end))
    logger.info(f"Found {len(segments_to_reconstruct)} segments to reconstruct")
    logger.info(f"Found {len(segments_to_remove)} segments to remove")

    ppg_reconstructed = reconstruct_short_gaps(raw_ppg, segments_to_reconstruct, fs)

    return ppg_reconstructed, segments_to_remove




def create_segments_with_gap_handling(subject_id, ppg_signal, rr_labels, original_len, removed_segments, ppg_fs,
                                      rr_fs, window_size_sec, step_size_sec):
    if subject_id == '45':
        print("subject id with no segments")
        print(len(ppg_signal))
    ppg_segments = []
    final_rr_labels = []

    window_samples = window_size_sec * ppg_fs
    step_samples = step_size_sec * ppg_fs
    fs_ratio = ppg_fs / rr_fs

    # --- Find Continuous Blocks (Bug Fixed) ---
    # BUG FIX: Initialize with the full signal range to start the process.
    continuous_blocks = [(0, original_len)]
    if removed_segments:
        
        for rem_start, rem_end in removed_segments:
            new_blocks = []
            for block_start, block_end in continuous_blocks:
                # If the removed segment is outside this block, keep the block
                if rem_end <= block_start or rem_start >= block_end:
                    new_blocks.append((block_start, block_end))
                    continue
                # If there's a valid portion before the removed segment
                if rem_start > block_start:
                    new_blocks.append((block_start, rem_start))
                # If there's a valid portion after the removed segment
                if rem_end < block_end:
                    new_blocks.append((rem_end, block_end))
            continuous_blocks = new_blocks

    # --- Create Index Mappings (Calculated Once) ---
    # Create mapping for PPG signal indices
    keep_mask_ppg = np.ones(original_len, dtype=bool)
    for start, end in removed_segments:
        keep_mask_ppg[start:end] = False
    clean_indices_map_ppg = np.cumsum(keep_mask_ppg) - 1

    # OPTIMIZATION: This block is now calculated only once for efficiency.
    # Create mapping for RR label indices
    rr_original_len = int(np.ceil(original_len / fs_ratio))
    keep_mask_rr = np.ones(rr_original_len, dtype=bool)
    for rs, re in removed_segments:
        start_rr = int(np.floor(rs / fs_ratio))
        end_rr = int(np.ceil(re / fs_ratio))
        if start_rr < end_rr: # Ensure valid slice
             keep_mask_rr[start_rr:end_rr] = False
    clean_rr_map = np.cumsum(keep_mask_rr) - 1

    # --- Main Segmentation Loop ---
    for block_start, block_end in continuous_blocks:
        for current_pos in range(block_start, block_end - window_samples + 1, step_samples):
            seg_start_orig = current_pos
            seg_end_orig = current_pos + window_samples

            # Map PPG indices and get segment
            seg_start_clean = clean_indices_map_ppg[seg_start_orig]
            # The length of the segment is fixed, so we just add window_samples
            seg_end_clean = seg_start_clean + window_samples
            ppg_segment = ppg_signal[seg_start_clean:seg_end_clean]

            # Map RR indices and get average label
            rr_start_idx_orig = int(np.floor(seg_start_orig / fs_ratio))
            rr_end_idx_orig = int(np.floor((seg_end_orig - 1) / fs_ratio))

            rr_start_idx_clean = clean_rr_map[rr_start_idx_orig]
            rr_end_idx_clean = clean_rr_map[rr_end_idx_orig]

            if rr_start_idx_clean <= rr_end_idx_clean:
                rr_slice = rr_labels[rr_start_idx_clean : rr_end_idx_clean + 1]
                
                if np.isnan(rr_slice).any():
                    continue
                # Check if len(rr_slice) > 0, corrected from len(rr_slice > 0)
                if len(rr_slice) > 0:
                    # average_rr = np.mean(rr_slice)
                    average_rr = rr_slice
                    final_rr_labels.append(average_rr)
                    # ppg_segment_normalized = normalize_signal(ppg_segment)
                    ppg_segments.append(ppg_segment)
                    # ppg_segments.append(ppg_segment)

    return ppg_segments, final_rr_labels


def create_segments_simple(subject_id, ppg_signal, rr_labels, ppg_fs, rr_fs, window_size_sec, overlap_sec):

    step_size_sec = window_size_sec - overlap_sec
    ppg_signal = np.asarray(ppg_signal).squeeze()
    rr_labels = np.asarray(rr_labels).squeeze()

    window_samples = int(window_size_sec * ppg_fs)
    step_samples = int(step_size_sec * ppg_fs)
    fs_ratio = ppg_fs / rr_fs

    ppg_segments = []
    rr_segments = []

    for start in range(0, len(ppg_signal) - window_samples + 1, step_samples):
        end = start + window_samples
        
        # Extract PPG window
        ppg_segment = ppg_signal[start:end]

        # Corresponding RR indices (1 Hz)
        rr_start = int(np.floor(start / fs_ratio))
        rr_end = int(np.floor((end - 1) / fs_ratio))
        rr_slice = rr_labels[rr_start:rr_end + 1]

        if len(rr_slice) == 0 or np.isnan(rr_slice).any() or np.any(rr_slice == 0):
            continue

        # # Optional: normalize each segment
        # ppg_segment_norm = (ppg_segment - np.mean(ppg_segment)) / (np.std(ppg_segment) + 1e-8)
        ppg_segment_norm = normalize_signal(ppg_segment)
        ppg_segments.append(ppg_segment_norm)
        rr_segments.append(rr_slice)
    
    return ppg_segments, rr_segments
# Place this new function in the same file as your other processing functions

def create_ssl_segments(subject_id, ppg_signal, original_len, removed_segments, ppg_fs, 
                        window_size_sec, step_size_sec):
    """
    Creates PPG segments for SSL pre-training. Does not require RR labels.
    Returns PPG segments and dummy labels.
    """
    ppg_segments = []
    window_samples = window_size_sec * ppg_fs
    step_samples = step_size_sec * ppg_fs

    # --- Find Continuous Blocks (same logic as before) ---
    continuous_blocks = [(0, original_len)]
    if removed_segments:
        for rem_start, rem_end in removed_segments:
            new_blocks = []
            for block_start, block_end in continuous_blocks:
                if rem_end <= block_start or rem_start >= block_end:
                    new_blocks.append((block_start, block_end))
                    continue
                if rem_start > block_start:
                    new_blocks.append((block_start, rem_start))
                if rem_end < block_end:
                    new_blocks.append((rem_end, block_end))
            continuous_blocks = new_blocks

    # --- Create Index Mapping for PPG signal (same logic as before) ---
    keep_mask_ppg = np.ones(original_len, dtype=bool)
    for start, end in removed_segments:
        keep_mask_ppg[start:end] = False
    clean_indices_map_ppg = np.cumsum(keep_mask_ppg) - 1

    # --- Main Segmentation Loop (Simplified) ---
    for block_start, block_end in continuous_blocks:
        if block_end - block_start < window_samples:
            continue
        for current_pos in range(block_start, block_end - window_samples + 1, step_samples):
            seg_start_orig = current_pos
            
            # Map PPG indices and get segment
            seg_start_clean = clean_indices_map_ppg[seg_start_orig]
            seg_end_clean = seg_start_clean + window_samples
            
            ppg_segment = ppg_signal[seg_start_clean:seg_end_clean]
            ppg_segments.append(ppg_segment)
    
    # Create dummy RR labels to match the number of PPG segments
    num_segments = len(ppg_segments)
    dummy_rr_segments = np.zeros(num_segments)

    return ppg_segments, dummy_rr_segments

def plot_cwt_scalogram(ppg_segment, fs, fmin=0.1, fmax=0.6, num_scales=50):
    dt = 1.0 / fs
    fc = pywt.central_frequency('morl')
    
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    scales = np.linspace(scale_min, scale_max, num_scales)

    cwt_coeffs, freqs = pywt.cwt(ppg_segment, scales, 'morl', sampling_period=dt)
    power = np.abs(cwt_coeffs)

    # Plot scalogram
    plt.figure(figsize=(10, 5))
    extent = [0, len(ppg_segment) / fs, freqs.min(), freqs.max()]
    plt.imshow(power, extent=extent, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('CWT Scalogram (Respiratory Band)')
    plt.show()

def extract_cwt_features(ppg_segment, fs, fmin=0.1, fmax=0.6, num_scales=50):
    dt = 1.0 / fs
    fc = pywt.central_frequency('morl')  # ≈0.8125; auto-computes for precision
    
    # Compute scales for the frequency band
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    scales = np.linspace(scale_min, scale_max, num_scales)
    
    # Apply CWT
    cwt_coeffs, freqs = pywt.cwt(ppg_segment, scales, 'morl', sampling_period=dt)
    
    # Filter to respiratory band (now fully covered)
    resp_mask = (freqs >= fmin) & (freqs <= fmax)
    cwt_resp = cwt_coeffs[resp_mask, :]
    
    # Pool: Mean absolute over time per scale (robust summary)
    features = np.mean(np.abs(cwt_resp), axis=1)
    
    # Safe print: Check mask before min/max
    # if np.any(resp_mask):
    #     print(f"Generated {features.shape[0]} features covering {freqs[resp_mask].min():.3f}–{freqs[resp_mask].max():.3f} Hz")
    # else:
    #     print("Warning: No frequencies in respiratory band—check scales/fs.")
    
    return features.astype(np.float32)

def extract_freq_features(ppg_segment, fs, fmin, fmax, nperseg):

    if nperseg is None:
        nperseg = min(1024, len(ppg_segment))
    
    freqs, psd = welch(ppg_segment, fs=fs, nperseg=nperseg, window='hann', scaling='density')
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    psd_band = psd[band_mask]
    psd_band = psd_band / (np.sum(psd_band) + 1e-12)

    return psd_band.astype(np.float32)

def augment_ppg_segment(ppg):
    # 1. Ensure Numpy
    ppg = np.array(ppg, dtype=np.float32)
    
    if ppg.ndim != 1:
        raise ValueError(f"augment_ppg_segment expected 1D array, got {ppg.shape}")

    aug_type = np.random.choice(['noise', 'drift', 'burst'])
    scale = np.random.uniform(0.7, 1.3)
    ppg_aug = ppg * scale 
    
    if aug_type == 'noise':
        # Gaussian White Noise
        noise = np.random.normal(0, 0.05, len(ppg)) * np.std(ppg)
        ppg_aug = ppg_aug + noise
        
    elif aug_type == 'drift':
        # Baseline Wander
        t = np.linspace(0, 1, len(ppg))
        freq = np.random.uniform(0.05, 0.2) 
        drift = np.sin(2 * np.pi * freq * t) * np.std(ppg) * 0.5
        ppg_aug = ppg_aug + drift
        
    elif aug_type == 'burst':
        # --- FIX START ---
        # 1. Decide length first
        burst_len = np.random.randint(10, 50)
        
        # 2. Calculate the maximum safe starting index
        max_start = len(ppg) - burst_len
        
        if max_start > 0:
            # 3. Pick start so it fits
            burst_start = np.random.randint(0, max_start)
            
            noise_burst = np.random.normal(0, 0.5, burst_len) * np.std(ppg)
            ppg_aug[burst_start : burst_start + burst_len] += noise_burst
        # --- FIX END ---

    return ppg_aug

def _synthesize_single_sample(src_ppg, src_rr):
    """Helper to generate 1 augmented sample (runs in parallel)."""
    # A. Augment PPG (Noise/Drift)
    aug_ppg = augment_ppg_segment(src_ppg)
    
    # B. Copy Label
    aug_rr = copy.deepcopy(src_rr)
    
    # C. Generate CWT (The slow part)
    # Ensure fmax matches your new config (0.8) and shape matches model (128, 60)
    aug_freq = generate_cwt_scalogram(
        aug_ppg, fs=125, target_shape=(128, 60), fmin=0.1, fmax=0.8
    )
    
    return aug_ppg, aug_rr, aug_freq
def balance_dataset_with_synthesis(ppg_list, rr_list, freq_list):
    print("--- Starting Dataset Balancing (Parallelized) ---")
    
    # Ensure inputs are lists
    if isinstance(ppg_list, np.ndarray): ppg_list = list(ppg_list)
    if isinstance(rr_list, np.ndarray): rr_list = list(rr_list)
    if isinstance(freq_list, np.ndarray): freq_list = list(freq_list)

    # 1. Organize Indices by Class
    class_indices = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for i, rr in enumerate(rr_list):
        mean_rr = np.mean(rr)
        if mean_rr < 10: bin_idx = 0
        elif 10 <= mean_rr < 15: bin_idx = 1
        elif 15 <= mean_rr < 20: bin_idx = 2
        elif 20 <= mean_rr < 25: bin_idx = 3
        else: bin_idx = 4
        class_indices[bin_idx].append(i)

    # 2. Find Majority Count
    counts = [len(idxs) for idxs in class_indices.values()]
    target_count = max(counts)
    print(f"Initial Counts: {counts}")
    print(f"Target per class: {target_count}")
    
    new_ppg, new_rr, new_freq = [], [], []
    
    # 3. Generate Data
    for bin_idx, indices in class_indices.items():
        current_count = len(indices)
        if current_count == 0:
            continue

        needed = target_count - current_count
        if needed <= 0:
            continue
            
        print(f"Class {bin_idx}: Generating {needed} samples using {current_count} sources...")

        # Prepare the list of source data to augment
        # We pre-select the source PPGs and RRs to avoid passing the huge main list to workers
        tasks = []
        for k in range(needed):
            source_idx = indices[k % current_count]
            tasks.append((ppg_list[source_idx], rr_list[source_idx]))

        # --- PARALLEL EXECUTION ---
        # n_jobs=-1 uses all available CPU cores
        njobs = max(1, cpu_count() - 6)
        results = Parallel(n_jobs=njobs)(
            delayed(_synthesize_single_sample)(p, r) for p, r in tqdm(tasks, desc=f"Class {bin_idx}")
        )
        
        # Unpack results
        for p, r, f in results:
            new_ppg.append(p)
            new_rr.append(r)
            new_freq.append(f)

    # 4. Combine
    final_ppg = ppg_list + new_ppg
    final_rr = rr_list + new_rr
    final_freq = freq_list + new_freq
    
    print(f"Balancing Complete. Added {len(new_ppg)} synthetic samples.")
    print(f"Final Dataset Size: {len(final_ppg)}")
    
    return final_ppg, final_rr, final_freq


def generate_cwt_scalogram(ppg_segment, fs=125, target_shape=(128, 60), fmin=0.1, fmax=0.8, wavelet='morl', use_fake=False):
    """
    Generates a CWT scalogram optimized for Seq2Seq RWKV.
    
    Args:
        ppg_segment: 1D array of PPG data (60 seconds).
        fs: Sampling rate (125 Hz).
        target_shape: (Freq_Bins, Time_Steps). 
                      For RWKV Seq2Seq, set (64, 60) -> 60 output steps = 1 per second.
        fmin: Min frequency (0.1 Hz).
        fmax: Max frequency (0.8 Hz to catch high RR).
    """
    if use_fake or ppg_segment is None or len(ppg_segment) < 2:
        return np.random.rand(*target_shape).astype(np.float32)

    # --- STEP 1: SAFE Z-SCORE (Time Domain) ---
    # Normalize the input signal so amplitudes are comparable across subjects.
    # We use a noise floor to prevent amplifying flat-lines (sensor off).
    mu = np.mean(ppg_segment)
    sigma = np.std(ppg_segment)
    NOISE_FLOOR = 0.001  # Adjust if your raw data is huge (e.g. ADC counts), use 1.0
    safe_sigma = max(sigma, NOISE_FLOOR)
    segment_norm = (ppg_segment - mu) / safe_sigma

    # --- STEP 2: CWT CONFIGURATION ---
    dt = 1.0 / fs
    # We compute scales based on the target FREQUENCY resolution (64 bins)
    num_scales = target_shape[0] 
    fc = pywt.central_frequency(wavelet)
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    scales = np.linspace(scale_min, scale_max, num_scales)

    # --- STEP 3: COMPUTE CWT ---
    # We assume the input length is ~7500 samples (60s * 125Hz).
    # The output 'cwt_coeffs' will have shape (64, 7500).
    cwt_coeffs, _ = pywt.cwt(segment_norm, scales, wavelet, sampling_period=dt)
    scalogram = np.abs(cwt_coeffs)

    # --- STEP 4: RESIZE TO TARGET SHAPE ---
    # We resize (64, 7500) -> (64, 60).
    # This effectively averages the energy within each 1-second block.
    # preserve_range=True is crucial to keep the intensity physics-based.
    resized_scalogram = resize(scalogram, target_shape, anti_aliasing=True, preserve_range=True)

    # --- STEP 5: FIXED SCALING (Image Domain) ---
    # Since input is Z-scored, a perfect wave maxes out at ~6.0 in CWT magnitude.
    # We divide by 6.0 to map "strong signal" to ~1.0.
    FIXED_MAX_VAL = 6.0
    normalized_scalogram = resized_scalogram / FIXED_MAX_VAL

    # Clip to [0, 1]. 
    # Result: Strong breaths are Bright. Noise is Dark.
    normalized_scalogram = np.clip(normalized_scalogram, 0.0, 1.0)

    return normalized_scalogram.astype(np.float32)
# def generate_cwt_scalogram(ppg_segment, fs=125, image_size=(64, 60), fmin=0.1, fmax=0.6, wavelet='morl', use_fake=False):
#     if use_fake or ppg_segment is None:
#         # Generate fake normalized data
#         fake_scalogram = np.random.rand(*image_size).astype(np.float32)
#         return fake_scalogram
#     # --- 1. Calculate CWT Coefficients ---
#     dt = 1.0 / fs
#     # Define the scales corresponding to the frequency range
#     # The number of scales will determine the initial height of the image
#     num_scales = image_size[0] 
#     fc = pywt.central_frequency(wavelet)
#     scale_min = fc / (fmax * dt)
#     scale_max = fc / (fmin * dt)
#     scales = np.linspace(scale_min, scale_max, num_scales)
    
#     # Perform the Continuous Wavelet Transform
#     cwt_coeffs, freqs = pywt.cwt(ppg_segment, scales, wavelet, sampling_period=dt)
    
#     # --- 2. Create the Image from Magnitude ---
#     # Take the absolute value to get the magnitude (energy)
#     scalogram = np.abs(cwt_coeffs)

#     # --- 3. Resize to Target Image Dimensions ---
#     # Use scikit-image for high-quality resizing
#     # anti_aliasing=True is recommended when downsampling
#     # Resize (Keep frequency 64, increase time to 128)
#     resized_scalogram = resize(scalogram, image_size, anti_aliasing=True)
    
#     # --- 4. THE FIX: FIXED SCALING ---
#     # Do NOT use (x - min) / (max - min).
#     # Since input is Z-scored, real signals rarely exceed a CWT magnitude of ~5.0 or 6.0.
#     # We divide by a fixed factor to bring typical signals to ~0.0 - 1.0 range,
#     # but we allow outliers to clip.
    
#     FIXED_MAX_VAL = 6.0  # Empirical value for Morlet CWT on Z-scored data
    
#     normalized_scalogram = resized_scalogram / FIXED_MAX_VAL
    
#     # Clip to ensure 0-1 range (Deep Learning loves 0-1)
#     # Real signals will be bright (near 1.0).
#     # Background noise will be dark (near 0.1).
#     normalized_scalogram = np.clip(normalized_scalogram, 0.0, 1.0)
#     # plt.imshow(normalized_scalogram, cmap='viridis', aspect='auto')
#     # plt.axis('off')
#     # plt.show()
#     # exit()
#     return normalized_scalogram.astype(np.float32)

def compute_freq_features(ppg_segments, fs, n_jobs=-1):  # -1 = all cores
    
    def process_single(segment):
        return generate_cwt_scalogram(segment, fs, use_fake=False)
        # return extract_cwt_features(segment, fs, num_scales=50)
    
    freq_features = Parallel(n_jobs=n_jobs)(
        delayed(process_single)(segment) for segment in tqdm(ppg_segments)
    )
    return np.stack(freq_features, axis=0)

# def compute_freq_features(ppg_segments, fs):
#     freq_features = []
#     for segment in tqdm(ppg_segments, desc="Computing frequency features"):
#         # features = extract_freq_features(segment, fs, fmin=0.1, fmax=0.6, nperseg=2048)
#         features  = extract_cwt_features(segment, fs, fmin=0.1, fmax=0.6, num_scales=50)
#         print("Single segment frequency feature shape:", features.shape)
#         freq_features.append(features)
#     freq_features = np.stack(freq_features, axis=0)
#     return freq_features

# def compute_freq_features(ppg_segments, fs):
#     freq_segments = []
#     for seg in ppg_segments:    # seg = 60s PPG
#         seg = np.asarray(seg)
#         per_second_feats = []
#         for i in range(60):
#             one_sec = seg[i*fs:(i+1)*fs]
#             psd_feat = extract_freq_features(one_sec, fs, fmin=0.1, fmax=0.6, nperseg=None)
#             per_second_feats.append(psd_feat)
#         per_second_feats = np.stack(per_second_feats)  # (60, n_bins)
#         freq_segments.append(per_second_feats)
#     return np.stack(freq_segments)   # shape (B, 60, n_bins)

def check_freq_features(freq_segments, rr_segments, subject_id):
    logger.info(f"Frequency features shape for subject {subject_id}: {freq_segments.shape}")
    plt.figure(figsize=(10,6))
    plt.plot(freq_segments[0])
    plt.title(f"Frequency-domain features for first PPG segment for subject {subject_id} with RR {rr_segments[0]}")
    plt.xlabel("Frequency bin")
    plt.ylabel("Normalized PSD")
    plt.show()

# --- NEW WAVELET DENOISING FUNCTION ---
def denoise_ppg_with_wavelet(ppg_signal, wavelet='sym8', level=5):
    """
    Denoises a PPG signal using the Wavelet Transform.

    Args:
        ppg_signal (np.array): The raw PPG signal.
        wavelet (str): The type of mother wavelet to use (e.g., 'db4', 'sym8').
                       'sym8' is often a good choice for PPG.
        level (int): The number of decomposition levels.

    Returns:
        np.array: The denoised PPG signal, same length as the input.
    """
    # 1. Decompose the signal
    coeffs = pywt.wavedec(ppg_signal, wavelet, level=level)

    # 2. Calculate the threshold value
    # Use the standard deviation of the finest detail coefficients as a noise estimate
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # Use the universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(ppg_signal)))

    # 3. Apply soft thresholding to the detail coefficients
    # We don't threshold the approximation coefficients (coeffs[0])
    thresholded_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        thresholded_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    # 4. Reconstruct the signal from the thresholded coefficients
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)

    # 5. Ensure the output signal has the same length as the input
    # Reconstruction can sometimes result in a slightly different length
    original_len = len(ppg_signal)
    if len(denoised_signal) > original_len:
        denoised_signal = denoised_signal[:original_len]
    elif len(denoised_signal) < original_len:
        # Pad with the last value if shorter
        padding = original_len - len(denoised_signal)
        denoised_signal = np.pad(denoised_signal, (0, padding), 'edge')

    return denoised_signal


def process_data(cfg, raw_data, dataset_name='bidmc'):
    # Code to process data goes here
    processed_data = {}
 
    # for subject_data in raw_data:
    for i in range(1,len(raw_data)+1):
        # Example processing: normalize PPG and RR signals
        if i==13:
            continue
        ppg = raw_data[f"{i:02}"][0]
        rr = raw_data[f"{i:02}"][1]
        
        if dataset_name == "bidmc":
            original_rate = 125

        check_effect = False
        
        if cfg.preprocessing.use_denoiser:
            logger.info(f"Subject {i:02}: Running with denoiser ENABLED.")
            if cfg.preprocessing.use_edpa:
                logger.info(f"Subject {i:02}: Running with EDPA denoising ENABLED.")
                _, merged_segments = edpa_denoiser(ppg, original_rate, check_effect=check_effect)
                if merged_segments:
                    ppg_reconstructed, segments_to_remove = reconstruct_noise(ppg, merged_segments, original_rate)
                    expanded_removed_segments = expand_removals_to_second_blocks(segments_to_remove, fs=original_rate)
                    ppg_denoised = apply_removals(ppg_reconstructed, expanded_removed_segments)
                    rr_labels_denoised = remove_corresponding_labels(rr, expanded_removed_segments, original_rate, 1)
                else:
                    logger.info(f"for this subject {i} there is no noise detected!")
                    expanded_removed_segments = []
                    ppg_denoised = ppg
                    rr_labels_denoised = rr
            elif cfg.preprocessing.use_wavelet_denoising:
                logger.info(f"Subject {i:02}: Running with wavelet denoising ENABLED.")
                ppg_denoised = denoise_ppg_with_wavelet(ppg)
                rr_labels_denoised = rr
                expanded_removed_segments = []

                # # Plot first 3 seconds
                # time_window = 0.1  # seconds
                # num_samples = int(time_window * original_rate)
                # t = np.arange(len(ppg)) / original_rate
                # plt.figure(figsize=(10, 5))
                # plt.plot(t[:num_samples], ppg[:num_samples], label='Raw PPG', alpha=0.7)
                # plt.plot(t[:num_samples], ppg_denoised[:num_samples], label='Denoised PPG', linewidth=2)
                # plt.title('PPG Signal Before and After Wavelet Denoising (First 100 ms)')
                # plt.xlabel('Time (s)')
                # plt.ylabel('Amplitude')
                # plt.legend()
                # plt.grid(True)
                # plt.tight_layout()
                # plt.show()
        else:
            # This is the ablation path: the denoiser is skipped entirely.
            logger.info(f"Subject {i:02}: Running with denoiser DISABLED (Ablation Study).")
            ppg_denoised = ppg
            rr_labels_denoised = rr
            expanded_removed_segments = []

        if cfg.preprocessing.filter:
            ppg_filtered = apply_bandpass_filter(cfg, ppg_denoised, original_rate)
            if np.any(np.isnan(ppg_filtered)):
                logger.info(f"after bandpass filter NaNs in PPG: {np.isnan(ppg_filtered).sum()} ")
        else:
            ppg_filtered = ppg_denoised

        # check_bandpass_filter_effect(ppg_denoised, ppg_filtered, original_rate, cfg.preprocessing.bandpass_filter.low_freq, cfg.preprocessing.bandpass_filter.high_freq)
        if cfg.preprocessing.remove_outliers:
            ppg_cliped,lower_band, higher_band = remove_outliers(ppg_filtered)
            # check_outliers_removal_effect(ppg_filtered, ppg_cliped, original_rate, lower_band, higher_band)
        else:
            ppg_cliped = ppg_filtered

        # ppg_normalized = normalize_signal(ppg_cliped)
        # check_normalization_effect(ppg_cliped, ppg_normalized, original_rate)

        subject_id = f"{i:02}"


        if cfg.preprocessing.use_denoiser and cfg.preprocessing.use_edpa:
            ppg_segments, rr_segments = create_segments_with_gap_handling(
                subject_id,
                ppg_cliped,
                rr_labels_denoised,
                original_len=len(ppg),
                removed_segments=expanded_removed_segments,
                ppg_fs=original_rate,
                rr_fs=1,
                window_size_sec=32,
                step_size_sec=16
            )
        else:
            ppg_segments, rr_segments = create_segments_simple(subject_id, ppg_cliped, rr_labels_denoised,ppg_fs=original_rate,
                                                     rr_fs=1, window_size_sec=cfg.training.window_size, overlap_sec=cfg.training.overlap)
        
        # 3. [NEW] Create Segments for SSL (Broadband / Cardiac Preserved)
        # We use ppg_denoised here because it still contains the Heart Rate (> 1.0 Hz)
        ppg_segments_ssl, _ = create_segments_simple(
            subject_id, ppg_denoised, rr_labels_denoised, 
            ppg_fs=original_rate, rr_fs=1, 
            window_size_sec=cfg.training.window_size, overlap_sec=cfg.training.overlap
        )
        # plot_cwt_scalogram(ppg_segments[0], original_rate)
        n_jobs = max(1, cpu_count() - 6)
        logger.info(f"Compute frequency segments for subject {subject_id}")
        freq_segments = compute_freq_features(ppg_segments, original_rate, n_jobs=n_jobs)
        # freq_segments = compute_freq_features(ppg_segments, original_rate)
        # check_freq_features(freq_segments, rr_segments, subject_id)
  

        processed_data[subject_id] = (ppg_segments, rr_segments, freq_segments, ppg_segments_ssl)
        # logger.info(f"processed data is {processed_data}")
        
    return processed_data


# This is your new function, dedicated to processing data for SSL
def process_ssl_data(cfg, raw_data):
    processed_data = {}

    # <<< FIX: Iterate directly over the dictionary's items (subject_id, data_tuple) >>>
    for subject_id, data_tuple in raw_data.items():
        # The first element of the tuple is the ppg signal
        ppg = data_tuple[0]
        
        # Determine sampling rate based on dataset (adjust if needed)
        original_rate = 125
        
        # --- Denoising (PPG only) ---
        if cfg.preprocessing.use_denoiser:
            logger.info(f"Subject {subject_id}: Running with denoiser ENABLED.")
            _, merged_segments = edpa_denoiser(ppg, original_rate, check_effect=False)
            if merged_segments:
                ppg_reconstructed, segments_to_remove = reconstruct_noise(ppg, merged_segments, original_rate)
                expanded_removed_segments = expand_removals_to_second_blocks(segments_to_remove, fs=original_rate)
                ppg_denoised = apply_removals(ppg_reconstructed, expanded_removed_segments)
            else:
                logger.info(f"for this subject {subject_id} there is no noise detected!")
                expanded_removed_segments = []
                ppg_denoised = ppg
        else:
            logger.info(f"Subject {subject_id}: Running with denoiser DISABLED (Ablation Study).")
            ppg_denoised = ppg
            expanded_removed_segments = []
        
        # --- Standard PPG processing pipeline ---
        ppg_filtered = apply_bandpass_filter(cfg, ppg_denoised, original_rate)
        if np.any(np.isnan(ppg_filtered)):
            logger.info(f"after bandpass filter NaNs in PPG: {np.isnan(ppg_filtered).sum()} ")
        ppg_cliped, _, _ = remove_outliers(ppg_filtered)
        ppg_normalized = normalize_signal(ppg_cliped)
        
        # The subject_id now comes directly from the loop, no need to create it
        
        # --- Use the new SSL segmentation function ---
        ppg_segments, rr_segments = create_ssl_segments(
            subject_id,
            ppg_normalized,
            original_len=len(ppg),
            removed_segments=expanded_removed_segments,
            ppg_fs=original_rate,
            window_size_sec=32,
            step_size_sec=16
        )
        
        if not ppg_segments:
            logger.warning(f"No valid PPG segments for subject {subject_id} after processing. Skipping.")
            continue

        # --- CWT Feature computation (remains the same) ---
        n_jobs = max(1, cpu_count() - 6)
        freq_segments = compute_freq_features(ppg_segments, original_rate, n_jobs=n_jobs)
        
        processed_data[subject_id] = (ppg_segments, rr_segments, freq_segments)
        
    return processed_data

def create_balanced_folds(processed_data, n_splits=5):



    subject_segment_counts = [(subject_id, len(ppg_segments)) for subject_id, (ppg_segments, rr_segments, freq_segments) in processed_data.items()]
    subject_segment_counts.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Subject segment counts (sorted): {subject_segment_counts}")
    folds = [[] for _ in range(n_splits)]
    fold_segment_totals = [0] * n_splits
    for subject_id, segment_count in subject_segment_counts:

        lightest_fold_index = np.argmin(fold_segment_totals)
        folds[lightest_fold_index].append(subject_id)
        fold_segment_totals[lightest_fold_index] += segment_count

    logger.info(f"Number of segments in each fold: {fold_segment_totals}")

    test_subjects = folds.copy()
    val_subjects = [test_subjects[i+1 if i+1 < n_splits else 0] for i in range(n_splits)]
    train_subjects = []
    for i in range(n_splits):
        train = []
        for j in range(n_splits):
            if j != i and j != (i + 1) % n_splits:
                train.extend(folds[j])
        train_subjects.append(train)
    logger.info(f"Test subjects in each fold: {test_subjects}")
    logger.info(f"Validation subjects in each fold: {val_subjects}")
    logger.info(f"Train subjects in each fold: {train_subjects}")

    all_subjects = set(subject_id for subject_id, (ppg_segments, rr_segments, freq_segments) in processed_data.items())

    cv_splits = []
    for i in range(n_splits):
        test_set = set(test_subjects[i])
        val_set = set(val_subjects[i])
        train_set = set(train_subjects[i])

        overlap_test_val = test_set & val_set
        overlap_test_train = test_set & train_set
        overlap_val_train = val_set & train_set

        logger.info(f"Fold {i+1} overlap between test and val: {overlap_test_val}")
        logger.info(f"Fold {i+1} overlap between test and train: {overlap_test_train}")
        logger.info(f"Fold {i+1} overlap between validation and train: {overlap_val_train}")

        union = train_set | val_set | test_set
        coverage = union == all_subjects
        missing_subjects = all_subjects - union if not coverage else set()
        
        logger.info(f"Fold {i+1} covers all subjects: {coverage}")
        logger.info(f"Fold {i+1} missing subjects: {missing_subjects}")
        cv_splits.append({
            "train_subjects": train_subjects[i],
            "val_subjects": val_subjects[i],
            "test_subjects": test_subjects[i],
            "fold_id": i+1
        })


    return cv_splits


def create_folds(processed_data, n_splits=10, seed=42):

    # Fix random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    all_subjects = set(subject_id for subject_id, (ppg_segments, rr_segments, freq_segments, ppg_segments_ssl) in processed_data.items())
    subjects_array = np.array(sorted(all_subjects))
            
    # Shuffle subjects
    shuffled_indices = np.random.permutation(len(list(all_subjects)))
    shuffled_subjects = subjects_array[shuffled_indices]
    
    # Create k-fold splits
    kfold = KFold(n_splits=n_splits, shuffle=False, random_state=None)  # Already shuffled
    cv_splits = []
    for fold_id, (train_val_indices, test_indices) in enumerate(kfold.split(shuffled_subjects)):
        train_val_subjects = shuffled_subjects[train_val_indices].tolist()
        test_subjects = shuffled_subjects[test_indices].tolist()
        
        n_val_subjects = max(1, int(len(train_val_subjects) * 0.2))
        # n_val_subjects = len(test_subjects)
        random.seed(seed + fold_id)  # make per-fold val split deterministic
        val_subjects = random.sample(train_val_subjects, n_val_subjects)
        train_subjects = [s for s in train_val_subjects if s not in val_subjects]
        

        test_set = set(test_subjects)
        val_set = set(val_subjects)
        train_set = set(train_subjects)

        overlap_test_val = test_set & val_set
        overlap_test_train = test_set & train_set
        overlap_val_train = val_set & train_set

        logger.info(f"Fold {fold_id+1} overlap between test and val: {overlap_test_val}")
        logger.info(f"Fold {fold_id+1} overlap between test and train: {overlap_test_train}")
        logger.info(f"Fold {fold_id+1} overlap between validation and train: {overlap_val_train}")

        union = train_set | val_set | test_set
        coverage = union == all_subjects
        missing_subjects = all_subjects - union if not coverage else set()

        logger.info(f"Fold {fold_id+1} covers all subjects: {coverage}")
        logger.info(f"Fold {fold_id+1} missing subjects: {missing_subjects}")

        cv_splits.append({
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "fold_id": fold_id+1
        })
    return cv_splits

def create_data_splits(cv_split, processed_data):

    train_subjects = cv_split["train_subjects"]
    validation_subjects = cv_split["val_subjects"]
    test_subjects = cv_split["test_subjects"]

    subject_ids = [subject_id for subject_id, (ppg_segments, rr_segments, freq_segments, ppg_segments_ssl) in processed_data.items()]

    train_ppg_list = []
    train_rr_list = []
    train_freq_list = []
    train_ppg_ssl_list = []
    for train_subject in train_subjects:
        if train_subject in subject_ids:
            train_ppg_list.append(processed_data[train_subject][0])
            train_rr_list.append(processed_data[train_subject][1])
            train_freq_list.append(processed_data[train_subject][2])
            train_ppg_ssl_list.append(processed_data[train_subject][3])
        else:
            logger.warning(f"Train subject {train_subject} not found in processed data.")
    
    val_ppg_list = []
    val_rr_list = []
    val_freq_list = []
    val_ppg_ssl_list = []
    for val_subject in validation_subjects:
        if val_subject in subject_ids:
            val_ppg_list.append(processed_data[val_subject][0])
            val_rr_list.append(processed_data[val_subject][1])
            val_freq_list.append(processed_data[val_subject][2])
            val_ppg_ssl_list.append(processed_data[val_subject][3])
        else:
            logger.warning(f"Validation subject {val_subject} not found in processed data.")

    test_ppg_list = []
    test_rr_list = []
    test_freq_list = []
    test_ppg_ssl_list = []
    for test_subject in test_subjects:
        if test_subject in subject_ids:
            test_ppg_list.append(processed_data[test_subject][0])
            test_rr_list.append(processed_data[test_subject][1])
            test_freq_list.append(processed_data[test_subject][2])
            test_ppg_ssl_list.append(processed_data[test_subject][3])
        else:
            logger.warning(f"Test subject {test_subject} not found in processed data.")

    
    train_ppg = np.concatenate(train_ppg_list, axis=0).tolist()
    train_rr = np.concatenate(train_rr_list, axis=0).tolist()
    train_freq = np.concatenate(train_freq_list, axis=0).tolist()
    
    # --- DEBUG CHECK ---
    # If this prints a shape like (141, 7500), we have a problem. 
    # It should print a length (e.g., 7500).
    first_sample = np.array(train_ppg[0])
    logger.info(f"DEBUG: First training sample shape: {first_sample.shape}")
    if first_sample.ndim > 1:
        raise ValueError(f"Data not flattened! Expected 1D segment, got shape {first_sample.shape}")

    # --- STEP 2: BALANCE DATASET ---
    #Now we pass the flattened lists.
    train_ppg_bal, train_rr_bal, train_freq_bal = balance_dataset_with_synthesis(
        train_ppg, train_rr, train_freq
    )
    val_ppg = np.concatenate(val_ppg_list, axis=0).tolist()
    val_rr = np.concatenate(val_rr_list, axis=0).tolist()
    val_freq = np.concatenate(val_freq_list, axis=0).tolist()
    test_ppg = np.concatenate(test_ppg_list, axis=0).tolist()
    test_rr = np.concatenate(test_rr_list, axis=0).tolist()
    test_freq = np.concatenate(test_freq_list, axis=0).tolist()

    # --- CHANGE 4: Concatenate SSL Data ---
    train_ppg_ssl = np.concatenate(train_ppg_ssl_list, axis=0).tolist()
    val_ppg_ssl = np.concatenate(val_ppg_ssl_list, axis=0).tolist()
    test_ppg_ssl = np.concatenate(test_ppg_ssl_list, axis=0).tolist()
    # logger.info(f"train ppg shape: {train_ppg.shape}, train rr shape: {train_rr.shape}")
    # logger.info(f"val ppg shape: {val_ppg.shape}, val rr shape: {val_rr.shape}")
    # logger.info(f"test ppg shape: {test_ppg.shape}, test rr shape: {test_rr.shape}")
    id = cv_split["fold_id"]
    logger.info(f"total number of segments in this fold {id} is {len(train_ppg)+len(val_ppg)+len(test_ppg)}")
    logger.info(f"number of train segments is {len(train_ppg)}, number of val segments is{len(val_ppg)}, number of test segments is{len(test_ppg)}")

    logger.info(f"total number of subjects in this fold {id} is {len(train_subjects)+len(validation_subjects)+len(test_subjects)}")
    logger.info(f"number of train subjects is {len(train_subjects)}, number of val subjects is{len(validation_subjects)}, number of test subjects is{len(test_subjects)}")
    return {
        'train_ppg': train_ppg_bal,
        'train_rr': train_rr_bal,
        'train_freq': train_freq_bal,
        'train_ppg_ssl': train_ppg_ssl,
        'val_ppg': val_ppg,
        'val_rr': val_rr,
        'val_freq': val_freq,
        'val_ppg_ssl': val_ppg_ssl,
        'test_ppg': test_ppg,
        'test_rr': test_rr,
        'test_freq': test_freq,
        'train_subjects': train_subjects,
        'val_subjects': validation_subjects,
        'test_subjects': test_subjects
    }



class FirstBatchTimer(pl.Callback):
    def __init__(self):
        self._epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0 and self._epoch_start_time is not None:
            gap = time.time() - self._epoch_start_time
            print(f"⏱️ Time from epoch start (0%) to first batch (1%): {gap:.2f} sec")
            self._epoch_start_time = None

# def train_single_fold(fold_data, fold_id):
def setup_callbacks(cfg,fold_id):
    filename_prefix = f"best-checkpoint-fold{fold_id}-"
    checkpoint_callback = ModelCheckpoint(
        monitor='val/MAE',
        dirpath=cfg.training.checkpoint_dir,
        filename=filename_prefix + '{epoch:02d}-{val_mae:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
        verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/MAE',
        patience=cfg.training.early_stopping_patience,
        mode='min',
        min_delta=0.001,
        verbose=True
    )
    return [checkpoint_callback, early_stopping_callback]


def extract_all_segments(processed_data_dict):
    """
    Combines all segments from a processed data dictionary into single NumPy arrays.
    This version is corrected to handle lists of segments from each subject.
    """
    # Initialize empty lists to store all segments from all subjects
    all_ppg_flat = []
    all_rr_flat = []
    all_freq_flat = []

    logger.info(f"Extracting and flattening segments from {len(processed_data_dict)} subjects...")

    for subject_id, segments in processed_data_dict.items():
        ppg_segments, rr_segments, freq_segments = segments
        
        if len(ppg_segments) > 0:
            # <<< FIX: Use .extend() to add all segments to a flat list >>>
            # Instead of appending a list of segments, we add each segment individually.
            all_ppg_flat.extend(ppg_segments)
            all_rr_flat.extend(rr_segments)
            all_freq_flat.extend(freq_segments)
        else:
            logger.warning(f"Subject {subject_id} has no valid segments, skipping.")
    
    if not all_ppg_flat:
        logger.error("No segments found in any subject. Returning empty arrays.")
        return np.array([]), np.array([]), np.array([])

    # <<< FIX: Use np.array() to stack the list of 1D arrays into a 2D array >>>
    # This correctly converts a list of (segment_length,) arrays into a single
    # (num_total_segments, segment_length) array.
    final_ppg = np.array(all_ppg_flat)
    final_rr = np.array(all_rr_flat)
    final_freq = np.array(all_freq_flat)

    logger.info(f"Total combined segments extracted: {final_ppg.shape[0]}")

    return final_ppg, final_rr, final_freq

class TimeWarpSSLDataset(Dataset):
    def __init__(self, ppg_list, fs=125):
        self.ppg_list = ppg_list
        self.fs = fs
        # Label 0: Slow (Stretched)
        # Label 1: Normal
        # Label 2: Fast (Squashed)

    def __len__(self):
        return len(self.ppg_list)

    def __getitem__(self, idx):
        raw_ppg = self.ppg_list[idx]
        L = len(raw_ppg)
        
        choice = np.random.choice([0, 1, 2])
        
        if choice == 0: # SLOW (Stretch 1.2x - 1.5x)
            factor = np.random.uniform(1.2, 1.5)
            new_len = int(L * factor)
            aug_ppg = signal.resample(raw_ppg, new_len)
            # Crop center to fit original window
            start = (new_len - L) // 2
            aug_ppg = aug_ppg[start : start + L]
            
        elif choice == 1: # NORMAL (Identity + Noise)
            aug_ppg = raw_ppg + np.random.normal(0, 0.05, L) * np.std(raw_ppg)
            
        elif choice == 2: # FAST (Squash 0.6x - 0.8x)
            factor = np.random.uniform(0.6, 0.8)
            new_len = int(L * factor)
            aug_ppg_short = signal.resample(raw_ppg, new_len)
            # Pad with edge values to fill window
            pad_len = L - new_len
            aug_ppg = np.pad(aug_ppg_short, (0, pad_len), mode='edge')

        # Generate Scalogram (Must match main training config!)
        scalogram = generate_cwt_scalogram(
            aug_ppg, fs=self.fs, target_shape=(128, 60), fmin=0.1, fmax=0.8
        )
        
        return torch.tensor(scalogram).float(), torch.tensor(choice).long()
def setup_ssl_datamodule(cfg, fold_data, processed_capnobase_ssl, finetune_val_dataset):
    """
    Creates the data module for the SSL pre-training stage.
    Conditionally adds CapnoBase data based on the config.
    """
    # Start with the BIDMC training data, which is a list of segments
    ssl_train_ppg_list = list(fold_data['train_ppg'])
    ssl_train_rr_list = list(fold_data['train_rr'])
    ssl_train_freq_list = list(fold_data['train_freq'])

    # Conditionally add CapnoBase data if the flag is set
    if cfg.ssl.use_capnobase and processed_capnobase_ssl is not None:
        logger.info("Adding CapnoBase data to the SSL training set.")
        
        # This returns stacked 2D NumPy arrays
        capnobase_ppg_array, capnobase_rr_array, capnobase_freq_array = extract_all_segments(processed_capnobase_ssl)
        
        # <<< FIX: Convert the 2D NumPy arrays back into lists of 1D arrays >>>
        # Then, extend the original lists with the new data.
        if capnobase_ppg_array.ndim > 1 and capnobase_ppg_array.shape[0] > 0:
            ssl_train_ppg_list.extend(list(capnobase_ppg_array))
            ssl_train_rr_list.extend(list(capnobase_rr_array))
            ssl_train_freq_list.extend(list(capnobase_freq_array))

    else:
        logger.info("Using only BIDMC fold data for SSL training.")

    # <<< FIX: Convert the final combined lists into single, stacked NumPy arrays >>>
    final_ssl_train_ppg = np.array(ssl_train_ppg_list)
    final_ssl_train_rr = np.array(ssl_train_rr_list)
    final_ssl_train_freq = np.array(ssl_train_freq_list)

    # Create the final SSL DataModule with the correctly shaped arrays
    ssl_train_dataset = PPGRRDataset(final_ssl_train_ppg, final_ssl_train_rr, final_ssl_train_freq)
    
    ssl_data_module = PPGRRDataModule(
        ssl_train_dataset, finetune_val_dataset, None,
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers
    )
    return ssl_data_module
# def setup_ssl_datamodule(cfg, fold_data, processed_capnobase_ssl, finetune_val_dataset):
#     """
#     Creates the data module for the SSL pre-training stage.
#     Conditionally adds CapnoBase data based on the config.
#     """
#     # Start with the BIDMC training data for the current fold
#     ssl_train_ppg = fold_data['train_ppg']
#     ssl_train_rr = fold_data['train_rr']
#     ssl_train_freq = fold_data['train_freq']

#     # Conditionally add CapnoBase data if the flag is set
#     if cfg.ssl.use_capnobase and processed_capnobase_ssl is not None:
#         logger.info("Adding CapnoBase data to the SSL training set.")
#         capnobase_ppg, capnobase_rr, capnobase_freq = extract_all_segments(processed_capnobase_ssl)
        
#         # Combine with the BIDMC fold data
#         ssl_train_ppg = np.concatenate([ssl_train_ppg, capnobase_ppg], axis=0)
#         ssl_train_rr = np.concatenate([ssl_train_rr, capnobase_rr], axis=0)
#         ssl_train_freq = np.concatenate([ssl_train_freq, capnobase_freq], axis=0)
#     else:
#         logger.info("Using only BIDMC fold data for SSL training.")

#     # Create the final SSL DataModule
#     ssl_train_dataset = PPGRRDataset(ssl_train_ppg, ssl_train_rr, ssl_train_freq)
    
#     # It's best practice to validate on the in-domain data (BIDMC validation set)
#     ssl_data_module = PPGRRDataModule(
#         ssl_train_dataset, finetune_val_dataset, None, # No test set needed for SSL
#         batch_size=cfg.training.batch_size, 
#         num_workers=cfg.training.num_workers
#     )
#     return ssl_data_module

class SSLDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size, num_workers):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # <--- KEEPS WORKERS ALIVE (Crucial)
            prefetch_factor=4         # <--- PRE-LOADS 4 BATCHES PER WORKER
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
def train(cfg, cv_splits, processed_data, processed_capnobase_ssl):

    all_fold_results = []
    for cv_split in cv_splits:

        fold_id = cv_split["fold_id"]
        logger.info(f"--- Starting Fold {fold_id} ---")

        fold_data = create_data_splits(cv_split, processed_data)
        train_dataset = PPGRRDataset(cfg,fold_data['train_ppg'], fold_data['train_rr'], fold_data['train_freq'],
        augment=cfg.training.use_augmentation)
        val_dataset = PPGRRDataset(cfg,fold_data['val_ppg'], fold_data['val_rr'], fold_data['val_freq'],
        augment=False)
        test_dataset = PPGRRDataset(cfg,fold_data['test_ppg'], fold_data['test_rr'], fold_data['test_freq'],
        augment=False)

        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers
        data_module = PPGRRDataModule(train_dataset, val_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers)
        logger.info(f"train dataset shape is {train_dataset[0][0].shape}")

        if cfg.mode.optuna:
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
            )

            # Optimize
            study.optimize(lambda trial: objective(trial, cfg, fold_data), n_trials=3)

            # Print best params
            print("Best hyperparameters:", study.best_params)
            print("Best value:", study.best_value)

            # Optional: Save as JSON (from previous)
            
            best_results = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials)
            }
            with open("best_hparams.json", "w") as f:
                json.dump(best_results, f, indent=4, default=str)

            exit()

        if cfg.training.use_ssl_pretraining:
            
            ssl_model = None
            ssl_data_module = None
            pretrained_path = ""

            # --- BRANCH 1: Time Domain SSL ---
            if cfg.training.ablation_mode in ["time_only"]:
                logger.info(f"[Fold {fold_id}] Starting TIME SSL Pre-training...")
                # ... Your existing time-domain SSL setup ...
                ssl_data_module = setup_ssl_datamodule(cfg, fold_data, processed_capnobase_ssl, val_dataset)
                ssl_model = SSLPretrainModule(cfg)
                pretrained_path = f"fold_{fold_id}_time_encoder.pth"

            # --- BRANCH 2: Frequency Domain SSL (NEW) ---
            elif cfg.training.ablation_mode in ["freq_only"]:
                logger.info(f"[Fold {fold_id}] Starting FREQ SSL Pre-training (Time-Warp Classification)...")
                
                # 1. Setup Time-Warp Dataset
                # Use the flattened SSL data lists
                ssl_train_ds = TimeWarpSSLDataset(fold_data['train_ppg_ssl'], fs=125)
                ssl_val_ds   = TimeWarpSSLDataset(fold_data['val_ppg_ssl'], fs=125)
                total_cores = os.cpu_count()
                optimal_workers = max(4, min(total_cores - 2, 16))
                # Create DataModule manually to control sampler/shuffle
                ssl_data_module = SSLDataModule(
                    ssl_train_ds, 
                    ssl_val_ds, 
                    batch_size=cfg.training.batch_size, 
                    num_workers=optimal_workers # <--- INCREASED WORKERS
                )
                # Note: from_datasets usually sets shuffle=False for val by default, which is fine.
                # For train, it usually sets shuffle=True.
                
                # 2. Setup Classification Model
                ssl_model = SSLPretrainModule(cfg)
                pretrained_path = f"fold_{fold_id}_ssl_encoder.pth"

            # --- Common Training Logic ---
            ssl_logger = TensorBoardLogger(
                save_dir=cfg.logging.log_dir,
                name=cfg.logging.experiment_name,
                version=f'fold_{cv_split["fold_id"]}_ssl_{cfg.training.ablation_mode}'
            )

            ssl_checkpoint_callback = ModelCheckpoint(
                monitor='ssl_val_loss',
                dirpath=ssl_logger.log_dir,
                filename='ssl-best-checkpoint',
                save_top_k=1,
                mode='min'
            )
            
            ssl_trainer = pl.Trainer(
                max_epochs=cfg.ssl.max_epochs,
                accelerator="auto",
                strategy='ddp_find_unused_parameters_true',
                devices=cfg.hardware.devices,
                logger=ssl_logger,
                log_every_n_steps=1,
                callbacks=[ssl_checkpoint_callback, TQDMProgressBar(leave=True)]
            )

            ssl_trainer.fit(ssl_model, datamodule=ssl_data_module)
            
            # Load best and save encoder weights
            best_ssl_model_path = ssl_checkpoint_callback.best_model_path
            
            if cfg.training.ablation_mode == "time_only":
                best_model = SSLPretrainModule.load_from_checkpoint(best_ssl_model_path)
            else: 
                best_model = SSLPretrainModule.load_from_checkpoint(best_ssl_model_path)

            if not dist.is_initialized() or dist.get_rank() == 0:
                # Both modules have self.encoder, so this line works for both
                torch.save(best_model.encoder.state_dict(), pretrained_path)
                logger.info(f"Saved best pre-trained encoder to {pretrained_path}")

            if dist.is_initialized():
                dist.barrier()

            cfg.training.pretrained_path = pretrained_path

        else:
             logger.info(f"[Fold {fold_id}] SSL Pre-training: DISABLED.")
             cfg.training.pretrained_path = None
        # if cfg.training.ablation_mode in ["fusion", "time_only"]:
        #     # Second, check the new flag to see if we should run SSL
        #     if cfg.training.use_ssl_pretraining:

        #         logger.info(f"[Fold {fold_id}] SSL Pre-training: ENABLED.")
        #         logger.info(f"[Fold {fold_id}] Starting SSL Pre-training for mode '{cfg.training.ablation_mode}'...")
        #         # 1. Setup Logger for the SSL phase
        #         ssl_logger = TensorBoardLogger(
        #             save_dir=cfg.logging.log_dir,
        #             name=cfg.logging.experiment_name,
        #             version=f'fold_{cv_split["fold_id"]}_ssl' # Appending '_ssl' to differentiate logs
        #         )

        #         # 2. Setup Callbacks for the SSL phase
        #         # This will save the model with the best validation loss
        #         ssl_checkpoint_callback = ModelCheckpoint(
        #             monitor='val_loss',
        #             dirpath=ssl_logger.log_dir, # Save checkpoints in the same folder as logs
        #             filename='ssl-best-checkpoint',
        #             save_top_k=1,
        #             mode='min'
        #         )

        #         # This will stop training if the validation loss doesn't improve for 5 epochs
        #         ssl_early_stopping_callback = EarlyStopping(
        #             monitor='val_loss',
        #             patience=5, # Number of epochs with no improvement after which training will be stopped.
        #             verbose=True,
        #             mode='min'
        #         )
        #         progress_bar = TQDMProgressBar(leave=True)
        #          # Create the specific datamodule for SSL
        #         ssl_data_module = setup_ssl_datamodule(cfg, fold_data, processed_capnobase_ssl, val_dataset)
        #         ssl_model = SSLPretrainModule(cfg)
        #         ssl_trainer = pl.Trainer(
        #             max_epochs=cfg.ssl.max_epochs,
        #             accelerator="auto",
        #             devices=cfg.hardware.devices,
        #             # strategy='ddp_find_unused_parameters_true',
        #             # detect_anomaly=True,
        #             logger=ssl_logger,
        #             log_every_n_steps=1,
        #             callbacks=[ssl_checkpoint_callback, ssl_early_stopping_callback, progress_bar]
        #         )
        #         # Start pre-training
        #         ssl_trainer.fit(ssl_model, datamodule=ssl_data_module)
        #         best_ssl_model = SSLPretrainModule.load_from_checkpoint(ssl_checkpoint_callback.best_model_path)
        #         pretrained_path = f"fold_{fold_id}_encoder.pth"
        #         # Save only on rank 0 to avoid corruption in DDP
        #         if not dist.is_initialized() or dist.get_rank() == 0:
        #             print(f"Rank {dist.get_rank()} is saving the model...")
        #             torch.save(best_ssl_model.encoder.state_dict(), pretrained_path)

        #         # This barrier should be OUTSIDE the if statement.
        #         # All processes must call it.
        #         if dist.is_initialized():
        #             dist.barrier()

        #         print(f"Rank {dist.get_rank()} has passed the barrier and will now proceed to loading.")
        #         # torch.save(best_ssl_model.encoder.state_dict(), pretrained_path)
        #         logger.info(f"[Fold {fold_id}]Saved best pre-trained encoder to {pretrained_path}")
        #         # Set the path in the config for the fine-tuning stage
        #         cfg.training.pretrained_path = pretrained_path
        #     else:
        #         # This is the new "ablation" path where we skip SSL
        #         logger.info(f"[Fold {fold_id}] SSL Pre-training: DISABLED (Training from scratch).")
        #         cfg.training.pretrained_path = None # Ensure no pre-trained model is loaded
        
        # else: # This block handles the 'freq_only' case
        #     logger.info(f"[Fold {fold_id}] Skipping SSL Pre-training for mode '{cfg.training.ablation_mode}'.")
        #     # Ensure the path is not set from a previous run
        #     cfg.training.pretrained_path = None

         # --- The fine-tuning stage proceeds from here ---
        logger.info(f"[Fold {fold_id}] Starting Supervised Fine-tuning...")

        callbacks = setup_callbacks(cfg, fold_id)
        # Setup logger
        tblogger = TensorBoardLogger(
            save_dir=cfg.logging.log_dir,
            name=cfg.logging.experiment_name,
            version=f'fold_{cv_split["fold_id"]}'
        )

        profiler = SimpleProfiler(dirpath=f"profiles/fold_{fold_id}", filename="profiler_summary.txt")  # Saves to file
        ddp_strategy = DDPStrategy(find_unused_parameters=False)
        fine_tune_trainer = pl.Trainer(max_epochs=cfg.training.max_epochs,
                             accelerator="auto",
                             devices=cfg.hardware.devices,
                             strategy='ddp_find_unused_parameters_true',
                            #  detect_anomaly=True,
                             callbacks=callbacks,
                             logger=tblogger,
                             enable_progress_bar=True,
                             log_every_n_steps=1,
                             gradient_clip_val=cfg.training.gradient_clip_val,
                             accumulate_grad_batches=1,
                             profiler=profiler,
                             benchmark=False
                             )
        model = RRLightningModule(cfg)
        
        fine_tune_trainer.fit(model, data_module)
        # Write profiler summary to file
        os.makedirs("profiles", exist_ok=True)
        summary = profiler.summary()
        with open("profiles/profiler_summary.txt", "w") as f:
            f.write(summary)
        test_reults = fine_tune_trainer.test(model, datamodule=data_module, ckpt_path="best")
        all_fold_results.append({
            "fold_id": fold_id,
            "test_results": test_reults[0]
        })
        logger.info(f"Completed training for fold {fold_id}. Test results: {test_reults}")

    return all_fold_results



@hydra.main(config_path="../", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    raw_data = read_data("data/")
    # print(raw_data['07'][1])
    # exit()
    processed_data = process_data(cfg, raw_data)


    print(f"processed data length: {len(processed_data)}")
    print(f"processed data for subjects")
    
    count_zero = 0
    segment_counts = {}
    for subject_id, (ppg_segments, rr_segments, freq_segments, ppg_segments_ssl) in processed_data.items():
        
        n_segments = len(rr_segments)   # or entry["PPG"].shape[0]
        segment_counts[subject_id] = n_segments
        if n_segments == 0: count_zero += 1

    print(segment_counts)
    logger.info(f"number of subjects with zero segments is: {count_zero}")

    min_len = min(segment_counts.values())
    logger.info(f"minimum number of segments across subjects: {min_len}")
    

    processed_capnobase_ssl = None  # Initialize to None
    if cfg.ssl.use_capnobase:
        logger.info("Loading and processing CapnoBase dataset for SSL pre-training...")
        capnobase_raw_data = read_capnobase_data("data/")
        processed_capnobase_ssl = process_ssl_data(cfg, capnobase_raw_data)
    else:
        logger.info("Skipping CapnoBase dataset loading as per config.")


    # cv_splits = create_balanced_folds(processed_data, n_splits=5)
    cv_splits = create_folds(processed_data, n_splits=5)
    logger.info(f"Created folds: {cv_splits}")


    # Collect all test subjects across folds
    all_test_subjects = set()
    for fold in cv_splits:
        all_test_subjects.update(fold["test_subjects"])

    # Collect all subjects in the dataset
    all_subjects = set(processed_data.keys())

    # Check coverage
    missing_subjects = all_subjects - all_test_subjects
    extra_subjects = all_test_subjects - all_subjects

    print(f"✅ Total subjects: {len(all_subjects)}")
    print(f"✅ Subjects covered in test sets: {len(all_test_subjects)}")
    print(f"🧩 Missing subjects in test folds: {missing_subjects if missing_subjects else 'None'}")
    print(f"⚠️ Unexpected subjects: {extra_subjects if extra_subjects else 'None'}")

    all_fold_results = train(cfg, cv_splits, processed_data, processed_capnobase_ssl)
    
    for fold_result in all_fold_results:
        logger.info(f"Fold {fold_result['fold_id']} test results: {fold_result['test_results']}")

    # Summarize all fold results

    print(f"\nTraining completed!")
    print(f"all fold results: {all_fold_results}")
    all_maes = [fold_result['test_results']['test/MAE'] for fold_result in all_fold_results]
    print(f"Average MAE across folds: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
    print("Hello, World!")


if __name__ == "__main__":
    main()