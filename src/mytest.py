import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import openpyxl
from tqdm import tqdm
import hydra
from utils import *
from dataset import PPGRRDataset, PPGRRDataModule
from model import RRLightningModule, SSLPretrainModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
logger = logging.getLogger("SleepData")

def rename2():
    ppg_folder = "data/sleep/csv"
    rr_folder = "data/sleep/Gold-Standard"
    for filename in os.listdir(ppg_folder):
        if filename.endswith(".csv"):
            for filename_rr in os.listdir(rr_folder):
                if filename_rr.endswith(".xlsx") and filename[:2] == filename_rr[:2]:
                    filename_rr_new = filename[:4] + ".xlsx"
                    os.rename(os.path.join(rr_folder, filename_rr), os.path.join(rr_folder, filename_rr_new))
                    print(f"Renamed: {filename_rr} → {filename_rr_new}")
def rename_files():
    # Path to your folder
    folder = "data/sleep/Gold-Standard"  # <-- change this to your folder path

    for filename in os.listdir(folder):
        if filename.endswith(".xlsx"):
            old_path = os.path.join(folder, filename)

            # Extract first 2 characters (before extension)
            prefix = filename[:2]
            new_filename = prefix + ".xlsx"
            new_path = os.path.join(folder, new_filename)

            # If multiple files share the same prefix, add a numeric suffix to avoid overwriting
            count = 1
            while os.path.exists(new_path):
                new_filename = f"{prefix}_{count}.xlsx"
                new_path = os.path.join(folder, new_filename)
                count += 1

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")


def load_subjects_sleep(path):
    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        return None
    subjects = []
    for file_name in os.listdir(path):
        if not file_name.endswith(".csv"):
            continue
        subject_id = os.path.splitext(file_name)[0]
        if subject_id not in subjects:
            subjects.append(subject_id)
    return subjects


def load_files_sleep(path,subjects):
    print(f"subjects are {subjects}")
    raw_data = {}
    min_RR = float("inf")
    min_subject = []
    for subject in tqdm.tqdm(subjects):
        print(f"Loading data for subject: {subject}")
        signal_file = path+"/csv/"+subject+".csv"
        numeric_file = path+"/Gold-Standard/"+subject+".xlsx"

        print(f"Signal file: {signal_file}")
        print(f"Numeric file: {numeric_file}")
        signal_df = pd.read_csv(signal_file, sep='\t', low_memory=False) if os.path.exists(signal_file) else None
        numeric_df = pd.read_excel(numeric_file) if os.path.exists(numeric_file) else None

        signal_df.columns = signal_df.columns.str.strip()
        numeric_df.columns = numeric_df.columns.str.strip()

        signal_df = signal_df.iloc[1:].reset_index(drop=True)

        for col in signal_df.columns:
            signal_df[col] = pd.to_numeric(signal_df[col], errors='coerce')
        
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

        print(signal_df.columns)
        ppg = signal_df["PPG"].values
        rr = numeric_df["RR count, actual"].values
        min_RR = min(min_RR, np.nanmin(rr))
        if np.nanmin(rr) == 0:
            min_subject.append(subject)
        if np.any(np.isnan(ppg)) or np.any(np.isinf(ppg)):
            logger.info(f"PPG data contains NaN or Inf values for subject: {subject}")

        if np.any(np.isnan(rr)) or np.any(np.isinf(rr)):
            logger.info(f"RR data contains NaN or Inf values for subject: {subject}")

        raw_data[subject] = (ppg, rr)

    logger.info(f"Minimum RR subjects for sleep dataset: {min_subject}")
    logger.info(f"Minimum RR for sleep dataset: {min_RR}")
    for subject, (ppg, rr) in raw_data.items():
        logger.debug(f"for subject {subject} PPG length: {len(ppg)}, RR length: {len(rr)}")

    return raw_data

def read_data_sleep(path):
    # Code to read data goes here
    for dataset_name in os.listdir(path):
        dataset_path = os.path.join(path,dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if dataset_name == "sleep":
            subjects = load_subjects_sleep(dataset_path+"/csv")
            raw_data = load_files_sleep(dataset_path, subjects)
            print(len(subjects))
            break
            # print(raw_data)
    return raw_data

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

def read_data_bidmc(path):
    # Code to read data goes here
    for dataset_name in os.listdir(path):
        dataset_path = os.path.join(path,dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if dataset_name == "bidmc":
            subjects = load_subjects_bidmc(dataset_path)
            raw_data = load_files_bidmc(dataset_path, subjects)
            print(len(subjects))
            break
            # print(raw_data)
    return raw_data

def process_data(cfg, raw_data, dataset_name='sleep'):
    # Code to process data goes here
    processed_data = {}
    # for subject_data in raw_data:
    for subject_id, data_tuple in raw_data.items():
        # Example processing: normalize PPG and RR signals
        # ppg = subject_data["PPG"]
        # rr = subject_data["RR"]

        ppg = data_tuple[0]
        rr = data_tuple[1]
        if dataset_name == "sleep":
            original_rate = 256
        if dataset_name == "bidmc":
            original_rate = 125

        check_effect = False
        
        if cfg.preprocessing.use_denoiser:
            logger.info(f"Subject {subject_id}: Running with denoiser ENABLED.")
            _, merged_segments = edpa_denoiser(ppg, original_rate, check_effect=check_effect)
            if merged_segments:
                ppg_reconstructed, segments_to_remove = reconstruct_noise(ppg, merged_segments, original_rate)
                expanded_removed_segments = expand_removals_to_second_blocks(segments_to_remove, fs=original_rate)
                ppg_denoised = apply_removals(ppg_reconstructed, expanded_removed_segments)
                rr_labels_denoised = remove_corresponding_labels(rr, expanded_removed_segments, original_rate, 1)
            else:
                logger.info(f"for this subject {subject_id} there is no noise detected!")
                expanded_removed_segments = []
                ppg_denoised = ppg
                rr_labels_denoised = rr
        else:
            # This is the ablation path: the denoiser is skipped entirely.
            logger.info(f"Subject {subject_id}: Running with denoiser DISABLED (Ablation Study).")
            ppg_denoised = ppg
            rr_labels_denoised = rr
            expanded_removed_segments = []
        
        ppg_filtered = apply_bandpass_filter(cfg, ppg_denoised, original_rate)
        if np.any(np.isnan(ppg_filtered)):
            logger.info(f"after bandpass filter NaNs in PPG: {np.isnan(ppg_filtered).sum()} ")

        # check_bandpass_filter_effect(ppg_denoised, ppg_filtered, original_rate, cfg.preprocessing.bandpass_filter.low_freq, cfg.preprocessing.bandpass_filter.high_freq)

        ppg_cliped,lower_band, higher_band = remove_outliers(ppg_filtered)
        # check_outliers_removal_effect(ppg_filtered, ppg_cliped, original_rate, lower_band, higher_band)

        ppg_normalized = normalize_signal(ppg_cliped)
        # check_normalization_effect(ppg_cliped, ppg_normalized, original_rate)

        
        ppg_segments, rr_segments = create_segments_with_gap_handling(
            subject_id,
            ppg_normalized,
            rr_labels_denoised,
            original_len=len(ppg),
            removed_segments=expanded_removed_segments,
            ppg_fs=original_rate,
            rr_fs=1,
            window_size_sec=32,
            step_size_sec=2
        )
        logger.info(f"for this subject {subject_id} the number of segments is {len(ppg_segments)}")
        # plot_cwt_scalogram(ppg_segments[0], original_rate)
        n_jobs = max(1, cpu_count() - 6)
        freq_segments = compute_freq_features(ppg_segments, original_rate, n_jobs=n_jobs)
        if freq_segments is None:
            logger.info(f"for this subject {subject_id} there is no frequency segments detected!")
            continue
        # check_freq_features(freq_segments, rr_segments, subject_id)
        # processed_data.append({
        #     "subject_id": subject_id,
        #     "PPG": X,
        #     "RR": y
        # })

        processed_data[subject_id] = (ppg_segments, rr_segments, freq_segments)
        # logger.info(f"processed data is {processed_data}")
        
    return processed_data

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
            step_size_sec=2
        )
        
        if not ppg_segments:
            logger.warning(f"No valid PPG segments for subject {subject_id} after processing. Skipping.")
            continue

        # --- CWT Feature computation (remains the same) ---
        n_jobs = max(1, cpu_count() - 6)
        freq_segments = compute_freq_features(ppg_segments, original_rate, n_jobs=n_jobs)
        
        processed_data[subject_id] = (ppg_segments, rr_segments, freq_segments)
        
    return processed_data



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


def setup_ssl_datamodule(cfg, fold_data, processed_bidmc_ssl, processed_capnobase_ssl, finetune_val_dataset):
    """
    Creates the data module for the SSL pre-training stage.
    Conditionally adds CapnoBase data based on the config.
    """
    # Start with the BIDMC training data, which is a list of segments
    ssl_train_ppg_list = list(fold_data['train_ppg'])
    ssl_train_rr_list = list(fold_data['train_rr'])
    ssl_train_freq_list = list(fold_data['train_freq'])

    use_capnobase = cfg.ssl.use_capnobase and processed_capnobase_ssl is not None
    use_bidmc = cfg.ssl.use_bidmc and processed_bidmc_ssl is not None
    if use_bidmc and use_capnobase:
        logger.info("Adding BIDMC and CapnoBase data to the SSL training set.")
        bidmc_ppg_array, bidmc_rr_array, bidmc_freq_array = extract_all_segments(processed_bidmc_ssl)
        if bidmc_ppg_array.ndim > 1 and bidmc_ppg_array.shape[0] > 0:
            ssl_train_ppg_list.extend(list(bidmc_ppg_array))
            ssl_train_rr_list.extend(list(bidmc_rr_array))
            ssl_train_freq_list.extend(list(bidmc_freq_array))
        capnobase_ppg_array, capnobase_rr_array, capnobase_freq_array = extract_all_segments(processed_capnobase_ssl)
        if capnobase_ppg_array.ndim > 1 and capnobase_ppg_array.shape[0] > 0:
            ssl_train_ppg_list.extend(list(capnobase_ppg_array))
            ssl_train_rr_list.extend(list(capnobase_rr_array))
            ssl_train_freq_list.extend(list(capnobase_freq_array))
    elif use_bidmc and not use_capnobase:
        logger.info("Adding BIDMC data to the SSL training set.")
        bidmc_ppg_array, bidmc_rr_array, bidmc_freq_array = extract_all_segments(processed_bidmc_ssl)
        if bidmc_ppg_array.ndim > 1 and bidmc_ppg_array.shape[0] > 0:
            ssl_train_ppg_list.extend(list(bidmc_ppg_array))
            ssl_train_rr_list.extend(list(bidmc_rr_array))
            ssl_train_freq_list.extend(list(bidmc_freq_array))
    # Conditionally add CapnoBase data if the flag is set
    elif not use_bidmc and use_capnobase:
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
        logger.info("Using only SLEEP fold data for SSL training.")

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

def create_data_splits(cv_split, processed_data):

    train_subjects = cv_split["train_subjects"]
    validation_subjects = cv_split["val_subjects"]
    test_subjects = cv_split["test_subjects"]

    subject_ids = [subject_id for subject_id, (ppg_segments, rr_segments, freq_segments) in processed_data.items()]

    train_ppg_list = []
    train_rr_list = []
    train_freq_list = []
    for train_subject in train_subjects:
        if train_subject in subject_ids:
            train_ppg_list.append(processed_data[train_subject][0])
            train_rr_list.append(processed_data[train_subject][1])
            train_freq_list.append(processed_data[train_subject][2])
        else:
            logger.warning(f"Train subject {train_subject} not found in processed data.")
    
    val_ppg_list = []
    val_rr_list = []
    val_freq_list = []
    for val_subject in validation_subjects:
        if val_subject in subject_ids:
            val_ppg_list.append(processed_data[val_subject][0])
            val_rr_list.append(processed_data[val_subject][1])
            val_freq_list.append(processed_data[val_subject][2])
        else:
            logger.warning(f"Validation subject {val_subject} not found in processed data.")

    test_ppg_list = []
    test_rr_list = []
    test_freq_list = []
    for test_subject in test_subjects:
        if test_subject in subject_ids:
            test_ppg_list.append(processed_data[test_subject][0])
            test_rr_list.append(processed_data[test_subject][1])
            test_freq_list.append(processed_data[test_subject][2])
        else:
            logger.warning(f"Test subject {test_subject} not found in processed data.")
    train_ppg = np.concatenate(train_ppg_list, axis=0)
    train_rr = np.concatenate(train_rr_list, axis=0)
    train_freq = np.concatenate(train_freq_list, axis=0)
    val_ppg = np.concatenate(val_ppg_list, axis=0)
    val_rr = np.concatenate(val_rr_list, axis=0)
    val_freq = np.concatenate(val_freq_list, axis=0)
    test_ppg = np.concatenate(test_ppg_list, axis=0)
    test_rr = np.concatenate(test_rr_list, axis=0)
    test_freq = np.concatenate(test_freq_list, axis=0)
    logger.info(f"train ppg shape: {train_ppg.shape}")

    return {
        'train_ppg': train_ppg,
        'train_rr': train_rr,
        'train_freq': train_freq,
        'val_ppg': val_ppg,
        'val_rr': val_rr,
        'val_freq': val_freq,
        'test_ppg': test_ppg,
        'test_rr': test_rr,
        'test_freq': test_freq,
        'train_subjects': train_subjects,
        'val_subjects': validation_subjects,
        'test_subjects': test_subjects
    }

def train(cfg, cv_splits, processed_data, processed_bidmc_ssl, processed_capnobase_ssl):

    all_fold_results = []
    for cv_split in cv_splits:

        fold_id = cv_split["fold_id"]
        logger.info(f"--- Starting Fold {fold_id} ---")

        fold_data = create_data_splits(cv_split, processed_data)
        train_dataset = PPGRRDataset(fold_data['train_ppg'], fold_data['train_rr'], fold_data['train_freq'])
        val_dataset = PPGRRDataset(fold_data['val_ppg'], fold_data['val_rr'], fold_data['val_freq'])
        test_dataset = PPGRRDataset(fold_data['test_ppg'], fold_data['test_rr'], fold_data['test_freq'])

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

        if cfg.training.ablation_mode in ["fusion", "time_only"]:
            # Second, check the new flag to see if we should run SSL
            if cfg.training.use_ssl_pretraining:

                logger.info(f"[Fold {fold_id}] SSL Pre-training: ENABLED.")
                logger.info(f"[Fold {fold_id}] Starting SSL Pre-training for mode '{cfg.training.ablation_mode}'...")
                # 1. Setup Logger for the SSL phase
                ssl_logger = TensorBoardLogger(
                    save_dir=cfg.logging.log_dir,
                    name=cfg.logging.experiment_name,
                    version=f'fold_{cv_split["fold_id"]}_ssl' # Appending '_ssl' to differentiate logs
                )

                # 2. Setup Callbacks for the SSL phase
                # This will save the model with the best validation loss
                ssl_checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=ssl_logger.log_dir, # Save checkpoints in the same folder as logs
                    filename='ssl-best-checkpoint',
                    save_top_k=1,
                    mode='min'
                )

                # This will stop training if the validation loss doesn't improve for 5 epochs
                ssl_early_stopping_callback = EarlyStopping(
                    monitor='val_loss',
                    patience=5, # Number of epochs with no improvement after which training will be stopped.
                    verbose=True,
                    mode='min'
                )
                progress_bar = TQDMProgressBar(leave=True)
                 # Create the specific datamodule for SSL
                ssl_data_module = setup_ssl_datamodule(cfg, fold_data, processed_bidmc_ssl, processed_capnobase_ssl, val_dataset)
                ssl_model = SSLPretrainModule(cfg)
                ssl_trainer = pl.Trainer(
                    max_epochs=cfg.ssl.max_epochs,
                    accelerator="auto",
                    devices=cfg.hardware.devices,
                    # strategy='ddp_find_unused_parameters_true',
                    logger=ssl_logger,
                    log_every_n_steps=1,
                    callbacks=[ssl_checkpoint_callback, ssl_early_stopping_callback, progress_bar]
                )
                # Start pre-training
                ssl_trainer.fit(ssl_model, datamodule=ssl_data_module)
                best_ssl_model = SSLPretrainModule.load_from_checkpoint(ssl_checkpoint_callback.best_model_path)
                pretrained_path = f"fold_{fold_id}_encoder.pth"
                torch.save(best_ssl_model.encoder.state_dict(), pretrained_path)
                logger.info(f"[Fold {fold_id}]Saved best pre-trained encoder to {pretrained_path}")
                # Set the path in the config for the fine-tuning stage
                cfg.training.pretrained_path = pretrained_path
            else:
                # This is the new "ablation" path where we skip SSL
                logger.info(f"[Fold {fold_id}] SSL Pre-training: DISABLED (Training from scratch).")
                cfg.training.pretrained_path = None # Ensure no pre-trained model is loaded
        
        else: # This block handles the 'freq_only' case
            logger.info(f"[Fold {fold_id}] Skipping SSL Pre-training for mode '{cfg.training.ablation_mode}'.")
            # Ensure the path is not set from a previous run
            cfg.training.pretrained_path = None

         # --- The fine-tuning stage proceeds from here ---
        logger.info(f"[Fold {fold_id}] Starting Supervised Fine-tuning...")

        callbacks = setup_callbacks(cfg)
        # Setup logger
        tblogger = TensorBoardLogger(
            save_dir=cfg.logging.log_dir,
            name=cfg.logging.experiment_name,
            version=f'fold_{cv_split["fold_id"]}'
        )

        fine_tune_trainer = pl.Trainer(max_epochs=cfg.training.max_epochs,
                             accelerator="auto",
                             devices=cfg.hardware.devices,
                            #  strategy='ddp_find_unused_parameters_true',
                             callbacks=callbacks,
                             logger=tblogger,
                             enable_progress_bar=True,
                             log_every_n_steps=1,
                             gradient_clip_val=cfg.training.gradient_clip_val,
                             accumulate_grad_batches=1,
                             benchmark=False
                             )
        model = RRLightningModule(cfg)
        
        fine_tune_trainer.fit(model, data_module)
        test_reults = fine_tune_trainer.test(model, datamodule=data_module, ckpt_path="best")
        all_fold_results.append({
            "fold_id": fold_id,
            "test_results": test_reults[0]
        })
        logger.info(f"Completed training for fold {fold_id}. Test results: {test_reults}")

    return all_fold_results




@hydra.main(config_path="../", config_name="config", version_base="1.3")
def main(cfg):

    logger.info("Starting SLEEP data loading and processing...")
    raw_data_sleep = read_data_sleep("data")
    processed_data_sleep = process_data(cfg,raw_data_sleep,dataset_name='sleep')

    print(f"processed data length: {len(processed_data_sleep)}")
    print(f"processed data for SLEEP subjects")
    
    count_zero = 0
    segment_counts = {}
    for subject_id, (ppg_segments, rr_segments, freq_segments) in processed_data_sleep.items():
        
        n_segments = len(rr_segments)   # or entry["PPG"].shape[0]
        segment_counts[subject_id] = n_segments
        if n_segments == 0: count_zero += 1

    print(segment_counts)
    logger.info(f"number of subjects with zero segments is: {count_zero}")
    min_len = min(segment_counts.values())
    logger.info(f"minimum number of segments across subjects: {min_len}")

    processed_bidmc_ssl = None
    if cfg.ssl.use_bidmc:
        logger.info("Loading and processing BIDMC dataset for SSL pre-training...")
        raw_data_bidmc = read_data_bidmc("data")
        processed_bidmc_ssl = process_data(cfg,raw_data_bidmc,dataset_name='bidmc')

        print(f"processed data length: {len(processed_bidmc_ssl)}")
        print(f"processed data for BIDMC subjects")
    else:
        logger.info("Skipping BIDMC dataset loading as per config.")

    processed_capnobase_ssl = None  # Initialize to None
    if cfg.ssl.use_capnobase:
        logger.info("Loading and processing CapnoBase dataset for SSL pre-training...")
        capnobase_raw_data = read_capnobase_data("data/")
        processed_capnobase_ssl = process_ssl_data(cfg, capnobase_raw_data)

        print(f"processed data length: {len(processed_capnobase_ssl)}")
        print(f"processed data for CapnoBase subjects")
    else:
        logger.info("Skipping CapnoBase dataset loading as per config.")

    cv_splits = create_balanced_folds(processed_data_sleep, n_splits=5)
    logger.info(f"Created folds: {cv_splits}")

    all_fold_results = train(cfg, cv_splits, processed_data_sleep, processed_bidmc_ssl, processed_capnobase_ssl)

    for fold_result in all_fold_results:
        logger.info(f"Fold {fold_result['fold_id']} test results: {fold_result['test_results']}")

    # Summarize all fold results

    print(f"\nTraining completed!")
    print(f"all fold results: {all_fold_results}")
    all_maes = [fold_result['test_results']['test_mae'] for fold_result in all_fold_results]
    print(f"Average MAE across folds: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
    print("Hello, World!")



if __name__ == "__main__":
    main()




