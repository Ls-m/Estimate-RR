from torch.utils.data import Dataset
import numpy as np
import logging
import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger("Dataset")

import torchaudio.transforms as T

class PPGRRDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, ppg_data, rr_labels, freq_data, augment=False):
        # ... existing init ...
        self.augment = augment
        self.ppg_data = ppg_data
        self.rr_data = rr_labels
        self.freq_data = freq_data
        self.cfg = cfg

        if np.any(np.isnan(ppg_data)) or np.any(np.isinf(ppg_data)):
            logger.info(f"nan or inf in ppg_data")
        
        if np.any(np.isnan(rr_labels)) or np.any(np.isinf(rr_labels)):
            logger.info(f"nan or inf in rr_data")

        # Define augmentations
        # Mask up to 20 frequency bins (out of 128)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)
        # Mask up to 10 time steps (out of 60)
        self.time_masking = T.TimeMasking(time_mask_param=10)

    def __getitem__(self, idx):
        # ... load data ...
        # Let's assume 'scalogram' is your (128, 60) numpy array
        ppg_segment = self.ppg_data[idx]
        rr = self.rr_data[idx]
        freq = self.freq_data[idx]
        scalogram_tensor = torch.tensor(freq) # (128, 60)
        
        if self.augment:
            # SpecAugment expects (Channel, Freq, Time)
            scalogram_tensor = scalogram_tensor.unsqueeze(0)
            
            # Apply masking
            scalogram_tensor = self.freq_masking(scalogram_tensor)
            scalogram_tensor = self.time_masking(scalogram_tensor)
            
            # Remove channel dim
            scalogram_tensor = scalogram_tensor.squeeze(0)

        ppg_tensor = torch.tensor(ppg_segment, dtype=torch.float32)
        rr_tensor = torch.tensor(rr, dtype=torch.float32)
        return ppg_tensor, rr_tensor, scalogram_tensor

# class PPGRRDataset(Dataset):
#     def __init__(self, cfg, ppg_data, rr_data, freq_data, augment=False):
#         self.ppg_data = ppg_data
#         self.rr_data = rr_data
#         self.freq_data = freq_data
#         self.augment = augment
#         self.cfg = cfg

#         if np.any(np.isnan(ppg_data)) or np.any(np.isinf(ppg_data)):
#             logger.info(f"nan or inf in ppg_data")
        
#         if np.any(np.isnan(rr_data)) or np.any(np.isinf(rr_data)):
#             logger.info(f"nan or inf in rr_data")


#     def __len__(self):
#         return len(self.ppg_data)

#     def __getitem__(self, idx):
#         ppg_segment = self.ppg_data[idx]
#         rr = self.rr_data[idx]
#         freq = self.freq_data[idx]

#         if self.augment:
#             ppg_segment = np.array(ppg_segment, dtype=np.float32)
#             if torch.rand(1) < 0.5:
#                 noise_std = self.cfg.training.noise_std * np.std(ppg_segment)
#                 if noise_std > 1e-4: # Avoid adding noise to a flat-line signal
#                     noise = np.random.normal(loc=0, scale=noise_std, size=ppg_segment.shape)
#                     ppg_segment += noise
                
            
#             if torch.rand(1) < 0.5:
#                 scale_factor = np.random.uniform(0.8, 1.2)  
#                 ppg_segment = ppg_segment * scale_factor

#             ppg_segment = ppg_segment.tolist()
        
#         ppg_tensor = torch.tensor(ppg_segment, dtype=torch.float32)
#         rr_tensor = torch.tensor(rr, dtype=torch.float32)
#         freq_tensor = torch.tensor(freq, dtype=torch.float32)
#         return ppg_tensor, rr_tensor, freq_tensor


class PPGRRDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    


import pywt
import numpy as np
import torch
import scipy.signal as signal

def generate_2d_scalogram(ppg_segment, fs=30, fmin=0.1, fmax=0.6, num_scales=64):
    """
    Generates a 2D CWT scalogram for the SSL task.
    Output shape: (num_scales, time_steps)
    """
    dt = 1.0 / fs
    fc = pywt.central_frequency('morl')
    
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    scales = np.linspace(scale_min, scale_max, num_scales)
    
    # Apply CWT
    cwt_coeffs, _ = pywt.cwt(ppg_segment, scales, 'morl', sampling_period=dt)
    
    # Take magnitude
    cwt_img = np.abs(cwt_coeffs)
    
    # Resize to fixed size if necessary (e.g., 64x64) to satisfy CNN
    # We assume input length is fixed (e.g. 60s * 30Hz = 1800 samples). 
    # We simply downsample the time axis to 64 to make the image square-ish for the CNN.
    target_time_dim = 64
    
    # Simple linear interpolation to resize time dimension
    cwt_img_tensor = torch.tensor(cwt_img).unsqueeze(0) # (1, H, W_old)
    cwt_resized = torch.nn.functional.interpolate(
        cwt_img_tensor.unsqueeze(0), # (1, 1, H, W_old)
        size=(num_scales, target_time_dim), 
        mode='bilinear', 
        align_corners=False
    )
    
    return cwt_resized.squeeze(0).squeeze(0) # Returns (64, 64)


class FrequencySSLDataset(torch.utils.data.Dataset):
    def __init__(self, ppg_segments, fs=30):
        """
        ppg_segments: list or array of raw PPG time-domain segments (N, time_steps)
        """
        self.ppg_segments = ppg_segments
        self.fs = fs

    def __len__(self):
        return len(self.ppg_segments)

    def __getitem__(self, idx):
        raw_ppg = self.ppg_segments[idx]
        
        # 1. REMOVE REAL RESPIRATION
        # High-pass filter > 0.8 Hz to keep only cardiac, removing real breath
        sos = signal.butter(4, 0.8, 'hp', fs=self.fs, output='sos')
        clean_cardiac = signal.sosfilt(sos, raw_ppg)
        
        # 2. GENERATE SYNTHETIC BREATH
        # Random freq between 0.1 (6 bpm) and 0.5 (30 bpm)
        target_freq = np.random.uniform(0.1, 0.5)
        
        # Create modulation wave
        t = np.arange(len(raw_ppg)) / self.fs
        mod_depth = np.random.uniform(0.1, 0.5) # Random intensity
        resp_wave = 1 + mod_depth * np.sin(2 * np.pi * target_freq * t)
        
        # 3. INJECT
        augmented_ppg = clean_cardiac * resp_wave
        
        # 4. CONVERT TO 2D SCALOGRAM
        # Result is (64, 64) tensor
        scalogram = generate_2d_scalogram(augmented_ppg, fs=self.fs)
        
        # Return Input (Image) and Label (Frequency float)
        return scalogram.float(), torch.tensor(target_freq, dtype=torch.float32)