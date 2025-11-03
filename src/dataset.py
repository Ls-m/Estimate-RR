from torch.utils.data import Dataset
import numpy as np
import logging
import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger("Dataset")


class PPGRRDataset(Dataset):
    def __init__(self, ppg_data, rr_data, freq_data, augment=False):
        self.ppg_data = ppg_data
        self.rr_data = rr_data
        self.freq_data = freq_data
        self.augment = augment

        if np.any(np.isnan(ppg_data)) or np.any(np.isinf(ppg_data)):
            logger.info(f"nan or inf in ppg_data")
        
        if np.any(np.isnan(rr_data)) or np.any(np.isinf(rr_data)):
            logger.info(f"nan or inf in rr_data")


    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        ppg_segment = self.ppg_data[idx]
        rr = self.rr_data[idx]
        freq = self.freq_data[idx]

        if self.augment:
            if torch.rand(1) < 0.5:
                noise_std = 0.05 * np.std(ppg_segment)
                if noise_std > 1e-4: # Avoid adding noise to a flat-line signal
                    noise = np.random.normal(mean=0, scale=noise_std, size=ppg_segment.shape)
                    ppg_segment += noise
                
            
            if torch.rand(1) < 0.5:
                scale_factor = np.random.uniform(0.8, 1.2)  
                ppg_segment = ppg_segment * scale_factor
        
        ppg_tensor = torch.tensor(ppg_segment, dtype=torch.float32)
        rr_tensor = torch.tensor(rr, dtype=torch.float32)
        freq_tensor = torch.tensor(freq, dtype=torch.float32)
        return ppg_tensor, rr_tensor, freq_tensor


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