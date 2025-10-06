
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pytorch_lightning import LightningDataModule

# This simple wrapper resolves the inheritance issue with PyTorch Lightning 2.x
class OptunaPruningCallback(PyTorchLightningPruningCallback, pl.Callback):
    pass

class OptunaDataset(Dataset):
    def __init__(self, ppg_data, rr_data, freq_data, augment=False):
        self.ppg_data = ppg_data
        self.rr_data = rr_data
        self.freq_data = freq_data
        self.augment = augment

        if np.any(np.isnan(ppg_data)) or np.any(np.isinf(ppg_data)):
            print(f"nan or inf in ppg_data")
        
        if np.any(np.isnan(rr_data)) or np.any(np.isinf(rr_data)):
            print(f"nan or inf in rr_data")


    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        ppg_segment = self.ppg_data[idx]
        rr = self.rr_data[idx]
        freq = self.freq_data[idx]

        if self.augment:
            if torch.rand(1) < 0.5:
                noise_std = 0.05 * torch.std(ppg_segment)
                noise = np.random.normal(mean=0, std=noise_std, size=ppg_segment.shape)
                ppg_segment += noise
            
            if torch.rand(1) < 0.5:
                scale_factor = torch.FloatTensor(1).uniform_(0.8, 1.2)
                ppg_segment = ppg_segment * scale_factor

        ppg_tensor = torch.tensor(ppg_segment, dtype=torch.float32).unsqueeze(-1)
        rr_tensor = torch.tensor(rr, dtype=torch.float32)
        freq_tensor = torch.tensor(freq, dtype=torch.float32)
        return ppg_tensor, rr_tensor, freq_tensor


class OptunaDataModule(LightningDataModule):
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

class OptunaTimeModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(OptunaTimeModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
    
class OptunaFreqModel(nn.Module):
    def __init__(self, n_bins, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class OptunaLightningModule(pl.LightningModule):
    def __init__(self, cfg, learning_rate, weight_decay, scheduler, dropout, hidden_size1, hidden_size2, freq_bins):
        super().__init__()
        self.cfg = cfg.training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.dropout = dropout
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.freq_bins = freq_bins
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        fusion_dim = hidden_size1 + hidden_size2
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )

        model_name = cfg.training.model_name
        if model_name == "LSTMRR":
            model = OptunaTimeModel(hidden_size=self.hidden_size1, dropout=self.dropout)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.time_model = model
        pretrained_path = cfg.training.get("fold_1_encoder.pth")
        if pretrained_path:
            print(f"Loading pretrained weights from: {pretrained_path}")
            # Load the saved state dictionary of the encoder
            pretrained_dict = torch.load(pretrained_path)
            self.time_model.load_state_dict(pretrained_dict)
        else:
            print("Training time_model from scratch.")


        self.freq_model = OptunaFreqModel(n_bins=self.freq_bins, hidden=self.hidden_size2)
        if cfg.training.criterion == "MSELoss":
            self.criterion = nn.MSELoss()
        elif cfg.training.criterion == "L1Loss":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion: {cfg.training.criterion}")
        

    

    def forward(self, ppg, freq):
        t = self.time_model(ppg)
        f = self.freq_model(freq)    # (B, 32)
        z = torch.cat([t, f], dim=1)
        # print("------------- z shape:", z.shape)
        out = self.head(z).squeeze(-1)   # (B,)
        return out

    
    def training_step(self, batch, batch_idx):
        ppg, rr, freq = batch
        rr_pred = self.forward(ppg, freq)
        rr_pred = rr_pred.squeeze(-1)
        loss = self.criterion(rr_pred, rr)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-end calculations
        self.training_step_outputs.append({
            'train_loss': loss,
            'pred': rr_pred.detach().cpu(),
            'target': rr.detach().cpu()
        })

        return loss
    

    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            return
        avg_loss = torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean()
        preds = torch.cat([x['pred'] for x in self.training_step_outputs], dim=0)
        targets = torch.cat([x['target'] for x in self.training_step_outputs], dim=0)
        mae = torch.mean(torch.abs(preds - targets))

        self.log('train_loss_epoch', avg_loss, prog_bar=False)
        self.log('train_mae', mae, on_epoch=True, prog_bar=True)

        # Clear outputs
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        ppg, rr, freq = batch
        rr_pred = self.forward(ppg, freq)
        rr_pred = rr_pred.squeeze(-1)
        loss = self.criterion(rr_pred, rr)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-end calculations
        self.validation_step_outputs.append({
            'val_loss': loss,
            'pred': rr_pred.detach().cpu(),
            'target': rr.detach().cpu()
        })

        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        preds = torch.cat([x['pred'] for x in self.validation_step_outputs], dim=0)
        targets = torch.cat([x['target'] for x in self.validation_step_outputs], dim=0)
        mae = torch.mean(torch.abs(preds - targets))

        self.log('val_loss_epoch', avg_loss, prog_bar=False)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)

        # Clear outputs
        self.validation_step_outputs.clear()

    
    def test_step(self, batch, batch_idx):
        ppg, rr, freq = batch
        rr_pred = self.forward(ppg, freq)
        rr_pred = rr_pred.squeeze(-1)
        loss = self.criterion(rr_pred, rr)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-end calculations
        self.test_step_outputs.append({
            'test_loss': loss,
            'pred': rr_pred.detach().cpu(),
            'target': rr.detach().cpu()
        })

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        preds = torch.cat([x['pred'] for x in self.test_step_outputs], dim=0)
        targets = torch.cat([x['target'] for x in self.test_step_outputs], dim=0)
        mae = torch.mean(torch.abs(preds - targets))

        self.log('test_loss_epoch', avg_loss, prog_bar=False)
        self.log('test_mae', mae, on_epoch=True, prog_bar=True)

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        if self.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                }
            }
        elif self.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss_epoch'
                }
            }
        elif self.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                }
            }
        else:       
            return optimizer



def objective(trial: optuna.Trial, cfg, fold_data):
    # Suggest hyperparameters (no loaders here—handled by DataModule)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    scheduler = trial.suggest_categorical("scheduler", ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size1 = trial.suggest_int("hidden_size1", 32, 64)
    hidden_size2 = trial.suggest_int("hidden_size2", 32, 64)
    num_scales = cfg.training.n_freq_bins  # Tune CWT scales
    
    train_dataset = OptunaDataset(fold_data['train_ppg'], fold_data['train_rr'], fold_data['train_freq'])
    val_dataset = OptunaDataset(fold_data['val_ppg'], fold_data['val_rr'], fold_data['val_freq'])
    test_dataset = OptunaDataset(fold_data['test_ppg'], fold_data['test_rr'], fold_data['test_freq'])

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    data_module = OptunaDataModule(train_dataset, val_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers)
    # Instantiate model
    model = OptunaLightningModule(cfg, learning_rate=learning_rate, weight_decay=weight_decay, scheduler=scheduler, dropout=dropout, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
                            freq_bins=num_scales)  # Pass tuned n_freq_bins if needed

    # Trainer with pruning callback
    trainer = pl.Trainer(
        max_epochs=2,  # Short for HPO
        accelerator="auto",
        devices=1,
        logger=False,  # Skip for speed
        enable_progress_bar=True,
        callbacks=[
            OptunaPruningCallback(trial, monitor="val_loss_epoch")
        ]
    )
    
    # Train and validate via DataModule
    trainer.fit(model, data_module)
    
    # Return metric to minimize
    val_loss = trainer.callback_metrics["val_loss_epoch"].item()  # Or e.g., "val_mae" if you log MAE for RR
    return val_loss