import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from rwkv import RWKVRRModel



def ppg_augmentation(x, crop_ratio=0.8):
    """
    Applies a simple random temporal crop augmentation to a batch of PPG signals.
    
    Args:
        x (Tensor): Input batch of PPG signals with shape (B, SeqLen, Features).
        crop_ratio (float): The ratio of the original sequence length to crop.
        
    Returns:
        Tensor: The augmented (cropped and resized) PPG signals.
    """
    batch_size, seq_len, _ = x.shape
    crop_len = int(seq_len * crop_ratio)
    
    # Generate a random starting point for the crop for each item in the batch
    start = torch.randint(0, seq_len - crop_len + 1, (batch_size,))
    
    # Create cropped segments using advanced indexing
    cropped_segments = [x[i, start[i]:start[i]+crop_len, :] for i in range(batch_size)]
    cropped_x = torch.stack(cropped_segments)
    
    # Interpolate back to the original sequence length
    # Note: Interpolation requires (N, C, L) format
    augmented_x = F.interpolate(cropped_x.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False)
    
    return augmented_x.permute(0, 2, 1)


class LSTMRRModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=64, dropout=0.2):
        super(LSTMRRModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
    
class FreqEncoder(nn.Module):
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


class RRLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.training
        self.learning_rate = cfg.training.learning_rate
        self.weight_decay = cfg.training.weight_decay
        self.scheduler = cfg.training.scheduler
        self.freq_bins = cfg.training.n_freq_bins
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        fusion_dim = 64 + 32
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        model_name = cfg.training.model_name
        if model_name == "LSTMRR":
            model = LSTMRRModel()
        elif model_name == "RWKV":
            model = RWKVRRModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.time_model = model
        pretrained_path = cfg.training.get("pretrained_path")
        if pretrained_path:
            print(f"Loading pretrained weights from: {pretrained_path}")
            # Load the saved state dictionary of the encoder
            pretrained_dict = torch.load(pretrained_path)
            self.time_model.load_state_dict(pretrained_dict)
        else:
            print("Training time_model from scratch.")


        self.freq_model = FreqEncoder(n_bins=self.freq_bins, hidden=32)
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
        



class SSLPretrainModule(pl.LightningModule):

    def __init__(self, cfg, learning_rate=1e-3, weight_decay=1e-5, temperature=0.07):
        super().__init__()
        self.save_hyperparameters()

        if cfg.training.model_name == "LSTMRR":
            self.encoder = LSTMRRModel(output_size=64)
        elif cfg.training.model_name == "RWKV":
            self.encoder = RWKVRRModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)

        self.projection_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    
    def forward(self, x):
        return self.encoder(x)
    

    def info_nce_loss(self, features, temperature):
        # Create labels that identify positive pairs
        labels = torch.cat([torch.arange(features.shape[0] // 2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        
        # Discard self-similarity from the matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # Select positive similarities
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # Select negative similarities
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        # The first column (0) corresponds to the positive pair
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        logits = logits / temperature
        return F.cross_entropy(logits, labels)
    
    def _shared_step(self, batch):
        """
        A shared step for training, validation, and testing to avoid code duplication.
        """
        # In SSL, we only need the PPG signal from the batch
        ppg, _, _ = batch

        # Create two augmented views of the input PPG
        # NOTE: It's important that augmentation is applied here to get different views
        # even for the validation and test sets.
        view1 = ppg_augmentation(ppg)
        view2 = ppg_augmentation(ppg)

        # Get embeddings from the encoder
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)

        # Get projections from the projection head
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        # Concatenate projections for loss calculation
        features = torch.cat([z1, z2], dim=0)

        # Calculate contrastive loss
        loss = self.info_nce_loss(features, self.hparams.temperature)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        # Log the validation loss. `prog_bar=True` makes it appear in the progress bar.
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        # Log the test loss.
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0
        )
        return [optimizer], [scheduler]
