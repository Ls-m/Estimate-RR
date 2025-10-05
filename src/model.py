import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMRRModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMRRModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
    



class RRLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.training
        self.learning_rate = cfg.training.learning_rate
        self.weight_decay = cfg.training.weight_decay
        self.scheduler = cfg.training.scheduler
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        model_name = cfg.training.model_name
        if model_name == "LSTMRR":
            model = LSTMRRModel()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.model = model

        if cfg.training.criterion == "MSELoss":
            self.criterion = nn.MSELoss()
        elif cfg.training.criterion == "L1Loss":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion: {cfg.training.criterion}")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        ppg, rr = batch
        rr_pred = self.forward(ppg)
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
        ppg, rr = batch
        rr_pred = self.forward(ppg)
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
        ppg, rr = batch
        rr_pred = self.forward(ppg)
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
