import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
# from rwkv import RWKVRRModel
from rwkv2 import RWKVTimeModel
# from rwkv2_opt import RWKVTimeModelOPT
# from rwkv_opt import OptimizedRWKVRRModel
# from rwkv_opt2 import OptimizedRWKVRRModel
# from rwkv_opt3 import RWKVRRModel 
from rwkv_version2 import RWKVRRModel
import torch.distributed as dist
from typing import Tuple, Optional
import torchmetrics

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
    def __init__(self, input_size=1, hidden_size=128, num_layers=4, output_size=32, dropout=0.2):
        super(LSTMRRModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        out = self.fc(lstm_out[:, -1, :])
        return out
    
class LinearModel(nn.Module):
    def __init__(self, input_size=60*125, hidden_size=2048, output_size=512, dropout=0):
        super(LinearModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
    
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


# class SEBlock(nn.Module):
#     def __init__(self, ch, reduction=8):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(ch, ch // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch // reduction, ch, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         scale = self.fc(x)
#         return x * scale

# class ResidualConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch)
#         )
#         self.skip = (
#             nn.Conv2d(in_ch, out_ch, 1, stride=stride)
#             if in_ch != out_ch else nn.Identity()
#         )

#     def forward(self, x):
#         return F.relu(self.conv(x) + self.skip(x))

# class AdvancedScalogramEncoder(nn.Module):
#     def __init__(self, in_channels=1, image_size=(64, 64), output_features=64, dropout_rate=0.3):
#         super().__init__()
#         self.conv_base = nn.Sequential(
#             ResidualConvBlock(in_channels, 16),
#             SEBlock(16),
#             nn.MaxPool2d(2),
            
#             ResidualConvBlock(16, 32),
#             SEBlock(32),
#             nn.MaxPool2d(2),
            
#             ResidualConvBlock(32, 64),
#             SEBlock(64),
#             nn.MaxPool2d(2),
            
#             nn.Dropout2d(dropout_rate)
#         )

#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(64, 128),
#             nn.SiLU(),
#             nn.Linear(128, output_features)
#         )

#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         x = self.conv_base(x)
#         x = self.head(x)
#         return x

# class ResidualBlock(nn.Module):
#     """Residual block for better gradient flow."""
#     def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.3):
#         super().__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                               stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout2d(dropout_rate)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
#                               stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         # Shortcut connection
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, 
#                          stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
    
#     def forward(self, x):
#         identity = self.shortcut(x)
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         out += identity  # Residual connection
#         out = self.relu(out)
        
#         return out


# class SEBlock(nn.Module):
#     """Squeeze-and-Excitation block for channel attention."""
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         B, C, _, _ = x.size()
#         # Squeeze: global spatial information
#         y = self.squeeze(x).view(B, C)
#         # Excitation: channel-wise attention weights
#         y = self.excitation(y).view(B, C, 1, 1)
#         # Scale features by attention weights
#         return x * y.expand_as(x)


# class SpatialAttention(nn.Module):
#     """Spatial attention mechanism."""
#     def __init__(self, kernel_size: int = 7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
#                              padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         # Channel-wise statistics
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
        
#         # Concatenate and compute spatial attention
#         attention = torch.cat([avg_out, max_out], dim=1)
#         attention = self.conv(attention)
#         attention = self.sigmoid(attention)
        
#         return x * attention


# class RobustScalogramEncoder(nn.Module):
#     """
#     Highly robust scalogram encoder with multiple improvements:
#     - Residual connections for better gradient flow
#     - SE blocks for channel attention
#     - Spatial attention
#     - Layer normalization option
#     - Stochastic depth for regularization
#     - Progressive widening
#     """
#     def __init__(
#         self, 
#         in_channels: int = 1, 
#         image_size: Tuple[int, int] = (64, 64), 
#         output_features: int = 64,
#         dropout_rate: float = 0.3,
#         use_se: bool = True,
#         use_spatial_attention: bool = True,
#         stochastic_depth_rate: float = 0.1
#     ):
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.image_size = image_size
#         self.use_se = use_se
#         self.use_spatial_attention = use_spatial_attention
        
#         # Initial convolution
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True)
#         )
        
#         # Residual blocks with progressive channel widening
#         self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=2, dropout_rate=dropout_rate)
#         self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout_rate=dropout_rate)
#         self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        
#         # Attention modules
#         if use_se:
#             self.se1 = SEBlock(64)
#             self.se2 = SEBlock(128)
#             self.se3 = SEBlock(256)
        
#         if use_spatial_attention:
#             self.spatial_attn = SpatialAttention()
        
#         # Global pooling
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
        
#         # Calculate flattened size
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, in_channels, *image_size)
#             features = self._forward_features(dummy_input)
#             flattened_size = features.shape[1]
        
#         # Enhanced MLP head with skip connection
#         self.mlp_head = nn.Sequential(
#             nn.Linear(flattened_size, 512),
#             nn.LayerNorm(512),  # Layer norm instead of batch norm for stability
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate * 0.5),  # Less dropout in deeper layers
            
#             nn.Linear(256, output_features)
#         )
        
#         # Initialize weights
#         self.apply(self._init_weights)
    
#     def _make_layer(self, in_channels: int, out_channels: int, 
#                    num_blocks: int, stride: int, dropout_rate: float):
#         """Create a layer of residual blocks."""
#         layers = []
        
#         # First block with stride for downsampling
#         layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
        
#         # Remaining blocks
#         for _ in range(1, num_blocks):
#             layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_rate))
        
#         return nn.Sequential(*layers)
    
#     def _init_weights(self, m):
#         """Initialize weights with proper scaling."""
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0, 0.01)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
    
#     def _forward_features(self, x):
#         """Forward through feature extraction layers."""
#         # Add channel dimension if needed
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
        
#         # Stem
#         x = self.stem(x)
        
#         # Layer 1
#         x = self.layer1(x)
#         if self.use_se:
#             x = self.se1(x)
        
#         # Layer 2
#         x = self.layer2(x)
#         if self.use_se:
#             x = self.se2(x)
        
#         # Layer 3
#         x = self.layer3(x)
#         if self.use_se:
#             x = self.se3(x)
        
#         # Spatial attention
#         if self.use_spatial_attention:
#             x = self.spatial_attn(x)
        
#         # Global pooling
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)
        
#         return x
    
#     def forward(self, x):
#         """Forward pass."""
#         features = self._forward_features(x)
#         output = self.mlp_head(features)
#         return output

# class AdvancedScalogramEncoder(nn.Module):
#     """
#     Ultimate robustness with all bells and whistles.
#     """
#     def __init__(
#         self,
#         in_channels: int = 1,
#         image_size: Tuple[int, int] = (64, 64),
#         output_features: int = 64,
#         dropout_rate: float = 0.3,
#     ):
#         super().__init__()
        
#         self.encoder = RobustScalogramEncoder(
#             in_channels=in_channels,
#             image_size=image_size,
#             output_features=output_features,
#             dropout_rate=dropout_rate,
#             use_se=True,
#             use_spatial_attention=True
#         )
        
#         # Add auxiliary classifier for regularization during training
#         with torch.no_grad():
#             dummy = torch.zeros(1, in_channels, *image_size)
#             feat_size = self.encoder._forward_features(dummy).shape[1]
        
#         self.aux_classifier = nn.Sequential(
#             nn.Linear(feat_size, output_features)
#         )
        
#         self.training_with_aux = False
    
#     def forward(self, x, return_aux=False):
#         """
#         Forward pass with optional auxiliary output.
        
#         Args:
#             x: Input tensor
#             return_aux: If True, return both main and auxiliary outputs
#         """
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
        
#         # Get features
#         features = self.encoder._forward_features(x)
        
#         # Main output
#         main_output = self.encoder.mlp_head(features)
        
#         if return_aux and self.training:
#             aux_output = self.aux_classifier(features)
#             return main_output, aux_output
        
#         return main_output

class AdvancedScalogramEncoder(nn.Module):
    """
    A more robust CNN with Batch Normalization and Dropout for better training stability and regularization.
    """
    def __init__(self, in_channels=1, image_size=(64, 64), output_features=64, dropout_rate=0.3):
        super().__init__()
        self.conv_base = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # Normalize activations
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            flattened_size = self.conv_base(dummy_input).flatten().shape[0]

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Add dropout
            nn.Linear(128, output_features)
        )

    def forward(self, x):
        # --- FIX: Automatically add channel dimension if missing ---
        # Conv2d expects a 4D tensor (B, C, H, W). If a 3D tensor (B, H, W) is passed,
        # this adds the missing channel dimension.
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        features_2d = self.conv_base(x)
        feature_vector_1d = self.mlp_head(features_2d)
        return feature_vector_1d
    
class RRLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # self.test_mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)
        metrics = {
            "MAE": torchmetrics.MeanAbsoluteError(),
            "MSE": torchmetrics.MeanSquaredError()
        }
        self.train_metrics = torchmetrics.MetricCollection(metrics, prefix="train/")
        self.val_metrics = torchmetrics.MetricCollection(metrics, prefix="val/")
        self.test_metrics = torchmetrics.MetricCollection(metrics, prefix="test/")

        self.cfg = cfg.training
        self.learning_rate = cfg.training.learning_rate
        self.weight_decay = cfg.training.weight_decay
        self.scheduler = cfg.training.scheduler
        self.freq_bins = cfg.training.n_freq_bins
        # self.training_step_outputs = []
        # self.validation_step_outputs = []
        # self.test_step_outputs = []
        self.ablation_mode = cfg.training.ablation_mode

        print(f"---------- Initializing with Ablation Mode: {self.ablation_mode} ----------")

        self.time_model = None
        self.freq_model = None
        fusion_dim = 0

        if self.ablation_mode in ["fusion", "time_only"]:
            model_name = cfg.training.model_name
            if model_name == "Linear":
                model = LinearModel(input_size=cfg.training.window_size*125, hidden_size=2048, output_size=cfg.training.time_model_output_dim, dropout=cfg.training.dropout)
            elif model_name == "LSTMRR":
                model = LSTMRRModel(input_size=1, hidden_size=128, num_layers=4, output_size=cfg.training.window_size, dropout=cfg.training.dropout)
            elif model_name == "RWKV":
                model = RWKVRRModel(input_size=1, hidden_size=128, num_layers=2, dropout=cfg.training.dropout)
            elif model_name == "RWKVTime":
                model = RWKVTimeModel(input_size=1, embed_size=64, output_size=64, num_layers=2, dropout=cfg.training.dropout)
            # elif model_name == "OptimizedRWKVRRModel":
            #     model = OptimizedRWKVRRModel(input_size=1,
            #         hidden_size=cfg.training.time_model_output_dim,
            #         num_layers=cfg.training.time_model_num_layers,
            #         dropout=cfg.training.dropout,
            #         output_size=cfg.training.time_model_output_dim)
                # model.enable_optimizations()
            # elif model_name == "RWKVTimeOPT":
            #     model = RWKVTimeModelOPT(input_size=1, embed_size=64, output_size=64, num_layers=2, dropout=0.2)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            self.time_model = model
            fusion_dim += self.cfg.time_model_output_dim
            pretrained_path = cfg.training.get("pretrained_path")
            if pretrained_path:
                # print(f"Loading pretrained weights for time_model from: {pretrained_path}")
                # # Load the saved state dictionary of the encoder
                # pretrained_dict = torch.load(pretrained_path,weights_only=False)
                # self.time_model.load_state_dict(pretrained_dict)

                print(f"Loading pretrained weights for time_model from: {pretrained_path}")

                # Determine the correct device for the current process
                # In PyTorch Lightning, the rank is often stored in self.global_rank
                # Or you can get it from the environment.
                # For simplicity, we'll assume a GPU setup.
                map_location = {'cuda:0': f'cuda:{dist.get_rank()}'} if torch.cuda.is_available() else 'cpu'

                try:
                    # Load the saved state dictionary of the encoder
                    pretrained_dict = torch.load(
                        pretrained_path,
                        map_location=map_location,
                        weights_only=True # Safer and recommended
                    )
                    self.time_model.load_state_dict(pretrained_dict)
                    print(f"Rank {dist.get_rank()} successfully loaded weights.")

                except FileNotFoundError:
                    print(f"Error: Pretrained file not found at {pretrained_path}")
                    # Handle error appropriately
                except RuntimeError as e:
                    print(f"Error loading file on Rank {dist.get_rank()}: {e}")
                    # This will now only happen if the file is truly corrupt, not because of a race condition.
                    raise e
            else:
                print("Training time_model from scratch.")

        if self.ablation_mode in ["fusion", "freq_only"]:
            # self.freq_model = FreqEncoder(n_bins=self.freq_bins, hidden=self.cfg.freq_model_output_dim)
            self.freq_model = AdvancedScalogramEncoder(
                image_size=(64, 64), # Make sure this matches your generated images
                output_features=self.cfg.freq_model_output_dim # The output vector size
            )
            fusion_dim += self.cfg.freq_model_output_dim

        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.training.dropout),
            nn.Linear(128, 60)
        )

        if cfg.training.criterion == "MSELoss":
            self.criterion = nn.MSELoss()
        elif cfg.training.criterion == "L1Loss":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion: {cfg.training.criterion}")
        

    

    def forward(self, ppg, freq):
        features = []
        if self.time_model is not None:
            features.append(self.time_model(ppg))

        if self.freq_model is not None:
            features.append(self.freq_model(freq))

        z = torch.cat(features, dim=1)  # (B, fusion_dim)
        
        out = self.head(z)  # (B,)
        return out
        # return z
 
    
    def training_step(self, batch, batch_idx):
        ppg, rr, freq = batch
        bs = ppg.shape[0]
        # if len(ppg.shape) > 2:
        #     ppg = ppg.squeeze()
        # else:
        #     print("this is ppg shape: ", ppg.shape)
        rr_pred = self.forward(ppg, freq)
        # rr_pred = rr_pred.squeeze(-1)
        loss = self.criterion(rr_pred, rr)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)

        # --- FIX 3: Use torchmetrics ---
        # This is DDP-safe. It logs 'train/MAE' and 'train/MSE'.
        metrics = self.train_metrics(rr_pred, rr)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        # --- End Fix 3 ---

        # We log 'train/MAE' now, so ModelCheckpoint can use 'val_mae' (which we will log in val_step)
        # We MUST log 'train/MAE' (or 'val_mae') with 'on_epoch=True' for the callbacks to work.
        # Naming convention: PL will auto-name 'val/MAE' to 'val_mae' for callbacks.
        
        return loss

    

    # def on_train_epoch_end(self):
    #     if not self.training_step_outputs:
    #         return
    #     avg_loss = torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean()
    #     preds = torch.cat([x['pred'] for x in self.training_step_outputs], dim=0)
    #     targets = torch.cat([x['target'] for x in self.training_step_outputs], dim=0)
    #     mae = torch.mean(torch.abs(preds - targets))

    #     # self.log('train_loss_epoch', avg_loss.to(self.device), prog_bar=False, sync_dist=True)
    #     self.log('train_mae', mae.to(self.device), on_epoch=True, prog_bar=True, sync_dist=True)

    #     # Clear outputs
    #     self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        ppg, rr, freq = batch
        bs = ppg.shape[0]
        rr_pred = self.forward(ppg, freq)
        # rr_pred = rr_pred.squeeze(-1)
        loss = self.criterion(rr_pred, rr)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)

        # --- FIX 5: Use torchmetrics ---
        # This is DDP-safe. It logs 'val/MAE' and 'val/MSE'.
        # PL automatically creates 'val_mae' for callbacks from 'val/MAE'.
        metrics = self.val_metrics(rr_pred, rr)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        # --- End Fix 5 ---

        return loss
    
    # def on_validation_epoch_end(self):
    #     if not self.validation_step_outputs:
    #         return
    #     avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
    #     preds = torch.cat([x['pred'] for x in self.validation_step_outputs], dim=0)
    #     targets = torch.cat([x['target'] for x in self.validation_step_outputs], dim=0)
    #     mae = torch.mean(torch.abs(preds - targets))

    #     # self.log('val_loss_epoch', avg_loss.to(self.device), prog_bar=False, sync_dist=True)
    #     self.log('val_mae', mae.to(self.device), on_epoch=True, prog_bar=True, sync_dist=True)

    #     # Clear outputs
    #     self.validation_step_outputs.clear()

    
    def test_step(self, batch, batch_idx):
        ppg, rr, freq = batch
        bs = ppg.shape[0]
        rr_pred = self.forward(ppg, freq)
        # rr_pred = rr_pred.squeeze(-1)
        loss = self.criterion(rr_pred, rr)

        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        metrics = self.test_metrics(rr_pred, rr)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        # self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # # Store outputs for epoch-end calculations
        self.test_step_outputs.append({
            'test_loss': loss.detach().cpu(),
            'pred': rr_pred.detach().cpu(),
            'target': rr.detach().cpu()
        })
        
        return loss
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        preds = torch.cat([x['pred'] for x in self.test_step_outputs], dim=0)
        targets = torch.cat([x['target'] for x in self.test_step_outputs], dim=0)
        print(f"preds are: {preds}")
        print(f"targets are: {targets}")
        # mae = torch.mean(torch.abs(preds - targets))

        # self.log('test_loss_epoch', avg_loss.to(self.device), prog_bar=False, sync_dist=True)
        # self.log('test_mae', mae.to(self.device), on_epoch=True, prog_bar=True, sync_dist=True)

        # # Clear outputs
        # self.test_step_outputs.clear()

        # mae = self.test_mae_metric.compute()
        # self.log('test_mae', mae, sync_dist=True)
        # self.test_mae_metric.reset()

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)

    def configure_optimizers(self):
        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss_epoch'
                }
            }
        elif self.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-7)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch', # Make sure it updates every epoch
                    'frequency': 1,
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
            self.encoder = RWKVRRModel(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
        elif cfg.training.model_name == "RWKVTime":
            self.encoder = RWKVTimeModel(input_size=1, embed_size=64, output_size=64, num_layers=2, dropout=0.2)
        # elif cfg.training.model_name == "OptimizedRWKVRRModel":
        #     self.encoder = OptimizedRWKVRRModel(input_size=1,
        #             hidden_size=cfg.training.time_model_output_dim,
        #             num_layers=cfg.training.time_model_num_layers,
        #             dropout=cfg.training.dropout,
        #             output_size=cfg.training.time_model_output_dim)
            # self.encoder.enable_optimizations()
        # elif cfg.training.model_name == "RWKVTimeOPT":
        #     self.encoder = RWKVTimeModelOPT(input_size=1, embed_size=64, output_size=64, num_layers=2, dropout=0.2)

        self.projection_head = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        # Log the validation loss. `prog_bar=True` makes it appear in the progress bar.
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        # Log the test loss.
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
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
            eta_min=1e-5
        )
        return [optimizer], [scheduler]
