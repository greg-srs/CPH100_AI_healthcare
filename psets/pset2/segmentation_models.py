"""
Simple U-Net architecture for box segmentation.
Includes MLP baseline, CNN, and U-Net for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSegmentation(nn.Module):
    """
    Simple linear model.
    """
    
    def __init__(self,  in_channels=3, out_channels=1):
        super().__init__()
        # PathMNIST images are 3x28x28 = 2352 features
        self.in_channels = in_channels
        self.out_channels = out_channels
        input_size = 3 * 28 * 28
        self.linear = nn.Linear(input_size, 28*28)
        self.sigm = nn.Sigmoid()
        self.unflatten = nn.Unflatten(1, (1,28,28))
    
    def forward(self, x):
        x_flatten = x.view(x.size(0), -1)
        output = self.linear(x_flatten)
        output = self.sigm(output)
        output = self.unflatten(output)
        return output

class MLPSegmentation(nn.Module):
    """
    Simple MLP model for segmentation.
    Flattens input, applies multiple linear layers, reshapes to output mask.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(MLPSegmentation, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 input features
        # Output should be 1x28x28 = 784 features
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*28*28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1*28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
        # TODO: Add your own MLP architecture here
    
    def forward(self, x):
        output = self.seq(x)
        return output

class CNNSegmentation(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(CNNSegmentation, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 1),
            nn.Sigmoid()
        )        
        # TODO: Add your own CNN architecture here
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TinyUNet(nn.Module):
    """
    Tiny U-Net for segmentation of 28x28 images.
    Optimized for small images and simple segmentation tasks.
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super(TinyUNet, self).__init__()
        
        # Encoder (contracting path)
        # TODO: Add your own encoder architecture here
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deep = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.out = nn.Conv2d(32, out_channels, 1)
        self.sigm = nn.Sigmoid()



    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.maxpool(enc1)
        enc2 = self.enc2(enc2)
        enc3 = self.maxpool(enc2)
        enc3 = self.deep(enc3)
        up2 = self.up2(enc3)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self.dec2(up2)
        up1 = self.up1(up2)
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self.dec1(up1)
        out = self.out(up1)
        output = self.sigm(out)
        return output        

def get_segmentation_model(model_name, in_channels=3, out_channels=1):
    """Get segmentation model by name."""
    if model_name == 'mlp':
        return MLPSegmentation(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'unet':
        return TinyUNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'linear':
        return LinearSegmentation(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'cnn':
        return CNNSegmentation(in_channels=in_channels, out_channels=out_channels)
    
    else:
        raise ValueError("Unknown segmentation model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Intersection over Union (IoU) metric for segmentation
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union for binary segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask [B, 1, H, W] or [B, H, W]
        true_mask: Ground truth segmentation mask [B, 1, H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score (float)
    """
    # Convert to binary
    if torch.is_tensor(pred_mask):
        pred_binary = (pred_mask > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).astype(float)
    
    if torch.is_tensor(true_mask):
        true_binary = (true_mask > 0.5).float()
    else:
        true_binary = (true_mask > 0.5).astype(float)
    
    # Flatten for easier computation
    if len(pred_binary.shape) > 2:
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        true_binary = true_binary.view(true_binary.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_binary * true_binary).sum(dim=-1)
    union = pred_binary.sum(dim=-1) + true_binary.sum(dim=-1) - intersection
    
    # Handle case where both masks are empty
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    return iou.mean().item() if torch.is_tensor(iou) else iou.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Better than BCE for imbalanced segmentation.
    """
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        #TODO: Compute the DICE loss
        loss = (2*(pred*target).sum()+self.smooth)/(pred.sum()+target.sum()+self.smooth) 
        return 1-loss 

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for better segmentation performance.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss() # TODO: Initialize the component losses and store the loss weights
        self.dice_loss = DiceLoss() # TODO: Initialize the component losses and store the loss weights
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
