import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Handle cases where targets have a channel dim, e.g., [B, 1, D, H, W]
        if targets.dim() == logits.dim():
            targets = targets.squeeze(1) 

        num_classes = logits.shape[1]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        # Reorder dims to [B, C, D, H, W]
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Apply softmax to logits to get probabilities
        probas = F.softmax(logits, dim=1)

        # Compute Dice per class (reduce over spatial and batch dims)
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probas * targets_one_hot, dims)
        cardinality = torch.sum(probas + targets_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Return 1 - average dice over all classes
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        if targets.dim() == logits.dim():
            targets = targets.squeeze(1)

        # Standard Cross Entropy Loss with no reduction
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        # Calculate pt (probability of target class)
        pt = torch.exp(-ce_loss)
        
        # Apply focal weight
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class BraTSCombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return (self.dice_weight * dice) + (self.focal_weight * focal)