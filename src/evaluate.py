import torch
import numpy as np
from tqdm import tqdm

from src.dataset import load_config, get_dataloaders
from src.model import get_model

def calculate_dice(preds, targets, num_classes=4):
    """Calculates the Dice score for each class individually."""
    dice_scores = []
    
    # We skip class 0 (background) and only evaluate the tumor regions (1, 2, 3)
    for cls in range(1, num_classes):
        pred_mask = (preds == cls).float()
        target_mask = (targets == cls).float()
        
        intersection = (pred_mask * target_mask).sum()
        total_pixels = pred_mask.sum() + target_mask.sum()
        
        if total_pixels == 0:
            # If the class isn't in the target or prediction, score is technically perfect 1.0
            dice = 1.0 
        else:
            # Calculate dice and instantly convert the PyTorch tensor to a standard float
            dice = ((2. * intersection) / total_pixels).item()
            
        dice_scores.append(dice)
        
    return dice_scores

def evaluate():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Starting Evaluation on device: {device}")

    # 1. Load the Validation Data
    print("Loading validation dataset...")
    _, val_loader = get_dataloaders(config)

    # 2. Initialize Model and Load Your 12th Epoch Weights
    model = get_model(config).to(device)
    
    # NOTE: Update this path if your weights were saved somewhere else!
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_12.pth" 
    print(f"Loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model.eval() # Set model to evaluation mode (turns off dropout/batchnorm updates)

    all_dice_scores = []

    print("Running inference on unseen patients...")
    with torch.no_grad(): # Turn off gradients to save massive amounts of VRAM
        val_loop = tqdm(val_loader, desc="Evaluating")
        for batch in val_loop:
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)
            
            # Remove the channel dimension from masks [B, 1, D, H, W] -> [B, D, H, W]
            if masks.dim() == 5:
                masks = masks.squeeze(1)

            # --- THE BRATS FIX ---
            masks[masks == 4] = 3

            # Forward pass
            outputs = model(images)
            
            # Convert logits to actual class predictions (0, 1, 2, or 3)
            # outputs shape: [B, 4, D, H, W] -> preds shape: [B, D, H, W]
            preds = torch.argmax(outputs, dim=1)

            # Calculate Dice for this patient
            batch_dice = calculate_dice(preds, masks, num_classes=4)
            all_dice_scores.append(batch_dice)

    # Calculate the average across all validation patients
    mean_dice = np.mean(all_dice_scores, axis=0)
    
    print("\n" + "="*40)
    print("🏆 FINAL VALIDATION DICE SCORES 🏆")
    print("="*40)
    print(f"NCR (Necrotic Core):   {mean_dice[0]:.4f}")
    print(f"ED  (Edema):           {mean_dice[1]:.4f}")
    print(f"ET  (Enhancing Tumor): {mean_dice[2]:.4f}")
    print(f"Average Tumor Score:   {np.mean(mean_dice):.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()

#python -m src.evaluate