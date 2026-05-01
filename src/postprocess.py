import torch
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

from src.dataset import load_config, get_dataloaders
from src.model import get_model

def calculate_dice(preds, targets, num_classes=4):
    """Calculates the Dice score (reused from our evaluate script)"""
    dice_scores = []
    for cls in range(1, num_classes):
        pred_mask = (preds == cls).float()
        target_mask = (targets == cls).float()
        
        intersection = (pred_mask * target_mask).sum()
        total_pixels = pred_mask.sum() + target_mask.sum()
        
        if total_pixels == 0:
            dice = 1.0 
        else:
            dice = ((2. * intersection) / total_pixels).item()
        dice_scores.append(dice)
    return dice_scores

def remove_small_islands(pred_numpy):
    """
    Keeps only the largest connected component of the predicted tumor.
    """
    # 1. Create a binary mask of ALL tumor classes (1, 2, or 3)
    binary_tumor = (pred_numpy > 0).astype(int)

    # 2. Label each distinct connected "island" with a unique number
    labeled_islands, num_islands = label(binary_tumor)

    # If there is 0 or 1 island, it's already perfectly clean!
    if num_islands <= 1:
        return pred_numpy

    # 3. Count the size of each island
    # bincount[0] is the background, so we set it to 0 so we don't accidentally keep the background
    island_sizes = np.bincount(labeled_islands.ravel())
    island_sizes[0] = 0

    # 4. Find the label number of the largest tumor island
    largest_island_label = island_sizes.argmax()

    # 5. Create a clean mask (True only for the largest island)
    clean_mask = (labeled_islands == largest_island_label)

    # 6. Apply the mask to the original predictions (zeros out the small noisy islands)
    clean_pred = pred_numpy * clean_mask

    return clean_pred

def run_postprocessing():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧹 Starting Post-Processing on device: {device}")

    _, val_loader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    # Load your 12th epoch weights
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_32.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    raw_dice_scores = []
    clean_dice_scores = []

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Cleaning Predictions")
        for batch in val_loop:
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)
            
            if masks.dim() == 5:
                masks = masks.squeeze(1)
            masks[masks == 4] = 3 # BraTS fix

            # Get Raw Predictions
            outputs = model(images)
            raw_preds = torch.argmax(outputs, dim=1)

            # Calculate Raw Dice
            raw_dice = calculate_dice(raw_preds, masks)
            raw_dice_scores.append(raw_dice)

            # --- APPLY POST-PROCESSING ---
            # Move to CPU numpy for SciPy, clean it, and move back to GPU tensor
            pred_np = raw_preds[0].cpu().numpy()
            clean_pred_np = remove_small_islands(pred_np)
            clean_preds = torch.tensor(clean_pred_np, device=device).unsqueeze(0)

            # Calculate Clean Dice
            clean_dice = calculate_dice(clean_preds, masks)
            clean_dice_scores.append(clean_dice)

    # Calculate averages
    mean_raw = np.mean(raw_dice_scores, axis=0)
    mean_clean = np.mean(clean_dice_scores, axis=0)
    
    print("\n" + "="*50)
    print("✨ POST-PROCESSING RESULTS (RAW vs CLEAN) ✨")
    print("="*50)
    print(f"NCR (Necrotic Core):   {mean_raw[0]:.4f}  --->  {mean_clean[0]:.4f}")
    print(f"ED  (Edema):           {mean_raw[1]:.4f}  --->  {mean_clean[1]:.4f}")
    print(f"ET  (Enhancing Tumor): {mean_raw[2]:.4f}  --->  {mean_clean[2]:.4f}")
    print("-" * 50)
    print(f"Average Tumor Score:   {np.mean(mean_raw):.4f}  --->  {np.mean(mean_clean):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_postprocessing()

#python -m src.postprocess