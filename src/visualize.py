import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from src.dataset import load_config, get_dataloaders
from src.model import get_model

def visualize():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎨 Starting Advanced Visualizer on device: {device}")

    # 1. Load Data and Model
    _, val_loader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    # Load your best weights
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_12.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 2. Grab just ONE patient from the validation set
    batch = next(iter(val_loader))
    images = batch['image'].to(device)
    masks = batch['seg'].to(device)

    # Format the mask and apply the BraTS fix
    if masks.dim() == 5:
        masks = masks.squeeze(1)
    masks[masks == 4] = 3

    # 3. Generate Prediction and Probabilities
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate Softmax Probabilities for the Heatmap
        probabilities = F.softmax(outputs, dim=1)
        # Sum the probabilities of classes 1, 2, and 3 to get overall "Tumor Confidence"
        tumor_probs = torch.sum(probabilities[:, 1:], dim=1)

    # Move tensors to CPU for plotting
    image_np = images[0, 0].cpu().numpy()       # [Depth, Height, Width]
    mask_np = masks[0].cpu().numpy()            # [Depth, Height, Width]
    pred_np = preds[0].cpu().numpy()            # [Depth, Height, Width]
    prob_np = tumor_probs[0].cpu().numpy()      # [Depth, Height, Width]

    # 4. Find the best slice! 
    best_slice_idx = np.argmax(np.sum(mask_np > 0, axis=(1, 2)))
    print(f"Found largest tumor on Depth Slice: {best_slice_idx}")

    # Extract that specific 2D slice
    img_slice = image_np[best_slice_idx, :, :]
    mask_slice = mask_np[best_slice_idx, :, :]
    pred_slice = pred_np[best_slice_idx, :, :]
    prob_slice = prob_np[best_slice_idx, :, :]

    # --- 5. Generate Error Map (Binary Tumor vs Background) ---
    gt_tumor = mask_slice > 0
    pred_tumor = pred_slice > 0
    
    error_map = np.zeros_like(img_slice)
    error_map[gt_tumor & pred_tumor] = 1   # True Positive (Correct)
    error_map[gt_tumor & ~pred_tumor] = 2  # False Negative (Missed)
    error_map[~gt_tumor & pred_tumor] = 3  # False Positive (Hallucinated)

    # --- 6. PLOTTING DASHBOARD ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Advanced BraTS 3D U-Net Analysis (Depth Slice {best_slice_idx})", fontsize=20, fontweight='bold')

    # Medical Colors: Black(BG), Red(NCR), Green(Edema), Yellow(ET)
    class_cmap = ListedColormap(['black', 'red', 'green', 'yellow'])
    
    # Error Colors: Transparent(BG), Green(TP), Blue(FN), Orange(FP)
    error_cmap = ListedColormap(['black', '#00ff00', '#0077ff', '#ff7700'])

    # --- Row 1, Col 1: Original MRI ---
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title("1. Original T1c MRI", fontsize=14)
    axes[0, 0].axis('off')

    # --- Row 1, Col 2: Ground Truth ---
    axes[0, 1].imshow(img_slice, cmap='gray')
    axes[0, 1].imshow(mask_slice, cmap=class_cmap, alpha=0.5, interpolation='none', vmin=0, vmax=3)
    axes[0, 1].set_title("2. Ground Truth (Expert)", fontsize=14)
    axes[0, 1].axis('off')

    # --- Row 1, Col 3: AI Prediction ---
    axes[0, 2].imshow(img_slice, cmap='gray')
    axes[0, 2].imshow(pred_slice, cmap=class_cmap, alpha=0.5, interpolation='none', vmin=0, vmax=3)
    axes[0, 2].set_title("3. AI Prediction", fontsize=14)
    axes[0, 2].axis('off')

    # --- Row 2, Col 1: Error Map ---
    axes[1, 0].imshow(img_slice, cmap='gray')
    # Mask out the background (0) so it's transparent over the MRI
    masked_error = np.ma.masked_where(error_map == 0, error_map)
    axes[1, 0].imshow(masked_error, cmap=error_cmap, alpha=0.7, interpolation='none', vmin=0, vmax=3)
    axes[1, 0].set_title("4. Error Analysis Map", fontsize=14)
    axes[1, 0].axis('off')

    # --- Row 2, Col 2: Probability Heatmap ---
    axes[1, 1].imshow(img_slice, cmap='gray')
    im = axes[1, 1].imshow(prob_slice, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
    axes[1, 1].set_title("5. Model Confidence Heatmap", fontsize=14)
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Tumor Probability")

    # --- Row 2, Col 3: Volumetric Bar Chart ---
    classes = ['NCR (Red)', 'Edema (Green)', 'ET (Yellow)']
    gt_counts = [(mask_slice == 1).sum(), (mask_slice == 2).sum(), (mask_slice == 3).sum()]
    pred_counts = [(pred_slice == 1).sum(), (pred_slice == 2).sum(), (pred_slice == 3).sum()]
    
    x = np.arange(len(classes))
    width = 0.35

    axes[1, 2].bar(x - width/2, gt_counts, width, label='Ground Truth', color='gray')
    axes[1, 2].bar(x + width/2, pred_counts, width, label='AI Prediction', color='dodgerblue')
    
    axes[1, 2].set_title("6. Class Pixel Volume Comparison", fontsize=14)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(classes)
    axes[1, 2].set_ylabel("Pixel Count")
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', linestyle='--', alpha=0.7)

    # Add Legends
    from matplotlib.patches import Patch
    class_legend = [
        Patch(facecolor='red', label='NCR'),
        Patch(facecolor='green', label='Edema'),
        Patch(facecolor='yellow', label='ET')
    ]
    error_legend = [
        Patch(facecolor='#00ff00', label='Correct (TP)'),
        Patch(facecolor='#0077ff', label='Missed (FN)'),
        Patch(facecolor='#ff7700', label='Hallucinated (FP)')
    ]
    
    axes[0, 1].legend(handles=class_legend, loc='lower right', fontsize=8)
    axes[1, 0].legend(handles=error_legend, loc='lower right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    visualize()
#python -m src.visualize