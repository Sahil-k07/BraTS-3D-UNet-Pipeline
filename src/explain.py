import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.dataset import load_config, get_dataloaders
from src.model import get_model

def generate_saliency_map():
    # 1. Load the configuration correctly within the function
    config = load_config("configs/config.yaml")
    
    # 2. Clear GPU memory to make room (just in case)
    torch.cuda.empty_cache()
    gc.collect()
    
    # 3. Use CPU to avoid the 4GB VRAM OutOfMemoryError
    device = torch.device("cpu")
    print(f"🔍 Running XAI Engine on {device} (Memory Optimized Mode)...")

    # 4. Load Data
    print("Loading validation dataset...")
    _, val_loader = get_dataloaders(config)

    # 5. Initialize Model
    model = get_model(config).to(device)
    
    # Use your finalized 32nd epoch weights
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_32.pth"
    print(f"Loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model.eval() # Must be in eval mode for stable gradients

    # 6. Get a single patient
    batch = next(iter(val_loader))
    images = batch['image'].to(device)
    masks = batch['seg'].to(device)

    # 7. CRITICAL XAI STEP: Tell PyTorch to track gradients on the INPUT image
    images.requires_grad_()

    # 8. Forward Pass
    print("Generating predictions and calculating gradients...")
    outputs = model(images)
    
    # We want to know what the AI looked at to predict "Enhancing Tumor" (Class 3)
    tumor_score = outputs[0, 3, :, :, :].sum()

    # 9. Backward Pass (The Magic)
    model.zero_grad()
    tumor_score.backward()

    # 10. Extract the Saliency Map
    saliency_3d = images.grad[0].abs().max(dim=0)[0].cpu().numpy()
    
    # 11. Prep data for visualization
    image_np = images[0, 1].detach().cpu().numpy() # Look at T1c (channel 1)
    mask_np = masks[0].cpu().numpy()
    
    if mask_np.ndim == 4:
        mask_np = mask_np.squeeze(0)
    
    # --- THE BRATS FIX ---
    mask_np[mask_np == 4] = 3 
    
    pred_np = torch.argmax(outputs, dim=1)[0].detach().cpu().numpy()

    # Find the best slice (the one with the most predicted tumor)
    best_slice_idx = np.argmax(np.sum(pred_np > 0, axis=(1, 2)))
    print(f"Visualizing Depth Slice: {best_slice_idx}")
    
    img_slice = image_np[best_slice_idx, :, :]
    mask_slice = mask_np[best_slice_idx, :, :]
    pred_slice = pred_np[best_slice_idx, :, :]
    saliency_slice = saliency_3d[best_slice_idx, :, :]

    # Normalize the saliency map for a pretty heatmap
    saliency_slice = (saliency_slice - saliency_slice.min()) / (saliency_slice.max() - saliency_slice.min() + 1e-8)

    # --- PLOTTING ---
    print("Plotting results...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Explainable AI (XAI): Tumor Detection Saliency Map (Epoch 32)", fontsize=18, fontweight='bold')
    
    # Custom colormap: 0=Black(BG), 1=Red(NCR), 2=Green(ED), 3=Yellow(ET)
    class_cmap = ListedColormap(['black', 'red', 'green', 'yellow'])

    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title("Original T1c MRI")
    axes[0].axis('off')

    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap=class_cmap, alpha=0.5, interpolation='none', vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap=class_cmap, alpha=0.5, interpolation='none', vmin=0, vmax=3)
    axes[2].set_title("AI Prediction")
    axes[2].axis('off')

    axes[3].imshow(img_slice, cmap='gray')
    im = axes[3].imshow(saliency_slice, cmap='hot', alpha=0.6)
    axes[3].set_title("AI Attention (Saliency Map)")
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_saliency_map()