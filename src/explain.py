import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.dataset import load_config, get_dataloaders
from src.model import get_model

def generate_saliency_map():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Initializing XAI Engine on {device}...")

    # 1. Load Data and Model
    _, val_loader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_12.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval() # Must be in eval mode for stable gradients

    # 2. Get a single patient
    batch = next(iter(val_loader))
    images = batch['image'].to(device)
    masks = batch['seg'].to(device)

    # 3. CRITICAL XAI STEP: Tell PyTorch to track gradients on the INPUT image
    images.requires_grad_()

    # 4. Forward Pass
    outputs = model(images)
    
    # We want to know what the AI looked at to predict "Enhancing Tumor" (Class 3)
    # We sum up all the logits for Class 3 across the entire 3D volume
    tumor_score = outputs[0, 3, :, :, :].sum()

    # 5. Backward Pass (The Magic)
    # This flows the 'tumor_score' backwards through the network to the input image
    model.zero_grad()
    tumor_score.backward()

    # 6. Extract the Saliency Map
    # Get the gradients of the input image, take absolute value, and find the max across the 4 MRI channels
    saliency_3d = images.grad[0].abs().max(dim=0)[0].cpu().numpy()
    
    # 7. Prep data for visualization
    image_np = images[0, 1].detach().cpu().numpy() # Let's look at T1c (channel 1)
    mask_np = masks[0].cpu().numpy()
    if mask_np.ndim == 4:
        mask_np = mask_np.squeeze(0)
    mask_np[mask_np == 4] = 3
    pred_np = torch.argmax(outputs, dim=1)[0].detach().cpu().numpy()

    # Find the best slice
    best_slice_idx = np.argmax(np.sum(pred_np > 0, axis=(1, 2)))
    
    img_slice = image_np[best_slice_idx, :, :]
    mask_slice = mask_np[best_slice_idx, :, :]
    pred_slice = pred_np[best_slice_idx, :, :]
    saliency_slice = saliency_3d[best_slice_idx, :, :]

    # Normalize the saliency map for a pretty heatmap
    saliency_slice = (saliency_slice - saliency_slice.min()) / (saliency_slice.max() - saliency_slice.min() + 1e-8)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Explainable AI (XAI): Tumor Detection Saliency Map", fontsize=18, fontweight='bold')
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

    #python -m src.explain