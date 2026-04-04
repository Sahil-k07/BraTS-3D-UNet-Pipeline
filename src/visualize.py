import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from src.dataset import load_config, get_dataloaders
from src.model import get_model

def visualize():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎨 Starting Visualizer on device: {device}")

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

    # 3. Generate Prediction
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # Move tensors to CPU for plotting
    image_np = images[0, 0].cpu().numpy() # [Depth, Height, Width] (Using Modality 0: T1c)
    mask_np = masks[0].cpu().numpy()      # [Depth, Height, Width]
    pred_np = preds[0].cpu().numpy()      # [Depth, Height, Width]

    # 4. Find the best slice! 
    # Look along the Depth axis (dim=0) and find the slice with the most tumor pixels (>0)
    best_slice_idx = np.argmax(np.sum(mask_np > 0, axis=(1, 2)))
    print(f"Found largest tumor on Depth Slice: {best_slice_idx}")

    # Extract that specific 2D slice
    img_slice = image_np[best_slice_idx, :, :]
    mask_slice = mask_np[best_slice_idx, :, :]
    pred_slice = pred_np[best_slice_idx, :, :]

    # 5. Setup Professional Medical Colors
    # 0: Black (Background), 1: Red (NCR), 2: Green (Edema), 3: Yellow (ET)
    colors = ['black', 'red', 'green', 'yellow']
    cmap = ListedColormap(colors)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"BraTS 3D U-Net Segmentation (Slice {best_slice_idx})", fontsize=16)

    # Plot 1: Raw MRI
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title("Original T1c MRI")
    axes[0].axis('off')

    # Plot 2: Ground Truth
    # We overlay the mask on top of the MRI using alpha (transparency)
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap=cmap, alpha=0.6, interpolation='none', vmin=0, vmax=3)
    axes[1].set_title("Ground Truth (Expert Doctor)")
    axes[1].axis('off')

    # Plot 3: AI Prediction
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap=cmap, alpha=0.6, interpolation='none', vmin=0, vmax=3)
    axes[2].set_title("AI Prediction (Model)")
    axes[2].axis('off')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Necrotic Core'),
        Patch(facecolor='green', label='Edema'),
        Patch(facecolor='yellow', label='Enhancing Tumor')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    visualize()
#python -m src.visualize