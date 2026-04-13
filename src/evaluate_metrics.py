import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tqdm import tqdm

from src.dataset import load_config, get_dataloaders
from src.model import get_model

def evaluate_metrics():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Starting Advanced Metrics Evaluation on: {device}")

    # 1. Load Data and Model
    _, val_loader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    # Load your most recent weights (update this filename if needed)
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_1.pth" 
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Classes for our BraTS dataset
    class_names = ['Background', 'NCR (Red)', 'Edema (Green)', 'ET (Yellow)']
    num_classes = len(class_names)

    # Accumulators for memory-safe processing
    y_true_all = []
    y_prob_all = []
    
    # Memory Guard: We will only process the first 3 patients for these heavy plots
    max_patients_to_plot = 3 
    
    print(f"🧠 Extracting Voxel Probabilities from {max_patients_to_plot} patients...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=max_patients_to_plot)):
            if i >= max_patients_to_plot:
                break
                
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)

            if masks.dim() == 5:
                masks = masks.squeeze(1)
            masks[masks == 4] = 3  # BraTS label correction

            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1) # Get probabilities [B, C, D, H, W]

            # Flatten the 3D tensors into 1D arrays of voxels to feed into scikit-learn
            masks_flat = masks.view(-1).cpu().numpy()
            probs_flat = probabilities.permute(0, 2, 3, 4, 1).reshape(-1, num_classes).cpu().numpy()

            # Sub-sample voxels to prevent RAM crashes (Take 10% of voxels from each patient)
            # Medical images are mostly background, so this speeds up plotting drastically
            sample_indices = np.random.choice(len(masks_flat), size=int(len(masks_flat)*0.1), replace=False)
            
            y_true_all.extend(masks_flat[sample_indices])
            y_prob_all.extend(probs_flat[sample_indices])

    # Convert to standard numpy arrays for sklearn
    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = np.argmax(y_prob, axis=1)

    print("📈 Generating Plots...")
    
    # --- PLOTTING DASHBOARD ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("BraTS 3D U-Net Model Evaluation Metrics", fontsize=22, fontweight='bold')

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    # Normalize the matrix to show percentages instead of raw voxel counts (which are in the millions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title('Normalized Confusion Matrix', fontsize=16)
    axes[0, 0].set_ylabel('True Medical Ground Truth')
    axes[0, 0].set_xlabel('AI Prediction')

    # 2. Multi-Class ROC Curves (One-vs-Rest)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=2) # Diagonal guessing line
    colors = ['black', 'red', 'green', 'orange']
    
    for i in range(1, num_classes): # Skip background (0) for cleaner clinical plots
        # Create binary labels: 1 if true class is i, 0 otherwise
        binary_y_true = (y_true == i).astype(int)
        binary_y_prob = y_prob[:, i]
        
        fpr, tpr, _ = roc_curve(binary_y_true, binary_y_prob)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('Receiver Operating Characteristic (ROC)', fontsize=16)
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # 3. Precision-Recall Curves
    for i in range(1, num_classes):
        binary_y_true = (y_true == i).astype(int)
        binary_y_prob = y_prob[:, i]
        
        precision, recall, _ = precision_recall_curve(binary_y_true, binary_y_prob)
        pr_auc = auc(recall, precision)
        
        axes[1, 0].plot(recall, precision, color=colors[i], lw=2, 
                        label=f'{class_names[i]} (AUC = {pr_auc:.3f})')

    axes[1, 0].set_xlabel('Recall (Sensitivity)')
    axes[1, 0].set_ylabel('Precision (Positive Predictive Value)')
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=16)
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(alpha=0.3)

    # 4. Class Distribution / Voxel Imbalance
    unique, counts = np.unique(y_true, return_counts=True)
    axes[1, 1].bar(class_names, counts, color=['gray', 'red', 'green', 'orange'])
    axes[1, 1].set_yscale('log') # Log scale because background voxels usually outnumber tumor voxels 1000 to 1
    axes[1, 1].set_title('Sampled Voxel Class Distribution (Log Scale)', fontsize=16)
    axes[1, 1].set_ylabel('Voxel Count')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    evaluate_metrics()