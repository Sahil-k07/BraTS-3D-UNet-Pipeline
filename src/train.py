import os
import csv
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import load_config, get_dataloaders
from src.model import get_model
from src.losses import BraTSCombinedLoss 

def train():
    config = load_config("configs/config.yaml")
    
    # --- HARDWARE OPTIMIZATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  
    print(f"🚀 Training initiated on device: {device}")

    # 1. Load Data (Now grabbing both train AND val loaders)
    train_loader, val_loader = get_dataloaders(config)

    # 2. Initialize Model, Loss, Optimizer, and Scaler
    model = get_model(config).to(device)
    criterion = BraTSCombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-5)
    scaler = GradScaler('cuda') 

    # 3. Setup Checkpoint Directory & Logging
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_epochs = config["training"]["max_epochs"]

    os.makedirs("outputs/logs", exist_ok=True)
    csv_path = "outputs/logs/metrics.csv"

    # Create fresh CSV and write headers
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_metric'])

    # 4. Main Training & Validation Loop
    for epoch in range(1, num_epochs + 1):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss_accum = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [TRAIN]")
        for batch in train_loop:
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)

            if images.shape[1] != 4:
                # Skip this iteration entirely and move to the next patient
                continue

            if masks.dim() == 5:
                masks = masks.squeeze(1)
            masks[masks == 4] = 3

            
            
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- VALIDATION PHASE ---
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_accum = 0.0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [VAL]")
        with torch.no_grad(): 
            for batch in val_loop:
                try:
                    # Move data to GPU
                    images = batch['image'].to(device)
                    masks = batch['seg'].to(device)

                    # 1. 🛡️ CHANNEL SHIELD: Ensure we have exactly 4 MRI modalities
                    if images.shape[1] != 4:
                        continue 

                    # 2. Pre-process masks to match training logic
                    if masks.dim() == 5:
                        masks = masks.squeeze(1)
                    
                    # Convert BraTS label 4 (Enhancing Tumor) to 3 to match our 4-class output
                    masks[masks == 4] = 3

                    # 3. Inference with Mixed Precision
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, masks)

                    val_loss_accum += loss.item()
                    val_loop.set_postfix(val_loss=loss.item())

                except Exception as e:
                    # If a specific NIfTI file is missing or corrupted, skip this patient
                    print(f"\n⚠️ Skipping validation patient due to error: {e}")
                    continue

        # Avoid division by zero if the entire validation set was skipped
        if len(val_loader) > 0:
            avg_val_loss = val_loss_accum / len(val_loader)
        else:
            avg_val_loss = 0.0
        
        print(f"\n✅ Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- LOGGING TO CSV ---
        # We write the real calculated losses to the file
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Setting val_metric to 0.0 for now, as we are mainly tracking loss
            writer.writerow([epoch, avg_train_loss, avg_val_loss, 0.0])

        # Save model checkpoint
        save_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint saved: {save_path}\n")

if __name__ == "__main__":
    train()

    #python -m src.train