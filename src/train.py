import os
import torch
import torch.optim as optim
from tqdm import tqdm

# Make sure this import matches your dataset.py functions!
from src.dataset import load_config, get_dataloaders 
from src.model import get_model
from src.losses import BraTSCombinedLoss

def train():
    # 1. Setup
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # Use paths from your config
    save_dir = config['paths']['checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load Data
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(config)

    # 3. Initialize Model, Loss, and Optimizer
    model = get_model(config).to(device)
    
    # Grab loss weights from config
    d_weight = config['loss']['dice_weight']
    f_weight = config['loss']['focal_weight']
    criterion = BraTSCombinedLoss(dice_weight=d_weight, focal_weight=f_weight)
    
    # Optimizer settings from config
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scaler for Mixed Precision (speeds up training and saves VRAM)
    use_amp = config['training']['mixed_precision']
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    num_epochs = config['training']['max_epochs']

    # 4. The Epoch Loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        # Progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_loop:
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)

            # --- THE BRATS FIX ---
            # Map label 4 (Enhancing Tumor) to 3 so it fits in our 4 classes
            masks[masks == 4] = 3 

            optimizer.zero_grad()

            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"End of Epoch {epoch+1} | Average Train Loss: {avg_train_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()

#python -m src.train