import os
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
        # Unlocks the fastest convolution algorithms for your specific GPU architecture
        torch.backends.cudnn.benchmark = True  
    print(f"🚀 Training initiated on device: {device}")

    # 1. Load Data
    train_loader, _ = get_dataloaders(config)

    # 2. Initialize Model, Loss, Optimizer, and Scaler (for AMP)
    model = get_model(config).to(device)
    criterion = BraTSCombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-5)
    
    # Modern PyTorch AMP initialization
    scaler = GradScaler('cuda') 

    # 3. Setup Checkpoint Directory
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Grab the max_epochs directly from your config
    num_epochs = config["training"]["max_epochs"]

    # 4. Main Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in loop:
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)

            if masks.dim() == 5:
                masks = masks.squeeze(1)
            masks[masks == 4] = 3
            
            optimizer.zero_grad()

            # Modern PyTorch autocast syntax
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            # AMP: Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint saved: {save_path}\n")

if __name__ == "__main__":
    train()