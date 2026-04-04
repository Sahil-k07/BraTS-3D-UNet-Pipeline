import torch
from src.losses import BraTSCombinedLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulate dummy model outputs (logits) [Batch=1, Classes=4, D=128, H=128, W=128]
dummy_logits = torch.randn(1, 4, 128, 128, 128).to(device)

# Simulate dummy ground truth masks (class indices 0 to 3) [Batch=1, D=128, H=128, W=128]
dummy_targets = torch.randint(0, 4, (1, 128, 128, 128)).to(device)

# Initialize our combined loss
criterion = BraTSCombinedLoss(dice_weight=0.5, focal_weight=0.5)

# Calculate loss
with torch.no_grad():
    loss = criterion(dummy_logits, dummy_targets)

print(f"Calculated Combined Loss: {loss.item():.4f}")
print("✅ Loss function works successfully!")