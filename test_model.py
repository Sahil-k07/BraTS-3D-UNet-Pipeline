import torch
from src.model import get_model, count_parameters
from src.dataset import load_config

config = load_config("configs/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = get_model(config).to(device)
count_parameters(model)

# Simulate one batch
dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)

with torch.no_grad():
    output = model(dummy_input)

print(f"\nInput shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # should be [1, 4, 128, 128, 128]
print("✅ Model forward pass successful!")