from src.dataset import load_config, get_dataloaders

config = load_config("configs/config.yaml")
train_loader, val_loader = get_dataloaders(config)

batch = next(iter(train_loader))
print(f"Image shape:       {batch['image'].shape}")
print(f"Seg shape:         {batch['seg'].shape}")
print(f"Seg unique labels: {batch['seg'].unique()}")