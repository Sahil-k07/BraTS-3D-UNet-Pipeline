import os
import glob
import yaml
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, RandSpatialCropd
)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_dataloaders(config):
    # 1. Locate the data files
    data_dir = config["paths"]["data_dir"]
    
    # Locate all patient folders
    patient_folders = sorted(glob.glob(os.path.join(data_dir, "*")))
    
    # Split into train and val (e.g., 80/20 split)
    split_idx = int(len(patient_folders) * 0.8)
    train_folders = patient_folders[:split_idx]
    val_folders = patient_folders[split_idx:]

    # Create dictionaries for MONAI
    train_files = [{"image": [os.path.join(f, "t1n.nii.gz"), 
                              os.path.join(f, "t1c.nii.gz"), 
                              os.path.join(f, "t2w.nii.gz"), 
                              os.path.join(f, "t2f.nii.gz")],
                    "seg": os.path.join(f, "seg.nii.gz")} for f in train_folders]
    
    val_files = [{"image": [os.path.join(f, "t1n.nii.gz"), 
                            os.path.join(f, "t1c.nii.gz"), 
                            os.path.join(f, "t2w.nii.gz"), 
                            os.path.join(f, "t2f.nii.gz")],
                  "seg": os.path.join(f, "seg.nii.gz")} for f in val_folders]

    # 2. Define transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "seg"], roi_size=[128, 128, 128], random_size=False)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    ])

    print("⏳ Starting Safe Memory Caching... (Optimized for system RAM)")
    
    # 3. Use CacheDataset for safe memory acceleration
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=0.05,  # 5% cache keeps RAM stable
        num_workers=4     # Optimal thread count for external storage throughput
    )
    
    val_ds = CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=0.10,  # 10% cache for validation
        num_workers=2
    )

    # 4. Create the DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=0,   # Set to 0 for Windows stability during the main loop
        pin_memory=True  # Enables fast transfer to the GPU
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader