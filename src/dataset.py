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
    # 1. Locate the data files directly from your config's "data" section
    data_dir = config["data"]["data_dir"]
    train_ratio = config["data"]["train_ratio"]
    
    # Locate all patient folders inside the data directory
    patient_folders = sorted(glob.glob(os.path.join(data_dir, "*")))
    
    # Dynamically split into train and val using your config's ratio
    split_idx = int(len(patient_folders) * train_ratio)
    train_folders = patient_folders[:split_idx]
    val_folders = patient_folders[split_idx:]

    # Helper function to auto-find files ignoring prefixes (like "BraTS-GLI-00005-100-t1n.nii.gz")
    def find_file(folder, suffix):
        match = glob.glob(os.path.join(folder, f"*{suffix}"))
        if not match:
            raise FileNotFoundError(f"Could not find a file ending in {suffix} inside {folder}")
        return match[0]

    # Create dictionaries for MONAI using the auto-finder
    train_files = [{"image": [find_file(f, "t1n.nii.gz"), 
                              find_file(f, "t1c.nii.gz"), 
                              find_file(f, "t2w.nii.gz"), 
                              find_file(f, "t2f.nii.gz")],
                    "seg": find_file(f, "seg.nii.gz")} for f in train_folders]
    
    val_files = [{"image": [find_file(f, "t1n.nii.gz"), 
                            find_file(f, "t1c.nii.gz"), 
                            find_file(f, "t2w.nii.gz"), 
                            find_file(f, "t2f.nii.gz")],
                  "seg": find_file(f, "seg.nii.gz")} for f in val_folders]
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