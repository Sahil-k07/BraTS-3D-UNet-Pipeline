import yaml
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    DeleteItemsd,
    Spacingd,
    Orientationd,
    CropForegroundd,
)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_patient_files(data_dir, modalities):
    """
    Scans data_dir for patient folders and builds a list of dicts,
    each containing paths to all 4 modality files + segmentation mask.
    """
    data_dir = Path(data_dir)
    patient_folders = sorted([
        f for f in data_dir.iterdir()
        if f.is_dir() and f.name.startswith("BraTS")
    ])

    data_list = []
    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        entry = {}

        all_found = True
        for mod in modalities:
            mod_file = patient_folder / f"{patient_id}-{mod}.nii.gz"
            if not mod_file.exists():
                print(f"⚠️  Missing: {mod_file}")
                all_found = False
                break
            entry[mod] = str(mod_file)

        seg_file = patient_folder / f"{patient_id}-seg.nii.gz"
        if not seg_file.exists():
            print(f"⚠️  Missing segmentation: {seg_file}")
            all_found = False

        if all_found:
            entry["seg"] = str(seg_file)
            data_list.append(entry)

    print(f"✅ Found {len(data_list)} complete patient cases")
    return data_list


def get_transforms(config, mode="train"):
    modalities = config["modalities"]
    patch_size = config["training"]["patch_size"]

    all_keys = modalities + ["seg"]
    image_keys = modalities

    base_transforms = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear",) * len(image_keys) + ("nearest",),
        ),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=image_keys, name="image"),
        DeleteItemsd(keys=image_keys),
    ]

    if mode == "train":
        aug_transforms = [
            CropForegroundd(keys=["image", "seg"], source_key="image"),
            RandSpatialCropd(
                keys=["image", "seg"],
                roi_size=patch_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "seg"]),
        ]
    else:
        aug_transforms = [
            CropForegroundd(keys=["image", "seg"], source_key="image"),
            ToTensord(keys=["image", "seg"]),
        ]

    return Compose(base_transforms + aug_transforms)


def get_dataloaders(config):
    data_list = get_patient_files(
        config["data"]["data_dir"],
        config["modalities"]
    )

    train_files, val_files = train_test_split(
        data_list,
        test_size=config["data"]["val_ratio"],
        random_state=42,
    )
    print(f"📊 Train: {len(train_files)} | Val: {len(val_files)}")

    train_transforms = get_transforms(config, mode="train")
    val_transforms   = get_transforms(config, mode="val")

    # Use regular Dataset instead of CacheDataset (no RAM caching)
    from monai.data import Dataset

    train_ds = Dataset(
        data=train_files,
        transform=train_transforms,
    )
    val_ds = Dataset(
        data=val_files,
        transform=val_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,   # 0 = no multiprocessing, avoids memory issues on Windows
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader