# 🧠 BraTS 3D U-Net: Brain Tumor Segmentation

An end-to-end PyTorch computer vision pipeline for the semantic segmentation of brain tumors using 3D MRI scans. This project takes raw, multi-modal NIfTI files and outputs precise 3D boundary maps of three distinct tumor sub-regions.

## 🚀 Project Overview
This repository contains a custom 3D U-Net architecture built from scratch to tackle the **Brain Tumor Segmentation (BraTS)** challenge. Medical 3D imaging requires massive computational resources and highly specialized loss functions to handle extreme class imbalances (where tumor pixels make up <1% of the total volume). 

### Key Features
* **Custom 3D U-Net Architecture:** Built dynamically with configurable depths and feature channels, including robust dynamic padding to prevent odd-dimension skip-connection crashes.
* **Advanced Loss Functions:** Utilizes a custom `BraTSCombinedLoss` module combining **Dice Loss** (to maximize spatial overlap) and **Focal Loss** (to force the network to learn difficult boundary pixels).
* **Hardware Optimized Training:** Leverages `torch.cuda.amp` (Automatic Mixed Precision) to accelerate training and reduce VRAM footprint on large 128x128x128 3D voxel patches.
* **Edge Deployment Ready:** Includes a Post-Training Quantization (PTQ) pipeline designed to compress FP32 weights into INT8 for lightweight CPU inference.

## 📊 Performance (12 Epochs)
The model achieved highly competitive Dice scores on unseen validation patients after a rapid 12-epoch training cycle:
* **Peritumoral Edema (ED):** `0.8017`
* **Enhancing Tumor (ET):** `0.7147`
* **Necrotic Core (NCR):** `0.6383`
* **Average Tumor Score:** `0.7182`

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Medical Imaging:** MONAI, NiBabel
* **Data Processing:** NumPy, SciPy
* **Visualization:** Matplotlib

## 📁 Repository Structure
```text
brats_project/
├── configs/
│   └── config.yaml          # Hyperparameters and file paths
├── src/
│   ├── dataset.py           # 3D DataLoaders and MONAI Augmentations
│   ├── model.py             # 3D U-Net Architecture
│   ├── losses.py            # Dice + Focal Loss implementation
│   ├── train.py             # Main training loop with AMP
│   ├── evaluate.py          # Dice score calculation on validation set
│   ├── postprocess.py       # Connected Component Analysis (Noise removal)
│   ├── visualize.py         # 2D slice visualization of 3D predictions
│   └── quantize.py          # FP32 to INT8 model compression
└── README.md