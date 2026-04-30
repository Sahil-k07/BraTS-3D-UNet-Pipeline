# 🧠 3D Brain Tumor Segmentation (BraTS)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-00A4E4.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

An end-to-end deep learning pipeline for 3D volumetric medical image segmentation, trained on the BraTS dataset. This project utilizes a **3D U-Net** architecture to process multi-modal MRI scans (T1, T1c, T2, FLAIR) and predict three distinct tumor sub-regions.

## 🌟 Project Highlights

* **Architecture:** 3D U-Net implemented via MONAI.
* **Loss Function:** Custom Combined Focal + Dice Loss to combat severe class imbalance (background vs. tumor voxels).
* **Hardware Optimization:** Engineered a safe memory-caching `DataLoader` with multi-threading and cuDNN benchmarking, reducing epoch training time by over 30% on consumer hardware.
* **Clinical UI:** Interactive web frontend built with Streamlit for real-time, in-browser MRI inference.

## 🖥️ Streamlit Frontend
Upload 4 NIfTI (`.nii.gz`) modalities to generate an AI-powered tumor boundary map in seconds. 

![Streamlit UI](ui_screenshot.png)
*(Legend: 🔴 Necrotic Core | 🟢 Peritumoral Edema | 🟡 Enhancing Tumor)*

## 📊 Model Performance & Metrics
Evaluated using multi-class ROC, Precision-Recall curves, and normalized confusion matrices.

![Metrics Dashboard](metrics_dashboard.png)

## 🚀 Quick Start

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt