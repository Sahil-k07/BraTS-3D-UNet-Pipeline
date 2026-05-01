import streamlit as st
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
import gc

from src.dataset import load_config
from src.model import get_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="BraTS AI Segmentation", page_icon="🧠", layout="wide")
st.title("🧠 3D Brain Tumor Segmentation (BraTS)")
st.markdown("Upload a patient's 4 MRI modalities (T1, T1c, T2, FLAIR) to generate an AI-powered 3D tumor boundary map.")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_ai_model():
    config = load_config("configs/config.yaml")
    # Force to CPU for safe web deployment
    device = torch.device("cpu") 
    model = get_model(config).to(device)
    
    try:
        # 1. Try Loading INT8
        weight_path = os.path.join(config['paths']['checkpoint_dir'], "unet_epoch_32_INT8.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            st.sidebar.success("Loaded Lightweight INT8 Model")
        else:
            raise FileNotFoundError

    except Exception:
        try:
            # 2. Fallback to FP32
            weight_path = os.path.join(config['paths']['checkpoint_dir'], "unet_epoch_32.pth")
            if os.path.exists(weight_path):
                # Use strict=False just in case there are slight architecture mismatches
                model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
                st.sidebar.success("Loaded FP32 Model")
            else:
                st.sidebar.error("Model weights not found. Check GitHub LFS.")
                
        except Exception as e:
            st.sidebar.error(f"Critical Error loading models: {e}")
            
    model.eval()
    return model, device

model, device = load_ai_model()

# --- HELPER FUNCTION: READ UPLOADED NIFTI ---
def load_nifti_upload(uploaded_file):
    import tempfile
    # Create a temporary file to save the uploaded bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Load the MRI and immediately cast to float32 to save 50% RAM!
        img = nib.load(tmp_path)
        data = img.get_fdata().astype(np.float32) 
    finally:
        # Always delete the temp file after loading to save RAM
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return data

# --- UI: FILE UPLOADS ---
st.sidebar.header("📂 Patient MRI Upload")
t1_file = st.sidebar.file_uploader("Upload T1 (t1n.nii.gz)", type=['nii.gz', 'nii'])
t1c_file = st.sidebar.file_uploader("Upload T1-Contrast (t1c.nii.gz)", type=['nii.gz', 'nii'])
t2_file = st.sidebar.file_uploader("Upload T2 (t2w.nii.gz)", type=['nii.gz', 'nii'])
flair_file = st.sidebar.file_uploader("Upload FLAIR (t2f.nii.gz)", type=['nii.gz', 'nii'])

# --- MAIN EXECUTION ---
if st.sidebar.button("🚀 Run AI Segmentation", type="primary"):
    if not all([t1_file, t1c_file, t2_file, flair_file]):
        st.warning("⚠️ Please upload all 4 modalities to run the AI.")
    else:
        with st.spinner("Processing 3D Volumes & Running AI... This may take a few seconds."):
            
            # 1. Read files into NumPy arrays
            t1_vol = load_nifti_upload(t1_file)
            t1c_vol = load_nifti_upload(t1c_file)
            t2_vol = load_nifti_upload(t2_file)
            flair_vol = load_nifti_upload(flair_file)
            
            # 2. Stack into a single tensor [Channels, Depth, Height, Width]
            stacked_volume = np.stack([t1_vol, t1c_vol, t2_vol, flair_vol], axis=0)
            
            # 🚨 RAM DIET: Delete unused individual volumes IMMEDIATELY 
            # (We keep t1c_vol because we need it for the matplotlib visual later)
            del t1_vol, t2_vol, flair_vol
            gc.collect() # Force clear RAM
            
            # Add Batch dimension: [1, 4, D, H, W]
            input_tensor = torch.tensor(stacked_volume, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 🚨 RAM DIET: Delete stacked_volume now that we have the PyTorch tensor
            del stacked_volume
            gc.collect()

            # 3. AI Prediction
            with torch.no_grad(): # CRITICAL: Tells PyTorch not to store gradients (saves massive memory)
                output = model(input_tensor)
                # Squeeze out batch dimension and cast to uint8 (super lightweight)
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) 

            # 🚨 RAM DIET: Clear tensor and output from memory
            del input_tensor, output
            gc.collect()

            # 4. Find the best slice to display (largest tumor area)
            best_slice_idx = np.argmax(np.sum(pred_mask > 0, axis=(1, 2)))
            
            # 5. Extract 2D slices for visualization
            t1c_slice = t1c_vol[best_slice_idx, :, :]
            pred_slice = pred_mask[best_slice_idx, :, :]

            # --- VISUALIZATION DASHBOARD ---
            st.success(f"✅ Prediction Complete! Displaying Cross-Section Depth: {best_slice_idx}")
            
            col1, col2 = st.columns(2)
            class_cmap = ListedColormap(['black', 'red', 'green', 'yellow'])
            
            with col1:
                st.subheader("Original T1-Contrast")
                fig1, ax1 = plt.subplots()
                ax1.imshow(t1c_slice, cmap='gray')
                ax1.axis('off')
                st.pyplot(fig1)
                
            with col2:
                st.subheader("AI Tumor Map")
                fig2, ax2 = plt.subplots()
                
                # 1. Plot the base MRI scan FIRST (This acts as the background layer)
                ax2.imshow(t1c_slice, cmap='gray')
                
                # 2. Mask out the '0' values (background) so they are completely invisible
                masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
                
                # 3. Plot the AI mask ON TOP. 
                # alpha=0.4 means it will be 40% opaque and 60% transparent.
                ax2.imshow(masked_pred, cmap=class_cmap, alpha=0.4, interpolation='none', vmin=0, vmax=3)
                
                ax2.axis('off')
                st.pyplot(fig2)
            
            st.markdown("""
            **Legend:**
            * 🔴 **Red:** Necrotic Core (Dead Tissue)
            * 🟢 **Green:** Peritumoral Edema (Swelling)
            * 🟡 **Yellow:** Enhancing Tumor (Active Edge)
            """)

            #streamlit run app.py