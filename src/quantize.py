import os
import time
import torch
import torch.quantization

from src.dataset import load_config
from src.model import get_model

def quantize_model():
    config = load_config("configs/config.yaml")
    
    # NOTE: Quantized models are designed to run on CPUs, so we force the device to CPU here!
    print("🧠 Loading original FP32 model...")
    model_fp32 = get_model(config).to("cpu")
    
    weight_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_12.pth"
    model_fp32.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model_fp32.eval()

    # 1. Calculate Original File Size
    original_size = os.path.getsize(weight_path) / (1024 * 1024)
    print(f"📦 Original Model Size (FP32): {original_size:.2f} MB")

    # 2. Apply Dynamic Quantization
    print("\n🗜️ Compressing model to INT8...")
    # We tell PyTorch to dynamically compress our heavy 3D Convolutions
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, 
        {torch.nn.Conv3d, torch.nn.ConvTranspose3d, torch.nn.Linear}, 
        dtype=torch.qint8
    )

    # 3. Save the Quantized Model
    quantized_path = f"{config['paths']['checkpoint_dir']}/unet_epoch_12_INT8.pth"
    torch.save(model_int8.state_dict(), quantized_path)
    
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"📦 Quantized Model Size (INT8): {quantized_size:.2f} MB")
    
    # Prevent division by zero if saving failed or backend didn't compress
    if quantized_size > 0:
        print(f"📉 Size Reduction: {original_size / quantized_size:.1f}x smaller!")

    # 4. CPU Speed Test (Hospital Laptop Simulation)
    print("\n⏱️ Running standard CPU Speed Test...")
    dummy_input = torch.randn(1, 4, 128, 128, 128) # One simulated brain patch
    
    # Test FP32 Speed
    start = time.time()
    with torch.no_grad():
        _ = model_fp32(dummy_input)
    fp32_time = time.time() - start

    # Test INT8 Speed
    start = time.time()
    with torch.no_grad():
        _ = model_int8(dummy_input)
    int8_time = time.time() - start

    print("-" * 40)
    print(f"🐢 FP32 CPU Inference Time: {fp32_time:.3f} seconds")
    print(f"🐇 INT8 CPU Inference Time: {int8_time:.3f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    quantize_model()

#python -m src.quantize