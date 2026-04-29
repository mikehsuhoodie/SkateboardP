import numpy as np
import torch
import os
import math
import imageio.v3 as iio
import sys
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
from depth_anything_3.api import DepthAnything3

def main():
    # --- Configuration ---
    image_path = 'assets/examples/SOH/Oakland.jpg'
    model_name = 'depth-anything/DA3METRIC-LARGE'
    
    # Unity requirements
    depth_out_name = 'depth_16bit.png'
    scale_out_name = 'scale.txt'
    fov_out_name = 'fov.txt'

    print(f"--- Unity Mesh Data Generator ---")
    print(f"Target Image: {image_path}")
    print(f"Model: {model_name}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # --- Load Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    model = DepthAnything3.from_pretrained(model_name).to(device)

    # --- Inference ---
    # Note: Passing list [image_path] to avoid batched processing issues
    print("Running inference...")
    prediction = model.inference([image_path])
    
    # Extract Raw Data
    depth = prediction.depth     # Float32 tensor (H, W) or (B, H, W) or numpy
    intrinsics = prediction.intrinsics # (B, 3, 3) or (3, 3)

    # Convert to Numpy if Tensor
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()

    # Handle Batch Dimension (if present)
    if depth.ndim == 3:
        depth = depth[0]
    if intrinsics is not None and intrinsics.ndim == 3:
        intrinsics = intrinsics[0]

    H, W = depth.shape
    print(f"Output Resolution: {W}x{H}")

    # --- 1. Scale Extraction ---
    max_val = depth.max()
    print(f"Max Depth (Scale): {max_val:.4f} m")
    
    with open(scale_out_name, 'w') as f:
        f.write(f"{max_val:.6f}")
    print(f"Saved: {scale_out_name}")

    # --- 2. Normalization & 16-bit Conversion ---
    if max_val > 0:
        norm_depth = depth / max_val
    else:
        norm_depth = depth 

    # Convert to 16-bit (0 - 65535)
    # Using uint16 ensures strict 16-bit save
    depth_16bit = (norm_depth * 65535).astype(np.uint16)
    
    iio.imwrite(depth_out_name, depth_16bit)
    print(f"Saved: {depth_out_name} (16-bit grayscale)")

    # --- 3. Intrinsics (FOV) ---
    # Intrinsic Matrix K:
    # [fx  0  cx]
    # [ 0  fy  cy]
    # [ 0   0   1]
    
    if intrinsics is not None:
        fx = intrinsics[0, 0]
        # Calculate horizontal FOV
        # tan(theta/2) = (W / 2) / fx
        # theta = 2 * atan(W / (2 * fx))
        
        # Note: 'W' here must match the resolution of the Intrinsics. 
        # The model output 'intrinsics' should correspond to the 'depth' map resolution.
        
        fov_rad = 2 * math.atan(W / (2 * fx))
        fov_deg = math.degrees(fov_rad)
        
        print(f"Estimated Focal Length (px): {fx:.2f}")
        print(f"Estimated Horizontal FOV: {fov_deg:.2f} degrees")
        
        with open(fov_out_name, 'w') as f:
            f.write(f"{fov_deg:.6f}")
        print(f"Saved: {fov_out_name}")
    else:
        print("Warning: No intrinsics returned. Skipping FOV.")

    print("--- Done ---")

if __name__ == '__main__':
    main()
