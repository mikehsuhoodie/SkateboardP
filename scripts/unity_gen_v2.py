import numpy as np
import torch
import os
import math
import imageio.v3 as iio
import cv2
from depth_anything_3.api import DepthAnything3

def main():
    # --- Configuration ---
    image_path = 'assets/examples/SOH/Oakland.jpg'
    # Switch to Nested model for Pose + Metric
    model_name = 'depth-anything/DA3NESTED-GIANT-LARGE' 
    
    # Outputs
    depth_out_name = 'depth_16bit.png'
    depth_inpainted_out_name = 'depth_inpainted_16bit.png'
    scale_out_name = 'scale.txt'
    pose_out_name = 'camera_pose.txt'
    intrinsics_out_name = 'intrinsics.txt'

    print(f"--- Unity Mesh Data Generator V2 ---")
    print(f"Target Image: {image_path}")
    print(f"Model: {model_name}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # --- Load Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    # NOTE: DA3NESTED-GIANT-LARGE is large. Ensure GPU memory is sufficient.
    model = DepthAnything3.from_pretrained(model_name).to(device)

    # --- Inference ---
    print("Running inference...")
    # Passing list to avoid batch issues
    prediction = model.inference([image_path])
    
    # Extract Raw Data
    depth = prediction.depth     # (H, W) or (B, H, W)
    intrinsics = prediction.intrinsics # (B, 3, 3)
    extrinsics = prediction.extrinsics # (B, 4, 4) - Camera-to-World usually for this repo's API

    # Handle Batch/Tensor conversion
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.cpu().numpy()

    if depth.ndim == 3: depth = depth[0]
    if intrinsics is not None and intrinsics.ndim == 3: intrinsics = intrinsics[0]
    if extrinsics is not None and extrinsics.ndim == 3: extrinsics = extrinsics[0]

    H, W = depth.shape
    print(f"Output Resolution: {W}x{H}")

    # --- 1. Pose & Intrinsics Export ---
    # Save Intrinsics (3x3)
    if intrinsics is not None:
        np.savetxt(intrinsics_out_name, intrinsics, fmt='%.8f')
        print(f"Saved: {intrinsics_out_name}")
    else:
        print("Warning: Intrinsics not found.")

    # Save Extrinsics (4x4)
    if extrinsics is not None:
        np.savetxt(pose_out_name, extrinsics, fmt='%.8f')
        print(f"Saved: {pose_out_name} (Camera Pose)")
    else:
        print("Warning: Extrinsics not found.")

    # --- 2. Scale ---
    max_val = depth.max()
    print(f"Max Depth (Scale): {max_val:.4f} m")
    with open(scale_out_name, 'w') as f:
        f.write(f"{max_val:.6f}")

    # --- 3. Normalization ---
    norm_depth = depth / max_val if max_val > 0 else depth
    
    # Normal Save
    depth_16bit = (norm_depth * 65535).astype(np.uint16)
    iio.imwrite(depth_out_name, depth_16bit)
    print(f"Saved: {depth_out_name}")

    # --- 4. Occlusion Inpainting ---
    print("Inpainting depth map for occlusion filling...")
    
    # Edge Detection on Depth (Discontinuities)
    # Normalize for CV2 (0-255 uint8) to find edges
    depth_u8 = (norm_depth * 255).astype(np.uint8)
    
    # Calculate Gradient
    grad_x = cv2.Sobel(depth_u8, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_u8, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Threshold to find sharp edges (objects)
    # High gradient = potential occlusion boundary
    # Threshold value needs tuning, 50 is a starting point for 0-255 range
    _, edge_mask = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
    
    # Dilate edges to create the "hole" mask
    # We treat edges as "unknown" regions to be filled by background
    kernel = np.ones((5,5), np.uint8) # 5x5 dilation
    inpaint_mask = cv2.dilate(edge_mask, kernel, iterations=2)
    
    # Inpaint
    # CV2 Inpaint works on 8-bit. For 16-bit float/uint16, we might need manual method or approximation.
    # cv2.inpaint only supports 8-bit 1-channel or 3-channel.
    # Workaround: Inpaint the *normalized* float depth? No, cv2.inpaint expects 8-bit src.
    
    # Strategy: Inpaint on 16-bit is hard with standard OpenCV.
    # We will use 'Telea' algorithm on 8-bit simply for the mask, but for Depth?
    # Actually, we can implement a simple filling:
    # Use standard cv2.inpaint on the 16_bit data by tricking it? No.
    # Let's perform inpainting on the *floating point* depth if possible, using Navier-Stokes or Telea
    # usually requires 8-bit.
    
    # Better approach for Unity "sliding": 
    # We want to fill "far" values into "near" edges? No, usually we want to fill background.
    # Simple Inpaint:
    depth_inpainted_u16 = cv2.inpaint(depth_16bit, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    iio.imwrite(depth_inpainted_out_name, depth_inpainted_u16)
    print(f"Saved: {depth_inpainted_out_name} (Inpainted)")
    
    print("--- Done V2 ---")

if __name__ == '__main__':
    main()
