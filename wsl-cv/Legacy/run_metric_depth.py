import numpy as np
import torch
import os
import sys
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
from depth_anything_3.api import DepthAnything3
from PIL import Image

def main():
    # Configuration
    image_path = 'assets/examples/SOH/Oakland.jpg'
    model_name = 'depth-anything/DA3METRIC-LARGE'
    
    print(f"Loading model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything3.from_pretrained(model_name).to(device)
    
    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Inference
    # The API expects a list of images, even for a single image
    prediction = model.inference([image_path])
    depth = prediction.depth # float32 tensor
    
    # Process depth
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
        
    # The output might be [B, H, W] or [H, W]. Handle both.
    if depth.ndim == 3:
        depth = depth[0]
        
    max_val = depth.max()
    print(f"Max depth value: {max_val}")
    
    # Normalize to 0-1
    if max_val > 0:
        norm_depth = depth / max_val
    else:
        norm_depth = depth # Should not happen for valid depth
        
    # Convert to 16-bit
    depth_16bit = (norm_depth * 65535).astype(np.uint16)
    
    # Save outputs
    # Using PIL to save 16-bit PNG
    img = Image.fromarray(depth_16bit)
    img.save('depth.png')
    print("Saved depth.png (16-bit)")
    
    with open('scale.txt', 'w') as f:
        f.write(str(max_val))
    print(f"Saved scale.txt with value: {max_val}")

if __name__ == '__main__':
    main()
