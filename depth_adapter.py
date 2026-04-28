import sys
import os
import torch

# Ensure the current directory is in sys.path so we can import from src/
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the vendor model code
from src.depth_anything_3.api import DepthAnything3

def get_depth_model(model_name: str, device: str) -> DepthAnything3:
    """
    Loads and returns the DepthAnything3 model.
    """
    return DepthAnything3.from_pretrained(model_name).to(device)

def predict_depth(model: DepthAnything3, image_path: str, input_size: int = 1008):
    """
    Runs inference on a single image and returns the raw prediction.
    """
    return model.inference([image_path], process_res=input_size)
