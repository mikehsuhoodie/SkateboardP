# run_inference.py
import os, glob, cv2
import numpy as np, torch
from depth_anything_3.api import DepthAnything3

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    dmin, dmax = float(np.min(depth)), float(np.max(depth))
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    d = (depth - dmin) / (dmax - dmin)
    return (d * 255.0).astype(np.uint8)

def colorize_gray(img8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)

def segment_objects_by_depth_edges(depth16: np.ndarray, sobel_percentile):
    """Segments objects based on depth 1st derivative (Sobel)."""
    depth_f = depth16.astype(np.float32)
    depth_f_blurred = cv2.GaussianBlur(depth_f, (3, 3), 0)
    
    # 1. 一次微分 (Sobel) 
    grad_x = cv2.Sobel(depth_f_blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f_blurred, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(grad_x, grad_y)
    
    sobel_thresh = np.percentile(sobel_mag, sobel_percentile)
    _, edges = cv2.threshold(sobel_mag, sobel_thresh, 255, cv2.THRESH_BINARY)
    edges = edges.astype(np.uint8)
    
    # kernel = np.ones((3,3), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)
    
    objects_mask = cv2.bitwise_not(edges)
    num_labels, labels = cv2.connectedComponents(objects_mask, connectivity=8)
    
    return labels, num_labels, edges

def colorize_labels(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Assigns random colors to each label id for visualization."""
    np.random.seed(42)  # For consistent colors across frames
    colors = np.random.randint(0, 255, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # edges/background as black
    return colors[labels]

def main():
    # --- Configuration ---
    # IMAGE_PATH = "./assets/examples/SOH/000.png"  # Input image or folder
    IMAGE_PATH = "./assets/examples/SOH/street2d.jpg"
    OUTDIR     = "./outputs/inference_results"        # Output directory
    MODEL_NAME = "depth-anything/DA3METRIC-LARGE"      # V3 Model Name
    INPUT_SIZE = 1008                                  # Input resolution (e.g., 518, 700, 1008)
    SAVE_16BIT = True                                 # Also save 16-bit raw depth
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    model = DepthAnything3.from_pretrained(MODEL_NAME).to(DEVICE)

    # Prepare inputs
    img_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')
    inputs = []
    if os.path.isfile(IMAGE_PATH):
        inputs = [IMAGE_PATH]
    else:
        paths = glob.glob(os.path.join(IMAGE_PATH, '**/*'), recursive=True)
        for p in paths:
            if os.path.splitext(p)[1].lower() in img_exts:
                inputs.append(p)

    if not inputs:
        print(f"No images found at {IMAGE_PATH}")
        return

    os.makedirs(OUTDIR, exist_ok=True)

    for idx, src in enumerate(inputs):
        print(f'[{idx+1}/{len(inputs)}] Processing {src}')
        
        # Inference
        with torch.no_grad():
            prediction = model.inference([src], process_res=INPUT_SIZE)
        
        # Get original size from source image
        orig_img = cv2.imread(src)
        orig_h, orig_w = orig_img.shape[:2]
        
        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if depth.ndim == 3: depth = depth[0]

        # Resize depth map to original image size
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # # 8-bit outputs
        # depth8 = normalize_depth(depth)
        # depth_color = colorize_gray(depth8)
        
        base = os.path.splitext(os.path.basename(src))[0]
        
        # # Save colorized
        # cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_color.png'), depth_color)
        # # Save 8-bit gray
        # cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_bw.png'), depth8)
        
        if SAVE_16BIT:
            # 16-bit raw depth (normalized by max available in frame)
            dmax = depth.max()
            if dmax > 0:
                depth16 = (depth / dmax * 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_16bit.png'), depth16)
                
                # --- Depth Edge Segmentation (Sobel Only) ---
                labels, num_labels, edges = segment_objects_by_depth_edges(depth16, sobel_percentile=95)
                
                # Save Edge Map
                cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_edges.png'), edges)
                
                # Save Colored Segmentation
                labels_color = colorize_labels(labels, num_labels)
                cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_segments.png'), labels_color)
                
                # --- Export raw mask data for downstream precise cropping ---
                cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_mask_raw.png'), labels.astype(np.uint8))
                np.save(os.path.join(OUTDIR, f'{base}_depth_mask.npy'), labels)
                
                print(f'  ✔ Depth Edge Segmentation Saved (Found {num_labels - 1} regions)')

    print(f"--- Done. Results saved to {OUTDIR} ---")

if __name__ == '__main__':
    main()
