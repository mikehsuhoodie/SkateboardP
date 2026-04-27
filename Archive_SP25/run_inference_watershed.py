# run_inference_watershed.py
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

def segment_objects_by_watershed(depth16: np.ndarray, magnitude_percentile=85):
    """Segments objects using depth gradients and Watershed algorithm."""
    # 1. Convert depth to 8-bit 3-channel (Watershed needs this as topological base)
    depth_f = depth16.astype(np.float32)
    depth8 = (depth_f / 65535 * 255).astype(np.uint8)
    # We use depth8 as base. Watershed will stop where depth8 has steep visual changes.
    img_bgr = cv2.cvtColor(depth8, cv2.COLOR_GRAY2BGR)
    
    # 2. Calculate true depth gradients
    grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 3. Find "sure foreground" (flat areas with minimal depth change)
    magnitude_thresh = np.percentile(magnitude, magnitude_percentile)
    _, flat_areas = cv2.threshold(magnitude, magnitude_thresh, 255, cv2.THRESH_BINARY_INV)
    flat_areas = flat_areas.astype(np.uint8)
    
    # Erode flat areas to be strictly inside the objects
    kernel = np.ones((3,3), np.uint8)
    sure_fg = cv2.erode(flat_areas, kernel, iterations=2)
    
    # 4. Find "unknown regions" (the steep borders where water will collide)
    unknown = cv2.bitwise_not(sure_fg)
    
    # 5. Label markers (seed points for watershed)
    num_labels, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    
    # Shift labels by 1 so 0 becomes 1
    # because cv2.watershed treats 0 as the unknown region
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 6. Apply Watershed
    # Water flows from the flat areas and builds ridges at the steep borders
    markers = cv2.watershed(img_bgr, markers)
    
    # Boundaries are marked with -1
    edges = np.zeros_like(depth8)
    edges[markers == -1] = 255
    
    # 7. Normalize labels
    labels = np.copy(markers)
    labels[labels == -1] = 0 # boundaries drawn as 0
    
    return labels, np.max(labels) + 1, edges

def colorize_labels(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Assigns random colors to each label id for visualization."""
    np.random.seed(42)  # For consistent colors across frames
    colors = np.random.randint(0, 255, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # edges/background as black
    return colors[labels]

def main():
    # --- Configuration ---
    IMAGE_PATH = "./assets/examples/SOH/000.png"  # Input image or folder
    OUTDIR     = "./outputs/inference_results"        # Output directory
    MODEL_NAME = "depth-anything/DA3METRIC-LARGE"      # V3 Model Name
    INPUT_SIZE = 518                                  # Input resolution (e.g., 518, 700, 1008)
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
        
        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if depth.ndim == 3: depth = depth[0]

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
                
                # --- Depth Edge Segmentation (Watershed) ---
                labels, num_labels, edges = segment_objects_by_watershed(depth16, magnitude_percentile=90)
                
                # Save Edge Map
                cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_edges_watershed.png'), edges)
                
                # Save Colored Segmentation
                labels_color = colorize_labels(labels, num_labels)
                cv2.imwrite(os.path.join(OUTDIR, f'{base}_depth_segments_watershed.png'), labels_color)
                
                print(f'  ✔ Watershed Segmentation Saved (Found {num_labels - 1} regions)')

    print(f"--- Done. Results saved to {OUTDIR} ---")

if __name__ == '__main__':
    main()
