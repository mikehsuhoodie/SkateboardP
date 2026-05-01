# run_inference_sobal.py
import os, glob, cv2
import numpy as np, torch
from depth_adapter import get_depth_model, predict_depth
from sam2_segmentation import generate_sam2_label_map

def colorize_labels(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Assigns random colors to each label id for visualization."""
    np.random.seed(42)  # For consistent colors across frames
    colors = np.random.randint(0, 255, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # edges/background as black
    return colors[labels]

def run_inference(image_path, out_dir="./outputs/inference_results", model_name="depth-anything/DA3METRIC-LARGE", input_size=1008, save_16bit=True, target_width=1024):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model {model_name} on {device}...")
    model = get_depth_model(model_name, device)

    # Prepare inputs
    img_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')
    inputs = []
    if os.path.isfile(image_path):
        inputs = [image_path]
    else:
        paths = glob.glob(os.path.join(image_path, '**/*'), recursive=True)
        for p in paths:
            if os.path.splitext(p)[1].lower() in img_exts:
                inputs.append(p)

    if not inputs:
        print(f"No images found at {image_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    for idx, src in enumerate(inputs):
        print(f'[{idx+1}/{len(inputs)}] Processing {src}')
        
        # Get original size from source image and resize if needed
        orig_img = cv2.imread(src)
        if orig_img is None:
            print(f"Failed to read {src}")
            continue
            
        orig_h, orig_w = orig_img.shape[:2]
        if orig_w != target_width:
            scale = target_width / float(orig_w)
            new_w = target_width
            new_h = int(orig_h * scale)
            orig_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Overwrite source image so downstream scripts use the standardized resolution
            cv2.imwrite(src, orig_img)
            print(f'  ✔ Resized original image to {new_w}x{new_h}')
            orig_w, orig_h = new_w, new_h
        
        # Inference
        with torch.no_grad():
            prediction = predict_depth(model, src, input_size=input_size)
        
        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if depth.ndim == 3: depth = depth[0]

        # Resize depth map to original image size
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        base = os.path.splitext(os.path.basename(src))[0]
        
        if save_16bit:
            # 16-bit raw depth (normalized by max available in frame)
            dmax = depth.max()
            if dmax > 0:
                depth16 = (depth / dmax * 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(out_dir, f'{base}_depth_16bit.png'), depth16)
                
                # --- SAM2 Automatic Masks ---
                labels, num_regions = generate_sam2_label_map(orig_img, depth16)
                
                # Save Colored Segmentation
                labels_color = colorize_labels(labels, num_regions + 1)
                cv2.imwrite(os.path.join(out_dir, f'{base}_depth_segments.png'), labels_color)
                
                # --- Export raw mask data for downstream precise cropping ---
                cv2.imwrite(os.path.join(out_dir, f'{base}_depth_mask_raw.png'), labels.astype(np.uint8))
                np.save(os.path.join(out_dir, f'{base}_depth_mask.npy'), labels)
                
                print(f'  ✔ SAM2 Segmentation Saved (Found {num_regions} regions)')

    print(f"--- Done. Results saved to {out_dir} ---")

def main():
    IMAGE_PATH = "./assets/pictures/whisky.jpg"
    OUTDIR     = "./outputs/inference_results"
    MODEL_NAME = "depth-anything/DA3METRIC-LARGE"
    INPUT_SIZE = 1008
    SAVE_16BIT = True
    TARGET_WIDTH = 1024
    run_inference(IMAGE_PATH, OUTDIR, MODEL_NAME, INPUT_SIZE, SAVE_16BIT, TARGET_WIDTH)

if __name__ == '__main__':
    main()
