import cv2
import numpy as np
import os
import math

def process_cut_img(image_path, depth16_path, mask_path, out_dir='./Send2Unity', max_layers=3, dilate_pixels=5, feather_pixels=2, inpaint_holes=True):
    # 1. 讀取彩色原圖 (把 BGR 轉成包含透明度的 BGRA)
    rgb_img = cv2.imread(image_path)
    if rgb_img is None:
        print(f"Failed to load image: {image_path}")
        return
    rgba_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)

    # 2. 讀取 16-bit 深度圖
    depth16 = cv2.imread(depth16_path, cv2.IMREAD_UNCHANGED)
    if depth16 is None:
        print(f"Failed to load depth image: {depth16_path}")
        return

    # 3. 讀取切割矩陣
    if not os.path.exists(mask_path):
        print(f"Failed to find mask: {mask_path}")
        return
    mask = np.load(mask_path)
    num_regions = np.max(mask)

    # 4. 對每一個區塊計算其「平均深度」並排序 (遠到近)
    region_depths = []
    for i in range(1, num_regions + 1):
        area_mask = (mask == i)
        if not np.any(area_mask): continue
        avg_depth = np.mean(depth16[area_mask])
        region_depths.append({'id': i, 'avg_depth': avg_depth})

    sorted_regions = sorted(region_depths, key=lambda x: x['avg_depth']) # 從小到大 (近到遠)

    # 將 Label 0 (邊緣線) 並入最後一個圖層 (最遠處)
    # --- 侵蝕 Label 0 (讓邊緣變細) ---
    if np.any(mask == 0):
        kernel = np.ones((5,5), np.uint8)
        mask_0 = (mask == 0).astype(np.uint8)
        # 執行侵蝕 (讓 Label 0 變細)
        eroded_0 = cv2.erode(mask_0, kernel, iterations=2)
        
        # 找出「原本是 Label 0，但現在被侵蝕掉」的邊緣區域
        to_be_inpainted = (mask_0 == 1) & (eroded_0 == 0)
        
        # 將這些區域標記為一個不存在的 ID (例如 -1)
        mask[to_be_inpainted] = -1

    # 更新深度給「剩下的」Label 0
    avg_depth_0 = np.mean(depth16[mask == 0]) if np.any(mask == 0) else 0
    sorted_regions.append({'id': 0, 'avg_depth': avg_depth_0})

    # 5. 分配區塊到 MAX_LAYERS
    chunk_size = math.ceil(len(sorted_regions) / max_layers)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Merging into {max_layers} layers and applying Hole Filling (Inpainting)...")

    for layer_idx in range(max_layers):
        start_idx = layer_idx * chunk_size
        end_idx = min((layer_idx + 1) * chunk_size, len(sorted_regions))
        if start_idx >= len(sorted_regions): break
        
        chunk = sorted_regions[start_idx:end_idx]
        
        # --- A. 合併區塊遮罩 ---
        combined_mask = np.zeros(mask.shape, dtype=np.uint8)
        for entry in chunk:
            combined_mask[mask == entry['id']] = 255
        
        # --- B. 決定製作邏輯 (補洞 vs. 裁切) ---
        if layer_idx == max_layers - 1 and inpaint_holes: # 最後一層最遠，執行補洞
            print(f"  ⚡ Inpainting Background Holes...")
            hole_mask = cv2.bitwise_not(combined_mask) # 洞 = 除了這一層以外的區域
            inpainted_bgr = cv2.inpaint(rgb_img, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            cut_img = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2BGRA)
            cut_img[:, :, 3] = 255 
        else:
            # 中景與前景：採用裁切 + 邊緣擴張 + 羽化
            if dilate_pixels > 0:
                kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
                dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            else:
                dilated_mask = combined_mask
                
            if feather_pixels > 0:
                feathered_alpha = cv2.GaussianBlur(dilated_mask, (0, 0), sigmaX=feather_pixels/3.0)
            else:
                feathered_alpha = dilated_mask
                
            cut_img = rgba_img.copy()
            cut_img[:, :, 3] = feathered_alpha

        filename = f'merged_layer_{layer_idx+1:02d}.png'
        save_path = os.path.join(out_dir, filename)
        cv2.imwrite(save_path, cut_img)
        
        if layer_idx == 0:
            np.save(os.path.join(out_dir, 'layer_00_mask.npy'), combined_mask)
        
        pos_desc = "NEAR" if layer_idx == 0 else "FAR(Inpainted)" if layer_idx == max_layers-1 else "MID"
        print(f"  ✔ Saved {filename} ({pos_desc}, Contains {len(chunk)} regions)")

    print(f"\n--- Done. Farthest layer is now solid (Inpainted)! ---")

if __name__ == '__main__':
    # --- 設定核心參數 ---
    base_name = 'street2d'
    
    # Setup inputs for infer.py backward compatibility
    rgb_img_path = f'./assets/examples/SOH/{base_name}.jpg'
    depth16_path_arg = f'./outputs/inference_results/{base_name}_depth_16bit.png'
    mask_npy_path = f'./outputs/inference_results/{base_name}_depth_mask.npy'
    
    process_cut_img(rgb_img_path, depth16_path_arg, mask_npy_path)
