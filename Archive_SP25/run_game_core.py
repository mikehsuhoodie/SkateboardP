# run_game_core.py
import argparse, os, glob, json, math
import cv2, numpy as np, torch
from depth_anything_3.api import DepthAnything3


# ---------- 幫手函式 ----------
def pixel_to_cam3d(u, v, z_m, fx, fy, cx, cy):
    # 相機座標系：Z 朝前，X 右，Y 下
    X = (u - cx) / fx * z_m
    Y = (v - cy) / fy * z_m
    return [float(X), float(Y), float(z_m)]

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    dmin, dmax = float(np.min(depth)), float(np.max(depth))
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    d = (depth - dmin) / (dmax - dmin)
    return (d * 255.0).astype(np.uint8)

def colorize_gray(img8: np.ndarray) -> np.ndarray:
    # BGR 彩色深度（用 OpenCV colormap，快）
    return cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)

def dp_simplify(cnt: np.ndarray, eps_ratio: float) -> np.ndarray:
    # cnt: (N,1,2) 或 (N,2)
    c = cnt.reshape(-1, 2)
    peri = cv2.arcLength(c, False)
    eps = max(1.0, eps_ratio * peri)
    approx = cv2.approxPolyDP(c, eps, False).reshape(-1, 2)
    return approx

def polyline_length(px: np.ndarray) -> float:
    if len(px) < 2: return 0.0
    return float(np.linalg.norm(np.diff(px, axis=0), axis=1).sum())

def split_by_turn(poly: np.ndarray, max_turn_deg: float) -> list:
    """若轉折角過尖，切成多段（避免不可滑的尖角）"""
    if len(poly) < 3: return [poly]
    segs, cur = [poly[0]], []
    for i in range(1, len(poly) - 1):
        a, b, c = poly[i-1], poly[i], poly[i+1]
        v1 = a - b; v2 = c - b
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = math.degrees(math.acos(cosang))  # 0~180，越小越尖
        segs.append(b)
        if angle < (180 - max_turn_deg):  # 例如 max_turn_deg=140 → 小於40度就切
            segs.append(None)
    segs.append(poly[-1])

    out, buf = [], []
    for p in segs:
        if p is None:
            if len(buf) >= 2: out.append(np.array(buf, dtype=np.int32))
            buf = []
        else:
            buf.append(p.tolist())
    if len(buf) >= 2: out.append(np.array(buf, dtype=np.int32))
    return out

def sample_with_depth(poly: np.ndarray, depth8: np.ndarray, step_px=3) -> list:
    """下採樣 polyline 並附加像素深度（之後 Unity 可做高度/坡度效果）"""
    if len(poly) == 0: return []
    sampled = [poly[0]]
    acc = 0.0
    for i in range(1, len(poly)):
        d = np.linalg.norm(poly[i] - poly[i-1])
        acc += d
        if acc >= step_px:
            sampled.append(poly[i])
            acc = 0.0
    if (sampled[-1] != poly[-1]).any():
        sampled.append(poly[-1])
    pts = []
    H, W = depth8.shape[:2]
    for x, y in sampled:
        xi = int(np.clip(x, 0, W-1)); yi = int(np.clip(y, 0, H-1))
        pts.append([float(x), float(y), float(depth8[yi, xi]) / 255.0])  # z=規範化深度
    return pts

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser("Skate AR V3 — depth->contour->polyline image-only")
    cv_group = parser.add_argument_group("CV (Computer Vision) — Depth & Model")
    cv_group.add_argument('--image-path', type=str, default="./assets/examples/SOH/Oakland.jpg", help="圖片路徑或資料夾")
    cv_group.add_argument('--outdir', type=str, default='./outputs/v3_results')
    cv_group.add_argument('--model-name', type=str, default='depth-anything/DA3METRIC-LARGE', help='V3 模型名稱')
    cv_group.add_argument('--input-size', type=int, default=518, help='輸入解析度 (例如 518, 700, 1008)')
    cv_group.add_argument('--pred-only', action='store_true')
    cv_group.add_argument('--fov-deg', type=float, default=60.0, help='假設的相機水平視角 (若模型未提供內參時使用)')
    cv_group.add_argument('--z-scale', type=float, default=5.0, help='把規範化深度[0,1]拉到可視的公尺尺度')
    cv_group.add_argument('--draw-edges', action='store_true', help='除輪廓外額外顯示邊緣圖')

    game_group = parser.add_argument_group("Game (JSON Output) — Polyline & Unity")
    game_group.add_argument('--grad-thresh', type=int, default=40, help='深度梯度二值化門檻')
    game_group.add_argument('--min-length', type=float, default=120.0, help='polyline 最短像素長度')
    game_group.add_argument('--dp-eps', type=float, default=0.015, help='Douglas–Peucker 比例')
    game_group.add_argument('--max-turn-deg', type=float, default=140.0, help='允許最大轉折角（越大越平滑）')
    game_group.add_argument('--topk', type=int, default=16, help='每幀輸出前 k 條最長 polyline')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model {args.model_name} on {DEVICE}...")
    model = DepthAnything3.from_pretrained(args.model_name).to(DEVICE)

    # 準備輸入清單
    img_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')
    inputs = []
    
    if os.path.isfile(args.image_path):
        inputs = [args.image_path]
    else:
        paths = glob.glob(os.path.join(args.image_path, '**/*'), recursive=True)
        for p in paths:
            if os.path.splitext(p)[1].lower() in img_exts:
                inputs.append(p)

    if not inputs:
        print(f"No images found at {args.image_path}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    for idx, src in enumerate(inputs):
        print(f'[{idx+1}/{len(inputs)}] Processing {src}')
        frame = cv2.imread(src)
        if frame is None:
            print(f'  !! 無法讀取圖片：{src}')
            continue
            
        H, W = frame.shape[:2]

        # --- Inference ---
        with torch.no_grad():
            prediction = model.inference([src], process_res=args.input_size) # V3 API supports path list
        
        depth = prediction.depth
        intrinsics = prediction.intrinsics

        # Convert to Numpy if Tensor
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()

        # Handle Batch Dimension
        if depth.ndim == 3: depth = depth[0]
        if intrinsics is not None and intrinsics.ndim == 3: intrinsics = intrinsics[0]

        depth8 = normalize_depth(depth)

        # FOV and Intrinsics Setup
        if intrinsics is not None:
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])
            fov_deg = math.degrees(2 * math.atan(W / (2 * fx)))
        else:
            print("  !! Warning: No intrinsics returned. Using fallback FOV.")
            cx, cy = W * 0.5, H * 0.5
            fx = (W / (2.0 * math.tan(math.radians(args.fov_deg) * 0.5)))
            fy = fx
            fov_deg = args.fov_deg

        # --- Polyline Extraction ---
        # 幾何邊界（深度不連續）
        gx = cv2.Sobel(depth8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth8, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges8 = cv2.convertScaleAbs(mag)
        edges8 = cv2.GaussianBlur(edges8, (5,5), 0)
        _, mask = cv2.threshold(edges8, args.grad_thresh, 255, cv2.THRESH_BINARY)

        # 拓樸清理
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 找輪廓
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 簡化 + 曲率切段 + 長度過濾
        polys = []
        for c in cnts:
            if len(c) < 8: continue
            simp = dp_simplify(c, args.dp_eps)
            for seg in split_by_turn(simp, args.max_turn_deg):
                if polyline_length(seg) >= args.min_length:
                    polys.append(seg)

        # 取最長前 K
        polys.sort(key=lambda p: polyline_length(p), reverse=True)
        polys = polys[:max(1, args.topk)]

        # --- Visualization ---
        vis_left = frame.copy()
        for p in polys:
            cv2.polylines(vis_left, [p.astype(np.int32)], isClosed=False, color=(0,255,0), thickness=2)

        depth_color = colorize_gray(depth8)
        
        base = os.path.splitext(os.path.basename(src))[0]
        base_no_ext = os.path.join(args.outdir, f'{base}_overlay')
        
        if args.pred_only:
            out_frame = vis_left
        else:
            sep_w = 16
            # Function to prepare panes (ensure uint8, BGR, and same height H)
            def prep(img, target_h=H):
                if len(img.shape) == 2: # Gray to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                curr_h, curr_w = img.shape[:2]
                if curr_h != target_h:
                    new_w = int(curr_w * (target_h / curr_h))
                    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
                return img.astype(np.uint8)

            canvas_elements = [vis_left, np.full((H, sep_w, 3), 255, np.uint8), depth_color]
            if args.draw_edges:
                edge_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                canvas_elements = [vis_left, np.full((H, sep_w, 3), 255, np.uint8), edge_vis,
                                  np.full((H, sep_w, 3), 255, np.uint8), depth_color]
            
            canvas = [prep(e) for e in canvas_elements]
            out_frame = cv2.hconcat(canvas)

        img_out = base_no_ext + ".png"
        cv2.imwrite(img_out, out_frame)
        print(f'  ✔ Overlay Saved: {img_out}')

        # --- Export JSON ---
        json_dir = os.path.join(args.outdir, f'{base}_polylines')
        os.makedirs(json_dir, exist_ok=True)
        
        lines = [sample_with_depth(p.astype(np.int32), depth8, step_px=3) for p in polys]
        lines = [ln for ln in lines if len(ln) >= 2]
        
        # New Unity format for the primary (longest) track
        from datetime import datetime
        primary_points_norm = []
        if lines:
            # lines[0] is the longest because we sorted polys
            for x, y, z_norm in lines[0]:
                primary_points_norm.append([round(float(x) / W, 6), round(float(y) / H, 6)])

        unity_payload = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat() + "Z",
            "source_image": os.path.basename(src),
            "aspect_ratio": round(float(W) / H, 4),
            "points": primary_points_norm,
            "occluded_segments": [],
            "terrain_type": "concrete",
            "friction_coefficient": 0.8
        }

        with open(os.path.join(json_dir, 'unity_track.json'), 'w', encoding='utf-8') as f:
            json.dump(unity_payload, f, ensure_ascii=False, indent=4)
        print(f'  ✔ Unity JSON Saved: {json_dir}/unity_track.json')

        # Keep the original detailed format for compatibility if needed, or just replace it
        # Here we replace/update the 000001.json if you still want the camera/3D data
        lines_cam = []
        for ln in lines: 
            pts3 = []
            for x, y, z_norm in ln:
                z_m = float(z_norm) * float(args.z_scale)
                pts3.append(pixel_to_cam3d(x, y, z_m, fx, fy, cx, cy))
            lines_cam.append(pts3)

        detailed_payload = {
            "frame": 1,
            "width": W, "height": H, "fps": 0,
            "polylines": [
                {
                    "points": ln,
                    "points_cam": lines_cam[i],
                    "length_px": float(polyline_length(np.array([[pt[0],pt[1]] for pt in ln], dtype=np.float32)))
                } for i, ln in enumerate(lines)
            ],
            "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "fov_deg": fov_deg, "z_scale": args.z_scale}
        }

        with open(os.path.join(json_dir, '000001.json'), 'w', encoding='utf-8') as f:
            json.dump(detailed_payload, f, ensure_ascii=False)

    print("--- Done ---")

if __name__ == '__main__':
    main()
