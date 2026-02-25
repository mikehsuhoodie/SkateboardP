from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import skeletonize
import networkx as nx
import json
from datetime import datetime
from collections import defaultdict

# Morphology kernel tuning (safe first-step optimization)
# Use an image-size-aware kernel so results are more stable across resolutions.
MORPH_KERNEL_RATIO = 0.06  # relative to min(image_width, image_height)
MORPH_KERNEL_MIN = 9
MORPH_KERNEL_MAX = 71

# Playability-oriented path extraction (safe incremental pass)
# Build a left-to-right x-monotonic path from skeleton pixels to avoid
# near-vertical tracks and stacked parallel lines in the same x region.
PLAYABLE_SCAN_STEP_RATIO = 0.01
PLAYABLE_SCAN_STEP_MIN = 2
PLAYABLE_SCAN_STEP_MAX = 8
PLAYABLE_CLUSTER_GAP_RATIO = 0.015
PLAYABLE_CLUSTER_GAP_MIN = 4
PLAYABLE_CLUSTER_GAP_MAX = 24
PLAYABLE_MAX_VERTICAL_STEP_RATIO = 0.08
PLAYABLE_MAX_VERTICAL_STEP_MIN = 10
PLAYABLE_MAX_VERTICAL_STEP_MAX = 80
PLAYABLE_MAX_GAP_BUCKETS = 2
PLAYABLE_MIN_RUN_POINTS = 12
PLAYABLE_SMOOTH_WINDOW = 5

# RDP simplification tuning (resolution-aware)
RDP_EPSILON_RATIO = 0.005  # relative to min(image_width, image_height)
RDP_EPSILON_MIN = 1.0
RDP_EPSILON_MAX = 8.0


def _to_odd_clamped(value, min_value, max_value):
    value = max(min_value, min(int(round(value)), max_value))
    if value % 2 == 0:
        value = value + 1 if value < max_value else value - 1
    return max(3, value)


def _to_int_clamped(value, min_value, max_value):
    return max(min_value, min(int(round(value)), max_value))


def get_morph_kernel_size(image_size):
    width, height = image_size
    min_dim = min(width, height)
    return _to_odd_clamped(min_dim * MORPH_KERNEL_RATIO, MORPH_KERNEL_MIN, MORPH_KERNEL_MAX)


def get_rdp_epsilon(image_size):
    width, height = image_size
    min_dim = min(width, height)
    epsilon = float(min_dim) * RDP_EPSILON_RATIO
    return float(np.clip(epsilon, RDP_EPSILON_MIN, RDP_EPSILON_MAX))


def _cluster_1d(values, gap_threshold):
    """Cluster sorted 1D coordinates to reduce duplicate parallel candidates."""
    if not values:
        return []

    clusters = []
    current = [int(values[0])]
    for v in values[1:]:
        v = int(v)
        if abs(v - current[-1]) <= gap_threshold:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)

    reps = []
    for cluster in clusters:
        reps.append((int(np.median(cluster)), len(cluster)))
    return reps


def _smooth_rows(path_points, window_size, image_height):
    if len(path_points) < 3 or window_size <= 1:
        return path_points
    if window_size % 2 == 0:
        window_size += 1

    rows = np.array([r for r, _ in path_points], dtype=np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    padded = np.pad(rows, (window_size // 2, window_size // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    result = []
    for (orig_r, c), new_r in zip(path_points, smoothed):
        r = int(np.clip(round(float(new_r)), 0, image_height - 1))
        result.append((r, c))
    return result


def _dedupe_nonincreasing_x(path_points):
    """Guarantee strictly increasing x so downstream segments cannot be vertical."""
    if not path_points:
        return []

    deduped = [path_points[0]]
    for r, c in path_points[1:]:
        prev_r, prev_c = deduped[-1]
        if c > prev_c:
            deduped.append((r, c))
        elif c == prev_c:
            deduped[-1] = (int(round((prev_r + r) / 2.0)), prev_c)
        # c < prev_c is dropped to preserve left-to-right monotonicity
    return deduped


def get_path_span_stats(path_points):
    if len(path_points) < 2:
        return {"x_span": 0, "y_span": 0, "ratio": 0.0}

    xs = [c for _, c in path_points]
    ys = [r for r, _ in path_points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    ratio = float(x_span) / max(1.0, float(y_span))
    return {"x_span": int(x_span), "y_span": int(y_span), "ratio": round(ratio, 4)}


def build_playable_scanline_path(skeleton_img, image_size):
    """
    Scan skeleton from left to right and keep at most one vertical segment per x bucket.
    This produces a single x-monotonic candidate track that is easier to skate on.
    """
    width, height = image_size
    points = np.argwhere(skeleton_img > 0)
    if len(points) == 0:
        return []

    min_dim = min(width, height)
    scan_step = _to_int_clamped(
        min_dim * PLAYABLE_SCAN_STEP_RATIO,
        PLAYABLE_SCAN_STEP_MIN,
        PLAYABLE_SCAN_STEP_MAX,
    )
    cluster_gap = _to_int_clamped(
        height * PLAYABLE_CLUSTER_GAP_RATIO,
        PLAYABLE_CLUSTER_GAP_MIN,
        PLAYABLE_CLUSTER_GAP_MAX,
    )
    max_vertical_step = _to_int_clamped(
        height * PLAYABLE_MAX_VERTICAL_STEP_RATIO,
        PLAYABLE_MAX_VERTICAL_STEP_MIN,
        PLAYABLE_MAX_VERTICAL_STEP_MAX,
    )

    bucket_rows = defaultdict(list)
    for r, c in points:
        bucket_rows[int(c) // scan_step].append(int(r))

    bucket_indices = sorted(bucket_rows.keys())
    if not bucket_indices:
        return []

    min_run_points = max(PLAYABLE_MIN_RUN_POINTS, max(8, width // max(1, scan_step * 10)))
    runs = []
    current_run = []
    last_bucket = None
    clamped_jump_count = 0

    def finalize_current_run():
        nonlocal current_run, last_bucket
        if len(current_run) >= min_run_points:
            runs.append(current_run)
        current_run = []
        last_bucket = None

    for bucket_idx in bucket_indices:
        y_candidates = _cluster_1d(sorted(bucket_rows[bucket_idx]), cluster_gap)
        if not y_candidates:
            continue

        if not current_run:
            # Prefer larger cluster first; tie-break to lower image area (larger row).
            chosen_y = max(y_candidates, key=lambda item: (item[1], item[0]))[0]
            candidate_x = min(width - 1, max(0, int(round((bucket_idx + 0.5) * scan_step))))
            current_run.append((chosen_y, candidate_x))
            last_bucket = bucket_idx
            continue

        gap_buckets = bucket_idx - last_bucket - 1
        if gap_buckets > PLAYABLE_MAX_GAP_BUCKETS:
            finalize_current_run()
            # Re-process this bucket as the start of a new run.
            chosen_y = max(y_candidates, key=lambda item: (item[1], item[0]))[0]
            candidate_x = min(width - 1, max(0, int(round((bucket_idx + 0.5) * scan_step))))
            current_run.append((chosen_y, candidate_x))
            last_bucket = bucket_idx
            continue

        prev_y, prev_x = current_run[-1]
        chosen_y, _ = min(
            y_candidates,
            key=lambda item: (abs(item[0] - prev_y), -item[1], item[0]),
        )

        if abs(chosen_y - prev_y) > max_vertical_step:
            chosen_y = int(np.clip(chosen_y, prev_y - max_vertical_step, prev_y + max_vertical_step))
            clamped_jump_count += 1

        candidate_x = min(width - 1, max(0, int(round((bucket_idx + 0.5) * scan_step))))

        # Small gaps are filled by interpolation to preserve a continuous track.
        if gap_buckets > 0:
            for i in range(1, gap_buckets + 1):
                t = i / float(gap_buckets + 1)
                interp_bucket = last_bucket + i
                interp_x = min(width - 1, max(0, int(round((interp_bucket + 0.5) * scan_step))))
                interp_y = int(round(prev_y + (chosen_y - prev_y) * t))
                interp_y = int(np.clip(interp_y, 0, height - 1))
                if interp_x > prev_x:
                    current_run.append((interp_y, interp_x))
                    prev_y, prev_x = current_run[-1]

        if candidate_x > current_run[-1][1]:
            current_run.append((chosen_y, candidate_x))
        elif candidate_x == current_run[-1][1]:
            avg_y = int(round((current_run[-1][0] + chosen_y) / 2.0))
            current_run[-1] = (avg_y, candidate_x)
        last_bucket = bucket_idx

    finalize_current_run()

    if not runs:
        print("Playable scanline path builder: no valid run found, fallback to longest path.")
        return []

    # Choose the run with the widest horizontal span, then the longest point count.
    best_run = max(
        runs,
        key=lambda run: ((run[-1][1] - run[0][1]), len(run)),
    )

    best_run = _smooth_rows(best_run, PLAYABLE_SMOOTH_WINDOW, height)
    best_run = _dedupe_nonincreasing_x(best_run)

    stats = get_path_span_stats(best_run)
    print(
        "Playable scanline path builder:",
        f"scan_step={scan_step}px, cluster_gap={cluster_gap}px, max_vertical_step={max_vertical_step}px,",
        f"runs={len(runs)}, selected_points={len(best_run)},",
        f"x_span={stats['x_span']}, y_span={stats['y_span']}, x/y={stats['ratio']},",
        f"clamped_jumps={clamped_jump_count}",
    )

    return best_run

# 1. 載入模型 (選用在 Cityscapes 預訓練過的 B0 版本)
# model_id = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
# model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
model_id = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
# 讀取 Access Token (優先從檔案，次之從環境變數)
token_file = "hf_token.txt"
if os.path.exists(token_file):
    with open(token_file, "r") as f:
        hf_token = f.read().strip()
else:
    hf_token = os.getenv("HF_TOKEN")

processor = SegformerImageProcessor.from_pretrained(model_id, token=hf_token)
model = SegformerForSemanticSegmentation.from_pretrained(model_id, token=hf_token)
# 列印出所有 ID 與 標籤 的對應關係
print(model.config.id2label)
# 2. 讀取照片並推論
input_dir = "images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

image_name = "street.jpg"
image_path = os.path.join(input_dir, image_name)
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # 取得分割結果

# 3. 提取「路面」類別
# 將 Logits 放大回原始圖片大小以獲得準確的面罩
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False,
)

# 在 Cityscapes 中，ID 0 是 Road, ID 1 是 Sidewalk
mask = (upsampled_logits.argmax(1) == 1).float()
print(f"Mask generated with shape: {mask.shape}")

# 4. 顯示與保存結果
# 將遮罩轉換為 PIL 圖片並保存
mask_np = mask[0].cpu().numpy()
mask_img = Image.fromarray((mask_np * 255).astype('uint8'))
mask_img.save(os.path.join(output_dir, "road_mask.png"))

# # 選項：創建一個半透明疊加圖 (Overlay)
# overlay = image.convert("RGBA")
# red_mask = Image.new("RGBA", image.size, (255, 0, 0, 100)) # 紅色半透明
# result = Image.composite(red_mask, overlay, mask_img)
# result.convert("RGB").save("result_overlay.jpg")
# print("Saved visualization to 'road_mask.png' and 'result_overlay.jpg'")

# 4. 形態學處理 (Morphology)
print("Performing morphology operations...")
mask_np_uint8 = (mask_np * 255).astype('uint8')
morph_kernel_size = get_morph_kernel_size(image.size)
print(
    f"Using adaptive morphology kernel: {morph_kernel_size}x{morph_kernel_size} "
    f"(min image dimension = {min(image.size)})"
)
kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
opening = cv2.morphologyEx(mask_np_uint8, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

Image.fromarray(closing).save(os.path.join(output_dir, "road_mask_closing.png"))
print("Saved cleaned morphology results to 'road_mask_closing.png'")

# 選項：創建一個半透明疊加圖 (Overlay)
overlay = image.convert("RGBA")
red_mask = Image.new("RGBA", image.size, (255, 0, 0, 100)) # 紅色半透明
result = Image.composite(red_mask, overlay, Image.fromarray(closing))
result.convert("RGB").save(os.path.join(output_dir, "result_overlay.jpg"))
print(f"Saved visualization to '{output_dir}/road_mask.png' and '{output_dir}/result_overlay.jpg'")

# 5. 骨架化與主幹提取 (Skeletonization & Path Search)
print("Performing skeletonization...")
skeleton = skeletonize(closing > 0)
skeleton_uint8 = (skeleton * 255).astype(np.uint8)
Image.fromarray(skeleton_uint8).save(os.path.join(output_dir, "skeleton.png"))
print(f"Saved skeleton to '{output_dir}/skeleton.png'")

def get_longest_path(skeleton_img):
    points = np.argwhere(skeleton_img > 0)
    if len(points) == 0:
        return []
    
    G = nx.Graph()
    point_set = set(map(tuple, points))
    
    # 建立 8-鄰接圖
    for r, c in points:
        G.add_node((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                neighbor = (r + dr, c + dc)
                if neighbor in point_set:
                    G.add_edge((r, c), neighbor)
    
    if len(G) == 0: return []
    
    # 找到最大連通分量
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_cc = G.subgraph(components[0])
    
    # 使用直徑演算法找到最長路徑
    nodes = list(largest_cc.nodes())
    if not nodes: return []
    
    # 兩次 BFS 找直徑 (適用於樹)
    u = nodes[0]
    lengths = nx.single_source_shortest_path_length(largest_cc, u)
    v = max(lengths, key=lengths.get)
    
    lengths_v = nx.single_source_shortest_path_length(largest_cc, v)
    w = max(lengths_v, key=lengths_v.get)
    
    path = nx.shortest_path(largest_cc, v, w)
    return path

print("Extracting longest main path...")
main_path = get_longest_path(skeleton)
main_path_stats = get_path_span_stats(main_path)
print(
    "Longest-path stats:",
    f"points={len(main_path)}, x_span={main_path_stats['x_span']}, "
    f"y_span={main_path_stats['y_span']}, x/y={main_path_stats['ratio']}"
)

print("Building left-to-right playable path (scanline constrained)...")
playable_path = build_playable_scanline_path(skeleton, image.size)
path_source_name = "playable_scanline" if len(playable_path) >= 2 else "longest_path_fallback"
path_for_simplify = playable_path if len(playable_path) >= 2 else main_path
if not path_for_simplify:
    raise RuntimeError("No valid path extracted from skeleton.")
print(f"Selected path source: {path_source_name} (points={len(path_for_simplify)})")

def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker algorithm for path simplification."""
    if len(points) < 3:
        return points

    dmax = 0
    index = 0
    end = len(points) - 1
    
    # Calculate shortest distance from points to the line segment (A, B)
    A = np.array(points[0])
    B = np.array(points[end])
    AB = B - A
    norm_AB = np.linalg.norm(AB)
    
    for i in range(1, end):
        C = np.array(points[i])
        if norm_AB == 0:
            d = np.linalg.norm(C - A)
        else:
            # Distance from point C to line AB: |det(AB, AC)| / |AB|
            # det([x1, y1], [x2, y2]) = x1*y2 - y1*x2
            d = np.abs(AB[0] * (C[1] - A[1]) - AB[1] * (C[0] - A[0])) / norm_AB
            
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        # Recursive call
        left = rdp_simplify(points[:index+1], epsilon)
        right = rdp_simplify(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[end]]

print("Simplifying path using RDP algorithm...")
rdp_epsilon = get_rdp_epsilon(image.size)
print(f"Using adaptive RDP epsilon: {rdp_epsilon:.2f} px (min image dimension = {min(image.size)})")
simplified_path = rdp_simplify(path_for_simplify, epsilon=rdp_epsilon)
simplified_path = _dedupe_nonincreasing_x(simplified_path)
print(f"Reduced points from {len(path_for_simplify)} to {len(simplified_path)}")
final_path_stats = get_path_span_stats(simplified_path)
print(
    "Final-path stats:",
    f"x_span={final_path_stats['x_span']}, y_span={final_path_stats['y_span']}, "
    f"x/y={final_path_stats['ratio']}"
)

# 6. 可視化最長路徑 (使用簡化後的路徑)
path_img = np.zeros_like(closing)
for r, c in simplified_path:
    path_img[r, c] = 255

# 將路徑加粗一點以便觀察
kernel_path = np.ones((3, 3), np.uint8)
path_img_dilated = cv2.dilate(path_img, kernel_path, iterations=1)
Image.fromarray(path_img_dilated).save(os.path.join(output_dir, "main_path.png"))

# 疊加到原始圖上
result_final = image.convert("RGBA")
draw_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
# 使用鮮艷的綠色標記路徑
from PIL import ImageDraw
draw = ImageDraw.Draw(draw_overlay)
if len(simplified_path) > 1:
    # 轉換座標為 (x, y)
    coords = [(c, r) for r, c in simplified_path]
    draw.line(coords, fill=(0, 255, 100, 255), width=5)

result_overlay = Image.alpha_composite(result_final, draw_overlay)
result_overlay.convert("RGB").save(os.path.join(output_dir, "result_final.jpg"))
print(f"Saved final path visualization to '{output_dir}/main_path.png' and '{output_dir}/result_final.jpg'")

# 7. 將 final skeleton 按照 schema.json 的格式輸出 JSON file
print(f"Exporting skeleton to '{output_dir}/output.json'...")
width, height = image.size
# 轉換為正規化座標 (x, y)，x 對應 c (column)，y 對應 r (row)
normalized_points = [[round(float(c) / width, 4), round(float(r) / height, 4)] for r, c in simplified_path]

output_data = {
    "version": "1.0",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "source_image": image_name,
    "aspect_ratio": round(width / height, 4),
    "points": normalized_points,
    "occluded_segments": [],
    "terrain_type": "asphalt",
    "friction_coefficient": 0.8
}

output_path = os.path.join(output_dir, "output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)

print(f"Successfully exported final skeleton to '{output_path}'")
