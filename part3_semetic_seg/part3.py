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

image_name = "street2d.jpg"
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
kernel = np.ones((50, 50), np.uint8)
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
# epsilon = 2.0 pixels distance threshold
simplified_path = rdp_simplify(main_path, epsilon=2.0)
print(f"Reduced points from {len(main_path)} to {len(simplified_path)}")

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