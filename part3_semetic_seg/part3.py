from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import skeletonize
import networkx as nx

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
image = Image.open("street.jpg")
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
mask_img.save("road_mask.png")

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

Image.fromarray(closing).save("road_mask_closing.png")
print("Saved cleaned morphology results to 'road_mask_closing.png'")

# 選項：創建一個半透明疊加圖 (Overlay)
overlay = image.convert("RGBA")
red_mask = Image.new("RGBA", image.size, (255, 0, 0, 100)) # 紅色半透明
result = Image.composite(red_mask, overlay, Image.fromarray(closing))
result.convert("RGB").save("result_overlay.jpg")
print("Saved visualization to 'road_mask.png' and 'result_overlay.jpg'")

# 5. 骨架化與主幹提取 (Skeletonization & Path Search)
print("Performing skeletonization...")
skeleton = skeletonize(closing > 0)
skeleton_uint8 = (skeleton * 255).astype(np.uint8)
Image.fromarray(skeleton_uint8).save("skeleton.png")
print("Saved skeleton to 'skeleton.png'")

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

# 6. 可視化最長路徑
path_img = np.zeros_like(closing)
for r, c in main_path:
    path_img[r, c] = 255

# 將路徑加粗一點以便觀察
kernel_path = np.ones((3, 3), np.uint8)
path_img_dilated = cv2.dilate(path_img, kernel_path, iterations=1)
Image.fromarray(path_img_dilated).save("main_path.png")

# 疊加到原始圖上
result_final = image.convert("RGBA")
draw_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
# 使用鮮艷的綠色標記路徑
from PIL import ImageDraw
draw = ImageDraw.Draw(draw_overlay)
if len(main_path) > 1:
    # 轉換座標為 (x, y)
    coords = [(c, r) for r, c in main_path]
    draw.line(coords, fill=(0, 255, 100, 255), width=5)

result_overlay = Image.alpha_composite(result_final, draw_overlay)
result_overlay.convert("RGB").save("result_final.jpg")
print("Saved final path visualization to 'main_path.png' and 'result_final.jpg'")