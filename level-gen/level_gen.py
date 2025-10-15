import cv2
import numpy as np
import sys, json

def chaikin_smooth(points, iterations=2, keep_ends=True):
    smoothed = np.array(points, dtype=float)
    for _ in range(iterations):
        new_pts = []
        if keep_ends and len(smoothed) > 0:
            new_pts.append(smoothed[0])
        for i in range(len(smoothed) - 1):
            p1 = smoothed[i]
            p2 = smoothed[i+1]
            Q = 0.75 * p1 + 0.25 * p2
            R = 0.25 * p1 + 0.75 * p2
            new_pts.extend([Q, R])
        if keep_ends and len(smoothed) > 0:
            new_pts.append(smoothed[-1])
        smoothed = np.array(new_pts, dtype=float)
    return smoothed

def auto_canny(gray, sigma=0.33):
    # 根據影像中位數自動估 Canny 門檻（更穩定）
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

# --- 主程式流程 ---
if len(sys.argv) < 2:
    print("用法: python main.py <輸入圖片路徑> [預覽輸出圖片] [地圖輸出JSON]")
    sys.exit(1)

input_path = sys.argv[1]
output_img_path = sys.argv[2] if len(sys.argv) > 2 else "output_preview.png"
output_json_path = sys.argv[3] if len(sys.argv) > 3 else "output_map.json"

# 1. 讀取輸入圖像
img = cv2.imread(input_path)
if img is None:
    print(f"讀取圖片失敗: {input_path}")
    sys.exit(1)

# 2. 灰階與去噪/對比
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 提升對比，讓邊緣更完整（可視需要開關）
gray = cv2.equalizeHist(gray)
gray = cv2.GaussianBlur(gray, (5,5), 0)

# 3. 邊緣（自動 Canny）+ 形態學閉運算把斷裂邊補起來
edges = auto_canny(gray, sigma=0.33)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# 4. 找「所有」輪廓（不是只有外層）
#    想保留階層關係可用 RETR_TREE
contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("未找到任何輪廓！")
    sys.exit(1)

# 5. 過濾掉太小的雜點（依圖像大小調整門檻）
h, w = gray.shape[:2]
min_area = 0.0005 * (h * w)      # 面積小於全圖 0.05% 略過
min_peri = 0.01  * (h + w)       # 周長小於圖寬高和的 1% 略過

kept_contours = []
for c in contours:
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if area >= min_area and peri >= min_peri:
        kept_contours.append(c)

if len(kept_contours) == 0:
    print("找到輪廓但全被過濾掉，請調低 min_area/min_peri。")
    sys.exit(1)

# 6. 對每條輪廓做多邊形逼近 + Chaikin 平滑
polylines = []
preview = img.copy()

for c in kept_contours:
    peri = cv2.arcLength(c, True)
    epsilon = 0.01 * peri  # 可視需要調 0.005~0.02
    approx = cv2.approxPolyDP(c, epsilon, True)  # True=視為封閉輪廓
    pts = approx.reshape(-1, 2)

    # 為了讓平滑更像「路徑」，將封閉輪廓打開或保留封閉：
    # - 若你想全部當開放折線畫：把 is_closed 設 False
    # - 想保持形狀外框：is_closed=True
    is_closed = True

    # Chaikin 平滑
    if len(pts) >= 2:
        smoothed = chaikin_smooth(pts, iterations=2, keep_ends=not is_closed)
        polylines.append(smoothed.tolist())

        # 繪製（視需求選擇 isClosed）
        draw_pts = np.round(smoothed).astype(np.int32).reshape((-1,1,2))
        cv2.polylines(preview, [draw_pts], isClosed=is_closed, color=(255,255,255), thickness=2)

# 7. 儲存預覽
cv2.imwrite(output_img_path, preview)
print(f"預覽圖已輸出: {output_img_path}")

# 8. JSON：輸出多條 polyline；spawn/goal 以「最長一條」的首尾當示範
def poly_length(poly):
    poly = np.array(poly, dtype=float)
    return np.sum(np.linalg.norm(poly[1:] - poly[:-1], axis=1)) if len(poly) > 1 else 0.0

longest_idx = int(np.argmax([poly_length(p) for p in polylines]))
spawn = polylines[longest_idx][0] if len(polylines[longest_idx]) else [0,0]
goal  = polylines[longest_idx][-1] if len(polylines[longest_idx]) else [0,0]

map_data = {
    "scale": 1.0,
    "polylines": polylines,  # 多條！
    "spawn": spawn,
    "goal": goal
}
with open(output_json_path, 'w') as f:
    json.dump(map_data, f, indent=4, ensure_ascii=False)
print(f"地圖資訊已輸出: {output_json_path}")
