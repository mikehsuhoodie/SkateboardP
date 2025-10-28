import cv2
import numpy as np
import sys, json, argparse
import math


def odd_at_least(n, lo=3, hi=21):
    n = int(n) // 2 * 2 + 1  # 轉成奇數
    return max(lo, min(n, hi))

def auto_params(h, w):
    diag = (h**2 + w**2) ** 0.5         # 影像對角線
    perim_ref = (h + w)                 # 尺度參考（你原本就用）

    params = {
        # ── 前處理 ───────────────────────────────────────
        "blur_ksize": odd_at_least(diag / 400),   # 小圖 3、大圖 7~11 左右
        "canny_sigma": 0.33,                      # 可依雜訊調 0.25~0.6

        # ── 輪廓過濾（已是比例制，保留） ─────────────────
        # "min_area": 0.0003 * (h * w),             # 佔全圖 0.03%
        # "min_peri": 0.01 * perim_ref,             # 佔 (h+w) 的 1%
        "min_area": 0,             # 佔全圖 0.03%
        "min_peri": 0,             # 佔 (h+w) 的 1%

        # ── 折線化與平滑 ────────────────────────────────
        "approx_epsilon_frac": 0.007,             # 仍用周長比例，與尺寸無關
        "chaikin_iters": 1,

        # ── 片段過濾（用對角線比例）──────────────────────
        "min_seg_len": 0.012 * diag,               # 約對角線的 2%（小圖≈10px，大圖自動放大）
        "max_slope_deg": 80.0,                    # 角度與尺寸無關
    }
    return params
# def auto_params(h=1080, w=1920):
#     diag = (h**2 + w**2) ** 0.5  # ≈ 2202.9
#     params = {
#         # 前處理
#         "blur_ksize": 3,           # 原本 diag/400 ≈ 5.5 → odd=7；改小到 3，保留細節
#         "canny_sigma": 0.55,       # 原 0.33 → 0.50；自動門檻範圍更寬，抓更多邊

#         # 輪廓過濾（小但不為 0，避免極小噪點）
#         "min_area": int(0.00005 * (h*w)),     # ≈ 104 px^2
#         "min_peri": int(0.004   * (h+w)),     # ≈ 12 px

#         # 折線化 + 平滑（保留更多節點）
#         "approx_epsilon_frac": 0.0045,        # 原 0.007 → 0.0045
#         "chaikin_iters": 2,                   # 原 2 → 1（少平滑一點）

#         # 片段過濾
#         "min_seg_len": 0.01 * diag,           # 原 0.02 → 0.01（≈ 22 px）
#         "max_slope_deg": 85.0,                # 原 60 → 85（幾乎不限制斜率）
#     }
#     return params



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

def auto_canny(gray, sigma=0.33):#sigma越小越敏感
    # 根據影像中位數自動估 Canny 門檻（更穩定）
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def filter_polylines_by_segment_rules(polylines,
                                      min_seg_len=8.0,      # 像素；短於此值的線段刪除
                                      max_slope_deg=60.0,   # 與水平夾角「大於」此度數視為過於傾斜
                                      min_poly_pts=2):      # 輸出子折線的最少點數
    """
    逐段檢查，每個折線被切成多條僅包含「合格線段」的子折線。
    - 過短：長度 < min_seg_len 的線段捨棄
    - 過傾斜：abs(angle w.r.t. horizontal) > max_slope_deg 的線段捨棄
    - 若一段不合格，就在此處斷開，繼續從下一段重新累積
    回傳：list[list[[x,y],...]]（只保留合格子折線）
    """
    def seg_len(p, q):
        return float(np.linalg.norm(q - p))

    def seg_angle_deg(p, q):
        # 與水平線的夾角（度），範圍 [0, 180)
        dy = float(q[1] - p[1])
        dx = float(q[0] - p[0])
        ang = abs(math.degrees(math.atan2(dy, dx)))  # 0=水平，90=垂直
        # 折線方向不分正負，因此用對稱角（>90 的等價為 180-ang）
        return ang if ang <= 90.0 else 180.0 - ang

    filtered = []

    for poly in polylines:
        pts = np.array(poly, dtype=float)
        if len(pts) < 2:
            continue

        curr = [pts[0]]  # 正在累積的子折線
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i+1]
            L = seg_len(p, q)
            ang = seg_angle_deg(p, q)

            is_short = (L < min_seg_len)
            is_short_for_steep = (L < 1.5 * min_seg_len)
            is_too_steep = (ang > max_slope_deg) and is_short_for_steep

            if not is_short and not is_too_steep:
                # 合格：加入下一點，延續目前子折線
                curr.append(q)
            else:
                # 不合格：先把當前子折線收掉（若足夠長）
                if len(curr) >= min_poly_pts:
                    filtered.append(np.array(curr))
                # 重新開始新的子折線，從 q 當起點
                curr = [q]

        # 收尾
        if len(curr) >= min_poly_pts:
            filtered.append(np.array(curr))

    # 轉回 list[list]
    return [sub.tolist() for sub in filtered]


TARGET_W, TARGET_H = 1920, 1080

def resize_letterbox(img, tw=TARGET_W, th=TARGET_H):
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw,3), dtype=img.dtype)   # 灰階；彩色請改 (th, tw, 3)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas



# --- 主程式流程 ---

parser = argparse.ArgumentParser(description="Generate level from image")
parser.add_argument("input_path", type=str, help="輸入圖片路徑")
parser.add_argument("-u", "--unity", action="store_true", help="開啟輸出到 Unity ")
args = parser.parse_args()

input_path = args.input_path
paste_to_unity = args.unity
output_img_path = "output_preview.png"
output_beforeFilter_path = "output_beforeFilter.png"
output_json_path = "polylines.json"

# 設輸出路徑為 Unity Assets
unity_assets_path = "/mnt/d/Project/SkateboardP/SkateboardP/Assets/json/"  # 注意尾端加 / 分隔



# 1. 讀取輸入圖像
ori_img = cv2.imread(input_path)
if ori_img is None:
    print(f"讀取圖片失敗: {input_path}")
    sys.exit(1)

# 1-2. Resize
img = resize_letterbox(ori_img)

# 2. 灰階與去噪/對比
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 提升對比，讓邊緣更完整（可視需要開關）
gray = cv2.equalizeHist(gray)

h, w = gray.shape[:2]
P = auto_params(h, w)

gray = cv2.GaussianBlur(gray, (P["blur_ksize"], P["blur_ksize"]), 0)

# 新增：保存模糊後的灰階圖片
cv2.imwrite("blurred_gray.png", gray)  # 輸出目前圖片的樣子（灰階模糊版）
print("模糊後灰階圖已輸出: blurred_gray.png")  # 提示訊息

# 3. 邊緣（自動 Canny
edges = auto_canny(gray, sigma=P["canny_sigma"])
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# 4. 找「所有」輪廓（不是只有外層）
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("未找到任何輪廓！")
    sys.exit(1)

# 5. 過濾掉太小的雜點（依圖像大小調整門檻）


min_area = 0.0003 * (h * w)      # 面積小於全圖 
min_peri = 0.01  * (h + w)       # 周長小於圖寬高和的 1% 略過

kept_contours = []
for c in contours:
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if area >= P["min_area"] and peri >= P["min_peri"]:
        kept_contours.append(c)

if len(kept_contours) == 0:
    print("找到輪廓但全被過濾掉，請調低 min_area/min_peri。")
    sys.exit(1)

# 6. 對每條輪廓做多邊形逼近 + Chaikin 平滑

polylines = []

preview = img.copy()
preview_beforeFilter = img.copy()

for c in kept_contours:
    is_closed = True

    peri = cv2.arcLength(c, is_closed)
    epsilon =P["approx_epsilon_frac"] * peri  # 0.005~0.02 越大越簡略
    approx = cv2.approxPolyDP(c, epsilon, is_closed)  # True=視為封閉輪廓
    pts = approx.reshape(-1, 2)

    # 為了讓平滑更像「路徑」，將封閉輪廓打開或保留封閉：
    # - 若你想全部當開放折線畫：把 is_closed 設 False
    # - 想保持形狀外框：is_closed=True
    
    # Chaikin 平滑
    if len(pts) >= 2:
        smoothed = chaikin_smooth(pts, iterations=P["chaikin_iters"], keep_ends=not is_closed)
        polylines.append(smoothed.tolist())

 # 過濾前的樣子
for poly in polylines:
    draw_pts = np.round(np.array(poly)).astype(np.int32).reshape((-1,1,2))
    cv2.polylines(preview_beforeFilter, [draw_pts], isClosed=False, thickness=2, color=(0,0,255))



polylines = filter_polylines_by_segment_rules(
    polylines,
    min_seg_len=P["min_seg_len"],
    max_slope_deg=P["max_slope_deg"],
    min_poly_pts=2
)

for poly in polylines:
    draw_pts = np.round(np.array(poly)).astype(np.int32).reshape((-1,1,2))
    cv2.polylines(preview, [draw_pts], isClosed=False, thickness=2, color=(0,0,255))

# 7. 儲存預覽

cv2.imwrite(output_beforeFilter_path, preview_beforeFilter)
print(f"過濾前預覽圖已輸出: {output_beforeFilter_path}")
cv2.imwrite(output_img_path, preview)
print(f"預覽圖已輸出: {output_img_path}")

# 8. JSON：輸出多條 polyline；spawn/goal 以「最長一條」的首尾當示範
def poly_length(poly):
    poly = np.array(poly, dtype=float)
    return np.sum(np.linalg.norm(poly[1:] - poly[:-1], axis=1)) if len(poly) > 1 else 0.0

# 檢查：處理 polylines 為空的情況
if not polylines:
    print("警告：沒有偵測到任何合格的 polylines！請檢查輸入圖片、調整參數（如降低 min_seg_len 或 max_slope_deg），或確認邊緣偵測是否正常。")
    longest_idx = -1  # 無效索引
    spawn = [0, 0]    # 預設起點
    goal = [0, 0]     # 預設終點
else:
    lengths = [poly_length(p) for p in polylines]
    longest_idx = int(np.argmax(lengths))
    spawn = polylines[longest_idx][0] if len(polylines[longest_idx]) else [0, 0]
    goal = polylines[longest_idx][-1] if len(polylines[longest_idx]) else [0, 0]

map_data = {
    #"scale": 1.0,
    "polylines": polylines  # 多條！即使空也輸出 []
}
with open(output_json_path, 'w') as f:
    json.dump(map_data, f, indent=2, ensure_ascii=False)
print(f"地圖資訊已輸出: {output_json_path}")

if paste_to_unity :
    output_unity_img_path = unity_assets_path + "output_preview.png"
    output_unity_json_path = unity_assets_path + "polylines.json"
    output_unity_ori_path = unity_assets_path + "original.png"

    # 儲存預覽圖
    cv2.imwrite(output_unity_img_path, preview)
    print(f"預覽圖已輸出到 Unity: {output_unity_img_path}")

    cv2.imwrite(output_unity_ori_path, img)
    print(f"原圖已輸出到 Unity: {output_unity_ori_path}")

    # 儲存 JSON
    with open(output_unity_json_path, 'w') as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False)
    print(f"JSON 已輸出到 Unity: {output_unity_json_path}")
