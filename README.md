# SkateP 2D to Unity 軌道生成管線

這個專案基於 Depth Anything 3，提供將一般 2D 街景照片轉換為 Unity 3D 滑板遊戲軌道與視差背景圖層的完整流程。
我們提供了兩種使用方式：**手動執行單步腳本 (Pipeline)** 與 **啟動 FastAPI 後端服務 (Server)**。

## 🛠️ 環境設定

請確保您的環境中安裝了 Python 3.10+，並且安裝了相關的依賴套件。
如果使用虛擬環境（例如 `.venv`），請先啟動它：

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart  # FastAPI 伺服器所需套件
```

---

## 🚀 方式一：手動執行三步處理管線 (Pipeline)

此方法適合測試單張圖片或是除錯。原始圖片預設放置於 `./assets/examples/SOH/` 目錄下（例如 `street2d.jpg`）。

### 第一步：深度推論與邊緣切割
使用 AI 模型計算圖片深度，並透過 Sobel 演算法找出深度邊界。

```bash
python run_inference_sobal.py
```
* **輸入：** 原始圖片
* **輸出：** 存於 `./outputs/inference_results/`，包含 `_depth_16bit.png` (高精度深度圖) 與 `_depth_mask.npy` (切割遮罩)。

### 第二步：圖片分層與背景補洞
讀取上一步的深度圖與遮罩，將原始圖片拆解成 前景、中景、遠景 三個圖層，並將被前景擋住的背景進行補洞 (Inpainting)。

```bash
python cut_img.py
```
* **輸入：** 第一步產生的 `.png` 與 `.npy` 檔案。
* **輸出：** 存於 `./Send2Unity/`，包含 `merged_layer_01.png`, `merged_layer_02.png`, `merged_layer_03.png`，以及前景遮罩 `layer_00_mask.npy`。

### 第三步：提取平滑軌道資料
讀取前景遮罩的上緣，經過高斯濾波與三次方樣條插值，生成一條完美的平滑滑板軌道，並輸出為 Unity 可讀的 JSON 格式。

```bash
python extract_track.py
```
* **輸入：** 第二步產生的 `layer_00_mask.npy` 與原始圖片（用來採樣軌道顏色）。
* **輸出：** 存於 `./Send2Unity/`，包含 `track_points.json` (Unity 軌道點資料) 與視覺化預覽圖。

---

## 🌐 方式二：啟動 FastAPI 後端服務 (Server)

我們提供了一個 `infer.py` 作為常駐的 API 伺服器（適合跑在 WSL/Linux），用來接收來自 Windows 系統或其他前端的請求，全自動跑完上述三個步驟，並直接回傳結果給呼叫端。

### 啟動伺服器

```bash
uvicorn infer:app --host 0.0.0.0 --port 9000
```
*(伺服器啟動後，將會監聽 9000 port)*

### API 規格
* **Endpoint:** `POST /infer`
* **Content-Type:** `multipart/form-data`
* **欄位:** `photo` (夾帶要處理的圖片檔案)

**回傳格式 (JSON):**
伺服器不會回傳本機檔案路徑，而是直接將處理好的圖片轉為 Base64 字串，並附帶解析好的 JSON 軌道數據：
```json
{
  "foreground_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "gameplay_base64": "...",
  "background_base64": "...",
  "metadata": {
    "version": "1.0",
    "timestamp": "2026-04-23T...",
    "aspect_ratio": 1.7778,
    "points": [ [0.1, 0.5], [0.2, 0.45], ... ]
  }
}
```

### 伺服器運作機制：
1. 接收上傳圖片後，會在 `jobs/<UUID>/` 建立一個暫存工作目錄以避免並發請求衝突。
2. 循序在背景執行 `run_inference_sobal.py`, `cut_img.py`, `extract_track.py`。
3. 讀取最終產生的圖層與 JSON 檔案。
4. 將圖片打包為 Base64 字串後回傳給客戶端，並自動清除暫存目錄。
