# 2.5D Skateboard Path Vision Project (Mike v2.0)

## 1. 專案願景 (Project Goal)
這是一個將現實照片轉化為 2.5D 滑板遊戲關卡的系統。利用電腦視覺技術提取路徑，並轉換為遊戲引擎可用的資料格式。

## 2. 環境設定 (Environment Setup)

### 軟體需求
- **Unity**: 6.2 (2D Core Template)
- **Python**: 3.10.x (建議使用 Conda 環境 `main310`)

### Python 環境初始化
我們建議使用 Conda 來管理環境。所有必要的套件都已記錄在 `part3_semetic_seg/requirements.txt` 中。

```bash
# 1. 建立並啟動環境
conda create -n main310 python=3.10
conda activate main310

# 2. 安裝必要的套件
cd part3_semetic_seg
pip install -r requirements.txt
```

---

# Semantic Segmentation & Path Extraction (Part 3)

此模組負責從照片中自動識別可用語意類別，並為每個類別提取候選骨架路徑，最後輸出 1~3 條可玩 track。

## 使用說明 (Usage)

1. **配置 Token**:
   在 `part3_semetic_seg/` 目錄下建立 `hf_token.txt`，並貼入你的 Hugging Face Access Token。
2. **準備圖片**:
   將街道照片（例如 `street.jpg`）放置於 `part3_semetic_seg/images/` 資料夾內。
3. **執行腳本**:
   ```bash
   cd part3_semetic_seg
   python part3.py
   ```
4. **可選參數（環境變數）**:
   - `INPUT_IMAGE`：指定輸入檔名（預設 `street2d.jpg`）
   - `SEGFORMER_MODEL_ID`：指定模型（預設 `nvidia/segformer-b2-finetuned-ade-512-512`）

## 輸出結果 (Outputs)

執行完成後，所有分析結果將儲存於 `part3_semetic_seg/results/`：

- `results/class_<id>_<label>_mask.png`: 各語意類別的原始遮罩。
- `results/class_<id>_<label>_mask_closing.png`: 各類別形態學清理結果。
- `results/class_<id>_<label>_skeleton.png`: 各類別骨架圖。
- `results/class_<id>_<label>_path.png`: 各類別最終路徑。
- `results/main_path.png`: 已選中 track 的合併路徑圖。
- `results/result_final.jpg`: 已選中 track 疊加預覽圖。
- `results/output.json`: 多 track JSON（含 `tracks`、`stage_index`、`semantic_label`）。

## 關鍵技術
- **SegFormer**: 用於高品質的語義分割。
- **Skeletonization**: 提取道路中心結構。
- **RDP Algorithm**: 簡化路徑點數，優化資料傳輸。

## 近期優化紀錄 (Safe Optimization Pass)

### 1) Morphology Kernel 改為依圖片尺寸自適應
- 位置：`part3_semetic_seg/part3.py`
- 原本使用固定 `50x50` kernel，對不同解析度圖片可能清理過度或不足。
- 現在改為依圖片短邊比例自動計算 kernel（並設最小/最大值上限），讓不同解析度下的結果更穩定。

### 2) 影響範圍（重要）
- **JSON 已升級為多 track 格式**（新增 `stage_count` / `tracks`）
- **保留舊相容欄位**：頂層 `points` 仍存在（預設為 `tracks[0].points`）
- **不改變 Unity 座標映射邏輯**
- 可能改變的是：遮罩清理與路徑評分後的最終軌跡

### 3) 目前參數（可調）
- `MORPH_KERNEL_RATIO = 0.06`
- `MORPH_KERNEL_MIN = 9`
- `MORPH_KERNEL_MAX = 71`

若不同場景（室內/公園/街道）清理效果差異大，建議先微調上述參數，再考慮更大的演算法改動。

### 4) 新增可玩性導向的 x-monotonic 路徑修正（避免鉛直軌道）
- 位置：`part3_semetic_seg/part3.py`
- 在骨架化之後，新增「由左到右掃描」的路徑候選器（scanline constrained path）。
- 規則（低風險版本）：
  - 每個 x 掃描區間只保留一條候選線段（避免上下平行雙線同時進入最終 track）
  - 限制相鄰區間的最大高度跳動（避免近鉛直段）
  - 允許少量空白區間並以插值補齊（保持單一路徑連續）
  - 只保留夠長的連續 run（避免短碎段）
- 若掃描式候選器失敗，會自動 fallback 回原本的「最長路徑」流程。

### 5) 目前限制（已知）
- Unity 端目前使用單一 `EdgeCollider2D`，因此 JSON 雖然有 `occluded_segments` 欄位，但尚未實作真正的「斷軌 / 跳躍 gap collider」。
- 現階段的「空格」是以小範圍插值保持連續可滑，避免生成完全斷裂的 collider。

### 6) RDP epsilon 改為依圖片尺寸自適應（避免點數過多/過少）
- 位置：`part3_semetic_seg/part3.py`
- 原本固定使用 `epsilon = 2.0 px`，在高解析度圖片可能保留太多點、低解析度圖片可能過度簡化。
- 現在改為依圖片短邊比例計算 epsilon，並設最小/最大值夾限，讓簡化密度在不同解析度下更一致。

### 7) 新增可調參數（RDP）
- `RDP_EPSILON_RATIO = 0.005`
- `RDP_EPSILON_MIN = 1.0`
- `RDP_EPSILON_MAX = 8.0`

### 8) 多類別自動 Track Generation（不再限定 road/sidewalk）
- 位置：`part3_semetic_seg/part3.py`
- 腳本會先自動統計圖片內出現的語意類別，再對每個類別各自產生候選 track。
- 會依可玩性與品質評分篩選，輸出最多 3 條 track。
- 每條 track 都包含：
  - `stage_index`（第幾階段）
  - `semantic_label`
  - `points`
- 為相容舊版 Unity，頂層 `points` 仍保留，內容為 `tracks[0].points`。

### 9) 類別涵蓋範圍注意事項
- 類別能否被偵測取決於模型標籤空間。
- 目前預設模型為 ADE20K 版本（可涵蓋較多室內/室外物件）。
- 若要改模型，可設定：
  - `export SEGFORMER_MODEL_ID=\"<your-model-id>\"`
