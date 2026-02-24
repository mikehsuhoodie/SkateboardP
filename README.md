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

此模組負責從照片中識別「路面」，並提取其骨架（Skeleton）與簡化後的主幹道路徑。

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

## 輸出結果 (Outputs)

執行完成後，所有分析結果將儲存於 `part3_semetic_seg/results/`：

- `results/road_mask_closing.png`: 形態學處理後的路面遮罩。
- `results/skeleton.png`: 道路中心骨架線。
- `results/main_path.png`: 最終提取的最長路徑視覺化（經 RDP 演算法簡化）。
- `results/result_final.jpg`: 將提取路徑疊加於原圖的預覽圖。
- `results/output.json`: 符合 `schema.json` 規範的路徑點位資料。

## 關鍵技術
- **SegFormer**: 用於高品質的語義分割。
- **Skeletonization**: 提取道路中心結構。
- **RDP Algorithm**: 簡化路徑點數，優化資料傳輸。
