# Semantic Segmentation & Path Extraction (Part 3)

此模組主要負責從街道照片中識別出「人行道 / 道路」，並提取其骨架（Skeleton）與最長主幹道。

## 環境配置 (Environment)

此模組建議在 `main310` 環境下運行：

```bash
conda activate main310
pip install scikit-image networkx transformers torch opencv-python
```

## 使用說明 (Usage)

1. **配置 Token**:
   請確保在此目錄下有 `hf_token.txt`，內容為你的 Hugging Face Access Token。
2. **準備圖片**:
   放置一張名為 `street.jpg` 的圖片。
3. **執行腳本**:
   ```bash
   python part3.py
   ```

## 輸出結果 (Outputs)

- `road_mask_closing.png`: 經過形態學清理後的遮罩。
- `skeleton.png`: 道路的單像素中心線（骨架）。
- `main_path.png`: 提取出的最長主幹道路徑。
- `result_final.jpg`: 將綠色路徑疊加在原圖上的最終視覺化結果。
