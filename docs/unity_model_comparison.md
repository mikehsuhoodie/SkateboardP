# Unity Gen Scripts Comparison

這份文件記錄了 `unity_gen_v2.py` 與 `unity_depth_gen.py` 的主要差異，供開發參考。

## 1. 使用模型 (Model Selection)

| Script | Model ID | 描述 |
| :--- | :--- | :--- |
| **`unity_depth_gen.py`** | `depth-anything/DA3METRIC-LARGE` | 適合一般 Metric Depth 生成，速度較快。 |
| **`unity_gen_v2.py`** | `depth-anything/DA3NESTED-GIANT-LARGE` | 模型更大，支援更精準的 Pose 估計與細節，對顯卡記憶體要求較高。 |

## 2. 輸出資料 (Output Data)

| 輸出項目 | `unity_depth_gen.py` | `unity_gen_v2.py` | 備註 |
| :--- | :--- | :--- | :--- |
| **Depth Map** | ✅ (16-bit PNG) | ✅ (16-bit PNG) | 兩者皆輸出標準化的 16-bit 深度圖。 |
| **Scale** | ✅ (`scale.txt`) | ✅ (`scale.txt`) | 紀錄最大深度值 (公尺)。 |
| **FOV / Intrinsics** | ✅ (`fov.txt`) | ✅ (`intrinsics.txt`) | 舊版僅計算水平 FOV (角度)；V2 輸出完整的 3x3 內參矩陣。 |
| **Camera Pose** | ❌ | ✅ (`camera_pose.txt`) | V2 輸出 4x4 Extrinsics 矩陣，用於還原相機位置。 |
| **Inpainted Depth** | ❌ | ✅ (`depth_inpainted_16bit.png`) | V2 額外輸出一張經修補的深度圖。 |

## 3. 後處理功能 (Post-Processing)

### `unity_gen_v2.py` 特有的 Inpainting 流程：
V2 版本包含了一個針對遮擋 (Occlusion) 的修補流程，旨在優化 Unity 中的 Mesh 生成效果，避免物體邊緣出現破洞或過度拉伸。

1.  **邊緣檢測**：使用 Sobel 算子計算深度圖的梯度。
2.  **Mask 生成**：針對高梯度區域 (邊緣) 進行膨脹 (Dilation) 處理，標記為「待修補區域」。
3.  **Inpainting**：使用 `cv2.inpaint` (Telea 算法) 填補遮擋區域的深度值。

## 建議
*   **一般用途**：若僅需深度圖製作 parallax 或簡單特效，使用 `unity_depth_gen.py` 即可。
*   **3D 重建/Mesh 生成**：若需在 Unity 中生成準確的 Mesh 並對齊相機，務必使用 `unity_gen_v2.py` 以獲得 Pose 資訊與修補後的深度圖。
