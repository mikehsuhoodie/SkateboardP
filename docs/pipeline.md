# Pipeline

## Current WSL CV Flow

1. `POST /infer`
2. -> `infer.py`
3. -> `run_inference_sobal.py`
4. -> `cut_img.py`
5. -> `extract_track.py`
6. -> payload response

## Full Future Flow

Unity -> Windows server -> WSL CV -> Windows server -> Unity

## File Responsibilities

*   **Depth Image:** `run_inference_sobal.py`
*   **Depth Mask:** `run_inference_sobal.py`
*   **Visual Layers:** `cut_img.py`
*   **track_points.json:** `extract_track.py`
*   **Final Response Payload:** `infer.py`

## Known Fragile Points

*   **Hardcoded paths:** Legacy scripts have hardcoded paths to image locations and output directories.
*   **Generated output folders:** Reliance on specific folder structures (`outputs/`, `Send2Unity/`) during execution.
*   **WSL/Windows boundary:** File paths may conflict if shared directly, though Base64 payloads mitigate this.
*   **DA3 vendor dependency:** `src/depth_anything_3/` is vendor code. Any changes to DA3's API require updates to our `depth_adapter.py`.
*   **Subprocess orchestration:** `infer.py` orchestrates the pipeline using `subprocess` calls and dynamic string patching of hardcoded paths, which is fragile to code changes in the legacy scripts.
