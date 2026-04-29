# WSL CV Backend Documentation

This module handles the computer vision pipeline for the SkateP project. It receives images, extracts depth maps using Depth Anything 3, segments them into visual layers, and generates track points for Unity.

## 1. Environment Setup & Execution

We use `uv` for dependency management.

### Installation
```bash
# Inside the wsl-cv directory
uv sync
```

### Running the API Server
Start the FastAPI backend server:
```bash
uv run uvicorn scripts.infer:app --host 0.0.0.0 --port 9000
```
Outputs and temporary job directories (`_jobs/`) are managed automatically during API execution.

### Running Scripts Manually
```bash
uv run python scripts/pipeline.py ./assets/pictures/whisky.jpg
```
Intermediate outputs will be saved to `outputs/` and `Send2Unity/`.

## 2. Repository Map

*   `scripts/infer.py`: Active runtime API entrypoint (FastAPI server).
*   `scripts/pipeline.py`: Standalone pipeline execution script.
*   `scripts/run_inference_sobal.py`: Depth and mask generation.
*   `scripts/cut_img.py`: Visual layer generation (foreground, gameplay, background).
*   `scripts/extract_track.py`: Track point extraction from masks.
*   `scripts/depth_adapter.py`: Wrapper and integration point for the DA3 model.
*   `models/depth_anything_3/`: Vendor/model code (Do not modify).
*   `Legacy/`: Older scripts (e.g., `unity_depth_gen.py`) and experiments.
*   `samples/output/`: Generated Unity-facing outputs / samples.
*   `outputs/`: Generated intermediate outputs.

## 3. Pipeline Flow & Responsibilities

### Execution Flow
1. Windows Server sends `POST /infer` with an image.
2. `scripts/infer.py` creates a sandbox environment in `_jobs/` and orchestrates subprocesses.
3. `scripts/run_inference_sobal.py` generates 16-bit depth maps and edge masks.
4. `scripts/cut_img.py` cuts the image into layers and performs occlusion inpainting.
5. `scripts/extract_track.py` extracts a JSON array of track points.
6. `scripts/infer.py` packages everything into Base64 JSON and responds.

### Known Fragile Points
*   **Subprocess Orchestration**: `scripts/infer.py` uses `subprocess` and dynamic string patching to override hardcoded paths in the legacy scripts. This is sensitive to script changes.
*   **DA3 Vendor Dependency**: `models/depth_anything_3/` must remain untouched. Any API changes there require updates to `scripts/depth_adapter.py`.
*   **Generated Folders**: Manual runs rely on `outputs/` and `Send2Unity/` folders being created correctly.

## 4. API Contract

The Windows server communicates with this module via a `POST` request to `http://<WSL_IP>:9000/infer`, attaching the image as `multipart/form-data` under the key `photo`.

### Current Response Format
```json
{
  "foreground_base64": "iVBORw0KGgo...",
  "gameplay_base64": "...",
  "background_base64": "...",
  "metadata": {
    "version": "1.0",
    "timestamp": "2026-04-23T12:00:00.000Z",
    "source_image": "input.jpg",
    "aspect_ratio": 1.7778,
    "track_color": "#9e5752",
    "points": [ [0.1, 0.5], [0.2, 0.45] ]
  }
}
```

### Intended Future `LevelPayload` Format
*Note: The current response does not yet match this intended format for Unity.*
```json
{
  "status": "ok",
  "images": {
    "foreground_base64": "...",
    "gameplay_base64": "...",
    "background_base64": "..."
  },
  "track": {
    "points": [ {"x": 0.1, "y": 0.8} ],
    "terrain_type": "road",
    "friction": 0.8
  },
  "metadata": {
    "source_width": 1024,
    "source_height": 768
  }
}
```
**Mismatches to resolve in the future:**
1. Grouping images under `images` and adding a `status`.
2. Changing `points` from `[x, y]` arrays to `{"x": x, "y": y}` objects under a `track` block.
3. Adding `terrain_type` and `friction`.
4. Using `source_width`/`source_height` instead of `aspect_ratio`/`timestamp`.

## 5. TODOs
*   **Model Checkpoints:** Clarify if `DepthAnything3.from_pretrained` downloads the model automatically to a cache directory on first run, or if manual placement of weights is required in a disconnected environment. Currently, it assumes the model is downloadable or cached via Hugging Face Hub.
