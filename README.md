# SkateboardP
SkateboardP Project

---

# SkateP WSL CV Backend
This repository is the WSL CV backend module for the Unity photo-to-level pipeline.

## Overview

This module takes a 2D image, extracts a depth map, segments it into visual layers (with occlusion inpainting), and generates a smooth 2D track path for the Unity skateboard game.

## Active Runtime Files

The active pipeline is orchestrated by `wsl-cv/scripts/infer.py` and consists of:
*   `wsl-cv/scripts/infer.py`: FastAPI server entrypoint.
*   `wsl-cv/scripts/run_inference_sobal.py`: Depth and mask generation.
*   `wsl-cv/scripts/cut_img.py`: Layer generation.
*   `wsl-cv/scripts/extract_track.py`: Track point extraction.

*Note: `wsl-cv/models/depth_anything_3/` is vendor/model code and must not be modified.*

## Documentation

*   [WSL CV Documentation](wsl-cv/README.md): Details on the API contract, pipeline flow, environment setup, and repository structure.
