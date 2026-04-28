# SkateP WSL CV Backend

This repository is the WSL CV backend module for the Unity photo-to-level pipeline.

## Overview

This module takes a 2D image, extracts a depth map, segments it into visual layers (with occlusion inpainting), and generates a smooth 2D track path for the Unity skateboard game.

## Active Runtime Files

The active pipeline is orchestrated by `infer.py` and consists of:
*   `infer.py`: FastAPI server entrypoint.
*   `run_inference_sobal.py`: Depth and mask generation.
*   `cut_img.py`: Layer generation.
*   `extract_track.py`: Track point extraction.

*Note: `src/depth_anything_3/` is vendor/model code and must not be modified.*

## Documentation

*   [API Contract](docs/api_contract.md): Protocol reference and payload formats.
*   [Monorepo Integration](docs/monorepo_integration.md): Details on how this module fits into the broader project structure.
*   [Setup & Execution](docs/setup_wsl.md): Instructions for setting up the environment and running the API.
*   [Pipeline Flow](docs/pipeline.md): Details on the data flow and file responsibilities.
