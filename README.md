# SkateboardP Project

Welcome to the SkateboardP monorepo! This project implements a photo-to-level pipeline for a Unity skateboard game. It takes a real-world 2D image, extracts a depth map, segments it into visual layers (with occlusion inpainting), and generates a smooth 2D track path that is fully playable in Unity.

## Project Architecture

This repository is structured as a monorepo containing several interconnected modules:

### 1. WSL CV Backend (`wsl-cv/`)
The computer vision core running on Linux/WSL. It hosts a FastAPI server that orchestrates the `Depth Anything 3` model and custom image processing scripts to generate tracks and segmented layers from raw images.
*   **Key Tech:** Python, FastAPI, PyTorch, DA3
*   **Documentation:** [wsl-cv/README.md](wsl-cv/README.md)

### 2. Windows Server (`win-server/`)
The gateway server running on Windows. It acts as the bridge between the Unity client and the WSL CV backend, handling payload routing and network communication.

### 3. Unity Client (`unity_client/`)
The frontend game engine client. It captures images (e.g., from a mobile device), sends them to the backend, and procedurally generates the playable skateboard level based on the returned JSON track points and layered images.
*   **Key Tech:** Unity (C#)

### 4. Legacy Experiments (`_legacy_2d/`)
Contains older 2D track generation scripts and legacy experimental code.

---

## Quick Start
For details on setting up and running the core CV pipeline, please refer to the [WSL CV Documentation](wsl-cv/README.md).
