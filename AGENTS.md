# AI Agent Guidelines for wsl-cv

* This repository is the WSL CV backend module for a Unity photo-to-level game.
* **Future monorepo layout:**
  * `unity-client/`
  * `server-win/`
  * `wsl-cv/`
  * `shared/`
  * `docs/`
* Do NOT put backend Python code inside Unity `Assets/Scripts/`.
* `src/depth_anything_3/` is vendor/model code. Do NOT modify it.
* Active runtime files are:
  * `infer.py`
  * `run_inference_sobal.py`
  * `cut_img.py`
  * `extract_track.py`
* `depth_adapter.py` is the only intended new runtime abstraction for now.
* Do NOT run heavy GPU inference unless explicitly requested.
* Do NOT download models.
* Preserve `POST /infer` compatibility.
