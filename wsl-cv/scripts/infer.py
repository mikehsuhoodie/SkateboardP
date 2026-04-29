"""
infer.py - FastAPI backend for the SkateP depth-to-track pipeline.

Exposes:
    GET  /health   - liveness check
    POST /infer    - accept an image, run the 3-step pipeline, return results

Run with:
    uvicorn infer:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ======================================================================
# PIPELINE CONFIGURATION
# Adjust paths / filenames here to match your environment.
# ======================================================================


class PipelineConfig:
    """Central place for every tuneable path and filename."""

    # Root of the workspace (where models/, assets/ live)
    WORKSPACE_ROOT: Path = Path(__file__).resolve().parent.parent

    # Where the scripts live
    SCRIPTS_ROOT: Path = Path(__file__).resolve().parent
    # Python interpreter used to call sub-scripts (same venv).
    PYTHON: str = sys.executable

    # -- Pipeline script paths (relative to SCRIPTS_ROOT) --
    STEP1_SCRIPT: str = "run_inference_sobal.py"
    STEP2_SCRIPT: str = "cut_img.py"
    STEP3_SCRIPT: str = "extract_track.py"

    # -- Temporary job directory root --
    JOBS_ROOT: Path = WORKSPACE_ROOT / "_jobs"

    # -- Timeout per pipeline step (seconds) --
    STEP1_TIMEOUT: int = 300  # depth inference can be slow on CPU
    STEP2_TIMEOUT: int = 120
    STEP3_TIMEOUT: int = 60

    # -- Input filename written inside each job directory --
    # The legacy scripts derive the "base_name" from the file stem,
    # so we use a fixed name that is safe and predictable.
    INPUT_FILENAME: str = "input.jpg"

    # -- Step 1 outputs (written to <job>/outputs/inference_results/) --
    # Pattern: {base_name}_depth_16bit.png  /  {base_name}_depth_mask.npy
    # base_name is derived from INPUT_FILENAME stem -> "input"
    DEPTH_16BIT_FILENAME: str = "input_depth_16bit.png"
    DEPTH_MASK_FILENAME: str = "input_depth_mask.npy"

    # -- Step 2 outputs (written to <job>/Send2Unity/) --
    # TODO: ADAPT THESE OUTPUT FILENAMES TO YOUR CURRENT PIPELINE
    # cut_img.py produces merged_layer_01.png ... merged_layer_03.png
    # Mapping:  layer 01 = nearest (foreground)
    #           layer 02 = middle  (gameplay)
    #           layer 03 = farthest (background, inpainted)
    FOREGROUND_FILENAME: str = "merged_layer_01.png"
    GAMEPLAY_FILENAME: str = "merged_layer_02.png"
    BACKGROUND_FILENAME: str = "merged_layer_03.png"
    LAYER_MASK_FILENAME: str = "layer_00_mask.npy"

    # -- Step 3 outputs (written to <job>/Send2Unity/) --
    # TODO: ADAPT THESE OUTPUT FILENAMES TO YOUR CURRENT PIPELINE
    TRACK_JSON_FILENAME: str = "track_points.json"


CFG = PipelineConfig()

# ======================================================================
# LOGGING
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("infer")

# ======================================================================
# FASTAPI APP
# ======================================================================

app = FastAPI(
    title="SkateP Depth Pipeline API",
    version="1.0.0",
    description="Accepts an image and returns depth-layered PNGs + track metadata.",
)

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================


def _create_job_dir() -> Path:
    """Create a unique per-request working directory.

    Format: _jobs/<timestamp>_<short-uuid>/
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = uuid.uuid4().hex[:8]
    job_dir = CFG.JOBS_ROOT / f"{ts}_{tag}"
    job_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created job directory: %s", job_dir)
    return job_dir


async def save_upload_file(upload: UploadFile, dest: Path) -> None:
    """Stream an uploaded file to *dest* on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    contents = await upload.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(contents)
    logger.info("Saved uploaded file -> %s (%d bytes)", dest, len(contents))


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
    step_label: str,
    env: dict[str, str] | None = None,
) -> None:
    """Execute a subprocess; raise HTTP 500 on failure or timeout.

    Parameters
    ----------
    cmd:
        Command tokens, e.g. ["python", "run_inference_sobal.py"].
    cwd:
        Working directory for the child process.
    timeout:
        Max seconds before the process is killed.
    step_label:
        Human-readable label used in log / error messages.
    env:
        Optional extra environment variables merged with os.environ.
    """
    full_env = {**os.environ, **(env or {})}
    logger.info(
        "[%s] Running: %s  (cwd=%s, timeout=%ds)",
        step_label,
        " ".join(cmd),
        cwd,
        timeout,
    )

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("[%s] TIMEOUT after %ds", step_label, timeout)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline step '{step_label}' timed out after {timeout}s.",
        ) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected error", step_label)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline step '{step_label}' could not be started: {exc}",
        ) from exc

    # Log stdout / stderr regardless of outcome
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("[%s:stdout] %s", step_label, line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.warning("[%s:stderr] %s", step_label, line)

    if result.returncode != 0:
        logger.error("[%s] Exited with code %d", step_label, result.returncode)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Pipeline step '{step_label}' failed (exit code {result.returncode}).  "
                f"stderr: {result.stderr[-2000:] if result.stderr else '(empty)'}"
            ),
        )

    logger.info("[%s] Completed successfully.", step_label)


def encode_image_base64(path: Path) -> str:
    """Read a binary image file and return its base64-encoded string."""
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Expected output file not found: {path.name}",
        )
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file; raise HTTP 500 on failure."""
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Expected output JSON not found: {path.name}",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse JSON from {path.name}: {exc}",
        ) from exc


def find_required_outputs(job_dir: Path) -> dict[str, Path]:
    """Locate the four required output files and return their paths.

    Raises HTTP 500 if any file is missing.
    """
    send2unity = job_dir / "Send2Unity"
    paths = {
        "foreground": send2unity / CFG.FOREGROUND_FILENAME,
        "gameplay": send2unity / CFG.GAMEPLAY_FILENAME,
        "background": send2unity / CFG.BACKGROUND_FILENAME,
        "metadata": send2unity / CFG.TRACK_JSON_FILENAME,
    }
    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Missing pipeline outputs: {', '.join(missing)}. "
                f"Expected in {send2unity}"
            ),
        )
    return paths


# ======================================================================
# JOB-LEVEL SCRIPT PATCHING
# ======================================================================
# The legacy scripts use hard-coded relative paths such as:
#   ./assets/examples/SOH/street2d.jpg
#   ./outputs/inference_results/
#   ./Send2Unity/
#
# To isolate each request we:
#   1. Create symlinks inside the job dir so the scripts resolve their
#      paths relative to the job dir as if it were the project root.
#   2. Copy/symlink only what each step needs.
# ======================================================================


def _prepare_job_environment(job_dir: Path, input_path: Path) -> None:
    """Set up the job directory so legacy scripts can run unmodified.

    Creates the folder structure the scripts expect:
        <job>/assets/examples/SOH/input.jpg
        <job>/outputs/inference_results/      (empty, will be written)
        <job>/Send2Unity/                     (empty, will be written)
        <job>/models  -> WORKSPACE_ROOT/models        (symlink to model code)
        <job>/depth_anything_3 -> ...         (symlink if exists)
        <job>/<script>.py -> ...              (symlink for each step)

    This way every script's hard-coded relative paths just work.
    """
    base_name = input_path.stem  # "input"

    # -- Directory structure --
    soh_dir = job_dir / "assets" / "examples" / "SOH"
    soh_dir.mkdir(parents=True, exist_ok=True)

    inf_dir = job_dir / "outputs" / "inference_results"
    inf_dir.mkdir(parents=True, exist_ok=True)

    s2u_dir = job_dir / "Send2Unity"
    s2u_dir.mkdir(parents=True, exist_ok=True)

    # -- Symlink the uploaded image if not already at the expected location --
    soh_link = soh_dir / input_path.name
    if not soh_link.exists():
        soh_link.symlink_to(input_path.resolve())

    # -- Symlink project source packages so imports work --
    for pkg_name in ("models", "depth_anything_3"):
        pkg = CFG.WORKSPACE_ROOT / pkg_name
        link = job_dir / pkg_name
        if pkg.exists() and not link.exists():
            link.symlink_to(pkg.resolve())

    # -- Symlink each pipeline script --
    for script in (CFG.STEP1_SCRIPT, CFG.STEP2_SCRIPT, CFG.STEP3_SCRIPT, "depth_adapter.py"):
        src = CFG.SCRIPTS_ROOT / script
        dst = job_dir / script
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    # -- Create thin wrapper scripts that override the hard-coded base_name --
    _write_step1_wrapper(job_dir, base_name)
    _write_step2_wrapper(job_dir, base_name)
    _write_step3_wrapper(job_dir, base_name)


def _write_step1_wrapper(job_dir: Path, base_name: str) -> None:
    """Wrapper that reads run_inference_sobal.py, patches the hard-coded
    IMAGE_PATH, and exec's it so depth inference targets the uploaded image."""
    script_name = CFG.STEP1_SCRIPT  # "run_inference_sobal.py"
    wrapper = job_dir / "_run_step1.py"
    content = (
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "\n"
        f'src_path = os.path.join(os.path.dirname(__file__), "{script_name}")\n'
        "with open(src_path) as f:\n"
        "    source = f.read()\n"
        "\n"
        "# Patch the hard-coded IMAGE_PATH to point at our uploaded image\n"
        "source = source.replace(\n"
        '    \'IMAGE_PATH = "./assets/examples/SOH/street2d.jpg"\',\n'
        f'    \'IMAGE_PATH = "./assets/examples/SOH/{base_name}.jpg"\',\n'
        ")\n"
        "\n"
        'ns = {"__name__": "__main__", "__file__": src_path}\n'
        "exec(compile(source, src_path, 'exec'), ns)\n"
    )
    wrapper.write_text(content, encoding="utf-8")


def _write_step2_wrapper(job_dir: Path, base_name: str) -> None:
    """Wrapper that execs cut_img.py with the correct base_name."""
    wrapper = job_dir / "_run_step2.py"
    content = (
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "\n"
        'src_path = os.path.join(os.path.dirname(__file__), "cut_img.py")\n'
        "with open(src_path) as f:\n"
        "    source = f.read()\n"
        "\n"
        "# Override the hard-coded base_name\n"
        f"source = source.replace(\"base_name = 'street2d'\", \"base_name = '{base_name}'\")\n"
        "\n"
        'ns = {"__name__": "__main__", "__file__": src_path}\n'
        "exec(compile(source, src_path, 'exec'), ns)\n"
    )
    wrapper.write_text(content, encoding="utf-8")


def _write_step3_wrapper(job_dir: Path, base_name: str) -> None:
    """Wrapper that calls extract_track with the correct paths."""
    wrapper = job_dir / "_run_step3.py"
    content = (
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "\n"
        "from extract_track import extract_smooth_track\n"
        "\n"
        "npy_file = './Send2Unity/layer_00_mask.npy'\n"
        "out_json = './Send2Unity/track_points.json'\n"
        f"original_img = './assets/examples/SOH/{base_name}.jpg'\n"
        f"img_name = '{base_name}.jpg'\n"
        "\n"
        "extract_smooth_track(npy_file, out_json, rgb_image_path=original_img, source_img_name=img_name)\n"
    )
    wrapper.write_text(content, encoding="utf-8")


# ======================================================================
# API ENDPOINTS
# ======================================================================


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/infer")
async def infer(photo: UploadFile = File(...)):
    """Run the full depth -> layers -> track pipeline on an uploaded image.

    Expects multipart/form-data with field name **photo**.

    Returns
    -------
    JSON with keys:
        foreground_base64  - base64 PNG of the nearest layer
        gameplay_base64    - base64 PNG of the middle layer
        background_base64  - base64 PNG of the farthest (inpainted) layer
        metadata           - parsed track_points.json dict
    """
    job_dir: Path | None = None

    try:
        # 1. Validate upload
        if not photo.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        # 2. Create isolated job directory
        job_dir = _create_job_dir()

        # 3. Save uploaded image
        input_path = job_dir / "assets" / "examples" / "SOH" / CFG.INPUT_FILENAME
        await save_upload_file(photo, input_path)

        # 4. Prepare the job environment (symlinks + wrapper scripts)
        _prepare_job_environment(job_dir, input_path)

        # 5. Run pipeline -- Step 1: Depth inference + Sobel segmentation
        run_command(
            [CFG.PYTHON, "_run_step1.py"],
            cwd=job_dir,
            timeout=CFG.STEP1_TIMEOUT,
            step_label="Step1:DepthInference",
        )

        # 6. Run pipeline -- Step 2: Layer cutting
        run_command(
            [CFG.PYTHON, "_run_step2.py"],
            cwd=job_dir,
            timeout=CFG.STEP2_TIMEOUT,
            step_label="Step2:CutLayers",
        )

        # 7. Run pipeline -- Step 3: Track extraction
        run_command(
            [CFG.PYTHON, "_run_step3.py"],
            cwd=job_dir,
            timeout=CFG.STEP3_TIMEOUT,
            step_label="Step3:ExtractTrack",
        )

        # 8. Locate required outputs
        outputs = find_required_outputs(job_dir)

        # 9. Build response
        response_data = {
            "foreground_base64": encode_image_base64(outputs["foreground"]),
            "gameplay_base64": encode_image_base64(outputs["gameplay"]),
            "background_base64": encode_image_base64(outputs["background"]),
            "metadata": load_json(outputs["metadata"]),
        }

        logger.info("Pipeline completed successfully for job %s", job_dir.name)
        return JSONResponse(content=response_data)

    except HTTPException:
        raise  # already formatted
    except Exception as exc:
        logger.exception("Unhandled error during inference")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {exc}"
        ) from exc
    finally:
        # 10. Clean up job directory
        if job_dir and job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info("Cleaned up job directory: %s", job_dir)
            except Exception:
                logger.warning(
                    "Failed to clean up %s", job_dir, exc_info=True
                )


# ======================================================================
# ENTRYPOINT
# ======================================================================

if __name__ == "__main__":
    # Example:  python infer.py
    # Or:       uvicorn infer:app --host 0.0.0.0 --port 9000
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
