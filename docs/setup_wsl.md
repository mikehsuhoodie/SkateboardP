# WSL Setup

## Environment Setup

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install fastapi uvicorn python-multipart
    ```

## Running the API

Start the FastAPI backend server using `infer.py`:

```bash
uvicorn infer:app --host 0.0.0.0 --port 9000
```

## Integration with Windows Server

The Windows server should call the WSL API by sending a `POST` request to `http://<WSL_IP>:9000/infer` with the image file attached as `multipart/form-data` under the key `photo`.

## Generated Outputs

During execution, `infer.py` creates temporary job directories under `_jobs/`. Once the pipeline finishes, the final layered images and metadata are extracted, encoded into the JSON payload, and the temporary `_jobs/` folder is deleted.

If running the scripts manually (e.g., `python run_inference_sobal.py`), outputs are generated in `outputs/` and `Send2Unity/`.

## TODOs

*   **Model Checkpoints:** Clarify if `DepthAnything3.from_pretrained` downloads the model automatically to a cache directory on first run, or if manual placement of weights is required in a disconnected environment. Currently, it assumes the model is downloadable or cached via Hugging Face Hub.
