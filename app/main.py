from pathlib import Path
import uuid
import shutil
import os
import logging
from typing import Optional

import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Globals for model and readiness
asr_model: Optional[object] = None
model_ready = False
model_name = "nvidia/parakeet-tdt-0.6b-v3"
logger = logging.getLogger("app.main")


@app.on_event("startup")
async def load_model():
    global asr_model, model_ready
    # ensure cache dirs exist (HF_HOME / TORCH_HOME set in Dockerfile)
    try:
        os.makedirs(os.environ.get("HF_HOME", "/app/.cache/huggingface"), exist_ok=True)
        os.makedirs(os.environ.get("TORCH_HOME", "/app/.cache/torch"), exist_ok=True)
    except Exception:
        logger.exception("Failed to create cache dirs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting model load for {model_name} on device={device}")
    try:
        # from_pretrained may accept map_location for some wrappers; if not, fallback to .to(device)
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location=device)
    except TypeError:
        # older/newer API may not accept map_location; try loading then move
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
            try:
                asr_model.to(device)
            except Exception:
                # some wrappers manage device internally; ignore if not supported
                logger.info("Could not call .to(device) on model; continuing")
        except Exception:
            logger.exception("Model load failed")
            asr_model = None
    except Exception:
        logger.exception("Model load failed")
        asr_model = None

    model_ready = asr_model is not None
    if model_ready:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded; service available but will return 503 for transcribe")


@app.get("/health")
async def health():
    return {"ready": model_ready}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not model_ready or asr_model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    upload_dir = Path("./data")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    def generator():
        out = asr_model.transcribe(
            [str(file_path)],
            timestamps=True,
            batch_size=1,
            return_timestamps="word",
            max_duration=90 * 60,
        )
        transcript = out[0]
        # Get text (fallback to string if attribute missing)
        try:
            text = transcript.text
        except Exception:
            text = str(transcript)
        yield f"Text: {text}\n\n"
        # Stream per-word timestamps if available. Adjust to your NeMo return shape.
        ts_obj = getattr(transcript, "timestamp", None)
        if isinstance(ts_obj, dict):
            word_list = ts_obj.get("word")
            if word_list:
                for ts in word_list:
                    start = ts.get("start") or ts.get("start_time")
                    end = ts.get("end") or ts.get("end_time")
                    txt = ts.get("text") or ts.get("word") or ""
                    if start is not None and end is not None:
                        yield f"{start:.2f}s – {end:.2f}s : {txt}\n"
    return StreamingResponse(generator(), media_type="text/plain")
