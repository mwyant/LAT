from pathlib import Path
import uuid
import shutil
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the Parakeet model once at container start.
model_name = "nvidia/parakeet-tdt-0.6b-v3"
print("Loading Parakeet …")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)

# Optional: try to set a local attention model for long audio inputs.
# This API may differ across NeMo releases; the try/except avoids crashes if unsupported.
try:
    asr_model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[512, 512],
    )
except Exception:
    print("change_attention_model not available or failed — continuing with defaults")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
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

