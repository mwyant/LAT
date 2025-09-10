# app/main.py
"""
Transcription assistant server (single-file).
Features:
- Single-page UI (WaveSurfer) to upload/record, show waveform, select regions.
- Server-side upload streaming with size limit (200 MiB).
- FFmpeg re-encode to 16kHz mono WAV for NeMo (parakeet) model.
- Background transcription jobs with SSE progress updates.
- /transcribe (POST) to create job, returns job_id.
- /status/{job_id} and /download/{job_id} endpoints.
- /admin/reload and /admin/switch_model endpoints (open in localhost).
- /metrics endpoint with basic counters.
- Saves transcriptions to /app/data/transcriptions/<TIMESTAMP>-<RND>.txt
Notes:
- Ensure ffmpeg is installed in the image and available on PATH.
- Ensure your Dockerfile created /app/data and that the container has write access.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
import logging
import random
import re
import string
import subprocess
import threading
import time
import app.diary as diary
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Third-party libs (ensure installed in venv)
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    import nemo.collections.asr as nemo_asr  # type: ignore
except Exception:
    nemo_asr = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcription_app")

# Constants / paths
APP_ROOT = Path("/app")
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TRANSCRIPT_DIR = DATA_DIR / "transcriptions"
CACHE_HF = APP_ROOT / ".cache" / "huggingface"
CACHE_TORCH = APP_ROOT / ".cache" / "torch"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MiB
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# Ensure dirs
for p in (DATA_DIR, UPLOAD_DIR, TRANSCRIPT_DIR, CACHE_HF, CACHE_TORCH):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Transcription Assistant (Parakeet)")

# CORS - useful for dev; restrict in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor for background transcriptions
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Global event loop handle (set at startup)
GLOBAL_LOOP: Optional[asyncio.AbstractEventLoop] = None

# Job storage in memory
@dataclass
class Job:
    id: str
    status: str  # queued, running, done, failed
    created_at: float
    updated_at: float
    progress: float  # 0-100
    message: str
    input_path: Optional[str] = None
    transcode_path: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    model_name: Optional[str] = None
    # internal asyncio queue for SSE (do not deep-copy)
    _queue: Optional[asyncio.Queue] = None

    def to_dict(self):
        """
        Return a plain serializable dict representation.
        Avoid dataclasses.asdict which tries to deepcopy asyncio primitives.
        """
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "message": self.message,
            "input_path": self.input_path,
            "transcode_path": self.transcode_path,
            "result_path": self.result_path,
            "error": self.error,
            "model_name": self.model_name,
        }


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

# Basic metrics
METRICS = {
    "jobs_created": 0,
    "jobs_completed": 0,
    "jobs_failed": 0,
    "bytes_uploaded": 0,
}

# Default model
DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
MODEL = {
    "instance": None,
    "name": DEFAULT_MODEL_NAME,
    "ready": False,
    "lock": threading.Lock(),
}


# ---- Helpers (queues, ffmpeg, transcription) ----

async def _ensure_job_queue(job: Job) -> None:
    if job._queue is None:
        job._queue = asyncio.Queue()


def now_ts() -> float:
    return time.time()


def gen_job_id() -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{ts}-{rnd}"


def gen_result_filename() -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"{ts}-{rnd}.txt"


def safe_ext(name: str) -> str:
    return Path(name).suffix.lower()


# SSE helper
async def sse_event_generator(job: Job):
    # Ensure job queue exists
    if job._queue is None:
        job._queue = asyncio.Queue()
    q = job._queue
    # start by sending initial state
    await q.put({"event": "status", "data": job.to_dict()})
    while True:
        msg = await q.get()
        if msg is None:
            break
        # format event
        ev = msg.get("event", "message")
        data = json.dumps(msg.get("data", {}), default=str)
        yield f"event: {ev}\ndata: {data}\n\n"
        if ev == "done" or ev == "error":
            break


def push_job_event(job: Job, event: str, data: Any):
    """
    Thread-safe push of events into the job's asyncio.Queue.
    This function is safe to call from background threads (worker threads).
    It schedules the queue put on the main event loop using call_soon_threadsafe.
    """
    job.updated_at = now_ts()

    # Ensure the queue exists on the main loop
    if job._queue is None:
        if GLOBAL_LOOP:
            try:
                fut = asyncio.run_coroutine_threadsafe(_ensure_job_queue(job), GLOBAL_LOOP)
                fut.result(timeout=5)
            except Exception:
                logger.exception("Failed to create job queue on main loop")
                return
        else:
            logger.debug("GLOBAL_LOOP not set; cannot create job queue")
            return

    # Now schedule a thread-safe put into the asyncio.Queue
    if GLOBAL_LOOP and job._queue:
        try:
            GLOBAL_LOOP.call_soon_threadsafe(job._queue.put_nowait, {"event": event, "data": data})
        except Exception:
            logger.exception("Failed to push event to job queue via call_soon_threadsafe")
    else:
        logger.debug("Missing GLOBAL_LOOP or job._queue; cannot push event")


# FFmpeg helpers: get duration via ffprobe
def get_duration_seconds(path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return float(out.strip())
    except Exception:
        return None


def transcode_to_wav(input_path: Path, output_path: Path, progress_cb: Optional[Callable[[float, str], None]] = None) -> None:
    """
    Transcode input to 16k mono WAV using ffmpeg. Calls progress_cb(percent, msg) periodically if provided.
    """
    duration = get_duration_seconds(input_path) or 0.0
    cmd = [
        "ffmpeg",
        "-y",
        "-nostats",
        "-hide_banner",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    last_pct = 0.0
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.strip()
            m = re.search(r"time=(\d+:\d+:\d+\.\d+|\d+:\d+\.\d+|\d+\.\d+)", line)
            if m:
                timestr = m.group(1)
                parts = timestr.split(":")
                try:
                    if len(parts) == 3:
                        s = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    elif len(parts) == 2:
                        s = int(parts[0]) * 60 + float(parts[1])
                    else:
                        s = float(parts[0])
                    pct = (s / duration * 100.0) if duration > 0 else 0.0
                    pct = max(0.0, min(100.0, pct))
                    if progress_cb and (pct - last_pct >= 1.0):
                        last_pct = pct
                        try:
                            progress_cb(pct, f"transcoding {pct:.1f}%")
                        except Exception:
                            pass
                except Exception:
                    pass
        proc.wait(timeout=300)
    finally:
        if progress_cb:
            try:
                progress_cb(100.0, "transcode complete")
            except Exception:
                pass


# Helper to run transcription robustly (signature differences)
def run_transcribe_model(asr_model, audio_path: str) -> List[Any]:
    """
    Try multiple possible signatures for asr_model.transcribe.
    Return the 'out' as a list-like of transcript objects.
    """
    try:
        out = asr_model.transcribe([audio_path], timestamps=True, batch_size=1, return_timestamps="word", max_duration=90 * 60)
        return out
    except TypeError:
        try:
            out = asr_model.transcribe([audio_path], batch_size=1)
            return out
        except Exception:
            out = asr_model.transcribe(audio_path)
            return out


def transcript_to_human_readable(transcript_obj: Any) -> (str, List[str]):
    """
    Convert a transcript object from NeMo to a text block and list of timestamped lines.
    Returns (plain_text, list_of_timestamped_lines)
    Timestamp format: HH:MM:SS - SPKR1: text
    """
    try:
        text = getattr(transcript_obj, "text", None) or getattr(transcript_obj, "transcript", None) or str(transcript_obj)
    except Exception:
        text = str(transcript_obj)

    lines = []
    ts_obj = getattr(transcript_obj, "timestamp", None) or getattr(transcript_obj, "timestamps", None) or getattr(transcript_obj, "word_timestamps", None) or getattr(transcript_obj, "segments", None)

    def format_time(t: float) -> str:
        dt = datetime.timedelta(seconds=float(t))
        total_seconds = int(dt.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = dt.total_seconds() - hours * 3600 - minutes * 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    if isinstance(ts_obj, dict):
        word_list = ts_obj.get("word") or ts_obj.get("words") or ts_obj.get("tokens")
        if word_list:
            speaker = "SPKR1"
            segment_words = []
            seg_start = None
            for w in word_list:
                start = w.get("start") or w.get("start_time")
                end = w.get("end") or w.get("end_time")
                txt = w.get("text") or w.get("word") or w.get("token") or ""
                if seg_start is None and start is not None:
                    seg_start = float(start)
                if txt:
                    segment_words.append(txt)
                if len(segment_words) >= 6 or (end is not None and (seg_start is not None) and (float(end) - seg_start >= 3.0)):
                    t0 = format_time(seg_start or 0.0)
                    lines.append(f"{t0} - {speaker}: {' '.join(segment_words)}")
                    segment_words = []
                    seg_start = None
            if segment_words:
                t0 = format_time(seg_start or 0.0)
                lines.append(f"{t0} - {speaker}: {' '.join(segment_words)}")
        else:
            lines.append(f"00:00:00 - SPKR1: {text}")
    elif isinstance(ts_obj, list):
        speaker = "SPKR1"
        for item in ts_obj:
            if isinstance(item, dict):
                start = item.get("start") or item.get("start_time")
                txt = item.get("text") or item.get("word") or str(item)
                t0 = format_time(float(start or 0.0))
                lines.append(f"{t0} - {speaker}: {txt}")
            elif isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    txt, start, end = item[0], item[1], item[2]
                    t0 = format_time(float(start or 0.0))
                    lines.append(f"{t0} - {speaker}: {txt}")
                else:
                    lines.append(f"00:00:00 - SPKR1: {text}")
    else:
        lines.append(f"00:00:00 - SPKR1: {text}")

    return text, lines


# Background job runner
def transcription_worker(job_id: str, remove_input_after: bool = True):
    """
    Runs in background thread. Updates job J and pushes events (progress/status) to SSE queue.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        logger.error("Job not found in worker: %s", job_id)
        return

    logger.info("Starting transcription job %s", job_id)
    job.status = "running"
    push_job_event(job, "status", job.to_dict())

    try:
        input_path = Path(job.input_path)
        transcode_out = UPLOAD_DIR / f"{job.id}-16k.wav"
        job.transcode_path = str(transcode_out)

        def transcode_progress(pct, msg):
            job.progress = 10.0 + pct * 0.4  # map to 10-50%
            job.message = msg
            push_job_event(job, "progress", {"progress": job.progress, "message": job.message})

        push_job_event(job, "progress", {"progress": 5.0, "message": "starting transcode"})
        transcode_to_wav(input_path, transcode_out, progress_cb=transcode_progress)

        job.progress = 50.0
        job.message = "transcoding complete; starting model transcription"
        push_job_event(job, "progress", {"progress": job.progress, "message": job.message})

        if not MODEL["ready"] or MODEL["instance"] is None:
            job.status = "failed"
            job.error = "model not ready"
            push_job_event(job, "error", {"error": job.error})
            METRICS["jobs_failed"] += 1
            return

        start_t = time.time()
        push_job_event(job, "progress", {"progress": 55.0, "message": "running transcription"})
        out = run_transcribe_model(MODEL["instance"], str(transcode_out))
        elapsed = time.time() - start_t
        job.progress = 90.0
        job.message = f"transcription finished in {elapsed:.1f}s; formatting output"
        push_job_event(job, "progress", {"progress": job.progress, "message": job.message})

        transcript_obj = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        plain, lines = transcript_to_human_readable(transcript_obj)

#        header = f"Transcription generated at {datetime.datetime.utcnow().isoformat()} UTC\nModel: {job.model_name or MODEL['name']}\n\n"
#        result_text = header + "\n".join(lines) + "\n\nFULL TEXT:\n" + plain
# Attempt diarization if transcode file exists\n            try:\n                segments = diary.diarize(job.transcode_path)\n                labeled_lines = diary.apply_diarization_to_lines(lines, segments)\n            except Exception:\n                logger.exception("Diarization failed; falling back")\n                labeled_lines = lines\n            # then write labeled_lines into result_text instead of lines

        header = f"Transcription generated at {datetime.datetime.utcnow().isoformat()} UTC\nModel: {job.model_name or MODEL['name']}\n\n"
        result_text = header + "\n".join(labeled_lines) + "\n\nFULL TEXT:\n" + plain



        fn = gen_result_filename()
        dest = TRANSCRIPT_DIR / fn
        dest.write_text(result_text, encoding="utf-8")
        job.result_path = str(dest)
        job.progress = 100.0
        job.message = "done"
        job.status = "done"
        push_job_event(job, "done", {"result_path": job.result_path, "filename": fn})
        METRICS["jobs_completed"] += 1

        if remove_input_after:
            try:
                input_path.unlink(missing_ok=True)
                transcode_out.unlink(missing_ok=True)
            except Exception:
                pass

    except Exception as e:
        logger.exception("Transcription job failed: %s", job_id)
        job.status = "failed"
        job.error = str(e)
        push_job_event(job, "error", {"error": job.error})
        METRICS["jobs_failed"] += 1


# Model loading routines
def load_model(model_name: Optional[str] = None) -> None:
    """
    Load (or reload) the NeMo ASR model. Sets MODEL['instance'] and MODEL['ready'].
    """
    if model_name is None:
        model_name = MODEL["name"]
    with MODEL["lock"]:
        MODEL["ready"] = False
        MODEL["name"] = model_name
        MODEL["instance"] = None
    try:
        logger.info("Loading model %s ...", model_name)
        inst = nemo_asr.models.ASRModel.from_pretrained(model_name)
        with MODEL["lock"]:
            MODEL["instance"] = inst
            MODEL["ready"] = True
        logger.info("Model %s loaded", model_name)
    except Exception:
        logger.exception("Failed to load model %s", model_name)
        with MODEL["lock"]:
            MODEL["instance"] = None
            MODEL["ready"] = False


# Startup: attempt to load the default model in background so server can start quickly
@app.on_event("startup")
async def startup_event():
    global GLOBAL_LOOP
    if GLOBAL_LOOP is None:
        GLOBAL_LOOP = asyncio.get_event_loop()
    # Ensure ffmpeg exists
    if not shutil_which("ffmpeg"):
        logger.warning("ffmpeg not found in PATH. Transcoding will fail unless ffmpeg is installed in image.")
    # launch model load in thread so server responds quickly
    loop = asyncio.get_event_loop()
    loop.run_in_executor(EXECUTOR, load_model, MODEL["name"])


# small helper to check binary in PATH
def shutil_which(name: str) -> Optional[str]:
    from shutil import which
    return which(name)


# Upload streaming helper with size limit
async def save_upload_stream(upload_file: UploadFile, dest: Path, size_limit: int = MAX_UPLOAD_BYTES) -> int:
    """
    Stream upload to disk with size check. Returns total bytes written.
    """
    total = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload_file.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > size_limit:
                raise HTTPException(status_code=413, detail=f"Uploaded file exceeds size limit of {size_limit} bytes")
            f.write(chunk)
    return total


# Routes
#@app.get("/", response_class=HTMLResponse)
#async def index():
#    # Single-file HTML UI using WaveSurfer from CDN. It interacts with endpoints below.
#    html = """
#    """
#    return HTMLResponse(content=html, status_code=200)
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload endpoint: streams file to disk and creates a job (but does not start transcription yet)
@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    # simple validation
    ext = safe_ext(file.filename)
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    job_id = gen_job_id()
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    try:
        bytes_written = await save_upload_stream(file, dest, size_limit=MAX_UPLOAD_BYTES)
    except HTTPException as e:
        raise e
    METRICS["bytes_uploaded"] += bytes_written
    job = Job(
        id=job_id,
        status="queued",
        created_at=now_ts(),
        updated_at=now_ts(),
        progress=0.0,
        message="uploaded",
        input_path=str(dest),
        transcode_path=None,
        result_path=None,
        error=None,
        model_name=MODEL["name"],
        _queue=asyncio.Queue(),
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    METRICS["jobs_created"] += 1
    # push initial event
    push_job_event(job, "status", job.to_dict())
    return JSONResponse({"job_id": job_id, "bytes": bytes_written})


# Start transcription endpoint: kicks off background worker for a job_id
@app.post("/transcribe")
async def transcribe_start(payload: Dict[str, Any]):
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id required")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in ("running", "done"):
        return JSONResponse({"job_id": job.id, "status": job.status})
    # submit to threadpool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(EXECUTOR, transcription_worker, job.id)
    job.status = "queued"
    push_job_event(job, "status", job.to_dict())
    return JSONResponse({"job_id": job.id, "status": "queued"})


# SSE events endpoint
@app.get("/events/{job_id}")
async def events(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job._queue is None:
        job._queue = asyncio.Queue()
    return StreamingResponse(sse_event_generator(job), media_type="text/event-stream")


# status endpoint
@app.get("/status/{job_id}")
async def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(job.to_dict())


# download endpoint
@app.get("/download/{job_id}")
async def download(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.result_path:
        raise HTTPException(status_code=404, detail="result not available")
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="result file missing")
    return FileResponse(path, media_type="text/plain", filename=path.name)


# metrics
@app.get("/metrics")
async def metrics():
    return JSONResponse(METRICS)


# admin: reload model
@app.post("/admin/reload")
async def admin_reload():
    # fire async reload
    loop = asyncio.get_event_loop()
    loop.run_in_executor(EXECUTOR, load_model, MODEL["name"])
    return JSONResponse({"status": "reloading", "model": MODEL["name"]})


# admin: switch model
@app.post("/admin/switch_model")
async def admin_switch_model(payload: Dict[str, Any]):
    m = payload.get("model_name")
    if not m:
        raise HTTPException(status_code=400, detail="model_name required")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(EXECUTOR, load_model, m)
    return JSONResponse({"status": "switching", "model": m})


# small utility endpoint to list jobs (dev)
@app.get("/jobs")
async def list_jobs():
    with JOBS_LOCK:
        return JSONResponse({k: v.to_dict() for k, v in JOBS.items()})


# Graceful shutdown (optional)
@app.on_event("shutdown")
def shutdown_event():
    EXECUTOR.shutdown(wait=False)
