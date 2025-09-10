# app/diary.py
"""
Lightweight diarization fallback (VAD + MFCC + clustering).

API:
- diarize(audio_path) -> list of segments: {"start": float, "end": float, "speaker": "SPKRn"}
- apply_diarization_to_lines(lines, segments) -> labeled lines (HH:MM:SS - SPKRn: text)

Notes:
- Uses librosa for audio I/O and feature extraction, sklearn AgglomerativeClustering with distance threshold
  to automatically determine number of speakers.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import numpy as np
import librosa
import math
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("diary_fallback")
logger.setLevel(logging.INFO)

# Config knobs
SR = 16000  # target sample rate for processing
MFCC_N_MELS = 40
MFCC_N_MFCC = 13
VAD_TOP_DB = 30  # librosa.effects.split sensitivity (lower = more sensitive)
MIN_SEG_MS = 200  # drop tiny segments


def _load_audio(path: str, sr: int = SR) -> Tuple[np.ndarray, int]:
    y, fs = librosa.load(path, sr=sr, mono=True)
    return y, fs


def _speech_segments(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    # returns list of (start_sec, end_sec)
    intervals = librosa.effects.split(y, top_db=VAD_TOP_DB)
    segs = []
    for a, b in intervals:
        dur = (b - a) / sr
        if dur * 1000.0 < MIN_SEG_MS:
            continue
        segs.append((a / sr, b / sr))
    return segs


def _mfcc_for_segment(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = int(math.floor(start * sr))
    e = int(math.ceil(end * sr))
    seg = y[s:e]
    if seg.size == 0:
        return np.zeros((MFCC_N_MFCC,))
    mf = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=MFCC_N_MFCC, n_mels=MFCC_N_MELS)
    # aggregate: mean + std
    feat = np.concatenate([mf.mean(axis=1), mf.std(axis=1)])
    return feat


def diarize(audio_path: str, min_clusters: int = 1, distance_threshold: float = 1.2) -> List[Dict[str, Any]]:
    """
    Very lightweight diarization.
    - VAD to get speech segments
    - compute MFCC features per segment
    - cluster (agglomerative) with distance threshold -> variable number of speakers
    Returns list of segments with speaker labels.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(audio_path)
    try:
        y, sr = _load_audio(str(path), sr=SR)
    except Exception as e:
        logger.exception("Failed to load audio for diarization: %s", e)
        return [{"start": 0.0, "end": float(len(y) / SR) if y is not None else 0.0, "speaker": "SPKR1"}]

    seg_times = _speech_segments(y, sr)
    if not seg_times:
        # no speech found; return whole-file SPKR1
        duration = len(y) / sr if y is not None else 0.0
        return [{"start": 0.0, "end": duration, "speaker": "SPKR1"}]

    feats = []
    for st, en in seg_times:
        feats.append(_mfcc_for_segment(y, sr, st, en))
    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)

    # Agglomerative clustering with a distance threshold gives variable # clusters
    # distance_threshold controls sensitivity; tweak if you get too many/too few speakers.
    try:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage="average")
        labels = clustering.fit_predict(X)
    except Exception as e:
        logger.exception("Clustering failed; falling back to 1 speaker: %s", e)
        labels = np.zeros(X.shape[0], dtype=int)

    # Build segments labeled by normalized SPKRn
    unique = {}
    speaker_idx = 1
    segments = []
    for (st, en), lbl in zip(seg_times, labels):
        key = str(lbl)
        if key not in unique:
            unique[key] = f"SPKR{speaker_idx}"
            speaker_idx += 1
        segments.append({"start": float(st), "end": float(en), "speaker": unique[key]})
    return segments


def _parse_time_prefix(line: str) -> float:
    import re
    m = re.match(r"\s*(\d{1,2}):(\d{2}):(\d{2}\.\d+|\d{2})", line)
    if m:
        h = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
        return h * 3600 + mm * 60 + ss
    m2 = re.match(r"\s*(\d{1,2}):(\d{2}\.\d+|\d{2})", line)
    if m2:
        mm = int(m2.group(1)); ss = float(m2.group(2))
        return mm * 60 + ss
    return 0.0


def apply_diarization_to_lines(lines: List[str], segments: List[Dict[str, Any]]) -> List[str]:
    """
    Assign a speaker label to each timestamped line.
    lines expected to start with a time prefix (HH:MM:SS or MM:SS).
    """
    labeled = []
    for ln in lines:
        t = _parse_time_prefix(ln)
        # find segment containing t
        sp = None
        if segments:
            for seg in segments:
                if t >= seg["start"] and t <= seg["end"]:
                    sp = seg["speaker"]
                    break
            if sp is None:
                # nearest by mid-point
                mid = lambda s: (s["start"] + s["end"]) / 2.0
                seg = min(segments, key=lambda s: abs(mid(s) - t))
                sp = seg["speaker"]
        if sp is None:
            sp = "SPKR1"
        # Replace existing speaker label if present, else insert
        import re
        new_line = re.sub(r"^\s*(\d{1,2}:\d{2}:\d{2}\.\d+|\d{1,2}:\d{2})\s*-\s*[^:]+:", lambda m: f"{m.group(0).split('-')[0].strip()} - {sp}:", ln)
        if new_line == ln:
            parts = ln.split(" - ", 1)
            if len(parts) == 2:
                rest = parts[1]
                if ":" in rest:
                    after = rest.split(":", 1)[1].lstrip()
                else:
                    after = rest
                new_line = f"{parts[0]} - {sp}: {after}"
            else:
                new_line = f"{ln} ({sp})"
        labeled.append(new_line)
    return labeled
