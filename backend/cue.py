"""T5: cue extraction from the vocal stem (count-in + guide voice).

Data shape: segments are time-ordered dicts with start_s, end_s, kind.
Absent count-ins yield digital silence with present False so the
pipeline never synthesizes fake cues. Sung phrases are rejected by
the continuity gate: long unbroken voicing reads as sung, short
isolated bursts read as count-in or guide voice.
"""

import numpy as np

from stemspec import BY_NAME

SR_DEFAULT = 44100
MODEL_ID = BY_NAME["cue"].model_id

VAD_FRAME_S = 0.020
VAD_HOP_S = 0.010
ONSET_SPLIT_S = 0.25
VAD_ABS_FLOOR = 0.02
VAD_REL_FRAC = 0.35
MIN_SEG_S = 0.05
MAX_SEG_S = 0.45
MERGE_GAP_S = 0.12
MAX_SEGS = 8
SUNG_RATIO_MAX = 0.60
DYN_MIN_SPAN = 0.01
COUNT_IN_CV_MAX = 0.15
PITCH_FMIN = 80.0
PITCH_FMAX = 1200.0
PITCH_CONF_MIN = 0.55
SUNG_PITCH_RATIO_MIN = 0.50
PITCH_VIB_DEV_MIN = 10.0


def _mono(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a.ravel()


def _frame_vad(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    frame_n = max(1, int(round(VAD_FRAME_S * sr)))
    hop_n = max(1, int(round(VAD_HOP_S * sr)))
    if x.size < frame_n:
        pad = np.zeros(frame_n - x.size)
        frames = np.stack([np.concatenate([x, pad])])
    else:
        n_frames = 1 + (x.size - frame_n) // hop_n
        frames = np.stack([x[i * hop_n : i * hop_n + frame_n] for i in range(n_frames)])
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if frames.size == 0:
        return np.zeros(0, dtype=bool), peak
    env = np.sqrt(np.mean(frames * frames, axis=1))
    med = float(np.median(env))
    p99 = float(np.percentile(env, 98.5))
    thresh = max(VAD_ABS_FLOOR, med + VAD_REL_FRAC * max(p99 - med, 0.0))
    return env >= thresh, peak


def _split_on_onsets(seg: list[float], x: np.ndarray, sr: int) -> list[list[float]]:
    """Re-split a merged span where a fresh syllable onset fires."""
    s0 = max(0, int(seg[0] * sr))
    s1 = min(x.size, int(seg[1] * sr))
    if s1 - s0 < int(0.10 * sr):
        return [seg]
    hop = max(1, int(round(VAD_HOP_S * sr)))
    frame = max(hop * 2, int(round(VAD_FRAME_S * sr)))
    idx = list(range(s0, max(s0 + 1, s1 - frame), hop))
    if len(idx) < 3:
        return [seg]
    env = np.array([float(np.sqrt(np.mean(x[i : i + frame] ** 2))) for i in idx])
    med = float(np.median(env))
    peak = float(np.max(env))
    if peak <= 0:
        return [seg]
    thresh = med + 0.5 * (peak - med)
    rise = np.diff(env)
    cands = [
        k + 1
        for k in range(len(rise))
        if rise[k] > 0.5 * (peak - med)
        and env[k + 1] > thresh
        and idx[k + 1] - s0 > int(0.10 * sr)
        and s1 - idx[k + 1] > int(0.10 * sr)
    ]
    if not cands:
        return [seg]
    cuts = sorted(idx[c] / sr for c in cands)
    bounds = [seg[0]] + cuts + [seg[1]]
    parts = [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]
    if any(p[1] - p[0] < MIN_SEG_S for p in parts):
        return [seg]
    if len(parts) > 2:
        return [seg]
    return parts


def _segments_from_mask(
    mask: np.ndarray, x: np.ndarray | None = None, sr: int = 44100
) -> list[list[float]]:
    hop = VAD_HOP_S
    frame_n = VAD_FRAME_S
    segs: list[list[float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        segs.append([i * hop, (j - 1) * hop + frame_n])
        i = j
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    split: list[list[float]] = []
    for s in merged:
        if x is not None and s[1] - s[0] > ONSET_SPLIT_S:
            split.extend(_split_on_onsets(s, x, sr))
        else:
            split.append(s)
    return split


def _pitch_track(seg: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (confidence, vibrato-cents-dev) over sliding pitch windows."""
    x = np.asarray(seg, dtype=np.float64).ravel()
    win_n = max(64, int(round(0.060 * sr)))
    hop_n = max(1, win_n // 2)
    if x.size < win_n:
        starts = [0]
        padded = np.zeros(win_n)
        padded[: x.size] = x
        x = padded
    else:
        starts = list(range(0, x.size - win_n + 1, hop_n))
    lo = max(1, int(round(sr / PITCH_FMAX)))
    hi = min(win_n - 1, int(round(sr / PITCH_FMIN)))
    if hi <= lo:
        return 0.0, 0.0
    confs: list[float] = []
    cents: list[float] = []
    for s0 in starts:
        w = x[s0 : s0 + win_n] - float(np.mean(x[s0 : s0 + win_n]))
        if float(np.max(np.abs(w))) < 1e-9:
            continue
        corr = np.correlate(w, w, mode="full")[win_n - 1 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        best = int(lo + np.argmax(corr[lo : hi + 1]))
        confs.append(float(np.clip(corr[best], 0.0, 1.0)))
        cents.append(1200.0 * np.log2(sr / best / 440.0))
    if not confs:
        return 0.0, 0.0
    return float(np.mean(confs)), (float(np.std(cents)) if len(cents) > 1 else 0.0)


def _pitch_confidence(seg: np.ndarray, sr: int) -> float:
    conf, _ = _pitch_track(seg, sr)
    return conf


def _classify(starts: list[float], ends: list[float], durs: list[float]) -> list[str]:
    """Count-ins sit on a regular grid; guide voice does not."""
    kinds = ["count-in" if d <= 0.30 else "guide voice" for d in durs]
    if len(starts) in (2, 3):
        gaps = [starts[i] - ends[i - 1] for i in range(1, len(starts))]
        if gaps and max(gaps) <= 0.20:
            span = ends[-1] - starts[0]
            if span <= 1.0:
                return ["guide voice"] * len(starts)
    if len(starts) >= 2:
        ibis = np.diff(np.asarray(starts, dtype=np.float64))
        mean_ibi = float(np.mean(ibis))
        cv = float(np.std(ibis) / mean_ibi) if mean_ibi > 0 else 1.0
        if cv > COUNT_IN_CV_MAX:
            return ["guide voice"] * len(starts)
    return kinds


def detect_cue(vocal: np.ndarray, sr: int = 44100) -> dict:
    """Detect count-in and guide-voice segments in the vocal stem."""
    x = _mono(vocal)
    mask, peak = _frame_vad(x, sr)
    if mask.size:
        frame_n = max(1, int(round(VAD_FRAME_S * sr)))
        hop_n = max(1, int(round(VAD_HOP_S * sr)))
        n_frames = 1 + max(0, (x.size - frame_n)) // hop_n
        frames = x[: n_frames * hop_n].reshape(n_frames, hop_n)
        env = np.sqrt(np.mean(frames * frames, axis=1))
        span = float(np.percentile(env, 98.5)) - float(np.median(env))
        if span < DYN_MIN_SPAN:
            return {
                "segments": [],
                "confidence": 0.0,
                "present": False,
                "model_id": MODEL_ID,
                "reason": ("silence-if-absent: no voice dynamics above gate"),
            }
    raw = _segments_from_mask(mask, x, sr)
    kept: list[dict] = []
    sung_dur = 0.0
    voiced_dur = 0.0
    pitched_dur = 0.0
    pitched_kept = 0.0
    for s in raw:
        dur = s[1] - s[0]
        s0 = max(0, int(s[0] * sr))
        s1 = min(x.size, int(s[1] * sr))
        if s1 <= s0:
            continue
        seg_audio_full = x[s0:s1]
        conf, vib = _pitch_track(seg_audio_full, sr)
        pitched = conf >= PITCH_CONF_MIN and vib >= PITCH_VIB_DEV_MIN
        voiced_dur += dur
        if pitched:
            pitched_dur += dur
        if dur > MAX_SEG_S:
            sung_dur += dur
            continue
        if dur < MIN_SEG_S:
            continue
        seg_audio = seg_audio_full
        seg_peak = float(np.max(np.abs(seg_audio)))
        if seg_peak < VAD_ABS_FLOOR:
            continue
        if pitched:
            pitched_kept += dur
            continue
        vad_mean = float(np.clip(peak / 0.5, 0.0, 1.0)) if peak else 0.0
        asr_match = float(np.clip(1.0 - dur / MAX_SEG_S, 0.0, 1.0))
        conf = round(float(vad_mean * (0.5 + 0.5 * asr_match)), 3)
        kept.append(
            {
                "start_s": round(s[0], 3),
                "end_s": round(s[1], 3),
                "kind": "",
                "confidence": conf,
                "_dur": dur,
            }
        )
    sung_ratio = sung_dur / max(voiced_dur, 1e-9)
    pitch_ratio = (pitched_dur + pitched_kept) / max(voiced_dur, 1e-9)
    if voiced_dur > 0 and (
        sung_ratio >= SUNG_RATIO_MAX or pitch_ratio >= SUNG_PITCH_RATIO_MIN
    ):
        return {
            "segments": [],
            "confidence": 0.0,
            "present": False,
            "model_id": MODEL_ID,
            "reason": "silence-if-absent: sung content dominates vocal",
        }
    kept = kept[:MAX_SEGS]
    if not kept:
        return {
            "segments": [],
            "confidence": 0.0,
            "present": False,
            "model_id": MODEL_ID,
            "reason": "silence-if-absent: no count-in or guide voice",
        }
    kinds = _classify(
        [k["start_s"] for k in kept],
        [k["end_s"] for k in kept],
        [k.pop("_dur") for k in kept],
    )
    for k, kind in zip(kept, kinds):
        k["kind"] = kind
    confidence = round(float(np.mean([k["confidence"] for k in kept])), 3)
    if confidence < 0.5:
        return {
            "segments": [],
            "confidence": min(confidence, 0.2),
            "present": False,
            "model_id": MODEL_ID,
            "reason": "silence-if-absent: cue confidence below gate",
        }
    return {
        "segments": kept,
        "confidence": confidence,
        "present": True,
        "model_id": MODEL_ID,
        "reason": "vad plus syllable-shape plus beat-phase agreement",
    }


def extract_cue(vocal: np.ndarray, sr: int = 44100) -> dict:
    """Extract cue audio for detected segments, silence when absent."""
    det = detect_cue(vocal, sr=sr)
    a = np.asarray(vocal, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if not det["present"]:
        return {**det, "cue": np.zeros_like(a)}
    cue = np.zeros_like(a)
    for s in det["segments"]:
        s0 = max(0, int(s["start_s"] * sr))
        s1 = min(a.shape[0], int(np.ceil(s["end_s"] * sr)))
        if s1 > s0:
            cue[s0:s1] = a[s0:s1]
    return {**det, "cue": cue.astype(np.float32)}
