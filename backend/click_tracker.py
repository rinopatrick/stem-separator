"""T4: beat-time click tracker + resynthesis (scipy + numpy only).

Data shape: beats are monotonically increasing seconds; tempo_map is
one dict with parallel arrays plus one bpm entry per 8 beats.
Absent metronome yields digital silence with present False so the
pipeline never synthesizes fake clicks.

Prototype decision (issue #6): clicks are always resynthesized,
1 kHz beats and 1.5 kHz downbeats with a raised-cosine envelope.

Design: onset-strength tracking on a highpassed transient envelope
with adaptive-median peak picking, then beat-adaptive bpm_over_time.
"""
import numpy as np
from scipy.signal import find_peaks

from stemspec import BY_NAME

SR_DEFAULT = 44100
CLICK_SR = 22050
BEAT_HZ = 1000.0
DOWNBEAT_HZ = 1500.0
CLICK_S = 0.03
CLICK_AMP = 0.8
BPM_MIN = 40.0
BPM_MAX = 220.0
PRESENT_PEAK = 0.10
PRESENT_MIN_BEATS = 4

MODEL_ID = BY_NAME["click"].model_id

BEAT_QUANTUM_S = 0.030


def _resample_mono(mix: np.ndarray, sr: int) -> np.ndarray:
    a = np.asarray(mix, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if sr == CLICK_SR:
        return a
    n_out = int(round(a.size * CLICK_SR / sr))
    if n_out < 2:
        return a
    idx = np.linspace(0, a.size - 1, n_out)
    return np.interp(idx, np.arange(a.size), a)


def _transient_envelope(x: np.ndarray) -> np.ndarray:
    hp = np.empty_like(x)
    hp[0] = x[0]
    hp[1:] = x[1:] - 0.98 * x[:-1]
    win = max(1, int(0.004 * CLICK_SR))
    ph = hp * hp
    k = np.ones(win) / win
    return np.convolve(ph, k, mode="same")


def _pick_onsets(env: np.ndarray) -> np.ndarray:
    med = float(np.median(env)) if env.size else 0.0
    p99 = float(np.percentile(env, 99)) if env.size else 0.0
    span = max(p99 - med, 1e-12)
    thresh = med + 0.25 * span
    min_gap = int(0.20 * CLICK_SR)
    peaks, props = find_peaks(env, height=thresh, distance=min_gap)
    if peaks.size == 0:
        return peaks
    heights = np.asarray(props["peak_heights"], dtype=np.float64)
    med_h = float(np.median(heights))
    keep = heights >= max(med_h * 0.35, med + 0.10 * span)
    return peaks[keep]


def _refine_onsets(peaks: np.ndarray, env: np.ndarray) -> list[float]:
    search = int(0.030 * CLICK_SR)
    out: list[float] = []
    for p in peaks:
        lo = max(0, int(p) - search)
        seg = env[lo:int(p) + 1]
        best = lo + int(np.argmax(seg)) if seg.size else int(p)
        while best > lo and env[best] < 0.5 * env[int(p)]:
            best -= 1
        out.append(best / CLICK_SR)
    if len(out) >= 2:
        ibi = float(np.median(np.diff(sorted(out))))
        if 60.0 / BPM_MAX <= ibi <= 60.0 / BPM_MIN:
            first = out[0] - round(out[0] / ibi) * ibi
            while first < -BEAT_QUANTUM_S:
                first += ibi
            if 0.0 <= first < out[0] - BEAT_QUANTUM_S:
                out = [first] + out
    return sorted(out)


def _quantize(beats: list[float]) -> list[float]:
    if not beats:
        return beats
    q = [beats[0]]
    for b in beats[1:]:
        if b - q[-1] >= BEAT_QUANTUM_S:
            q.append(b)
    return q


def _median_ibi(beats: list[float]) -> float:
    if len(beats) < 2:
        return 0.0
    return float(np.median(np.diff(np.asarray(beats, dtype=np.float64))))


def detect_click(mix: np.ndarray, sr: int = 44100) -> dict:
    """Detect metronome onsets. Returns beats/downbeats/bpm/confidence."""
    x = _resample_mono(mix, sr)
    env = _transient_envelope(x)
    onsets = _quantize(_refine_onsets(_pick_onsets(env), env))
    if len(onsets) < PRESENT_MIN_BEATS:
        return {"beats": [], "downbeats": None, "bpm_over_time": [],
                "meter": "4/4", "confidence_global": 0.0, "present": False,
                "model_id": MODEL_ID,
                "reason": "silence-if-absent: no periodic onsets above gate"}
    ibi = float(np.median(np.diff(np.asarray(onsets, dtype=np.float64))))
    if not (60.0 / BPM_MAX <= ibi <= 60.0 / BPM_MIN):
        return {"beats": [], "downbeats": None, "bpm_over_time": [],
                "meter": "4/4", "confidence_global": 0.0, "present": False,
                "model_id": MODEL_ID,
                "reason": "silence-if-absent: onset spacing outside 40-220 BPM"}
    peak = float(np.max(np.abs(np.asarray(mix, dtype=np.float64)))) if np.asarray(mix).size else 0.0
    if peak > 0 and peak < PRESENT_PEAK:
        return {"beats": [], "downbeats": None, "bpm_over_time": [],
                "meter": "4/4", "confidence_global": 0.0, "present": False,
                "model_id": MODEL_ID,
                "reason": "silence-if-absent: bed too quiet for metronome gate"}
    ibi_med = _median_ibi(onsets)
    devs = [abs(b - (onsets[0] + i * ibi_med)) for i, b in enumerate(onsets)]
    jitter = float(np.median(devs)) if devs else 0.0
    confidence = float(np.clip(1.0 - jitter / (0.5 * ibi_med), 0.0, 1.0))
    if confidence < 0.5:
        return {"beats": [], "downbeats": None, "bpm_over_time": [],
                "meter": "4/4", "confidence_global": round(confidence, 3),
                "present": False, "model_id": MODEL_ID,
                "reason": "silence-if-absent: onsets not periodic enough"}
    beats = [round(b, 3) for b in onsets]
    downbeats = [b for i, b in enumerate(beats) if i % 4 == 0]
    bpm_over_time = _bpm_over_time(beats)
    return {"beats": beats, "downbeats": downbeats,
            "bpm_over_time": bpm_over_time, "meter": "4/4",
            "confidence_global": round(confidence, 3), "present": True,
            "model_id": MODEL_ID,
            "reason": "beat-time tracker: onset strengths, per-8-beat tempo"}


def _bpm_over_time(beats: list[float], per: int = 8) -> list[dict]:
    entries: list[dict] = []
    for start in range(0, max(len(beats) - 1, 0), per):
        seg = beats[start:start + per + 1]
        if len(seg) < 2:
            continue
        ibis = np.diff(np.asarray(seg, dtype=np.float64))
        ibi = float(np.median(ibis))
        if ibi <= 0:
            continue
        bpm = float(np.clip(60.0 / ibi, BPM_MIN, BPM_MAX))
        entries.append({"start_s": seg[0], "end_s": seg[-1],
                        "bpm": round(bpm, 1)})
    return entries


def _pulse(freq: float, amp: float = CLICK_AMP) -> np.ndarray:
    n = max(1, int(CLICK_S * SR_DEFAULT))
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    t = np.arange(n) / SR_DEFAULT
    return (amp * np.sin(2.0 * np.pi * freq * t) * env).astype(np.float32)


_BEAT_PULSE = _pulse(BEAT_HZ)
_DOWN_PULSE = _pulse(DOWNBEAT_HZ)


def _place(track: np.ndarray, pulse: np.ndarray, at_s: float) -> None:
    c = int(round(at_s * SR_DEFAULT)) - len(pulse) // 2
    s = max(0, c)
    e = min(track.shape[0], c + len(pulse))
    if e > s:
        track[s:e, :] += pulse[s - c:e - c, None]


def resynthesize(beats: list[float], downbeats: list[float] | None,
                 n_frames: int, sr: int = 44100) -> np.ndarray:
    """Render click track at mix length. Beats 1 kHz, downbeats 1.5 kHz."""
    track = np.zeros((n_frames, 2), dtype=np.float32)
    if sr != SR_DEFAULT or n_frames <= 0:
        return track
    downs = set(downbeats or [])
    for b in beats:
        _place(track, _DOWN_PULSE if b in downs else _BEAT_PULSE, b)
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1.0:
        track = (track / peak * 0.99).astype(np.float32)
    return track


def track_click(mix: np.ndarray, sr: int = 44100) -> dict:
    """Detect beats and resynthesize a mix-length click stem."""
    det = detect_click(mix, sr=sr)
    n = np.asarray(mix).shape[0] if np.asarray(mix).size else 0
    if not det["present"]:
        return {**det, "click": np.zeros((n, 2), dtype=np.float32)}
    return {**det, "click": resynthesize(det["beats"], det["downbeats"], n, sr)}
