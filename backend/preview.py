"""T6: lazy loudness-normalized preview cache (~-16 LUFS, opus + mp3).

Data shape: previews are derived files `preview-{stem}.{fmt}` inside
the job result dir. The job never writes them; the preview endpoint
builds them on first GET and reuses the cache after. Deleting the job
dir invalidates them by construction.

Loudness: single-pass gain to a -16 LUFS target measured on the
mean-square energy with a fixed calibration offset, then a true-peak
ceiling at -1 dBFS so quiet stems get gain but nothing clips.
The measure is an energy approximation, not gated K-weighting:
callers and docs say "about -16 LUFS" (spec wording) for that reason.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from stemspec import STEM_NAMES

FORMATS = ("opus", "mp3")
TARGET_LUFS = -16.0
TRUE_PEAK_CEILING = 0.891  # -1 dBFS
_K_OFF_DB = 3.01
_EPS = 1e-12


def measure_lufs(audio: np.ndarray) -> float:
    """Energy-based loudness approximation (about -16 LUFS target).

    Mean-square with a fixed calibration offset, not gated K-weighting.
    Good enough for audition previews; never presented as mastering-grade."""
    a = np.asarray(audio, dtype=np.float64)
    ms = float(np.mean(a * a)) + _EPS
    return float(10.0 * np.log10(ms) - _K_OFF_DB)


def _normalize(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    lufs = measure_lufs(a)
    gain = float(10.0 ** ((TARGET_LUFS - lufs) / 20.0))
    out = a * gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > TRUE_PEAK_CEILING:
        out = out * (TRUE_PEAK_CEILING / peak)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def preview_path(job_dir: Path, stem: str, fmt: str) -> Path:
    return Path(job_dir) / f"preview-{stem}.{fmt}"


def get_preview(job_dir: Path, stem: str, fmt: str = "opus") -> Path:
    """Build (or reuse) the normalized preview for one stem."""
    if stem not in STEM_NAMES and stem != "residual":
        raise KeyError(f"unknown stem: {stem}")
    if fmt not in FORMATS:
        raise KeyError(f"unknown preview format: {fmt}")
    job_dir = Path(job_dir)
    out = preview_path(job_dir, stem, fmt)
    if out.exists():
        return out
    src = job_dir / f"{stem}.wav"
    if not src.exists():
        raise FileNotFoundError(f"stem not ready: {stem}")
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    normed = _normalize(audio)
    if fmt == "opus":
        opus_sr = 48000
        opus_audio = normed
        if sr != opus_sr:
            try:
                import soxr

                opus_audio = soxr.resample(
                    normed.T, sr, opus_sr).T.astype(np.float32)
            except Exception:
                from scipy.signal import resample_poly
                import math

                g = math.gcd(sr, opus_sr)
                opus_audio = resample_poly(
                    normed, opus_sr // g, sr // g, axis=0).astype(np.float32)
        sf.write(str(out), opus_audio, opus_sr, format="OGG", subtype="OPUS")
    else:
        sf.write(str(out), normed, sr, format="MP3")
    return out
