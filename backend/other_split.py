"""Stage-2 other-split: piano/guitar/strings specialists, synth leftover.

T2 heuristic stand-in behind the registry specialist ids (model
weights land in a later ticket): piano takes low harmonic body,
strings take sustained upper harmonics, guitar takes transient
midrange, all as CPU-only STFT masks. The trio masks are jointly
normalized; each masked slot is reconstructed by ISTFT; synth is
computed in the time domain as other minus piano minus guitar minus
strings, so the four slots sum to `other` exactly by construction.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

import qa as qa_module
from stemspec import BY_NAME

NPERSEG = 4096
NOVERLAP = 3072
EPS = 1e-8


def _hpss_weights(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harm = median_filter(mag, size=(1, 17))
    perc = median_filter(mag, size=(17, 1))
    denom = harm + perc + EPS
    return harm / denom, perc / denom


def _band(freqs: np.ndarray, lo: float, hi: float, edge: float = 120.0) -> np.ndarray:
    w = np.ones_like(freqs)
    w = w * np.clip((freqs - (lo - edge)) / edge, 0.0, 1.0)
    w = w * np.clip(((hi + edge) - freqs) / edge, 0.0, 1.0)
    return w


def _stft(x: np.ndarray, sr: int):
    return stft(x, fs=sr, window="hann", nperseg=NPERSEG,
                noverlap=NOVERLAP, boundary="zeros")


def _masks_for_channel(freqs: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    mag = np.abs(z)
    harm_w, perc_w = _hpss_weights(mag)
    piano = harm_w * _band(freqs, 55.0, 2100.0)[:, None]
    strings = harm_w * _band(freqs, 300.0, 8000.0)[:, None]
    guitar = (perc_w * _band(freqs, 150.0, 6000.0)[:, None]
              + 0.3 * harm_w * _band(freqs, 150.0, 6000.0)[:, None])
    total = piano + strings + guitar
    over = total > 1.0
    total_safe = np.where(over, total, 1.0)
    piano = piano / total_safe
    strings = strings / total_safe
    guitar = guitar / total_safe
    synth = np.clip(1.0 - (piano + strings + guitar), 0.0, 1.0)
    return {"piano": piano, "guitar": guitar, "strings": strings, "synth": synth}


def split_other(other: np.ndarray, sr: int = 44100) -> dict[str, np.ndarray]:
    audio = np.asarray(other, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    n, ch = audio.shape
    trio: dict[str, list] = {"piano": [], "guitar": [], "strings": []}
    for c in range(ch):
        freqs, _, z = _stft(audio[:, c], sr)
        masks = _masks_for_channel(freqs, z)
        for name in ("piano", "guitar", "strings"):
            _, rec = istft(masks[name] * z, fs=sr, window="hann",
                           nperseg=NPERSEG, noverlap=NOVERLAP)
            if rec.size < n:
                rec = np.pad(rec, (0, n - rec.size))
            trio[name].append(rec[:n])
    out: dict[str, np.ndarray] = {}
    for name, chans in trio.items():
        out[name] = np.stack(chans, axis=1).astype(np.float32)
    synth64 = audio.astype(np.float64) - np.stack(
        [out["piano"].astype(np.float64), out["guitar"].astype(np.float64),
         out["strings"].astype(np.float64)], axis=0).sum(axis=0)
    out["synth"] = synth64.astype(np.float32)
    return out


def split_report(other: np.ndarray, sr: int = 44100) -> dict[str, dict]:
    parts = split_other(other, sr=sr)
    report = {}
    for name in ("piano", "guitar", "strings"):
        spec = BY_NAME[name]
        report[name] = {
            "model_id": spec.model_id,
            "source": spec.source,
            "peak": round(float(np.max(np.abs(parts[name]))), 4)
            if parts[name].size
            else 0.0,
        }
    spec = BY_NAME["synth"]
    report["synth"] = {"model_id": spec.model_id, "source": spec.source}
    return report


def overlap_doc(
    parts: dict[str, np.ndarray],
    sr: int = 44100,
    clip: str = "strings-plus-synth-overlap",
) -> dict:
    return {
        "clip": clip,
        "sr": sr,
        "stems": qa_module.score_stems(parts, runs=1, spread_db=0.0, limiter="none"),
        "clips": qa_module.score_clips(parts),
    }
