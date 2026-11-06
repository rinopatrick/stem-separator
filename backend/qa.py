"""qa.json writer: SI-SDR estimates, leakage matrix, clip flags.

Reads per-job stems off disk output only. No model internals.
SI-SDR here is the scale-invariant projection of the target stem
onto the rest (alpha = dot(t, r) / dot(r, r)), without a clean
reference: it measures how much of the target is explained by the
other stems. Large negative values mean the stem is distinct; values
near zero or positive mean heavy bleed. Recorded with run count,
spread, and limiter so floors lock from measured runs.
"""
import json
from pathlib import Path

import numpy as np

CLIP_PEAK = 0.99


def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a.ravel()


def _sisdr_db(target: np.ndarray, rest: np.ndarray) -> float:
    t = _mono(target)
    r = _mono(rest)
    n = min(t.size, r.size)
    t, r = t[:n], r[:n]
    eps = 1e-8
    rr = float(np.dot(r, r)) + eps
    alpha = float(np.dot(t, r)) / rr
    proj = alpha * r
    sig = float(np.dot(proj, proj)) + eps
    noise = float(np.dot(t - proj, t - proj)) + eps
    return float(10.0 * np.log10(sig / noise))


def _leakage(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _mono(a), _mono(b)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))


def score_stems(
    stems: dict[str, np.ndarray],
    runs: int = 1,
    spread_db: float = 0.0,
    limiter: str = "none",
) -> dict:
    names = sorted(stems)
    out = {}
    for name in names:
        rest = sum(
            np.asarray(stems[o], dtype=np.float64) for o in names if o != name
        )
        out[name] = {
            "sisdr_db": round(_sisdr_db(stems[name], rest), 2),
            "spread_db": spread_db,
            "runs": runs,
            "limiter": limiter,
        }
    return out


def score_clips(stems: dict[str, np.ndarray]) -> list:
    clips = []
    for name in sorted(stems):
        peak = float(np.max(np.abs(np.asarray(stems[name], dtype=np.float64))))
        clips.append(
            {"stem": name, "peak": round(peak, 4), "flagged": bool(peak >= CLIP_PEAK)}
        )
    return clips


def score_leakage(stems: dict[str, np.ndarray]) -> dict:
    names = sorted(stems)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}|{b}"] = round(_leakage(stems[a], stems[b]), 4)
    return out


def write_qa(
    out_dir: Path,
    stems: dict[str, np.ndarray],
    sr: int = 44100,
    runs: int = 1,
    spread_db: float = 0.0,
    limiter: str = "none",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "sr": sr,
        "stems": score_stems(stems, runs=runs, spread_db=spread_db, limiter=limiter),
        "leakage": score_leakage(stems),
        "clips": score_clips(stems),
    }
    p = out_dir / "qa.json"
    p.write_text(json.dumps(doc, indent=2))
    return p
