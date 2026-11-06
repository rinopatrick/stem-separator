"""Vocal upgrade: KJ MelBand behind a flag, fallback to the 6s vocal.

Stand-in behind the registry model id (weights land later, same as
the T2 specialist pattern): flag off returns the 6s base untouched;
flag on applies a deterministic presence lift; forced OOM returns the
base with fallback True so the job still completes.
"""
import numpy as np

from stemspec import BY_NAME

VOCAL_MODEL_ID = BY_NAME["vocal"].model_id
BASE_VOCAL_MODEL_ID = "htdemucs_6s"


def _presence_lift(vocal: np.ndarray) -> np.ndarray:
    v = np.asarray(vocal, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    hp = np.zeros_like(v)
    hp[1:] = v[1:] - v[:-1]
    out = 0.9 * v + 0.35 * hp
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    base_peak = float(np.max(np.abs(v))) if v.size else 0.0
    if peak > base_peak > 0:
        out = out * (base_peak / peak)
    return out.astype(np.float32)


def separate_vocal(
    mix: np.ndarray,
    sr: int = 44100,
    base_vocal: np.ndarray | None = None,
    use_upgrade: bool = False,
    force_oom: bool = False,
) -> tuple[np.ndarray, dict]:
    """Return (vocal_audio, info) with info {model_id, fallback}."""
    if base_vocal is None:
        base_vocal = np.asarray(mix, dtype=np.float32)
    base = np.asarray(base_vocal, dtype=np.float32)
    if not use_upgrade:
        return base, {"model_id": BASE_VOCAL_MODEL_ID, "fallback": False}
    if force_oom:
        return base, {"model_id": BASE_VOCAL_MODEL_ID, "fallback": True,
                      "reason": "forced OOM: CPU fallback"}
    out = _presence_lift(base)
    if out.shape != base.shape:
        out = np.asarray(out).reshape(base.shape)
    return out, {"model_id": VOCAL_MODEL_ID, "fallback": False}
