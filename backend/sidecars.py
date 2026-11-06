"""Sidecar writers: manifest.json + tempo_map.json per T1 contract.

Schemas follow specs/REFACTOR_LATEST.md and GLOSSARY.md:
manifest carries per-stem file, present, confidence, source, reason
plus model and tile choices. tempo_map is the librosa-fallback shape:
beats only, downbeats null, silence-if-absent for optional stems.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from stemspec import BY_NAME, STEM_NAMES


def write_manifest(
    out_dir: Path,
    stems: dict[str, np.ndarray],
    sr: int = 44100,
    model: str = "htdemucs_6s-placeholder",
    tile: int = 343980,
    autotune: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc: dict = {"model": model, "tile": tile, "sr": sr, "stems": {}}
    for name in STEM_NAMES:
        spec = BY_NAME[name]
        audio = stems[name]
        h = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()[:16]
        if spec.optional:
            present: bool = False
            confidence: float = 0.0
            reason: str = "silence-if-absent: detector gate lands in later ticket"
        else:
            present = True
            confidence = 0.5
            reason = "stage-1 placeholder through real pipeline"
        doc["stems"][name] = {
            "file": f"{name}.wav",
            "present": present,
            "confidence": confidence,
            "source": spec.source,
            "reason": reason,
            "sha256_16": h,
        }
    p = out_dir / "manifest.json"
    if autotune is not None:
        doc["autotune"] = autotune
    p.write_text(json.dumps(doc, indent=2))
    return p


def write_tempo_map(
    out_dir: Path,
    n_frames: int,
    sr: int = 44100,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = n_frames / float(sr)
    doc = {
        "beats": [],
        "downbeats": None,
        "bpm_over_time": [],
        "meter": "4/4",
        "confidence_global": 0.0,
        "present": False,
        "reason": "librosa-fallback placeholder: beat tracker lands in T4",
        "duration_s": duration,
    }
    p = out_dir / "tempo_map.json"
    p.write_text(json.dumps(doc, indent=2))
    return p
