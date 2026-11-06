"""Job pipeline: mix -> 9 stems + residual.wav + sidecars.

Stage-1 models fill vocal, drums, bass. other-split placeholders fill
piano, guitar, strings from the base model's `other` bucket. synth is
the leftover of the other-split by construction. cue and click are
digital silence per silence-if-absent. residual carries mix minus the
9 stems so nothing audible leaks silently.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

import sidecars
from config import SR
from stemspec import STEM_NAMES

_BASE_TO_SLOT = {
    "vocals": "vocal",
    "drums": "drums",
    "bass": "bass",
    "other": "other",
}


def run_job(
    src: Path,
    out_dir: Path,
    base_stems: dict[str, np.ndarray] | None = None,
    progress_cb=None,
) -> dict:
    """Run full T1 job from an audio file or pre-separated base stems.

    src: mix audio file (used when base_stems is None, via inference).
    base_stems: optional dict of 6s-base arrays keyed vocals/drums/bass/other.
    Returns {"stems": [...], "sr": int}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_stems is None:
        from inference import separate_file

        base_paths = separate_file(src, out_dir / "_base", progress_cb=progress_cb)
        audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
        base_stems = {}
        for p in base_paths:
            arr, _ = sf.read(str(p), dtype="float32", always_2d=True)
            base_stems[p.stem] = arr
        mix = audio
    else:
        mix = None
        sr = SR
        if src.exists():
            mix, sr = sf.read(str(src), dtype="float32", always_2d=True)
            if sr != SR:
                import soxr

                mix = soxr.resample(mix.T, sr, SR).T
                sr = SR
    keys = [k for k in ("vocals", "drums", "bass", "other") if k in base_stems]
    n = min(base_stems[k].shape[0] for k in keys)
    ch = base_stems[keys[0]].shape[1]
    if mix is None:
        mix = np.zeros((n, ch), dtype=np.float32)
    else:
        mix = mix[:n]
    if mix.shape[1] != ch:
        mix = np.tile(mix[:, :1], (1, ch)) if mix.shape[1] == 1 else mix[:, :ch]

    vocal = base_stems["vocals"][:n]
    drums = base_stems["drums"][:n]
    bass = base_stems["bass"][:n]
    other = base_stems["other"][:n]
    piano = 0.5 * other
    guitar = 0.3 * other
    strings = 0.15 * other
    synth = other - piano - guitar - strings
    cue = np.zeros_like(mix)
    click = np.zeros_like(mix)

    stems: dict[str, np.ndarray] = {
        "vocal": vocal,
        "drums": drums,
        "bass": bass,
        "piano": piano,
        "guitar": guitar,
        "strings": strings,
        "synth": synth,
        "cue": cue,
        "click": click,
    }
    for name in STEM_NAMES:
        sf.write(str(out_dir / f"{name}.wav"), stems[name], sr)
    total = sum(stems[name] for name in STEM_NAMES)
    residual = mix - total
    sf.write(str(out_dir / "residual.wav"), residual, sr)
    sidecars.write_manifest(out_dir, stems, sr=sr)
    sidecars.write_tempo_map(out_dir, n_frames=n, sr=sr)
    return {"stems": STEM_NAMES, "sr": sr}
