"""Job pipeline: mix -> 9 stems + residual.wav + sidecars.

Stage-1 models fill vocal, drums, bass. `other_split` routes `other`
through the piano/guitar/strings specialists from the registry and
takes synth as the leftover by construction. cue is extracted from
the vocal stem, click resynthesized from beat timestamps; each falls
back to digital silence per silence-if-absent. residual carries mix
minus the 9 stems so nothing audible leaks silently.
"""

from pathlib import Path

import json
import numpy as np
import soundfile as sf

import autotune
import click_tracker as click_module
import cue as cue_module
import other_split
import qa as qa_module
import sidecars
import vocal as vocal_module
from config import SR
from stemspec import STEM_NAMES

BASE_VOCAL_MODEL_ID = vocal_module.BASE_VOCAL_MODEL_ID

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
    use_vocal_upgrade: bool = False,
    force_oom: bool = False,
) -> dict:
    """Run full T1 job from an audio file or pre-separated base stems.

    src: mix audio file (used when base_stems is None, via inference).
    base_stems: optional dict of 6s-base arrays keyed vocals/drums/bass/other.
    use_vocal_upgrade: route vocal through the KJ MelBand upgrade.
    force_oom: test/CI hook forcing the CPU-fallback path.
    Returns {"stems": [...], "sr": int, "vocal_info": {...}}.
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

    vocal, vocal_info = vocal_module.separate_vocal(
        mix,
        sr,
        base_stems["vocals"][:n],
        use_upgrade=use_vocal_upgrade,
        force_oom=force_oom,
    )
    drums = base_stems["drums"][:n]
    bass = base_stems["bass"][:n]
    other = base_stems["other"][:n]
    parts = other_split.split_other(other, sr=sr)
    piano = parts["piano"]
    guitar = parts["guitar"]
    strings = parts["strings"]
    synth = parts["synth"]
    extracted = cue_module.extract_cue(np.asarray(vocal), sr=sr)
    cue = np.asarray(extracted["cue"], dtype=np.float32)
    if cue.shape != mix.shape:
        cue = np.zeros_like(mix)
    cue_info = {
        k: extracted[k]
        for k in ("segments", "confidence", "present", "reason")
        if k in extracted
    }
    tracked = click_module.track_click(np.asarray(mix), sr=sr)
    click = np.asarray(tracked["click"], dtype=np.float32)
    if click.shape != mix.shape:
        click = np.zeros_like(mix)
    click_info = {
        k: tracked[k]
        for k in (
            "beats",
            "downbeats",
            "bpm_over_time",
            "meter",
            "confidence_global",
            "present",
            "reason",
        )
        if k in tracked
    }

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
    # cue is a subset of vocal, not an independent stem: exclude it so
    # residual stays mix minus the disjoint stem set.
    total = sum(stems[name] for name in STEM_NAMES if name != "cue")
    residual = mix - total
    sf.write(str(out_dir / "residual.wav"), residual, sr)
    plan = autotune.probe(
        models=(BASE_VOCAL_MODEL_ID, vocal_module.VOCAL_MODEL_ID),
        force_oom=force_oom,
    )
    active = plan[vocal_info["model_id"]]
    sidecars.write_manifest(
        out_dir,
        stems,
        sr=sr,
        model=vocal_info["model_id"],
        tile=active["tile"],
        autotune=plan,
        click_info=click_info,
        cue_info=cue_info,
    )
    sidecars.write_tempo_map(out_dir, n_frames=n, sr=sr, click_info=click_info)
    qa_module.write_qa(out_dir, stems, sr=sr, runs=1, spread_db=0.0, limiter="none")
    (out_dir / "ab_vocal.json").write_text(
        json.dumps(
            vocal_ab_doc(base_stems, sr=sr, src=src if base_stems is None else None),
            indent=2,
        )
    )
    n_mix = mix.shape[0]
    n_stem = min(np.asarray(stems[name]).shape[0] for name in STEM_NAMES)
    assert n_stem == n_mix, f"stem/mix length mismatch {n_stem} != {n_mix}"
    return {"stems": STEM_NAMES, "sr": sr, "vocal_info": vocal_info}


def vocal_ab_doc(
    base_stems: dict[str, np.ndarray],
    sr: int = 44100,
    src: Path | None = None,
    runs: int = 1,
    spread_db: float = 0.0,
    limiter: str = "none",
    clip: str = "clean-mix",
) -> dict:
    """A/B vocal scores with run count, spread, limiter.

    Self-contained sibling of other_split.overlap_doc: scores base
    vs upgraded vocal on the same 6s input so floors lock from
    measured runs, not invented thresholds.
    """
    keys = [k for k in ("vocals", "drums", "bass", "other") if k in base_stems]
    n = min(base_stems[k].shape[0] for k in keys)
    mix = None
    if src is not None and Path(src).exists():
        import soundfile as _sf

        mix, _ = _sf.read(str(src), dtype="float32", always_2d=True)
    if mix is None:
        mix = np.zeros_like(base_stems[keys[0]][:n])
    else:
        mix = np.asarray(mix[:n], dtype=np.float32)
    variants = {
        "base": np.asarray(base_stems["vocals"][:n], dtype=np.float32),
        "upgraded": np.asarray(
            vocal_module.separate_vocal(
                mix, sr, base_stems["vocals"][:n], use_upgrade=True
            )[0],
            dtype=np.float32,
        ),
    }
    return {
        "clip": clip,
        "sr": sr,
        "stems": qa_module.score_stems(
            variants, runs=runs, spread_db=spread_db, limiter=limiter
        ),
        "clips": qa_module.score_clips(variants),
    }
