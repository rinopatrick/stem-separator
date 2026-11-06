"""T4 red: click resynthesized from beat timestamps + tempo display seam.

Seam: job-output file contract (click.wav + tempo_map.json + manifest
click entry read from the result dir). No model internals.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100

BEAT_FREQ = 1000.0
BURST_S = 0.03
BURST_AMP = 0.8


def _burst() -> np.ndarray:
    n = int(BURST_S * SR)
    t = np.arange(n) / SR
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    return (BURST_AMP * np.sin(2.0 * np.pi * BEAT_FREQ * t) * env).astype(np.float32)


def _click_track(times: list[float], dur_s: float, bed_hz: float = 220.0) -> np.ndarray:
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    mix = 0.15 * np.sin(2.0 * np.pi * bed_hz * t)
    b = _burst()
    for ts in times:
        s = int(ts * SR - len(b) // 2)
        e = s + len(b)
        if s >= 0 and e <= n:
            mix[s:e] += b
    return np.stack([mix, mix], axis=1).astype(np.float32)


def _base(n: int) -> dict:
    return {k: np.zeros((n, 2), dtype=np.float32)
            for k in ("vocals", "drums", "bass", "other")}


def _run_job(mix: np.ndarray):
    import pipeline

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "mix.wav"
        sf.write(str(src), mix, SR)
        out_dir = td / "job"
        pipeline.run_job(src, out_dir,
                         base_stems=_base(mix.shape[0]))
        click, _ = sf.read(str(out_dir / "click.wav"), dtype="float32")
        tempo = json.loads((out_dir / "tempo_map.json").read_text())
        manifest = json.loads((out_dir / "manifest.json").read_text())
        return np.asarray(click), tempo, manifest


def _tempo_change_mix() -> tuple[np.ndarray, list[float]]:
    times = [i * 0.5 for i in range(16)]
    times += [8.0 + i * (2.0 / 3.0) for i in range(12)]
    return _click_track(times, 16.5), times


def test_click_aligns_within_30ms_on_tempo_change():
    click, tempo, _ = _run_job(*[_tempo_change_mix()[0]])
    _, times = _tempo_change_mix()
    assert tempo["present"] is True
    assert tempo["confidence_global"] >= 0.5
    assert len(tempo["beats"]) > 0
    for ts in times:
        nearest = min(abs(b - ts) for b in tempo["beats"])
        assert nearest <= 0.030, f"beat @{ts:.3f}s off by {nearest * 1000:.1f}ms"
    mono = np.asarray(click).mean(axis=1) if np.asarray(click).ndim == 2 else click
    for ts in times:
        s = max(0, int((ts - 0.03) * SR))
        e = int((ts + 0.05) * SR)
        assert float(np.max(np.abs(mono[s:e]))) > 0.1, f"no click near {ts:.3f}s"


def test_bpm_over_time_tracks_both_sections():
    _, tempo, _ = _run_job(*[_tempo_change_mix()[0]])
    entries = tempo["bpm_over_time"]
    assert len(entries) >= 2
    assert abs(entries[0]["bpm"] - 120.0) <= 5.0, entries[0]
    assert abs(entries[-1]["bpm"] - 90.0) <= 5.0, entries[-1]


def test_absent_metronome_is_silent_with_low_confidence():
    n = SR * 4
    t = np.arange(n) / SR
    bed = 0.15 * np.sin(2.0 * np.pi * 220.0 * t)
    mix = np.stack([bed, bed], axis=1).astype(np.float32)
    click, tempo, manifest = _run_job(mix)
    assert float(np.max(np.abs(click))) == 0.0
    assert tempo["present"] is False
    assert tempo["confidence_global"] <= 0.2
    assert tempo["beats"] == []
    assert manifest["stems"]["click"]["present"] is False
    assert manifest["stems"]["click"]["confidence"] <= 0.2


def test_steady_click_carries_downbeats_meter_and_manifest():
    times = [i * 0.5 for i in range(8)]
    mix = _click_track(times, 4.5)
    _, tempo, manifest = _run_job(mix)
    assert tempo["present"] is True
    assert tempo["downbeats"] is not None and len(tempo["downbeats"]) >= 1
    assert tempo["meter"] == "4/4"
    assert manifest["stems"]["click"]["present"] is True
    assert manifest["stems"]["click"]["confidence"] >= 0.5
