"""T5 red: cue extraction from vocal + silence gate.

Seam: job-output file contract (cue.wav + manifest cue entry with
segments read from the result dir). No model internals.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100


def _base(n: int) -> dict:
    return {
        k: np.zeros((n, 2), dtype=np.float32)
        for k in ("vocals", "drums", "bass", "other")
    }


def _place(track: np.ndarray, sig: np.ndarray, t0: float) -> None:
    s = int(t0 * SR)
    e = min(track.shape[0], s + sig.shape[0])
    if e > s:
        track[s:e] += sig[: e - s]


def _syllable(
    dur: float = 0.12, freqs=(700.0, 1050.0), amp: float = 0.5, seed: int = 0
) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    tone = sum(np.sin(2.0 * np.pi * f * t) for f in freqs) / len(freqs)
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 14.0)
    cons = np.zeros(n)
    m = int(0.02 * SR)
    cons[:m] = rng.standard_normal(m) * 0.4
    return (amp * (tone * env + cons * np.exp(-t * 60.0))).astype(np.float32)


def _sung_phrase(dur: float = 1.5, f0: float = 440.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 + 8.0 * np.sin(2.0 * np.pi * 5.5 * t)
    ph = 2.0 * np.pi * np.cumsum(f) / SR
    v = np.sin(ph) + 0.4 * np.sin(2.0 * ph) + 0.15 * np.sin(3.0 * ph)
    v = v / np.max(np.abs(v))
    r = int(0.05 * SR)
    env = np.ones(n)
    env[:r] = np.linspace(0.0, 1.0, r)
    env[-r:] = np.linspace(1.0, 0.0, r)
    return (0.5 * v * env).astype(np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _run_job(vocal: np.ndarray):
    import pipeline

    n = vocal.shape[0]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "mix.wav"
        sf.write(str(src), vocal, SR)
        base = _base(n)
        base["vocals"] = vocal
        out_dir = td / "job"
        pipeline.run_job(src, out_dir, base_stems=base)
        cue, _ = sf.read(str(out_dir / "cue.wav"), dtype="float32")
        manifest = json.loads((out_dir / "manifest.json").read_text())
        return np.asarray(cue), manifest["stems"]["cue"]


def _count_in_mix() -> tuple[np.ndarray, list[float]]:
    n = SR * 4
    mono = np.zeros(n, dtype=np.float32)
    times = [0.5, 1.0, 1.5, 2.0]
    for i, t0 in enumerate(times):
        _place(mono, _syllable(seed=i), t0)
    return _stereo(mono), times


def test_count_in_carries_segments_with_timestamps():
    cue, entry = _run_job(*[_count_in_mix()[0]])
    _, times = _count_in_mix()
    assert entry["present"] is True
    assert entry["confidence"] >= 0.5
    assert float(np.max(np.abs(cue))) > 0.1
    segs = entry["segments"]
    assert len(segs) >= len(times), f"expected >= {len(times)} segments, got {segs}"
    assert all(s["kind"] == "count-in" for s in segs)
    for t0 in times:
        nearest = min(abs(s["start_s"] - t0) for s in segs)
        assert nearest <= 0.06, f"count-in @{t0:.3f}s off by {nearest * 1000:.1f}ms"
    for s in segs:
        assert s["end_s"] > s["start_s"]
        assert s["kind"] in ("count-in", "guide voice")


def test_guide_voice_carries_segments_with_timestamps():
    n = SR * 4
    mono = np.zeros(n, dtype=np.float32)
    _place(mono, _syllable(dur=0.15, freqs=(500.0, 900.0), seed=11), 0.5)
    _place(mono, _syllable(dur=0.20, freqs=(600.0, 1200.0), seed=12), 0.9)
    _place(mono, _syllable(dur=0.25, freqs=(450.0, 800.0), seed=13), 1.6)
    cue, entry = _run_job(_stereo(mono))
    assert entry["present"] is True
    assert entry["confidence"] >= 0.5
    assert float(np.max(np.abs(cue))) > 0.1
    segs = entry["segments"]
    assert len(segs) >= 3
    assert any(s["kind"] == "guide voice" for s in segs)
    for t0 in (0.5, 0.9, 1.6):
        nearest = min(abs(s["start_s"] - t0) for s in segs)
        assert nearest <= 0.06, f"guide @{t0:.3f}s off by {nearest * 1000:.1f}ms"


def test_absent_count_in_is_silent_with_low_confidence():
    n = SR * 4
    t = np.arange(n) / SR
    bed = 0.15 * np.sin(2.0 * np.pi * 220.0 * t)
    cue, entry = _run_job(_stereo(bed))
    assert float(np.max(np.abs(cue))) == 0.0
    assert entry["present"] is False
    assert entry["confidence"] <= 0.2
    assert entry["segments"] == []


def test_sung_numbers_do_not_leak_into_cue():
    n = SR * 4
    mono = np.zeros(n, dtype=np.float32)
    _place(mono, _sung_phrase(dur=1.5, f0=392.0), 0.25)
    _place(mono, _sung_phrase(dur=1.5, f0=523.25), 2.25)
    vocal = _stereo(mono)
    cue, entry = _run_job(vocal)
    vocal_rms = float(np.sqrt(np.mean(np.asarray(vocal, dtype=np.float64) ** 2)))
    cue_rms = float(np.sqrt(np.mean(np.asarray(cue, dtype=np.float64) ** 2)))
    assert cue_rms / max(vocal_rms, 1e-9) <= 0.05, (
        f"sung leak {cue_rms / vocal_rms:.3f} above threshold"
    )
    assert entry["present"] is False
    assert entry["confidence"] <= 0.2


def _sung_staccato(dur: float = 0.20, gap: float = 0.10) -> np.ndarray:
    n = SR * 4
    mono = np.zeros(n, dtype=np.float32)
    t0 = 0.25
    seed = 0
    while t0 + dur < 3.75:
        _place(mono, _sung_phrase(dur=dur, f0=392.0 + 20 * seed), t0)
        t0 += dur + gap
        seed += 1
    return _stereo(mono)


def test_staccato_sung_counts_do_not_leak_into_cue():
    vocal = _sung_staccato()
    cue, entry = _run_job(vocal)
    vocal_rms = float(np.sqrt(np.mean(np.asarray(vocal, dtype=np.float64) ** 2)))
    cue_rms = float(np.sqrt(np.mean(np.asarray(cue, dtype=np.float64) ** 2)))
    assert cue_rms / max(vocal_rms, 1e-9) <= 0.05, (
        f"staccato sung leak {cue_rms / vocal_rms:.3f} above threshold"
    )
    assert entry["present"] is False
    assert entry["confidence"] <= 0.2
