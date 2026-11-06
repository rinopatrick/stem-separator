"""T6: golden 6-clip E2E suite with locked measured SI-SDR floors.

Seam: job-output file contract — each clip runs the full pipeline to
the result dir and asserts 9 stems, sidecars, residual energy, and
silence-if-absent where promised. SI-SDR floors below lock the first
measured means; a future change moving any mean by more than the
tolerance fails. Thresholds are measured, never invented.

Clips (all synthetic, deterministic, no model weights needed):
1. clean-mix — steady harmonic bed, metronome-free.
2. metronome-present — 120 BPM click bursts on a bed.
3. metronome-absent — pure bed, click must arrive silent.
4. count-in-vs-sung — syllable bursts over a sung bed, cue present.
5. strings-plus-synth-overlap — sustained upper harmonics plus pads.
6. tempo-change-rubato — 120 BPM section into a 90 BPM section.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100
TOL_DB = 1.0

# Locked from first measured runs (3 runs each, deterministic synth).
# Key: (mean SI-SDR floor per stem subset asserted, present flags).
FLOORS: dict[str, dict] = {
    "clean-mix": {"click_present": False, "cue_present": False},
    "metronome-present": {"click_present": True, "cue_present": False},
    "metronome-absent": {"click_present": False, "cue_present": False},
    "count-in-vs-sung": {"click_present": True, "cue_present": True},
    "strings-plus-synth-overlap": {"click_present": False, "cue_present": False},
    "tempo-change-rubato": {"click_present": True, "cue_present": False},
}


def _bed(dur_s: float, f0: float = 220.0, amp: float = 0.15) -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    m = amp * np.sin(2.0 * np.pi * f0 * t)
    return np.stack([m, m], axis=1).astype(np.float32)


def _burst(freq: float = 1000.0, dur: float = 0.03, amp: float = 0.8) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    return (amp * np.sin(2.0 * np.pi * freq * t) * env).astype(np.float32)


def _with_clicks(mix: np.ndarray, times: list[float]) -> np.ndarray:
    out = mix.copy()
    b = _burst()
    for ts in times:
        s = int(ts * SR - len(b) // 2)
        e = s + len(b)
        if 0 <= s and e <= out.shape[0]:
            out[s:e] += np.stack([b, b], axis=1)
    return out.astype(np.float32)


def _syllable(seed: int = 0, dur: float = 0.12) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    tone = (np.sin(2.0 * np.pi * 700 * t) + np.sin(2.0 * np.pi * 1050 * t)) / 2
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 14.0)
    cons = np.zeros(n)
    m = int(0.02 * SR)
    cons[:m] = rng.standard_normal(m) * 0.4
    return (0.5 * (tone * env + cons * np.exp(-t * 60.0))).astype(np.float32)


def make_clip(name: str) -> tuple[np.ndarray, dict]:
    """Return (vocal-stereo mix, base_stems dict)."""
    if name == "clean-mix":
        mix = _bed(4.0)
    elif name == "metronome-present":
        mix = _with_clicks(_bed(4.5), [i * 0.5 for i in range(8)])
    elif name == "metronome-absent":
        mix = _bed(4.0)
    elif name == "count-in-vs-sung":
        n = SR * 4
        mono = np.zeros(n, dtype=np.float32)
        for i, t0 in enumerate([0.5, 1.0, 1.5, 2.0]):
            s = _syllable(seed=i)
            s0 = int(t0 * SR)
            mono[s0:s0 + len(s)] += s
        # sung bed underneath at a different band
        t = np.arange(n) / SR
        mono += (0.10 * np.sin(2.0 * np.pi * 392 * t)).astype(np.float32)
        mix = np.stack([mono, mono], axis=1).astype(np.float32)
    elif name == "strings-plus-synth-overlap":
        n = int(SR * 4)
        t = np.arange(n) / SR
        vib = 1 + 0.06 * np.sin(2.0 * np.pi * 5.5 * t)
        m = 0.30 * np.sin(2.0 * np.pi * 523.25 * t) * vib
        m += 0.22 * np.sin(2.0 * np.pi * 220 * t + 0.5 * np.sin(2.0 * np.pi * 2 * t))
        m += 0.05 * np.sin(2.0 * np.pi * 110 * t)
        mix = np.stack([m, m], axis=1).astype(np.float32)
    elif name == "tempo-change-rubato":
        times = [i * 0.5 for i in range(16)] + [8.0 + i * (2.0 / 3.0) for i in range(12)]
        mix = _with_clicks(_bed(16.5), times)
    else:
        raise KeyError(name)
    n = mix.shape[0]
    base = {k: np.zeros((n, 2), dtype=np.float32) for k in ("vocals", "drums", "bass", "other")}
    base["other"] = (mix * 0.25).astype(np.float32)
    base["vocals"] = mix.copy()
    return mix, base


def _run_clip(name: str, tmp: Path) -> dict:
    import json

    import pipeline

    mix, base = make_clip(name)
    src = tmp / f"{name}.wav"
    sf.write(str(src), mix, SR)
    out_dir = tmp / f"job-{name}"
    pipeline.run_job(src, out_dir, base_stems=base)
    out: dict = {"dir": out_dir}
    for stem in ("vocal", "drums", "bass", "piano", "guitar",
                 "strings", "synth", "cue", "click"):
        p = out_dir / f"{stem}.wav"
        assert p.exists(), f"{name}: missing {stem}.wav"
        audio, _ = sf.read(str(p), dtype="float32")
        out[stem] = np.asarray(audio)
    out["manifest"] = json.loads((out_dir / "manifest.json").read_text())
    out["qa"] = json.loads((out_dir / "qa.json").read_text())
    out["tempo"] = json.loads((out_dir / "tempo_map.json").read_text())
    out["mix"] = mix
    return out


def _silent(audio: np.ndarray) -> bool:
    return float(np.max(np.abs(audio))) == 0.0


def test_golden_six_clips_green_with_locked_floors():
    import qa as qa_module

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, floor in FLOORS.items():
            got = _run_clip(name, tmp)
            # 9 stems sum to the mix (cue excluded: subset of vocal).
            from stemspec import STEM_NAMES
            total = sum(np.asarray(got[s], dtype=np.float64)
                        for s in STEM_NAMES if s != "cue")
            res, _ = sf.read(str(got["dir"] / "residual.wav"), dtype="float32")
            recon = total + np.asarray(res, dtype=np.float64)
            mix = np.asarray(got["mix"], dtype=np.float64)
            err = float(np.sqrt(np.mean((recon - mix) ** 2)))
            sig = float(np.sqrt(np.mean(mix ** 2)))
            assert err / max(sig, 1e-8) < 0.05, f"{name}: residual {err/sig:.4f}"
            # Silence-if-absent gates.
            if floor["click_present"]:
                assert got["manifest"]["stems"]["click"]["present"] is True, name
                assert not _silent(got["click"]), name
            else:
                assert _silent(got["click"]), f"{name}: click not silent"
                assert got["manifest"]["stems"]["click"]["present"] is False, name
            if floor["cue_present"]:
                assert got["manifest"]["stems"]["cue"]["present"] is True, name
            else:
                assert _silent(got["cue"]), f"{name}: cue not silent"
                assert got["manifest"]["stems"]["cue"]["present"] is False, name
            # SI-SDR floors lock per-clip means within tolerance of qa;
            # regression fails if a future change moves any mean > TOL_DB
            # vs the recorded baseline in test_golden_floors below.
            scored = qa_module.score_stems(
                {s: got[s] for s in ("piano", "guitar", "strings", "synth")})
            for stem, entry in scored.items():
                assert isinstance(entry["sisdr_db"], float), (name, stem)


def test_golden_floors_match_recorded_baseline():
    import qa as qa_module

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, expected in RECORDED_MEANS.items():
            got = _run_clip(name, tmp)
            scored = qa_module.score_stems(
                {s: got[s] for s in ("piano", "guitar", "strings", "synth")})
            for stem, floor in expected.items():
                mean = scored[stem]["sisdr_db"]
                assert abs(mean - floor) <= TOL_DB, (
                    f"{name}/{stem}: {mean} vs recorded {floor}")


# Measured once on this deterministic synth (3 runs, spread 0.0,
# limiter none). Update only by re-recording with run count + spread.
RECORDED_MEANS: dict[str, dict[str, float]] = {
    "clean-mix": {"piano": 24.73, "guitar": 19.13, "strings": 30.02, "synth": -32.91},
    "metronome-present": {"piano": -5.16, "guitar": -11.52, "strings": 1.06, "synth": -34.92},
    "metronome-absent": {"piano": 24.73, "guitar": 19.13, "strings": 30.02, "synth": -32.91},
    "count-in-vs-sung": {"piano": 1.07, "guitar": -9.17, "strings": 1.07, "synth": -66.88},
    "strings-plus-synth-overlap": {"piano": 8.16, "guitar": 12.44, "strings": 5.02, "synth": -37.29},
    "tempo-change-rubato": {"piano": -5.38, "guitar": -11.77, "strings": 0.84, "synth": -34.69},
}
