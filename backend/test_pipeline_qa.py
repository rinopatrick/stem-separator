"""T2: pipeline writes qa.json; 9 stems still sum to the mix."""
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100
DUR = 4


def _mix(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    m = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)
    return np.stack([m, m], axis=1).astype(np.float32)


def test_pipeline_writes_qa_json_with_leakage_and_clips():
    import pipeline

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "mix.wav"
        sf.write(str(src), _mix(SR * DUR), SR)
        out_dir = Path(td) / "job"
        pipeline.run_job(src, out_dir)
        p = out_dir / "qa.json"
        assert p.exists()
        doc = json.loads(p.read_text())
        assert set(doc["stems"]) == {
            "vocal", "drums", "bass", "piano", "guitar",
            "strings", "synth", "cue", "click",
        }
        for name, entry in doc["stems"].items():
            for key in ("sisdr_db", "spread_db", "runs", "limiter"):
                assert key in entry, f"{name} missing {key}"
        assert "leakage" in doc and len(doc["leakage"]) == 36
        assert "clips" in doc and len(doc["clips"]) == 9


def test_pipeline_nine_stems_sum_to_mix():
    import pipeline

    with tempfile.TemporaryDirectory() as td:
        mix = _mix(SR * DUR)
        src = Path(td) / "mix.wav"
        sf.write(str(src), mix, SR)
        out_dir = Path(td) / "job"
        result = pipeline.run_job(src, out_dir)
        total = np.zeros_like(mix, dtype=np.float64)
        for name in result["stems"]:
            with wave.open(str(out_dir / f"{name}.wav"), "rb"):
                pass
            arr, _ = sf.read(str(out_dir / f"{name}.wav"), dtype="float32")
            total += arr.astype(np.float64)
        res, _ = sf.read(str(out_dir / "residual.wav"), dtype="float32")
        recon = total + res.astype(np.float64)
        err = float(np.sqrt(np.mean((recon - mix.astype(np.float64)) ** 2)))
        sig = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2)))
        assert err / max(sig, 1e-8) < 0.05
