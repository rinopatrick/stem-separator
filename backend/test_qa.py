"""T2 red: qa.json writer — per-stem SI-SDR estimates, leakage matrix, clip flags."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100


def _stems(n: int) -> dict:
    rng = np.random.default_rng(7)
    return {
        name: rng.standard_normal((n, 2), dtype=np.float64) * 0.1
        for name in [
            "vocal", "drums", "bass", "piano",
            "guitar", "strings", "synth", "cue", "click",
        ]
    }


def test_qa_json_schema_on_disk():
    import qa

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        stems = _stems(SR * 2)
        p = qa.write_qa(out, stems, sr=SR)
        assert p.exists()
        doc = json.loads(p.read_text())
        assert set(doc["stems"]) == set(stems)
        for name, entry in doc["stems"].items():
            for key in ("sisdr_db", "spread_db", "runs", "limiter"):
                assert key in entry, f"{name} missing {key}"
            assert isinstance(entry["runs"], int) and entry["runs"] >= 1
        assert "leakage" in doc
        for pair, val in doc["leakage"].items():
            a, b = pair.split("|")
            assert a in stems and b in stems
            assert -1.0 <= val <= 1.0
        assert "clips" in doc and isinstance(doc["clips"], list)
        for clip in doc["clips"]:
            for key in ("stem", "peak", "flagged"):
                assert key in clip, f"clip missing {key}"


def test_qa_clip_flag_fires_on_hot_stem():
    import qa

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        stems = _stems(SR)
        stems["drums"] = np.ones((SR, 2)) * 1.2
        qa.write_qa(out, stems, sr=SR)
        doc = json.loads((out / "qa.json").read_text())
        drums = [c for c in doc["clips"] if c["stem"] == "drums"]
        assert len(drums) == 1 and drums[0]["flagged"] is True
        assert drums[0]["peak"] >= 1.0


def test_qa_leakage_sees_correlated_stems():
    import qa

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        n = SR
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 440 * t)
        stems = _stems(n)
        stems["piano"] = np.stack([tone, tone], axis=1) * 0.5
        stems["guitar"] = np.stack([tone, tone], axis=1) * 0.5
        qa.write_qa(out, stems, sr=SR)
        doc = json.loads((out / "qa.json").read_text())
        assert doc["leakage"]["guitar|piano"] > 0.9
