"""T3 red: pipeline vocal flag + autotune in manifest; A/B vocal scores."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100
DUR = 4


def _mix_file(td: Path) -> Path:
    t = np.arange(SR * DUR) / SR
    m = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 110 * t)
    stereo = np.stack([m, m], axis=1).astype(np.float32)
    src = td / "mix.wav"
    sf.write(str(src), stereo, SR)
    return src


def _base(n: int) -> dict:
    rng = np.random.default_rng(11)
    base = {}
    for k in ("vocals", "drums", "bass", "other"):
        base[k] = (rng.standard_normal((n, 2)) * 0.1).astype(np.float32)
    return base


def test_pipeline_vocal_upgrade_toggleable_with_fallback():
    import pipeline

    n = SR * DUR
    base = _base(n)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _mix_file(td)
        out_off = td / "off"
        r = pipeline.run_job(src, out_off, base_stems={k: v.copy() for k, v in base.items()},
                             use_vocal_upgrade=False)
        assert r["vocal_info"]["fallback"] is False
        v_off, _ = sf.read(str(out_off / "vocal.wav"), dtype="float32")
        out_on = td / "on"
        r = pipeline.run_job(src, out_on, base_stems={k: v.copy() for k, v in base.items()},
                             use_vocal_upgrade=True)
        assert r["vocal_info"]["fallback"] is False
        assert r["vocal_info"]["model_id"] == "kj-melband-smank-onnx"
        v_on, _ = sf.read(str(out_on / "vocal.wav"), dtype="float32")
        assert not np.array_equal(v_off, v_on)
        out_oom = td / "oom"
        r = pipeline.run_job(src, out_oom, base_stems={k: v.copy() for k, v in base.items()},
                             use_vocal_upgrade=True, force_oom=True)
        assert r["vocal_info"]["fallback"] is True
        v_oom, _ = sf.read(str(out_oom / "vocal.wav"), dtype="float32")
        assert np.array_equal(np.asarray(v_oom), np.asarray(v_off))


def test_pipeline_manifest_logs_autotune_per_model():
    import pipeline

    n = SR * DUR
    base = _base(n)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _mix_file(td)
        out_dir = td / "job"
        pipeline.run_job(src, out_dir, base_stems=base)
        doc = json.loads((out_dir / "manifest.json").read_text())
        assert "autotune" in doc
        assert set(doc["autotune"]) == {"htdemucs_6s", "kj-melband-smank-onnx"}
        for model, entry in doc["autotune"].items():
            assert isinstance(entry["fp16"], bool)
            assert entry["tile"] > 0
        assert doc["tile"] == doc["autotune"][doc["model"]]["tile"]


def test_ab_vocal_scores_recorded_with_spread_and_limiter():
    import pipeline

    n = SR * DUR
    base = _base(n)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _mix_file(td)
        doc = pipeline.vocal_ab_doc(base, sr=SR, src=src)
        assert doc["clip"] == "clean-mix"
        for variant in ("base", "upgraded"):
            entry = doc["stems"][variant]
            for key in ("sisdr_db", "spread_db", "runs", "limiter"):
                assert key in entry, f"{variant} missing {key}"
            assert entry["runs"] >= 1
        assert "clips" in doc and len(doc["clips"]) == 2


def test_pipeline_writes_ab_vocal_json():
    import pipeline

    n = SR * DUR
    base = _base(n)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _mix_file(td)
        out_dir = td / "job"
        pipeline.run_job(src, out_dir, base_stems=base)
        p = out_dir / "ab_vocal.json"
        assert p.exists()
        doc = json.loads(p.read_text())
        assert set(doc["stems"]) == {"base", "upgraded"}
