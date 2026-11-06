"""T3 red: autotune probe logs model, fp16, tile per job; forced OOM -> CPU."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_autotune_logs_model_fp16_tile_in_manifest():
    import autotune
    import sidecars

    plan = autotune.probe(models=("htdemucs_6s", "kj-melband-smank-onnx"))
    assert set(plan) == {"htdemucs_6s", "kj-melband-smank-onnx"}
    for model, entry in plan.items():
        assert isinstance(entry["fp16"], bool)
        assert entry["tile"] > 0
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        stems = {n: np.zeros((44100, 2), dtype=np.float32) for n in
                 ["vocal", "drums", "bass", "piano", "guitar",
                  "strings", "synth", "cue", "click"]}
        p = sidecars.write_manifest(out, stems, sr=44100, model="x", tile=1,
                                    autotune=plan)
        doc = json.loads(p.read_text())
        assert doc["autotune"]["htdemucs_6s"]["tile"] == plan["htdemucs_6s"]["tile"]
        assert doc["autotune"]["kj-melband-smank-onnx"]["fp16"] == \
            plan["kj-melband-smank-onnx"]["fp16"]


def test_forced_oom_falls_back_to_cpu_and_job_completes():
    import autotune

    plan = autotune.probe(models=("htdemucs_6s",), force_oom=True)
    entry = plan["htdemucs_6s"]
    assert entry["device"] == "cpu"
    assert entry["fp16"] is False
    assert entry["tile"] == autotune.CPU_TILE
    assert entry["fallback"] is True


def test_real_oom_hook_retries_on_cpu():
    import autotune

    plan = autotune.probe(models=("htdemucs_6s",), vram_mb=8000)
    assert plan["htdemucs_6s"]["device"] == "cuda"
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    out, entry = autotune.run_cpu_fallback(flaky, plan["htdemucs_6s"])
    assert out == "ok"
    assert entry["device"] == "cpu"
    assert entry["fallback"] is True
    assert len(calls) == 2
