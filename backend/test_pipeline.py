"""T1 red: job-output file contract — 9 stems + residual + sidecars."""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100
DUR = 4


def test_job_output_contract_without_model():
    """End-to-end shape without loading ONNX: every job writes 9 stem
    WAVs plus residual.wav, manifest.json, tempo_map.json with valid
    duration and channels. Uses pipeline helper, not the model."""
    import pipeline

    src = Path(__file__).resolve().parent / "_t1_probe.wav"
    t = np.arange(SR * DUR) / SR
    mix = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)
    stereo = np.stack([mix, mix], axis=1).astype(np.float32)
    import soundfile as sf

    sf.write(str(src), stereo, SR)
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "job"
            result = pipeline.run_job(src, out_dir)
            assert result["sr"] == SR
            for name in result["stems"]:
                p = out_dir / f"{name}.wav"
                assert p.exists(), f"missing {name}.wav"
                with wave.open(str(p), "rb") as w:
                    dur = w.getnframes() / w.getframerate()
                    assert abs(dur - DUR) < 0.1, f"{name} dur {dur}"
                    assert w.getnchannels() == 2, f"{name} not stereo"
            for sidecar in ("manifest.json", "tempo_map.json", "residual.wav"):
                assert (out_dir / sidecar).exists(), f"missing {sidecar}"
            with wave.open(str(out_dir / "residual.wav"), "rb") as w:
                dur = w.getnframes() / w.getframerate()
                assert abs(dur - DUR) < 0.1
                assert w.getnchannels() == 2
            import json

            manifest = json.loads((out_dir / "manifest.json").read_text())
            assert set(manifest["stems"]) == set(result["stems"])
            tempo = json.loads((out_dir / "tempo_map.json").read_text())
            assert "beats" in tempo and "bpm_over_time" in tempo
    finally:
        src.unlink(missing_ok=True)
