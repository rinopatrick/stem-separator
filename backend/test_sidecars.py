"""T1 red: manifest.json + tempo_map.json sidecar schemas on disk."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _write_manifest(out_dir: Path, n_frames: int):
    import sidecars

    stems = {
        name: np.zeros((n_frames, 2), dtype=np.float32)
        for name in [
            "vocal",
            "drums",
            "bass",
            "piano",
            "guitar",
            "strings",
            "synth",
            "cue",
            "click",
        ]
    }
    p = sidecars.write_manifest(out_dir, stems, sr=44100)
    return p


def _write_tempo_map(out_dir: Path, n_frames: int):
    import sidecars

    p = sidecars.write_tempo_map(out_dir, n_frames=n_frames, sr=44100)
    return p


def test_manifest_schema():
    import sidecars  # noqa: F401

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        n = 44100 * 2
        p = _write_manifest(out, n)
        assert p.exists()
        doc = json.loads(p.read_text())
        assert set(doc["stems"]) == {
            "vocal",
            "drums",
            "bass",
            "piano",
            "guitar",
            "strings",
            "synth",
            "cue",
            "click",
        }
        for name, entry in doc["stems"].items():
            for key in ("file", "present", "confidence", "source", "reason"):
                assert key in entry, f"{name} missing {key}"
            assert entry["file"] == f"{name}.wav"
        assert doc["stems"]["cue"]["present"] is False
        assert doc["stems"]["click"]["present"] is False
        assert doc["stems"]["cue"]["confidence"] <= 0.2
        assert doc["stems"]["click"]["confidence"] <= 0.2
        assert "model" in doc and "tile" in doc


def test_tempo_map_schema():
    import sidecars  # noqa: F401

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        p = _write_tempo_map(out, 44100 * 2)
        assert p.exists()
        doc = json.loads(p.read_text())
        for key in (
            "beats",
            "downbeats",
            "bpm_over_time",
            "meter",
            "confidence_global",
            "present",
            "reason",
        ):
            assert key in doc, f"missing {key}"
        assert doc["downbeats"] is None
        assert doc["present"] is False
        assert isinstance(doc["beats"], list)
