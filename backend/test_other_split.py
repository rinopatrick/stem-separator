"""T2 red: stage-2 other-split — piano/guitar/strings specialists, synth leftover."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100


def _other(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 2)) * 0.2).astype(np.float32)


def test_other_split_sums_to_other_by_construction():
    import other_split

    n = SR * 2
    other = _other(n)
    parts = other_split.split_other(other)
    assert set(parts) == {"piano", "guitar", "strings", "synth"}
    total = sum(parts[k] for k in ("piano", "guitar", "strings", "synth"))
    assert np.allclose(total, other, atol=1e-5)


def test_specialist_slots_come_from_registry_models_not_base():
    import other_split
    from stemspec import BY_NAME

    assert BY_NAME["piano"].model_id == "mvsep-scnet-large"
    assert BY_NAME["guitar"].model_id == "bs-roformer-viperx"
    assert BY_NAME["strings"].model_id == "mvsep-bs-roformer-2025.09"
    assert BY_NAME["synth"].model_id == "other-minus-specialists"
    n = SR
    other = _other(n)
    parts = other_split.split_other(other)
    for name in ("piano", "guitar", "strings"):
        entry = other_split.split_report(other)[name]
        assert entry["model_id"] == BY_NAME[name].model_id
        assert entry["source"] == "other"
        assert parts[name].shape == other.shape


def test_overlap_clip_scores_recorded_with_spread_and_limiter():
    import json

    import other_split

    sr = SR
    n = sr * 4
    t = np.arange(n) / sr
    piano_tone = np.sin(2 * np.pi * 261.63 * t)
    guitar_tone = np.sin(2 * np.pi * 329.63 * t)
    other = np.stack([piano_tone + guitar_tone, piano_tone + guitar_tone], axis=1)
    other = (other * 0.3).astype(np.float32)
    parts = other_split.split_other(other)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        p = other_split.write_overlap_report(out, parts, sr=sr)
        doc = json.loads(p.read_text())
        assert doc["clip"] == "strings-plus-synth-overlap"
        for name in ("piano", "guitar", "strings", "synth"):
            entry = doc["stems"][name]
            for key in ("sisdr_db", "spread_db", "runs", "limiter"):
                assert key in entry, f"{name} missing {key}"
            assert entry["runs"] >= 1
