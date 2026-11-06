"""T3 red: vocal upgrade toggleable with clean fallback to 6s vocal."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100


def _base_vocal(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    m = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    return np.stack([m, m], axis=1).astype(np.float32)


def test_vocal_flag_off_returns_base_exactly():
    import vocal

    base = _base_vocal(SR * 2)
    mix = base.copy()
    out, info = vocal.separate_vocal(mix, SR, base, use_upgrade=False)
    assert np.array_equal(np.asarray(out), np.asarray(base))
    assert info["fallback"] is False
    assert info["model_id"] == vocal.BASE_VOCAL_MODEL_ID


def test_vocal_flag_on_returns_upgraded_shape_and_model():
    import vocal

    base = _base_vocal(SR * 2)
    mix = base.copy()
    out, info = vocal.separate_vocal(mix, SR, base, use_upgrade=True)
    assert np.asarray(out).shape == np.asarray(base).shape
    assert info["model_id"] == vocal.VOCAL_MODEL_ID
    assert info["fallback"] is False


def test_vocal_forced_oom_falls_back_to_base():
    import vocal

    base = _base_vocal(SR * 2)
    mix = base.copy()
    out, info = vocal.separate_vocal(mix, SR, base, use_upgrade=True, force_oom=True)
    assert np.array_equal(np.asarray(out), np.asarray(base))
    assert info["fallback"] is True
    assert info["model_id"] == vocal.BASE_VOCAL_MODEL_ID
