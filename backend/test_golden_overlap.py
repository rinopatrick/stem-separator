"""T2: golden overlap clip scores recorded with run count, spread, limiter.

Locks the first measured SI-SDR estimates for the specialist
other-split. Deterministic splitter, so spread is 0.0 across runs.
Regression fails if a future change moves any mean by >0.5 dB.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100
DUR = 8
N_RUNS = 3
TOL_DB = 0.5

EXPECTED_MEANS = {
    "piano": 8.99,
    "guitar": -13.55,
    "strings": 1.41,
    "synth": -45.57,
}
EXPECTED_SPREAD = 0.0
EXPECTED_LIMITER = "none"
EXPECTED_FLAGGED: list = []


def _overlap_other() -> np.ndarray:
    n = SR * DUR
    t = np.arange(n) / SR
    vib = 1 + 0.06 * np.sin(2 * np.pi * 5.5 * t)
    strings = 0.30 * np.sin(2 * np.pi * 523.25 * t) * vib
    strings = strings + 0.18 * np.sin(2 * np.pi * 659.25 * t) * vib
    synth = 0.22 * np.sin(2 * np.pi * 220 * t + 0.5 * np.sin(2 * np.pi * 2 * t))
    synth = synth + 0.18 * np.sin(2 * np.pi * 880 * t)
    m = strings + synth + 0.05 * np.sin(2 * np.pi * 110 * t)
    return np.stack([m, m], axis=1).astype(np.float32)


def test_golden_overlap_scores_with_spread_and_limiter():
    import other_split
    import qa as qa_module

    other = _overlap_other()
    per_run = []
    for _ in range(N_RUNS):
        parts = other_split.split_other(other, sr=SR)
        per_run.append(
            {k: v["sisdr_db"] for k, v in qa_module.score_stems(parts).items()}
        )
    for name, expected in EXPECTED_MEANS.items():
        vals = [r[name] for r in per_run]
        mean = float(np.mean(vals))
        spread = float(np.std(vals))
        assert abs(mean - expected) <= TOL_DB, f"{name} {mean} != {expected}"
        assert spread <= 1e-9
    parts = other_split.split_other(other, sr=SR)
    clips = qa_module.score_clips(parts)
    flagged = sorted(c["stem"] for c in clips if c["flagged"])
    assert flagged == EXPECTED_FLAGGED

    scored = qa_module.score_stems(parts, runs=N_RUNS, limiter=EXPECTED_LIMITER)
    assert scored["piano"]["runs"] == N_RUNS
    assert scored["piano"]["limiter"] == EXPECTED_LIMITER
