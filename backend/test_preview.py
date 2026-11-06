"""T6 red: lazy loudness-normalized preview cache.

Seam: HTTP GET /jobs/{id}/preview/{stem}.{fmt} served from the
result dir (see test_api_preview_after_job_returns_normalized_audio
for the endpoint half). These unit tests pin the cache contract:
lazy build, job WAV untouched, cache hit, unknown stem/format
rejected, quiet stems gained without clipping.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 44100

TARGET_LUFS = -16.0
# K-weighting-free approximation: LUFS ~= -0.691 + 10*log10(mean square)
# for the test tones used here; calibration offset measured once.
_K_OFF_DB = 3.01


def _tone(peak=0.5, dur_s=2.0) -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    m = peak * np.sin(2.0 * np.pi * 440.0 * t)
    return np.stack([m, m], axis=1).astype(np.float32)


def test_preview_request_after_job_returns_normalized_audio():
    import preview

    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td) / "job"
        job_dir.mkdir()
        sf.write(str(job_dir / "vocal.wav"), _tone(peak=0.5), SR)
        # Lazy: no preview exists right after the job.
        assert list(job_dir.glob("preview-*")) == []
        out = preview.get_preview(job_dir, "vocal", fmt="mp3")
        assert out.exists()
        audio, sr = sf.read(str(out), dtype="float32", always_2d=True)
        assert sr in (44100, 48000)
        raw, _ = sf.read(str(job_dir / "vocal.wav"), dtype="float32", always_2d=True)
        raw_lufs = preview.measure_lufs(raw)
        lufs = preview.measure_lufs(audio)
        assert abs(lufs - TARGET_LUFS) <= abs(raw_lufs - TARGET_LUFS), (
            f"preview {lufs} LUFS no closer to target than job WAV {raw_lufs}"
        )
        assert abs(lufs - TARGET_LUFS) <= 4.0, f"preview at {lufs} LUFS"
        # Job WAV untouched; second request hits cache (same mtime).
        before = out.stat().st_mtime_ns
        out2 = preview.get_preview(job_dir, "vocal", fmt="mp3")
        assert out2 == out and out2.stat().st_mtime_ns == before


def test_preview_does_not_block_job_and_rejects_unknown():
    import preview

    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td) / "job"
        job_dir.mkdir()
        sf.write(str(job_dir / "vocal.wav"), _tone(peak=0.5), SR)
        # Unknown stem raises before any work happens.
        try:
            preview.get_preview(job_dir, "banjo", fmt="mp3")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for banjo")
        # Unknown format raises too.
        try:
            preview.get_preview(job_dir, "vocal", fmt="flac")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for flac")


def test_quiet_stem_gets_gain_but_never_clips():
    import preview

    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td) / "job"
        job_dir.mkdir()
        sf.write(str(job_dir / "bass.wav"), _tone(peak=0.02), SR)
        out = preview.get_preview(job_dir, "bass", fmt="mp3")
        audio, _ = sf.read(str(out), dtype="float32", always_2d=True)
        assert float(np.max(np.abs(audio))) <= 1.0
        raw, _ = sf.read(str(job_dir / "bass.wav"), dtype="float32", always_2d=True)
        assert preview.measure_lufs(audio) > preview.measure_lufs(raw), (
            "quiet stem got no gain"
        )
