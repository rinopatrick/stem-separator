"""E2E: upload sintetik 10 detik -> antrean sync (fakeredis) -> 6 WAV keluar."""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakeredis  # noqa: E402

import config  # noqa: E402
from worker_job import job_separate  # noqa: E402

SR = 44100
DUR = 10


def make_audio(p: Path):
    t = np.arange(SR * DUR) / SR
    mix = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)
    stereo = np.stack([mix, mix], axis=1).astype(np.float32)
    import soundfile as sf

    sf.write(str(p), stereo, SR)


def main():
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    src = config.UPLOAD_DIR / "e2e_test.wav"
    make_audio(src)
    print(f"input: {src} ({DUR}s synth)")
    out = job_separate("e2e_test", src.name)
    print("stems:", out["stems"])
    ok = True
    for name in out["stems"]:
        p = config.RESULT_DIR / "e2e_test" / name
        with wave.open(str(p), "rb") as w:
            dur = w.getnframes() / w.getframerate()
            ch = w.getnchannels()
            print(f"  {name}: {dur:.1f}s, {ch}ch, {p.stat().st_size // 1024}KB")
            if abs(dur - DUR) > 0.5 or ch != 2:
                ok = False
    print("E2E PASS" if ok else "E2E FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
