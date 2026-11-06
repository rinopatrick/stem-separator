"""E2E: 10s synth -> job_separate -> 9 stems + residual + sidecars."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def check_wav(p: Path) -> bool:
    with wave.open(str(p), "rb") as w:
        dur = w.getnframes() / w.getframerate()
        ch = w.getnchannels()
        print(f"  {p.name}: {dur:.1f}s, {ch}ch, {p.stat().st_size // 1024}KB")
        return abs(dur - DUR) < 0.5 and ch == 2


def main():
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    src = config.UPLOAD_DIR / "e2e_test.wav"
    make_audio(src)
    print(f"input: {src} ({DUR}s synth)")
    out = job_separate("e2e_test", src.name)
    print("stems:", out["stems"])
    ok = True
    expected = [
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
    if out["stems"] != expected:
        print(f"FAIL: stems {out['stems']} != {expected}")
        ok = False
    for name in expected + ["residual"]:
        p = config.RESULT_DIR / "e2e_test" / f"{name}.wav"
        if not p.exists():
            print(f"FAIL: missing {p}")
            ok = False
            continue
        if not check_wav(p):
            print(f"FAIL: bad audio {p}")
            ok = False
    for sidecar in ("manifest.json", "tempo_map.json", "qa.json"):
        p = config.RESULT_DIR / "e2e_test" / sidecar
        if not p.exists():
            print(f"FAIL: missing {p}")
            ok = False
            continue
        doc = json.loads(p.read_text())
        print(f"  {sidecar}: keys={sorted(doc)[:6]}")
    manifest = json.loads((config.RESULT_DIR / "e2e_test" / "manifest.json").read_text())
    if set(manifest["stems"]) != set(expected):
        print("FAIL: manifest stems mismatch")
        ok = False
    qa_doc = json.loads((config.RESULT_DIR / "e2e_test" / "qa.json").read_text())
    if set(qa_doc["stems"]) != set(expected):
        print("FAIL: qa stems mismatch")
        ok = False
    if "leakage" not in qa_doc or "clips" not in qa_doc:
        print("FAIL: qa missing leakage/clips")
        ok = False
    tempo = json.loads((config.RESULT_DIR / "e2e_test" / "tempo_map.json").read_text())
    for key in ("beats", "downbeats", "bpm_over_time", "meter",
                "confidence_global", "confidence", "present"):
        if key not in tempo:
            print(f"FAIL: tempo_map missing {key}")
            ok = False
    # T4 silence-if-absent: synth bed has no metronome, so the E2E
    # click must arrive silent with present false and low confidence.
    import soundfile as _sf
    e2e_click, _ = _sf.read(
        str(config.RESULT_DIR / "e2e_test" / "click.wav"), dtype="float32")
    if float(np.max(np.abs(e2e_click))) != 0.0:
        print("FAIL: e2e click not silent on metronome-free bed")
        ok = False
    if tempo.get("present") is not False:
        print("FAIL: e2e tempo present should be False")
        ok = False
    if tempo.get("confidence_global", 1.0) > 0.2:
        print("FAIL: e2e tempo confidence above 0.2")
        ok = False
    if manifest["stems"]["click"]["present"] is not False:
        print("FAIL: e2e manifest click present should be False")
        ok = False
    if manifest["stems"]["click"]["confidence"] > 0.2:
        print("FAIL: e2e manifest click confidence above 0.2")
        ok = False
    print("E2E PASS" if ok else "E2E FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
