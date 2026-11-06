"""T1 red: GET /jobs/{id}/stems/{stem} serves 9 names plus residual."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_api_serves_qa_sidecar():
    from fastapi.testclient import TestClient

    import app
    import config

    job_dir = config.RESULT_DIR / "t1_qa_probe"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "qa.json").write_text('{"stems": {}}')
    client = TestClient(app.app)
    try:
        r = client.get("/jobs/t1_qa_probe/sidecars/qa.json")
        assert r.status_code == 200
        assert r.json() == {"stems": {}}
        r = client.get("/jobs/t1_qa_probe/sidecars/manifest.json")
        assert r.status_code == 404
        r = client.get("/jobs/nope/sidecars/nope.json")
        assert r.status_code == 404
    finally:
        (job_dir / "qa.json").unlink(missing_ok=True)
        job_dir.rmdir()


def test_api_allows_nine_plus_residual():
    from fastapi.testclient import TestClient

    import app
    import config

    allowed = set(config.STEMS) | {"residual"}
    assert allowed == {
        "vocal",
        "drums",
        "bass",
        "piano",
        "guitar",
        "strings",
        "synth",
        "cue",
        "click",
        "residual",
    }
    client = TestClient(app.app)
    r = client.get("/jobs/nope/stems/piano")
    assert r.status_code == 404
    assert r.json()["detail"] in ("stem not ready", "unknown stem")
    r = client.get("/jobs/nope/stems/residual")
    assert r.status_code == 404
    assert r.json()["detail"] in ("stem not ready", "unknown stem")
    r = client.get("/jobs/nope/stems/banjo")
    assert r.status_code == 404
    assert r.json()["detail"] == "unknown stem"


def test_api_serves_all_ten_files():
    import wave

    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient

    import app
    import config

    sr = 44100
    job_dir = config.RESULT_DIR / "t1_probe"
    job_dir.mkdir(parents=True, exist_ok=True)
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    names = [
        "vocal",
        "drums",
        "bass",
        "piano",
        "guitar",
        "strings",
        "synth",
        "cue",
        "click",
        "residual",
    ]
    for name in names:
        sf.write(str(job_dir / f"{name}.wav"), stereo, sr)
    client = TestClient(app.app)
    try:
        for name in names:
            r = client.get(f"/jobs/t1_probe/stems/{name}")
            assert r.status_code == 200, name
            assert r.headers["content-type"].startswith("audio/")
    finally:
        for name in names:
            (job_dir / f"{name}.wav").unlink(missing_ok=True)
        job_dir.rmdir()

def test_api_preview_after_job_returns_normalized_audio():
    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient

    import app
    import config

    sr = 44100
    job_dir = config.RESULT_DIR / "t6_preview_probe"
    job_dir.mkdir(parents=True, exist_ok=True)
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    sf.write(str(job_dir / "vocal.wav"), np.stack([tone, tone], axis=1), sr)
    client = TestClient(app.app)
    try:
        assert list(job_dir.glob("preview-*")) == []
        r = client.get("/jobs/t6_preview_probe/preview/vocal.opus")
        assert r.status_code == 200, r.text[:200]
        assert (job_dir / "preview-vocal.opus").exists()
        r = client.get("/jobs/t6_preview_probe/preview/banjo.opus")
        assert r.status_code == 404
        r = client.get("/jobs/t6_preview_probe/preview/vocal.flac")
        assert r.status_code == 404
    finally:
        for p in job_dir.glob("*"):
            p.unlink()
        job_dir.rmdir()
