"""RQ job: separate one upload, report progress 0..1 via job.meta."""
from pathlib import Path

from rq import get_current_job

import config
from pipeline import run_job


def job_separate(job_id: str, src_name: str) -> dict:
    job = get_current_job()
    src = config.UPLOAD_DIR / src_name
    out_dir = config.RESULT_DIR / job_id
    aud_dur = 0.0
    try:
        import soundfile as sf

        info = sf.info(str(src))
        aud_dur = info.frames / info.samplerate
    except Exception:
        pass
    if aud_dur > config.MAX_MINUTES * 60:
        raise ValueError(f"file too long: {aud_dur:.0f}s > {config.MAX_MINUTES:.0f}min")

    def cb(frac: float):
        if job:
            job.meta["progress"] = round(frac, 3)
            job.save_meta()

    paths = run_job(src, out_dir, progress_cb=cb)
    if job:
        job.meta["progress"] = 1.0
        job.save_meta()
    return {"stems": paths["stems"]}
