"""FastAPI app: POST /jobs, SSE GET /jobs/{id}, GET stems, GET /jobs, DELETE."""
import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from redis import Redis
from rq import Queue
from rq.job import Job

import config

app = FastAPI(title="Stem Separator")
redis = Redis.from_url(config.REDIS_URL)
queue = Queue("separate", connection=redis)


@app.get("/health")
def health():
    return {"ok": True, "stems": config.STEMS}


@app.post("/jobs", status_code=202)
def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "upload").suffix.lower() or ".wav"
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff"}:
        raise HTTPException(400, f"unsupported format: {ext}")
    dest = config.UPLOAD_DIR / f"{job_id}{ext}"
    dest.write_bytes(file.file.read())
    job = queue.enqueue(worker_job.job_separate, job_id, dest.name, job_id=job_id)
    return {"job_id": job.id, "status": "queued"}


def _status(job: Job) -> dict:
    if job.is_queued:
        st = "queued"
    elif job.is_started:
        st = "running"
    elif job.is_finished:
        st = "done"
    elif job.is_failed:
        st = "error"
    else:
        st = job.get_status() or "unknown"
    d = {"job_id": job.id, "status": st, "progress": job.meta.get("progress", 0.0)}
    if job.is_failed and job.exc_info:
        d["error"] = job.exc_info.splitlines()[-1][:300]
    if job.is_finished and job.result:
        d["stems"] = job.result.get("stems", [])
    return d


@app.get("/jobs/{job_id}")
async def job_events(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis)
    except Exception:
        raise HTTPException(404, "job not found")

    async def gen():
        while True:
            job2 = Job.fetch(job_id, connection=redis)
            yield f"data: {json.dumps(_status(job2))}\n\n"
            if job2.is_finished or job2.is_failed:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/jobs")
def list_jobs():
    # RQ has no built-in history; finished registry keeps last N.
    from rq.registry import FinishedJobRegistry

    reg = FinishedJobRegistry("separate", connection=redis)
    out = []
    for jid in reg.get_job_ids()[:50]:
        try:
            out.append(_status(Job.fetch(jid, connection=redis)))
        except Exception:
            continue
    return {"jobs": out}


@app.get("/jobs/{job_id}/stems/{stem}")
def download_stem(job_id: str, stem: str):
    if stem not in config.STEMS and stem != "residual":
        raise HTTPException(404, "unknown stem")
    p = config.RESULT_DIR / job_id / f"{stem}.wav"
    if not p.exists():
        raise HTTPException(404, "stem not ready")
    return FileResponse(str(p), media_type="audio/wav", filename=f"{stem}.wav")


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis)
    except Exception:
        raise HTTPException(404, "job not found")
    if job.is_queued or job.is_started:
        job.cancel()
        return {"job_id": job_id, "status": "cancelled"}
    raise HTTPException(409, f"cannot cancel job in state {job.get_status()}")
