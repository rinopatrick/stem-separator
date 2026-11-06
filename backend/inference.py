"""Inference worker: chunked ONNX separation, CPU-first.

Spike result 2026-10-26 (Strix Halo 8060S): DML OOM on htdemucs_6s,
CPU 1.5s/chunk. So default provider list is CPU-only; DML stays
disabled until a model small enough to fit is chosen.
"""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import hf_hub_download

from config import BASE_STEMS, CHUNK_SAMPLES, MODEL_FILE, MODEL_ID, SR

_session = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        mp = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE)
        # CPU proven on Strix Halo; DML OOMs on htdemucs_6s (spike 2026-10-26).
        _session = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    return _session


def separate_file(src: Path, out_dir: Path, progress_cb=None) -> list[Path]:
    sess = get_session()
    inp_name = sess.get_inputs()[0].name
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if sr != SR:
        import soxr

        audio = soxr.resample(audio.T, sr, SR).T
    audio = audio.T  # (ch, n)
    n = audio.shape[1]
    outs = [np.zeros_like(audio) for _ in BASE_STEMS]
    counts = np.zeros(n, dtype=np.float32)
    n_chunks = (n + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    for i in range(n_chunks):
        s = i * CHUNK_SAMPLES
        e = min(s + CHUNK_SAMPLES, n)
        chunk = np.zeros((1, 2, CHUNK_SAMPLES), dtype=np.float32)
        L = e - s
        chunk[0, :, :L] = audio[:, s:e]
        (stems,) = sess.run(None, {inp_name: chunk})  # (1,6,2,N)
        for k in range(len(BASE_STEMS)):
            outs[k][:, s:e] += stems[0, k, :, :L]
        counts[s:e] += 1
        if progress_cb:
            progress_cb((i + 1) / n_chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, name in enumerate(BASE_STEMS):
        mix = outs[k] / np.maximum(counts, 1)
        p = out_dir / f"{name}.wav"
        sf.write(str(p), mix.T, SR)
        paths.append(p)
    return paths
