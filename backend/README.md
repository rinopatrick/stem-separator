"""Stem Separator backend: FastAPI + RQ job queue + SSE progress.

3-tier inference: ROCm-gfx1151 (experimental) -> ONNX-DML (disabled for
Demucs: proven OOM on Strix Halo 2026-10-24) -> CPU onnxruntime (proven:
1.5s per 7.8s chunk, RTF ~0.2).
"""
