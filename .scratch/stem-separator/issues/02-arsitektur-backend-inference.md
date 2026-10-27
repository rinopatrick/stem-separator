Status: resolved
Type: grilling
Blocked by: 01

## Question

Arsitektur backend inference seperti apa (framework, API shape, antrean, batas, siap-cloud)? Lihat Answer.

## Answer

FastAPI + Redis/RQ, SSE progress realtime, storage abstraction siap-S3, retry + cancel job. Kontrak: POST /jobs (upload) → GET /jobs/{id} (SSE status+progress) → unduh per-stem WAV + riwayat GET /jobs. Batas v1: 1 job/worker, maks ±10 mnt/file, antre sisanya.

## Adendum GPU (Strix Halo 8060S/gfx1151, 2026-10-24)

Mesin target: Ryzen AI MAX+ 395 + Radeon 8060S, Windows. PyTorch CUDA tidak tersedia. Jalur GPU: ROCm-gfx1151 Windows masih preview/nightly (belum stabil produksi); DirectML: Demucs gagal fused-LSTM, RoFormer fallback CPU (audio-separator v0.44.5 CI), MDX/VR jalan GPU. ONNX DmlExecutionProvider untuk Demucs-ONNX belum terbukti di AMD — butuh spike test di mesin target. Worker wajib 3-tier auto-detect: ROCm-gfx1151 (eksperimen) → ONNX-DML → CPU fallback. CPU 16-core Strix Halo tetap usable sebagai baseline.
