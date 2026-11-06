# Stem Separator

Web desktop: upload lagu → pisah 6-stem (vocals, drums, bass, piano, guitar, other) → preview per stem.

## Hasil riset + spike

- Engine: ensemble per-stem — vokal Mel-Band RoFormer, drum/bass/other Demucs `htdemucs_ft`,
  gitar/piano Banquet atau fine-tune `htdemucs_6s`. Detail: `.scratch/stem-separator/research/01-engine-6stem.md`
- Spike Strix Halo 8060S (2026-10-26): ONNX-DML **OOM** di `htdemucs_6s`; **CPU 1.5s/chunk 7.8s
  (RTF ~0.2)** — lagu 4 menit ≈ 45 detik. Worker default CPU.
- Cloud: web di Vercel gratis, worker HF Spaces / Modal ($30 kredit/bln) / RunPod (~$0,01/lagu).
  Detail: `.scratch/stem-separator/research/05-cloud-gratis.md`
- Peta keputusan: `.scratch/stem-separator/map.md`

## Jalankan lokal

Butuh Redis (antrean RQ):

```bash
docker run -d --name stem-redis -p 6379:6379 redis:7-alpine
```

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --port 8000        # API
rq worker separate                  # worker inference (proses terpisah)
```

Frontend:

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

E2E tanpa Redis (CPU inference beneran, audio sintetik 10 detik):

```bash
cd backend && python e2e_test.py    # E2E PASS = 6 WAV keluar
```

## API

- `POST /jobs` (multipart `file`) → `{job_id, status: queued}`
- `GET /jobs/{id}` — SSE: `{job_id, status, progress, stems?, error?}`
- `GET /jobs/{id}/stems/{stem}` — WAV per stem
- `GET /jobs` — riwayat 50 job terakhir
- `DELETE /jobs/{id}` — cancel antrean
