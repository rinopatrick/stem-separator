## Destination

Spec siap bangun untuk tool web desktop: upload lagu → pisah 6-stem (vokal, drum, bass, piano, gitar, other) → preview per stem. Lolos bila solo stem minim bleed/artefak di lagu real via uji dengar. Jalur engine bertahap: SOTA pretrained + ensemble dulu, improve belakangan, engine sendiri hanya bila mentok.

## Notes

Domain: audio source separation (Demucs, MDX, RoFormer). Setiap sesi konsultasi skill: grilling, domain-modeling. Preferensi: akurat > cepat; lokal dulu, siap cloud; UI web desktop.

## Decisions so far

<!-- kosong saat charting; diisi saat tiket resolved -->

- [Pilihan engine 6-stem](issues/01-pilihan-engine-6-stem.md): ensemble per-stem (RoFormer vokal, Demucs drum/bass/other, Banquet/fine-tune gitar/piano).
- [Arsitektur backend inference](issues/02-arsitektur-backend-inference.md): FastAPI + Redis/RQ, SSE progress, storage siap-S3, retry + cancel.
- [Kelayakan cloud gratis](issues/05-kelayakan-cloud-gratis.md): Vercel untuk web saja; inference di HF Spaces/Modal/RunPod; laptop mati = antrean cloud tetap jalan.
- [Spec UX web desktop](issues/03-spec-ux-web-desktop.md): 1 layar, 6 row stem + mixer, preview ringan + WAV, ZIP, riwayat; mockup sementara oke.
- [Protokol uji dengar](issues/04-protokol-uji-dengar.md): 20 lagu blind AB, ambang ≥4 (piano/gitar ≥3.5) di ≥17 lagu, penilai tunggal.
- [Jalur improve vs engine-sendiri](issues/06-jalur-improve-vs-engine-sendiri.md): gagal ambang → fine-tune MoisesDB; masih gagal/stem baru → hitung biaya baru custom.

## Not yet specified

- Versi model exact + strategi ensemble 6-stem piano/gitar.
- Arsitektur backend inference: framework, API shape, antrean job, GPU/CPU target.
- Spec UX detail: framework UI, waveform, mixer solo/mute, unduh zip, riwayat job.
- Protokol uji dengar: daftar lagu, skala nilai, ambang lolos.
- Jalur improve: fine-tune data apa, biaya training, kapan engine sendiri layak.
- Budget cloud bila naik dari lokal: biaya GPU, batas durasi/upload.

## Out of scope

- Latih foundation model from-scratch di v1 (jalur v2+ bila improve mentok).
- Mobile app native (fokus web desktop dulu).
- Real-time separation / streaming (batch file saja).
