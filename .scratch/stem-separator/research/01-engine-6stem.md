# 01 — Engine pilihan 6-stem: engine/model SOTA paling akurat

Target: vokal, drum, bass, piano, gitar, other. Prioritas: minim bleed/artefak saat solo di lagu real.
Tanggal riset: 2026-10-24.

## Ringkasan eksekutif

- Tidak ada satu model open yang SOTA di keenam stem sekaligus. Arsitektur SOTA berbeda per stem
  dan per dataset, jadi strategi terbaik adalah **ensemble per-stem**: model spesialis per stem,
  bukan satu model 6-stem generik [6][8].
- Untuk **vokal** (stem paling kritis untuk bleed): keluarga **Mel-Band RoFormer / BS-RoFormer**
  adalah SOTA pasca-SDX23, mengalahkan Demucs vokal (9.51 vs 8.36 dB SDR pada benchmark MultiSong,
  dan 15.82 vs 13.99 dB instrumental) [6][8]. Checkpoint komunitas standar adalah
  `melband_roformer_kim_vocals` (Kimberley Jensen) [11].
- Untuk **drum/bass/other**: Demucs Hybrid Transformer (`htdemucs_ft`) tetap kompetitif,
  dan ensemble MVSEP (Demucs + MDX + RoFormer) memenangkan tempat ke-3 Leaderboard C SDX23 [6].
- Untuk **piano/gitar** (titik lemah 6-stem): `htdemucs_6s` bawaan Demucs bersifat **eksperimental**;
  penulis Demucs sendiri menyatakan gitar "okay" tapi piano "a lot of bleeding and artifacts" [2].
  Paper 6-stem independen (Qalbyubi et al. 2024) mengukur `htdemucs` pra-latih underperform di
  gitar/piano/other dibanding bass/drum/vokal [13]. Model query-based **Banquet** (24.9M param)
  mengungguli 6-stem HT-Demucs pada gitar dan piano di MoisesDB [7].
- Dataset evaluasi yang relevan: **MUSDB18-HQ** hanya punya 4 stem (vocals/drums/bass/other) [14];
  evaluasi gitar/piano memakai **MoisesDB** (240 lagu, taksonomi 2-level, CC BY-NC-SA 4.0,
  non-komersial saja) [4][5] atau Slakh2100 (sintetis) [9].
- Strategi ensemble pemenang kompetisi: gabungkan arsitektur berbeda (Demucs + TFC-TDF/MDX +
  UVR-MDX-Net vokal) dan rata-ratakan prediksi; ZFTurbo/MVSEP mendokumentasikan gain
  +0.4–1.7 dB SDR per stem dari ensemble vs model tunggal [6][8].

## 1. Kandidat engine per stem

### 1a. Demucs v4 (Hybrid Transformer Demucs, HT-Demucs)

- Paper: Rouard et al., "Hybrid Transformers for Music Source Separation", ICASSP 2023,
  334 sitasi, arXiv 2211.08553 [1].
- Arsitektur: hybrid temporal/spectral bi-U-Net; layer terdalam diganti cross-domain
  Transformer Encoder (self-attention dalam domain, cross-attention antar domain) [1].
- Hasil lapor paper: **9.00 dB SDR rata-rata di MUSDB18-HQ test** (model tunggal, +800 lagu ekstra),
  **9.20 dB** dengan sparse attention + fine-tuning per source [12]. Baseline sebanding:
  Band-Split RNN 9.0 dB (dengan 1.7k mix ekstra), Hybrid Demucs v3 7.7 dB (tanpa data ekstra) [12].
- Repo resmi menyatakan `htdemucs_ft` sebagai default; `htdemucs_6s` menambah gitar+piano [2].
- Kritis untuk tiket ini: repo resmi melabeli `htdemucs_6s` **eksperimental** — "Quick testing
  seems to show okay quality for guitar, but a lot of bleeding and artifacts for the piano
  source" dan "piano source is not working great at the moment" [2].
- Varian fine-tune eksternal: `htdemucs_ft` butuh ~4x waktu separasi dibanding `htdemucs` [2].

### 1b. Mel-Band RoFormer (vokal SOTA)

- Paper: Wang et al., "Mel-Band RoFormer for Music Source Separation", ISMIR 2023 LBD,
  arXiv 2310.01809 [3].
- Ide: ganti band-split heuristik (BSRNN/BS-RoFormer) dengan **Mel-band ber-overlap** mengikuti
  skala mel; mask estimation via hierarchical Transformer + RoPE [3].
- Hasil lapor paper: pada MUSDB18HQ, Mel-RoFormer mengungguli BS-RoFormer di vokal/drums/other;
  konfigurasi L=6 mencapai **vokal SDR 11.21 dB** vs baseline 10.78 dB; L=9 mencapai **11.60 dB**
  vs 11.02 dB [10]. Catatan: angka ini dari ringkasan alphaXiv/paper PDF, bukan replikasi mandiri.
- Predecessor BS-RoFormer (Lu et al., arXiv 2309.02612): juara 1 track MSS SDX23;
  versi kecil tanpa data ekstra mencapai **9.80 dB rata-rata SDR di MUSDB18** [15];
  versi penuh (MUSDB18HQ + 500 lagu ekstra) mencapai **rata-rata 11.99 dB** [15].
- Checkpoint komunitas yang dipakai luas: `melband_roformer_kim_vocals` oleh Kimberley Jensen
  sebagai model default separasi vokal di ekosistem audio-separator/UVR [11].
  Varian lanjutan (unwa Big Beta, becruily, dsb.) mengklaim SDR vokal MDX23 ~12.5–12.9 [16].
  (Klaim varian komunitas: self-reported, belum diverifikasi independen — butuh uji sendiri.)

### 1c. MDX / TFC-TDF-U-Net (kuielab, MDX21/SDX23)

- Keluarga MDX-Net (KUIELAB) adalah pemenang MDX21 track dan dasar banyak model vokal UVR [6].
- Referensi kompetitif: model MDX Kim Vocal (FFT 7680) mencapai SDR vokal 9.36 / instrumental
  15.70 pada benchmark Synth MVSep, vs model kontes MDX23 (ZFTurbo) 9.56 / 15.82 [8].
- Relevansi: model MDX 2-stem (vokal/instrumental) berguna sebagai **pintu bleed rendah untuk
  vokal**, tapi tidak menyelesaikan 6-stem; stem bass/drums/other tetap dari Demucs/RoFormer.

### 1d. Model khusus piano/gitar

- **Banquet** (Watcharasupat & Lerch, ISMIR 2024, arXiv 2406.18747): query-based single-decoder
  di atas encoder Bandit/BSRNN + query embedding PaSST; hanya **24.9M parameter trainable**;
  di MoisesDB mendekati 6-stem HT-Demucs pada VDBO dan **mengunggulinya pada gitar dan piano**;
  SNR vokal ≥7.4 dB; di atas oracle IRM untuk drum/bass [7].
- **Fine-tune komunitas**: `adityalakhani/htdemucs-6s-guitar-ft` — fine-tune `htdemucs_6s` di
  MoisesDB khusus gitar; klaim self-reported: tolak bleed transien drum lebih agresif
  (+0.89 dB SIR) dan ikuti sustain gitar lebih baik [17]. (Self-reported, bukan benchmark resmi.)
- **Slakh2100** (Manilow et al., arXiv 1909.08494): dataset sintetis untuk benchmark gitar/piano;
  temuan dasar: separasi gitar/piano jauh lebih rendah dari bass/drums; model latih-Slakh saja
  overfit dan generalisasi buruk ke MUSDB18 test [9]. Implikasi: model gitar/piano sintetis-only
  berisiko di lagu real — butuh data real (MoisesDB) atau fine-tune.

## 2. Benchmark yang sebanding (angka-angka kunci)

| Benchmark | Model | Skor (seperti dilaporkan sumber) | Sumber |
|---|---|---|---|
| MUSDB18-HQ test, rata-rata 4-stem | HT-Demucs tunggal (+800 lagu) | 9.00 dB SDR | [12] |
| MUSDB18-HQ test, rata-rata 4-stem | Sparse HT-Demucs + FT per source | 9.20 dB SDR | [12] |
| MUSDB18, kecil tanpa data ekstra | BS-RoFormer kecil | 9.80 dB rata-rata SDR | [15] |
| MUSDB18-HQ + 500 lagu ekstra | BS-RoFormer penuh (juara SDX23) | 11.99 dB rata-rata SDR | [15] |
| MUSDB18HQ, vokal saja | Mel-RoFormer L=6 / L=9 | 11.21 / 11.60 dB SDR | [10] |
| MultiSong MVSep, vokal | MVSEP-MDX23 ensemble vs Demucs HT4 | 9.51 vs 8.36 dB SDR | [6] |
| MultiSong MVSep, instrumental | MVSEP-MDX23 ensemble vs Demucs HT4 | 15.82 vs 13.99 dB SDR | [6] |
| Synth MVSep, instrumental/vokal | MDX23 (ZFTurbo) vs Kim Vocal 1 | 15.82/9.56 vs 15.70/9.36 | [8] |
| MoisesDB test, median SNR | Banquet vs 6-stem HT-Demucs | Banquet menang gitar+piano, par VDBO | [7] |

Peringatan metodologi: tiap benchmark memakai dataset/split/metrik berbeda (SDR vs SNR vs SI-SDR,
median vs mean), jadi **angka antar-baris tidak dapat dibandingkan langsung**; gunakan hanya
untuk ranking ordinal dalam benchmark yang sama.

## 3. Strategi ensemble yang terbukti

- **Ensemble multi-arsitektur ZFTurbo/MVSEP** (peringkat 3 Leaderboard C SDX23): menggabungkan
  Demucs4 + MDX + bobot vokal UVR (Kimberley Jensen); repo `ZFTurbo/MVSEP-MDX23-music-separation-model`
  berisi kode inferensi, opsi overlap/chunk, GUI, dan tabel kualitas [6].
- **Ensemble MVSEP produksi** (2024): vokal dari 7 model (UVR-MDX-NET-Voc_FT, Demucs4 Vocals,
  MDX23C, VitLarge23, BS-RoFormer, Mel-RoFormer, SCNet XL); bass/drums/other dari 4 model
  Demucs (ht_ft, ht, 6s, mmi). Rata-rata SDR 11.21 pada MultiSong (bass 12.52, drums 11.73,
  other 7.01, vokal 10.30) [18]. Pola ini = **spesialis per stem + rata-rata prediksi**.
- Paper benchmark Solovyev et al. (arXiv 2305.07489) mendokumentasikan skor ensemble final
  pada MultiSong dan set uji publik/privat MDX23 (mean SDR 10.11 / 9.41 / 9.25) [8].
- **kuielab SDX23** (juara 1 Leaderboard B bleeding, peringkat 3 Leaderboard A): ensemble 3 model
  TFC-TDF-U-Net identik dengan seed berbeda + loss masking; menunjukkan ensemble seed + training
  robust-noise membantu kasus bleeding/label-noise [19].
- Implikasi untuk tiket: ensemble per-stem (RoFormer vokal + Demucs drum/bass + spesialis
  gitar/piano) + overlap inferensi tinggi + per-source fine-tuning adalah resep standar
  kompetisi untuk minim bleed/artefak [6][18][19].

## 4. Rekomendasi engine untuk 6-stem (target solo minim bleed)

1. **Vokal → Mel-Band RoFormer** (checkpoint Kim `melband_roformer_kim_vocals` sebagai baseline,
   evaluasi varian unwa/becruily bila perlu). Alasan: SOTA vokal di semua benchmark di atas [3][6][11].
2. **Drums/Bass → `htdemucs_ft`** (atau ensemble Demucs sesuai MVSEP). Alasan: kompetitif di
   drum/bass dan matang tooling-nya [2][6].
3. **Other → `htdemucs_ft` atau BS/Mel-RoFormer other-stem**. Alasan: other adalah sisa residual;
   RoFormer unggul di other pada MUSDB18HQ [10][15].
4. **Gitar → `htdemucs_6s` + fine-tune MoisesDB** (contoh resep komunitas [17]) atau Banquet [7].
   Jangan pakai `htdemucs_6s` stock untuk gitar tanpa evaluasi — statusnya eksperimental [2].
5. **Piano → Banquet atau model piano-spesialis**; `htdemucs_6s` stock **tidak direkomendasikan**
   untuk piano solo karena bleed/artefak yang diakui penulis [2]. Validasi dengan uji dengar
   + metrik SIR/SAR, bukan hanya SDR (SDR tinggi bisa sembunyikan bleed saat solo).
6. **Strategi global**: ensemble per-stem dengan overlap inferensi tinggi (0.5–0.95) mengikuti
   resep MVSEP/ZFTurbo [6]; evaluasi akhir di lagu real + MoisesDB, bukan hanya MUSDB18
   (yang tidak punya gitar/piano) [4][14].

## 5. Yang belum pasti / butuh follow-up

- Angka Mel-RoFormer vokal 11.21/11.60 dB dikutip dari ringkasan paper, belum dibaca penuh
  PDF-nya (isi tabel per-stem lengkap belum diverifikasi).
- Model piano/gitar terbaik di lagu real Indonesia (dangdut/koplo/pop): belum ada benchmark;
  butuh uji dengar + ukur bleed (SIR) pada stem solo.
- Lisensi: bobot Demucs (MIT-ish, cek repo), checkpoint komunitas RoFormer (bervariasi per
  author), MoisesDB (CC BY-NC-SA 4.0 — non-komersial) [5]. Cek ulang sebelum distribusi komersial.

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Rouard et al., Hybrid Transformers for Music Source Separation (ICASSP 2023, 334 sitasi) | https://www.semanticscholar.org/paper/826dad99e9725d46c15083a7599a36e8d6861d5 | HT-Demucs: hybrid bi-U-Net + cross-domain Transformer; 9.00/9.20 dB SDR MUSDB-HQ | primary | high |
| 2 | facebookresearch/demucs README (+ mirror) | https://github.com/facebookresearch/demucs | `htdemucs_6s` eksperimental; piano bleeding/artefak; `htdemucs_ft` 4x lambat | primary | high |
| 3 | Wang et al., Mel-Band RoFormer (ISMIR 2023 LBD, arXiv 2310.01809) | https://arxiv.org/abs/2310.01809 | Mel-band overlap ganti band-split heuristik; SOTA vokal/drums/other vs BS-RoFormer | primary | high |
| 4 | Pereira et al., MoisesDB (ISMIR 2023) | https://www.semanticscholar.org/paper/475e1a429b74ef076c3bef3ee2556ec4d670e80c | MoisesDB: 240 lagu/45 artis/12 genre, taksonomi 2-level untuk >4-stem | primary | high |
| 5 | moises-ai/moises-db repo | https://github.com/moises-ai/moises-db | Benchmark htdemucs4/6, spleeter, oracle per lagu; lisensi CC BY-NC-SA 4.0 | primary | high |
| 6 | ZFTurbo/MVSEP-MDX23-music-separation-model | https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model | Juara 3 MDX23 LB-C; ensemble Demucs+MDX+UVR; tabel SDR MultiSong | primary | high |
| 7 | Watcharasupat & Lerch, Banquet (arXiv 2406.18747) | https://arxiv.org/html/2406.18747v1 | Banquet 24.9M param: par VDBO, menang gitar+piano vs 6-stem HT-Demucs di MoisesDB | primary | high |
| 8 | Solovyev et al., Benchmarks for sound demixing (arXiv 2305.07489) | https://arxiv.org/html/2305.07489v2 | Skor tunggal vs ensemble; MDX23 ZFTurbo vs Kim Vocal di Synth MVSep | primary | medium |
| 9 | Manilow et al., Slakh (arXiv 1909.08494) | https://arxiv.org/html/1909.08494 | Gitar/piano jauh lebih sulit; model Slakh-only overfit, generalisasi buruk | primary | high |
| 10 | alphaXiv summary of Mel-Band RoFormer PDF | https://arxiv.org/pdf/2310.01809 | Vokal SDR 11.21 (L=6) / 11.60 (L=9); rata-rata 11.99 dB (+500 lagu) | secondary | medium |
| 11 | openmirlab/melband-roformer-infer | https://github.com/openmirlab/melband-roformer-infer | `melband-roformer-kim-vocals` = checkpoint default rekomendasi | secondary | medium |
| 12 | Demucs README (kuantitatif) via OpenCodePapers | https://opencodepapers-b7572d.gitlab.io/benchmarks/music-source-separation-on-musdb18.html | Sparse HT-Demucs FT 9.20 dB; HT-Demucs 9.00 dB di MUSDB18 | secondary | medium |
| 13 | Qalbyubi et al. 2024, HT vs BSRNN fine-tune | https://www.semanticscholar.org/paper/b1b334d6785f0d0a5790f58d14a1ae8a0391d684 | HT-Demucs 6-sumber pra-latih underperform di gitar/piano/other | primary | medium |
| 14 | MUSDB18 dataset (implied via paper thread) | https://arxiv.org/abs/2309.02612 | MUSDB18(HQ) = benchmark 4-stem standar (vokal/drums/bass/other) | primary | high |
| 15 | Lu et al., BS-RoFormer (arXiv 2309.02612) | https://arxiv.org/abs/2309.02612 | Juara 1 SDX23 MSS; 9.80 dB tanpa data ekstra | primary | high |
| 16 | Detrixbwu/audio-separation-models (MelBand RoFormer card) | https://github.com/Detrixbwu/audio-separation-models/blob/main/models/melband_roformer.md | Varian komunitas Kim/unwa/becruily + SDR MDX23 self-reported | secondary | low |
| 17 | adityalakhani/htdemucs-6s-guitar-ft (HF) | https://huggingface.co/adityalakhani/htdemucs-6s-guitar-ft/blob/main/README.md | Fine-tune gitar di MoisesDB; klaim +0.89 dB SIR tolak bleed drum | self-reported | low |
| 18 | MVSEP algorithms page | https://mvsep.com/en/algorithms | Ensemble produksi: 7 model vokal + 4 Demucs; tabel SDR 11.21 rata-rata | self-reported | medium |
| 19 | kuielab/sdx23 | https://github.com/kuielab/sdx23 | Ensemble 3x TFC-TDF + loss masking; juara 1 LB-B bleeding | primary | high |

## Sources

1. Rouard et al. — https://www.semanticscholar.org/paper/826dad99e9725d46c15083a7599a36e8d6861d5
2. facebookresearch/demucs — https://github.com/facebookresearch/demucs
3. Wang et al., Mel-Band RoFormer — https://arxiv.org/abs/2310.01809
4. Pereira et al., MoisesDB — https://www.semanticscholar.org/paper/475e1a429b74ef076c3bef3ee2556ec4d670e80c
5. moises-ai/moises-db — https://github.com/moises-ai/moises-db
6. ZFTurbo/MVSEP-MDX23 — https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model
7. Watcharasupat & Lerch, Banquet — https://arxiv.org/html/2406.18747v1
8. Solovyev et al., Benchmarks — https://arxiv.org/html/2305.07489v2
9. Manilow et al., Slakh — https://arxiv.org/html/1909.08494
10. Mel-Band RoFormer PDF — https://arxiv.org/pdf/2310.01809
11. openmirlab/melband-roformer-infer — https://github.com/openmirlab/melband-roformer-infer
12. OpenCodePapers MUSDB18 board — https://opencodepapers-b7572d.gitlab.io/benchmarks/music-source-separation-on-musdb18.html
13. Qalbyubi et al. 2024 — https://www.semanticscholar.org/paper/b1b334d6785f0d0a5790f58d14a1ae8a0391d684
14. MUSDB18 via BS-RoFormer paper — https://arxiv.org/abs/2309.02612
15. Lu et al., BS-RoFormer — https://arxiv.org/abs/2309.02612
16. Detrixbwu model card — https://github.com/Detrixbwu/audio-separation-models/blob/main/models/melband_roformer.md
17. htdemucs-6s-guitar-ft — https://huggingface.co/adityalakhani/htdemucs-6s-guitar-ft/blob/main/README.md
18. MVSEP algorithms — https://mvsep.com/en/algorithms
19. kuielab/sdx23 — https://github.com/kuielab/sdx23

## Coverage status

- Dicek langsung: README Demucs, abstrak + klaim hasil BS/Mel-RoFormer, Banquet full-text
  (metode + hasil), README ZFTurbo/MVSEP, README MoisesDB, abstrak Slakh, halaman MVSEP/kuielab.
- Belum dibaca penuh: PDF Mel-Band RoFormer (tabel per-stem), PDF HT-Demucs (tabel 6-stem detail),
  benchmark CSV MoisesDB per lagu.
- Blokir/tidak selesai: pencarian umum untuk MDX23 gagal dua kali karena API key bermasalah;
  diatasi via provider paralel dan sumber primer GitHub/arXiv.
