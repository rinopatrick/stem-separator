# 05 — Bisakah inference 6-stem jalan full-gratis di cloud? (Vercel dkk)

Tanggal riset: 2026-10-24. Semua angka dari docs resmi (diakses via fetch/search snippet); klaim komunitas ditandai.

## Jawaban singkat

- **Vercel Hobby (dan Cloudflare Workers) tidak layak sebagai mesin inference 6-stem.** Bukan cuma soal timeout — RAM maks 2 GB, bundle 250 MB (Python 500 MB), kuota CPU gratis kecil, dan tidak ada GPU [1][2][8].
- **Full-gratis yang realistis = Hugging Face Spaces (CPU Basic gratis atau ZeroGPU), atau notebook gratis (Colab/Kaggle) untuk pemakaian manual**, atau **Oracle Always Free VM (CPU ARM, kapasitas langka)** [3][4][5][6][9].
- **GPU bayar-per-pakai termurah yang terdokumentasi: RunPod Serverless kelas RTX 2000/A4000 ≈ $0,58/jam, billing per detik** — bukan gratis, tapi kandidat fallback kuasi-gratis (bayar per lagu, hitungan sen) [10]. Modal memberi **kredit gratis $30/bulan** dengan timeout fungsi s.d. 24 jam — kandidat kuat untuk job inference [11][12].

## Kebutuhan workload (acuan)

- Model `htdemucs_6s` ONNX: fp32 258 MB; runtime ≈ 1,1 GB RAM per segmen 7,8 dtk; RTF ≈ 0,20 di CPU Apple M4 Pro (≈ 22 dtk untuk lagu 3 menit); varian fp16-weights 136–166 MB tanpa beda runtime cost [13][14].
- Panduan repo Demucs: akselerasi GPU butuh VRAM ≥ 3 GB (≈ 7 GB dengan argumen default); fallback CPU via `-d cpu` [15] (catatan: URL sumber hasil pencarian menunjuk fork `Maricpl/htdemucs`, bukan repo upstream `facebookresearch/demucs` — perlakukan klaim ini sebagai medium confidence).
- Inferensi: model 258 MB + ffmpeg + onnxruntime + audio I/O. Estimasi kasar: muat dalam RAM 2 GB (Vercel) secara ketat, nyaman di ≥ 4 GB; waktu CPU untuk lagu 3–4 menit orde puluhan detik di CPU modern, menit-an di CPU shared kecil.

## Batas konkret per platform

### 1. Vercel (Hobby gratis) — tidak cocok untuk inference

- Durasi maks fungsi: **Hobby 300 dtk (default = maks)**; Pro/Enterprise default 300 dtk, maks 800 dtk, extended beta 1800 dtk [1].
- Memori maks: **Hobby 2 GB**; Pro/Enterprise s.d. 4 GB (2 vCPU) [1].
- Ukuran bundle (uncompressed): **250 MB** umum, **500 MB untuk Python**; beta "Large functions" s.d. 5 GB (perlu Fluid compute + Active CPU) [1].
- Kuota gratis Hobby: **Active CPU 4 CPU-jam**, **Provisioned Memory 360 GB-jam**, **invokasi fungsi 1 juta pertama** per bulan; Hobby non-komersial/personal saja; **kelebihan kuota tidak bisa beli tambahan — tunggu reset ±30 hari** [2].
- Konsekuensi: model 258 MB saja sudah lewat 250 MB (lolos hanya via runtime Python 500 MB atau Large Functions beta); lagu 3 menit (~22 dtk di M4 Pro) mungkin masih < 300 dtk di CPU Vercel, tapi lagu panjang + antrean + cold start berisiko timeout; kuota 4 CPU-jam/bulan habis dalam belasan lagu. **Peran realistis Vercel = frontend + orkestrasi/antrean, bukan inference.**

### 2. Cloudflare Workers (Free) — tidak cocok

- Free: **100.000 request/hari, CPU time 10 ms per request, memori 128 MB**; plan berbayar pun CPU time maks 5 menit (default 30 dtk) [8].
- 10 ms CPU time vs kebutuhan puluhan detik–menit: mustahil. Hanya relevan sebagai API gateway/webhook.

### 3. Hugging Face Spaces — kandidat full-gratis utama (dengan catatan)

- Spesifikasi resmi: **CPU Basic = 2 vCPU / 16 GB / disk 50 GB, harga $0 (FREE)**; CPU Upgrade 8 vCPU/32 GB $0,03/jam [3][4].
- **Catatan penting:** docs menyatakan pembuatan Space baru yang berjalan di atas compute (Gradio/Docker) **membutuhkan paid plan**, sedangkan Static Spaces gratis untuk semua [3][4]. Implikasi: akun gratis lama/tertentu masih bisa menjalankan CPU Basic gratis, tapi akun baru kemungkinan harus verifikasi/kartu atau langganan — **cek langsung di console sebelum mengandalkan ini.**
- Lifecycle: hardware gratis **tidur (sleep) setelah ±48 jam tidak aktif**, dikunjungi → restart otomatis; hardware upgrade jalan indefinite [3][4].
- GPU berbayar termurah: **T4-small (4 vCPU/15 GB/16 GB VRAM) $0,40/jam**; T4-medium $0,60/jam [4].
- **ZeroGPU (GPU gratis dinamis):** alokasi dinamis GPU Blackwell RTX Pro 6000, **hanya SDK Gradio**, ukuran `large` default (½ kartu, 48 GB VRAM, biaya kuota 1×) dan `xlarge` (penuh, 2× kuota); **durasi GPU default 60 dtk per fungsi** (bisa custom/dinamis); kuota harian **anon 2 mnt / akun gratis 5 mnt / PRO 40 mnt (prioritas tertinggi, bisa diperpanjang pakai kredit $1 per 10 mnt)**; hosting gratis maks **2 Space ZeroGPU** (email terverifikasi, umur akun > 30 hari), PRO 10, organisasi 50 [5].
- Konsekuensi: CPU Basic 16 GB RAM cukup untuk `htdemucs_6s` ONNX; ZeroGPU gratis 5 mnt/hari cukup untuk beberapa lagu 3 menit (RTF GPU ≫ CPU) tapi **fungsi > 60 dtk harus di-chunk/diatur duration**, dan antrean shared tidak ada SLA.

### 4. Google Colab (gratis) — manual/demo, bukan API produksi

- Google **tidak mempublikasi limit tetap** (dinamis, tergantung ketersediaan & riwayat pakai); **notebook gratis maks 12 jam runtime**; jenis GPU/TPA bervariasi, tidak dijamin; GPU premium untuk plan berbayar [6].
- Konsekuensi: cocok untuk demo/validasi manual, tidak untuk endpoint inference yang dipanggil web app (butuh keep-alive/akun, melanggar ToS bila dipakai sebagai server).

### 5. Kaggle Notebooks (gratis) — batch manual

- **GPU gratis s.d. 30 jam/minggu** (kadang lebih, tergantung demand; reset mingguan), akselerator P100 atau 2×T4 [7].
- Batasan sesi (sumber sekunder, perlu verifikasi di UI notebook): sesi CPU/GPU s.d. 12 jam, TPU 9 jam; verifikasi telepon untuk unlock akselerator [7].
- Konsekuensi: cocok untuk batch processing manual/kompetisi, bukan endpoint real-time.

### 6. Oracle Cloud Always Free — VM gratis selamanya (CPU saja, kapasitas langka)

- Docs resmi: Ampere A1 (Arm) gratis **1.500 OCPU-jam + 9.000 GB-jam per bulan** pada satu mirror docs (mirror lain menyebut 3.000/18.000 — inkonsisten antar mirror, cek console), total **±2 OCPU / 12 GB RAM** lintas instance [9].
- InfoQ (2026-07) melaporkan Oracle **memangkas jatah A1 dari 4 OCPU/24 GB menjadi 2 OCPU/12 GB per 15 Juni 2026** tanpa pengumuman [9].
- Konsekuensi: 12 GB RAM cukup untuk inference CPU 6-stem; tanpa GPU sehingga lambat (menit-an per lagu); instance gratis **sangat sulit didapat** di banyak region (kapasitas habis). Cocok sebagai worker antrean murah bila berhasil klaim.

### 7. Opsi kuasi-gratis (bayar-per-pakai, tanpa langganan)

- **RunPod Serverless:** billing **per detik** (dibayar sejak worker start s.d. selesai + idle sebelum scale-down), scale-to-zero; tier Flex termurah **RTX 2000/A4000/A4500/RTX 4000 (16 GB) $0,58/jam**, L4/A5000/3090 $0,69/jam (harga situs per Okt 2026) [10]. Per lagu 3 menit (≈ 1 menit GPU incl. cold start) ≈ **±$0,01**. Tanpa free tier — perlu deposit/kartu.
- **Modal:** plan Starter **$0 + compute, kredit gratis $30/bulan** (±3 seat, konkurensi 100 container/10 GPU) [11]; **timeout fungsi default 300 dtk, dapat s.d. 24 jam** [12]. $30 ≈ 50 jam T4-small atau ratusan lagu — kandidat terbaik untuk worker GPU antrean.
- **HF Inference Providers:** akun gratis hanya **$0,10 kredit/bulan** (PRO $2), habis itu pay-as-you-go — dan tidak ada model stem-separation siap pakai di katalog; endpoint dedikasi s.d. $0,50/jam+ [13→dipindahkan: sumber rate-limits]. Tidak relevan kecuali deploy endpoint sendiri (berbayar).

## Rekomendasi arsitektur (untuk tiket implementasi)

1. **Web app (Vercel Hobby, gratis):** upload → simpan ke object storage → enqueue job → polling hasil. Tetap dalam timeout 300 dtk karena request-nya pendek [1][2].
2. **Worker A — full gratis:** HF Space (CPU Basic 2 vCPU/16 GB, sleep 48 jam) atau ZeroGPU (Gradio, chunk ≤ 60 dtk, kuota 5 mnt/hari) [3][4][5].
3. **Worker B — kuasi-gratis sekali bayar:** Modal ($30 kredit/bulan, timeout 24 jam) atau RunPod Serverless (≈ $0,01/lagu) untuk lagu panjang/antrean penuh [10][11][12].
4. **Fallback manual:** Colab/Kaggle untuk batch awal; Oracle A1 bila berhasil klaim VM [6][7][9].

## Evidence table

| # | Sumber | URL | Klaim kunci | Tipe | Confidence |
|---|--------|-----|-------------|------|------------|
| 1 | Vercel Functions Limits (last updated 2026-08-24) | https://vercel.com/docs/functions/limitations | Hobby durasi maks 300 dtk; memori maks Hobby 2 GB; bundle 250 MB (Python 500 MB); Large Functions beta 5 GB | primer | high |
| 2 | Vercel Hobby Plan | https://vercel.com/docs/plans/hobby | Gratis; Active CPU 4 CPU-jam, Provisioned Memory 360 GB-jam, 1 jt invokasi; over-kuota tunggu reset, tak bisa top-up | primer | high |
| 3 | HF Spaces Overview | https://huggingface.co/docs/hub/spaces-overview | CPU Basic 2 vCPU/16 GB/50 GB $0; Space compute baru butuh paid plan; sleep bila idle lama | primer | high |
| 4 | HF Using GPU Spaces | https://huggingface.co/docs/hub/en/spaces-gpus | T4-small $0,40/jam; billing per menit; sleep cpu-basic 48 jam; upgrade jalan indefinite | primer | high |
| 5 | HF Spaces ZeroGPU | https://huggingface.co/docs/hub/en/spaces-zerogpu | GPU gratis dinamis Gradio-only; durasi default 60 dtk; kuota harian gratis 5 mnt / PRO 40 mnt; hosting gratis maks 2 Space | primer | high |
| 6 | Google Colab FAQ | https://research.google.com/colaboratory/faq.html | Limit dinamis tak dipublikasi; notebook gratis maks 12 jam; GPU tak dijamin | primer | high |
| 7 | Kaggle efficient GPU usage (+ agregator aifree) | https://www.kaggle.com/docs/efficient-gpu-usage | GPU s.d. 30 jam/minggu P100/2×T4; sesi ±12 jam (sekunder) | primer + sekunder | medium-high |
| 8 | Cloudflare Workers Limits | https://developers.cloudflare.com/workers/platform/limits | Free: 100 rb req/hari, CPU 10 ms, RAM 128 MB; Paid CPU s.d. 5 mnt | primer | high |
| 9 | Oracle Always Free Resources (+ InfoQ 2026-07) | https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm | A1 gratis ±2 OCPU/12 GB; Jun 2026 dipangkas dari 4/24 (InfoQ); mirror docs inkonsisten 1500 vs 3000 OCPU-jam | primer + sekunder | medium |
| 10 | RunPod pricing + serverless pricing | https://www.runpod.io/pricing | Serverless per-detik scale-to-zero; Flex termurah $0,58/jam (RTX 2000/A4000 16 GB) | primer (self-reported) | high |
| 11 | Modal pricing | https://modal.com/pricing | Starter $0 + compute; kredit gratis $30/bln; konkurensi 100 container/10 GPU | primer (self-reported) | high |
| 12 | Modal Timeouts | https://modal.com/docs/guide/timeouts | Timeout default 300 dtk, maks 24 jam per eksekusi | primer | high |
| 13 | StemSplitio htdemucs-6s-onnx README (HF Hub) | https://huggingface.co/StemSplitio/htdemucs-6s-onnx/blob/main/README.md | htdemucs_6s.onnx 258 MB; ~1,1 GB RAM/segmen; RTF 0,20 CPU; varian fp16weights 136 MB | sekunder (komunitas) | medium |
| 14 | demucs-onnx models page | https://stemsplit.github.io/demucs-onnx/models/ | htdemucs_6s ≈ 1,6 dtk/segmen, ~22 dtk/lagu 3 mnt, RTF 0,20 | sekunder | medium |
| 15 | Demucs GitHub (fork Maricpl/htdemucs) | https://github.com/Maricpl/htdemucs | GPU perlu VRAM ≥ 3 GB (≈ 7 GB default); fallback `-d cpu` | sekunder (fork) | medium |
| 16 | HF Inference Providers rate-limits/pricing | https://huggingface.co/docs/inference-providers/en/rate-limits | Akun gratis $0,10 kredit/bln; PRO $2; selebihnya PAYG | primer | high |

## Coverage status

- Dicek langsung (fetch): halaman Vercel Limits [1], HF spaces-gpus [4], HF spaces-zerogpu [5]. Sisanya via snippet provider search (isi dikutip apa adanya dari docs resmi) — untuk keputusan biaya final, buka URL sumber di tabel.
- Belum dicek: Vercel fair-use/acceptable-use (larangan eksplisit proses berat), harga egress Vercel untuk file audio besar, ketersediaan aktual Oracle A1 per region, kuota sesi Kaggle saat ini (sumber primernya di UI notebook, bukan docs statis).
- Inkonsistensi: mirror docs Oracle menyebut 1.500 vs 3.000 OCPU-jam — jadikan 2 OCPU/12 GB total sebagai acuan aman, verifikasi di console.
