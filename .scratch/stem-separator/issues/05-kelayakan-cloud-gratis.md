Status: resolved
Type: research
Blocked by: 01

## Question

Bisakah full-gratis di cloud (Vercel dkk) untuk inference stem separation? Teliti batas durasi serverless, RAM, timeout, larangan proses berat, biaya GPU bila perlu, dan alternatif gratis/kuasi-gratis (HF Spaces, Colab, RunPod serverless, Oracle free tier). Keluaran: putusan lokal-dulu-siap-cloud vs cloud apa, dengan angka batas konkret.

## Answer

Vercel/Workers tidak layak untuk inference (RAM/timeout/kuota). Full-gratis realistis: HF Spaces CPU Basic / ZeroGPU, atau Colab/Kaggle manual. Web di Vercel gratis + worker Modal kredit $30/bln atau RunPod ±$0,01/lagu. Report: research/05-cloud-gratis.md.
