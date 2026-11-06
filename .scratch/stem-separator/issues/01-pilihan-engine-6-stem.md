Status: resolved
Type: research
Blocked by:

## Question

Model/engine SOTA apa yang paling akurat untuk 6-stem (vokal, drum, bass, piano, gitar, other) dengan target minim bleed/artefak saat solo di lagu real? Bandingkan Demucs v4, Mel-Band RoFormer, MDX23, piano/gitar khusus (piano-sdk, guitar models), dan strategi ensemble. Keluaran: rekomendasi engine v1 + alasan + batas akurasi piano/gitar yang jujur.

## Answer

Ensemble per-stem: vokal Mel-Band RoFormer (kim checkpoint), drum/bass/other Demucs ht_ft, gitar Banquet/fine-tune 6s, piano Banquet/spesialis. htdemucs_6s stock tidak untuk piano solo (bleed diakui penulis). Report: research/01-engine-6stem.md.
