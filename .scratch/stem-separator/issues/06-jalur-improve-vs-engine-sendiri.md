Status: resolved
Type: grilling
Blocked by: 01

## Question

Jalur improve vs bikin engine sendiri: kriteria kapan ensemble + fine-tune SOTA dianggap mentok sehingga layak bangun engine custom? Butuh ambang ukur (skor uji dengar), estimasi biaya data + training, dan risiko. Keluaran: gerbang keputusan v2 vs engine-sendiri.

## Answer

Gerbang v2: piano/gitar di bawah 3.5 di ≥4 lagu setelah ensemble v1 → fine-tune MoisesDB dulu. Gerbang engine-sendiri: setelah fine-tune masih gagal, atau butuh stem baru (strings/synth) → hitung biaya training dulu, baru putuskan arsitektur custom. Engine dari nol bukan opsi v1.
