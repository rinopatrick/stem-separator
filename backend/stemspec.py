"""StemSpec registry: one 9-name stem contract replacing the STEMS list.

GLOSSARY.md is canonical: stem, other-split, tempo map, manifest.
Stage-1 models fill vocal, drums, bass. The other six slots are honest
placeholders wired through the real pipeline so later tickets swap in
real separators without touching the contract.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class StemSpec:
    name: str
    kind: str  # direct | specialist | synthesized | extracted | leftover
    stage: int  # 1 or 2
    source: str  # mix | other | vocal | timestamps
    optional: bool
    model_id: str


STEMS: tuple[StemSpec, ...] = (
    StemSpec("vocal", "specialist", 1, "mix", False, "kj-melband-smank-onnx"),
    StemSpec("drums", "direct", 1, "mix", False, "htdemucs_6s"),
    StemSpec("bass", "direct", 1, "mix", False, "htdemucs_6s"),
    StemSpec("piano", "specialist", 2, "other", False, "mvsep-scnet-large"),
    StemSpec("guitar", "specialist", 2, "other", False, "bs-roformer-viperx"),
    StemSpec("strings", "specialist", 2, "other", False, "mvsep-bs-roformer-2025.09"),
    StemSpec("synth", "leftover", 2, "other", False, "other-minus-specialists"),
    StemSpec("cue", "extracted", 2, "vocal", True, "silero-vad-whisper-tiny"),
    StemSpec("click", "synthesized", 2, "timestamps", True, "beat-this-small0"),
)

STEM_NAMES: list[str] = [s.name for s in STEMS]

BY_NAME: dict[str, StemSpec] = {s.name: s for s in STEMS}
