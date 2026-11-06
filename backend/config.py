import os
from pathlib import Path

from stemspec import STEM_NAMES

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", BASE_DIR / "results"))
MODEL_ID = os.getenv("MODEL_ID", "StemSplitio/htdemucs-6s-onnx")
MODEL_FILE = os.getenv("MODEL_FILE", "htdemucs_6s.onnx")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MAX_MINUTES = float(os.getenv("MAX_MINUTES", "10"))
CHUNK_SAMPLES = 343980  # fixed ONNX input (~7.8s @44.1kHz)
SR = 44100
# 9-name contract (T1): vocal/drums/bass + other-split placeholders + cue/click.
# inference still reads the 6s base internally; pipeline maps base -> contract.
STEMS = STEM_NAMES
# 6s base output names from the ONNX model (internal, not the API contract).
BASE_STEMS = ["vocals", "drums", "bass", "piano", "guitar", "other"]

for d in (UPLOAD_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)
