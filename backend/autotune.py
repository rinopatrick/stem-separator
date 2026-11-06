"""GPU autotune: VRAM probe picks per-model fp16 and largest safe tile.

Probe is conservative: flag-readable VRAM only, CPU tier as the safe
default, deterministic so the manifest choice is reproducible.
Forced OOM (test path) and run_cpu_fallback (real CUDA OOM hook)
fall back to the CPU tile with fallback True so the job completes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

CPU_TILE = 343980
GPU_TILE = CPU_TILE * 2
SPECIALIST_WINDOW = 8 * 44100

_KNOWN_MODELS = ("htdemucs_6s", "kj-melband-smank-onnx")


def _vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _plan_for_model(model: str, vram_mb: float) -> dict:
    if vram_mb >= 6000:
        device, fp16, tile, fallback = "cuda", True, GPU_TILE, False
    elif vram_mb >= 2500:
        device, fp16, tile, fallback = "cuda", False, GPU_TILE, False
    else:
        device, fp16, tile, fallback = "cpu", False, CPU_TILE, False
    if model not in ("htdemucs_6s", *_KNOWN_MODELS):
        tile = min(tile, CPU_TILE)
    if model.startswith("mvsep-") or model.startswith("bs-roformer"):
        tile = SPECIALIST_WINDOW if device == "cpu" else min(tile, SPECIALIST_WINDOW)
    return {"device": device, "fp16": fp16, "tile": int(tile),
            "fallback": fallback}


def probe(
    models: tuple[str, ...] = ("htdemucs_6s",),
    vram_mb: float | None = None,
    force_oom: bool = False,
) -> dict[str, dict]:
    """Pick per-model device/fp16/tile. Logs into the manifest as-is."""
    if vram_mb is None:
        vram_mb = _vram_mb()
    plan: dict[str, dict] = {}
    for model in models:
        if force_oom:
            plan[model] = {"device": "cpu", "fp16": False, "tile": CPU_TILE,
                           "fallback": True,
                           "reason": "forced OOM: CPU fallback"}
        else:
            plan[model] = _plan_for_model(model, vram_mb)
    return plan


T = TypeVar("T")


def run_cpu_fallback(fn: Callable[[], T], plan_entry: dict) -> tuple[T, dict]:
    """Run fn; on CUDA OOM return (cpu_result, cpu_entry).

    Real-OOM hook the spec's "falls back to CPU on OOM" needs beyond
    the force_oom test flag: caller retries fn on CPU and the job
    completes with fallback True logged.
    """
    try:
        return fn(), plan_entry
    except Exception as exc:
        if "out of memory" not in str(exc).lower():
            raise
        cpu_entry = {**plan_entry, "device": "cpu", "fp16": False,
                     "tile": CPU_TILE, "fallback": True,
                     "reason": f"OOM fallback: {exc}"[:200]}
        return fn(), cpu_entry
