"""GPU-side metrics for the request-level profiler.

These helpers are deliberately dependency-light: they lazily import
``torch`` only when CUDA is actually available, so that importing this
module on a CPU-only CI box never triggers CUDA initialization. The
sampling entry points are also designed to be *monkeypatchable* so that
unit tests can run without a GPU.

All functions here are safe to call from the hot path as long as the
profiler is active; when the profiler is disabled the caller should skip
sampling entirely (see :mod:`sglang_omni.profiler.event_recorder`).
"""

from __future__ import annotations

import math
from typing import Any, Optional

__all__ = [
    "sample_gpu_metrics",
    "compute_throughput_metrics",
    "GPU_METRIC_KEYS",
]

# Keys that may appear in an event's ``metadata`` dict when GPU metrics
# are sampled. Keep this list in sync with :func:`sample_gpu_metrics`.
GPU_METRIC_KEYS = (
    "gpu_mem_allocated_mb",
    "gpu_mem_reserved_mb",
)


def _cuda_module() -> Optional[Any]:
    """Return ``torch.cuda`` if CUDA is available, else ``None``.

    Importing ``torch`` is deferred so that merely importing this module
    on a CPU-only machine is cheap and side-effect free.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch missing in some envs
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda


def sample_gpu_metrics(device: Any = None) -> dict[str, float]:
    """Sample instantaneous GPU memory usage.

    Parameters
    ----------
    device:
        A CUDA device index or a ``torch.device`` object. When ``None`` the
        current default CUDA device is used. The caller is responsible for
        passing the scheduler's device; this keeps the helper pure and easy
        to test.

    Returns
    -------
    dict
        ``{"gpu_mem_allocated_mb": ..., "gpu_mem_reserved_mb": ...}`` on a
        CUDA-capable runtime, or ``{}`` when CUDA is unavailable. Memory is
        reported in **megabytes** (divided by ``1024 * 1024``) to keep the
        JSONL traces compact and human readable.

    Notes
    -----
    ``memory_allocated`` / ``memory_reserved`` do **not** force a device
    synchronize, so sampling is effectively free on the hot path.
    """
    cuda = _cuda_module()
    if cuda is None:
        return {}
    try:
        if device is not None:
            allocated = cuda.memory_allocated(device)
            reserved = cuda.memory_reserved(device)
        else:
            allocated = cuda.memory_allocated()
            reserved = cuda.memory_reserved()
    except Exception:  # pragma: no cover - defensive against driver errors
        return {}
    mb = 1024.0 * 1024.0
    return {
        "gpu_mem_allocated_mb": allocated / mb,
        "gpu_mem_reserved_mb": reserved / mb,
    }


def compute_throughput_metrics(
    duration_ms: float,
    output_tokens: Optional[int] = None,
    audio_seconds: Optional[float] = None,
) -> dict[str, float]:
    """Derive throughput metrics from a measured generation duration.

    Parameters
    ----------
    duration_ms:
        Wall-clock generation time in milliseconds (e.g. the interval
        between ``scheduler_prefill_start`` and ``scheduler_first_emit``).
    output_tokens:
        Number of generated tokens, if known. Enables ``tokens_per_sec``.
    audio_seconds:
        Duration of the generated/consumed audio in seconds. Enables ``rtf``
        (real-time factor = wall_time / audio_duration).

    Returns
    -------
    dict
        A subset of ``{"tokens_per_sec": ..., "rtf": ...}`` depending on
        which optional inputs are provided. Missing inputs are simply
        omitted, so callers can feed whatever they have.
    """
    result: dict[str, float] = {}
    if duration_ms is None or duration_ms <= 0:
        return result
    seconds = duration_ms / 1000.0
    if output_tokens is not None and output_tokens > 0:
        result["tokens_per_sec"] = output_tokens / seconds
    if audio_seconds is not None and audio_seconds > 0:
        # RTF < 1 means faster than real time (good); RTF > 1 means slower.
        result["rtf"] = seconds / audio_seconds
    return result


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return ``numerator / denominator`` or ``None`` on non-finite/zero."""
    if denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    if math.isnan(value) or math.isinf(value):
        return None
    return value
