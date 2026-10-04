# SPDX-License-Identifier: Apache-2.0
"""Stage factory for Parakeet ASR on macOS Apple Silicon."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from sglang_omni.models.parakeet.request_builders import (
    ParakeetASRRequest,
    build_parakeet_result,
    make_parakeet_request_builder,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)


def plan_padded_batches(
    lengths: Sequence[int], *, max_padded_samples: int
) -> list[list[int]]:
    """Group request indices so each padded batch stays within a sample budget.

    Requests are sorted longest first so neighbours have similar lengths and
    padding stays small. A single request longer than the budget still runs,
    alone.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        longest = max(current_max, lengths[index])
        if current and longest * (len(current) + 1) > max_padded_samples:
            batches.append(current)
            current, current_max = [index], lengths[index]
        else:
            current.append(index)
            current_max = longest
    if current:
        batches.append(current)
    else:
        pass
    return batches


def make_parakeet_batch_fn(
    *,
    request_builder: Callable[[StagePayload], ParakeetASRRequest],
    transcribe: Callable[[Sequence[np.ndarray]], list[str]],
    max_padded_samples: int,
) -> Callable[[list[StagePayload]], list[StagePayload | BaseException]]:
    """Build the batch compute function; a bad request fails only itself."""

    def transcribe_batch(
        payloads: list[StagePayload],
    ) -> list[StagePayload | BaseException]:
        results: list[StagePayload | BaseException | None] = [None] * len(payloads)
        ready: list[tuple[int, ParakeetASRRequest]] = []
        for index, payload in enumerate(payloads):
            try:
                ready.append((index, request_builder(payload)))
            except Exception as exc:
                results[index] = exc
        lengths = [request.waveform.shape[0] for _, request in ready]
        for group in plan_padded_batches(
            lengths, max_padded_samples=max_padded_samples
        ):
            requests = [ready[i] for i in group]
            started_at_s = time.perf_counter()
            try:
                texts = transcribe([request.waveform for _, request in requests])
            except Exception as exc:
                logger.exception(
                    "Parakeet ASR batch of %d requests failed", len(requests)
                )
                for index, _ in requests:
                    results[index] = exc
                continue
            model_latency_s = time.perf_counter() - started_at_s
            for (index, request), text in zip(requests, texts):
                results[index] = build_parakeet_result(
                    request, text=text, model_latency_s=model_latency_s
                )
        return results

    return transcribe_batch


def make_parakeet_runner(model_path: str, *, device: torch.device, dtype: str):
    """Native MLX when ``SGLANG_USE_MLX=1``, otherwise Transformers on Torch MPS."""
    from sglang.srt.hardware_backend.mlx.runtime import use_mlx

    if use_mlx():
        from sglang_omni.models.parakeet.mlx.runner import ParakeetMlxModelRunner

        return ParakeetMlxModelRunner(model_path, dtype=dtype)
    else:
        from sglang_omni.models.parakeet.model_runner import ParakeetModelRunner

        return ParakeetModelRunner(model_path, device=str(device), dtype=dtype)


def create_parakeet_asr_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 16,
    max_batch_wait_ms: float = 5.0,
    max_batch_audio_s: float = 600.0,
) -> SimpleScheduler[StagePayload, StagePayload]:
    from sglang_omni.utils.device import resolve_concrete_device

    if not current_platform.is_mps():
        raise ValueError(
            "Parakeet ASR runs only on macOS Apple Silicon (MPS); this host "
            f"resolved to {current_platform.device_type!r}"
        )
    elif max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    elif not math.isfinite(max_batch_wait_ms) or max_batch_wait_ms < 0:
        raise ValueError(
            f"max_batch_wait_ms must be finite and >= 0, got {max_batch_wait_ms}"
        )
    elif not math.isfinite(max_batch_audio_s) or max_batch_audio_s <= 0:
        raise ValueError(
            f"max_batch_audio_s must be finite and > 0, got {max_batch_audio_s}"
        )
    else:
        pass

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.type != "mps":
        raise ValueError(
            f"Parakeet ASR runs only on the MPS device, got device={device!r}"
        )
    else:
        pass
    runner = make_parakeet_runner(model_path, device=concrete_device, dtype=dtype)
    batch_fn = make_parakeet_batch_fn(
        request_builder=make_parakeet_request_builder(sample_rate=runner.sample_rate),
        transcribe=runner.transcribe,
        max_padded_samples=int(max_batch_audio_s * runner.sample_rate),
    )

    def transcribe_one(payload: StagePayload) -> StagePayload:
        (result,) = batch_fn([payload])
        if isinstance(result, BaseException):
            raise result
        else:
            return result

    return SimpleScheduler(
        transcribe_one,
        batch_compute_fn=batch_fn,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "create_parakeet_asr_executor",
    "make_parakeet_batch_fn",
    "make_parakeet_runner",
    "plan_padded_batches",
]
