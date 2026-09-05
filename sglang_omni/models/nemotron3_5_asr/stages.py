# SPDX-License-Identifier: Apache-2.0
"""Stage factory for Nemotron 3.5 ASR."""

from __future__ import annotations

from collections.abc import Sequence

from sglang_omni.proto import StagePayload
from sglang_omni.utils.device import resolve_device_spec

from .model_runner import Nemotron3_5ASRModelRunner
from .request_builders import (
    Nemotron3_5ASRRequest,
    make_nemotron3_5_asr_request_builder,
)
from .streaming import Nemotron3_5ASRStreamingScheduler


def create_nemotron3_5_asr_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    num_lookahead_tokens: int = 3,
    max_batch_size: int = 8,
    max_batch_wait_ms: float = 2.0,
    max_pending_stream_messages: int = 256,
) -> Nemotron3_5ASRStreamingScheduler:
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1")
    if max_batch_wait_ms < 0:
        raise ValueError("max_batch_wait_ms must be non-negative")
    if max_pending_stream_messages < 1:
        raise ValueError("max_pending_stream_messages must be at least 1")

    resolved_device = resolve_device_spec(device, gpu_id)
    runner = Nemotron3_5ASRModelRunner(
        model_path,
        device=resolved_device,
        dtype=dtype,
        num_lookahead_tokens=num_lookahead_tokens,
    )
    build_request = make_nemotron3_5_asr_request_builder(
        prompt_dictionary=runner.prompt_dictionary
    )

    def run_one(payload: StagePayload) -> StagePayload:
        return runner.run_one(build_request(payload))

    def run_batch(
        payloads: Sequence[StagePayload],
    ) -> list[StagePayload | BaseException]:
        results: list[StagePayload | BaseException | None] = [None] * len(payloads)
        valid: list[tuple[int, Nemotron3_5ASRRequest]] = []
        for index, payload in enumerate(payloads):
            try:
                valid.append((index, build_request(payload)))
            except Exception as exc:
                results[index] = exc

        if valid:
            try:
                batch_results: list[StagePayload | BaseException] = list(
                    runner.run_batch([request for _, request in valid])
                )
            except Exception as exc:
                batch_results = [exc] * len(valid)
            if len(batch_results) != len(valid):
                batch_results = [
                    RuntimeError(
                        "Nemotron runner returned "
                        f"{len(batch_results)} results for {len(valid)} requests"
                    )
                ] * len(valid)
            for (index, _), result in zip(valid, batch_results):
                results[index] = result

        if any(result is None for result in results):
            raise RuntimeError("Nemotron batch result isolation was incomplete")
        return [result for result in results if result is not None]

    return Nemotron3_5ASRStreamingScheduler(
        runner,
        run_one,
        batch_compute_fn=run_batch,
        prompt_dictionary=runner.prompt_dictionary,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        max_pending_messages=max_pending_stream_messages,
    )


__all__ = ["create_nemotron3_5_asr_executor"]
