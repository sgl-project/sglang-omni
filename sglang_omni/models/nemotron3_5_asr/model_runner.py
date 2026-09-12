# SPDX-License-Identifier: Apache-2.0
"""Model-owned batched RNN-T inference for Nemotron 3.5 ASR."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto import StagePayload
from sglang_omni.utils.checkpoint import resolve_checkpoint

from .hf_compat import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrProcessor,
)
from .request_builders import NEMOTRON_ASR_SAMPLE_RATE, Nemotron3_5ASRRequest


def _move_model_inputs(
    inputs: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            moved[name] = value
        elif value.is_floating_point():
            moved[name] = value.to(device=device, dtype=dtype)
        else:
            moved[name] = value.to(device=device)
    return moved


class Nemotron3_5ASRModelRunner:
    """Own one processor/model pair and serialize its mutable generate path."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        dtype: str | torch.dtype = "float32",
        num_lookahead_tokens: int = 3,
    ) -> None:
        checkpoint = str(Path(resolve_checkpoint(model_path)).resolve())
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is None:
            raise ValueError("dtype must resolve to a concrete torch dtype")

        self.device = torch.device(device)
        self.dtype = resolved_dtype
        self.processor = Nemotron3_5AsrProcessor.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        self.processor.set_num_lookahead_tokens(int(num_lookahead_tokens))
        config = Nemotron3_5AsrConfig.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        self.model = Nemotron3_5AsrForRNNT.from_pretrained(
            checkpoint,
            config=config,
            dtype=resolved_dtype,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        # Note (LG-0927): Upstream generate stores encoder/decoder progress on
        # the model. Even callers outside SimpleScheduler must never overlap calls.
        self._model_lock = threading.Lock()

    @property
    def prompt_dictionary(self) -> dict[str, int]:
        return dict(self.processor.prompt_dictionary)

    def _generate_compatible_batch(
        self,
        requests: Sequence[Nemotron3_5ASRRequest],
        *,
        max_new_tokens: int | None,
    ) -> list[StagePayload]:
        processor_inputs = self.processor(
            [request.waveform for request in requests],
            sampling_rate=NEMOTRON_ASR_SAMPLE_RATE,
            language=[request.language for request in requests],
            padding="longest",
            return_tensors="pt",
        )
        model_inputs = _move_model_inputs(
            dict(processor_inputs), device=self.device, dtype=self.dtype
        )
        generate_kwargs: dict[str, Any] = {"return_dict_in_generate": True}
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens

        started_at_s = time.perf_counter()
        with self._model_lock, torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generate_kwargs)
        elapsed_s = time.perf_counter() - started_at_s
        sequences = generated.sequences.detach().to("cpu")
        raw_texts = self.processor.batch_decode(
            sequences,
            skip_special_tokens=False,
        )
        if len(raw_texts) != len(requests):
            raise RuntimeError(
                "Nemotron processor returned "
                f"{len(raw_texts)} transcripts for {len(requests)} requests"
            )

        results: list[StagePayload] = []
        for request, raw_text in zip(requests, raw_texts):
            payload = request.stage_payload
            stage_latency_s = (
                time.perf_counter() - request.started_at_s
                if request.started_at_s
                else elapsed_s
            )
            results.append(
                StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data={
                        "text": str(raw_text).strip(),
                        "language": request.language,
                        "duration_s": request.duration_s,
                        "asr_latency_s": stage_latency_s,
                        "model_latency_s": elapsed_s,
                        "batch_size": len(requests),
                        "usage": {"engine_time_s": stage_latency_s},
                        "modality": "text",
                    },
                )
            )
        return results

    def run_batch(
        self, requests: Sequence[Nemotron3_5ASRRequest]
    ) -> list[StagePayload]:
        if not requests:
            return []

        # Note (LG-0927): GenerationConfig has one output cap for a whole tensor
        # batch. Keep explicit per-request caps exact by batching only compatible
        # requests; the normal endpoint path remains one true generate call.
        groups: OrderedDict[int | None, list[tuple[int, Nemotron3_5ASRRequest]]] = (
            OrderedDict()
        )
        for index, request in enumerate(requests):
            groups.setdefault(request.max_new_tokens, []).append((index, request))

        ordered_results: list[StagePayload | None] = [None] * len(requests)
        for max_new_tokens, indexed_requests in groups.items():
            compatible = [request for _, request in indexed_requests]
            batch_results = self._generate_compatible_batch(
                compatible,
                max_new_tokens=max_new_tokens,
            )
            for (index, _), result in zip(indexed_requests, batch_results):
                ordered_results[index] = result
        if any(result is None for result in ordered_results):
            raise RuntimeError("Nemotron batch result ordering was incomplete")
        return [result for result in ordered_results if result is not None]

    def run_one(self, request: Nemotron3_5ASRRequest) -> StagePayload:
        return self.run_batch([request])[0]

    def close(self) -> None:
        # Note (LG-0927): The worker process normally exits after shutdown;
        # dropping references here also makes explicit scheduler teardown release
        # model ownership.
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]


__all__ = ["Nemotron3_5ASRModelRunner"]
