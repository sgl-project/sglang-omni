# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 stages: request validation, then batched generation on one GPU."""

from pathlib import Path

import torch

from sglang_omni.models.stable_audio3.payload_types import StableAudio3State
from sglang_omni.models.stable_audio3.request_builders import build_state
from sglang_omni.models.stable_audio3.runtime import (
    GenerationRequest,
    StableAudio3Runtime,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device


def create_preprocessing_executor() -> SimpleScheduler[StagePayload, StagePayload]:
    return SimpleScheduler(lambda payload: store_state(payload, build_state(payload)))


def create_generation_executor(
    model_path: str,
    *,
    max_batch_size: int,
    max_batch_wait_ms: float,
    gpu_id: int | None = None,
    device: str | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    resolved = resolve_concrete_device(device, gpu_id)
    if resolved.type != "cuda":
        raise ValueError("Stable Audio 3 serving requires a CUDA device")
    else:
        pass
    # note (ruilin): the checkpoint's own inference runs the DiT and decoder in fp16.
    runtime = StableAudio3Runtime.from_pretrained(
        Path(resolve_checkpoint(model_path)), resolved, torch.float16
    )

    def generate(payloads: list[StagePayload]) -> list[StagePayload]:
        states = [load_state(payload, StableAudio3State) for payload in payloads]
        audio = runtime.generate(
            [
                GenerationRequest(
                    prompt=state.prompt, seconds=state.duration_seconds, seed=state.seed
                )
                for state in states
            ]
        )
        return [
            StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=audio_waveform_payload(
                    waveform,
                    sample_rate=runtime.config.sample_rate,
                    modality="audio",
                    source_hint="Stable Audio 3",
                    keep_channels=True,
                ),
            )
            for payload, waveform in zip(payloads, audio, strict=True)
        ]

    return SimpleScheduler(
        lambda payload: generate([payload])[0],
        batch_compute_fn=generate,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
    )
