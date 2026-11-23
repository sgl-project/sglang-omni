# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 stages: request validation, then batched generation on one GPU."""

import logging
import time
from pathlib import Path

import torch

from sglang_omni.models.stable_audio3.cuda_graphs import (
    DiTCudaGraphRunner,
    PromptEncoderCudaGraphRunner,
)
from sglang_omni.models.stable_audio3.payload_types import StableAudio3State
from sglang_omni.models.stable_audio3.request_builders import build_state
from sglang_omni.models.stable_audio3.runtime import (
    GenerationRequest,
    StableAudio3Runtime,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

# note (ruilin): one batch covers the DiT, a single-chunk decode and a multi-chunk decode, so serving starts compiled.
WARMUP_REQUESTS = [
    GenerationRequest(prompt="Warm piano chords", seconds=4.0, seed=0),
    GenerationRequest(prompt="Warm piano chords", seconds=30.0, seed=0),
]

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention_v3 import (  # noqa: leading-underscore  # upstream spelling
        _is_fa3_supported,
    )
else:
    pass


def create_preprocessing_executor() -> SimpleScheduler[StagePayload, StagePayload]:
    return SimpleScheduler(lambda payload: store_state(payload, build_state(payload)))


def create_generation_executor(
    model_path: str,
    *,
    max_batch_size: int,
    max_batch_wait_ms: float,
    enable_cuda_graph: bool,
    enable_torch_compile: bool,
    gpu_id: int | None = None,
    device: str | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    resolved = resolve_concrete_device(device, gpu_id)
    if resolved.type != "cuda":
        raise ValueError("Stable Audio 3 serving requires a CUDA device")
    else:
        pass
    # note (ruilin): the checkpoint's own inference runs the DiT and decoder in fp16, with FlashAttention where the GPU has it.
    runtime = StableAudio3Runtime.from_pretrained(
        Path(resolve_checkpoint(model_path)),
        resolved,
        torch.float16,
        attention_backend=(
            "flash"
            if current_platform.is_cuda() and _is_fa3_supported(resolved)
            else "sdpa"
        ),
    )
    if enable_torch_compile:
        runtime.compile_blocks()
    else:
        pass
    graph_backend = (
        current_platform.get_device_graph_backend(resolved)
        if enable_cuda_graph
        else None
    )
    if graph_backend is None:
        pass
    else:
        runtime.dit_graphs = DiTCudaGraphRunner(
            runtime.dit, graph_backend=graph_backend, max_batch_size=max_batch_size
        )
        runtime.prompt_graphs = PromptEncoderCudaGraphRunner(
            runtime.text_encoder,
            graph_backend=graph_backend,
            max_batch_size=max_batch_size,
        )
    if enable_torch_compile:
        started = time.perf_counter()
        runtime.generate(WARMUP_REQUESTS)
        logger.info(
            f"Stable Audio 3 blocks compiled in {time.perf_counter() - started:.1f}s"
        )
    else:
        pass

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
