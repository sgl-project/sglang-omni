# SPDX-License-Identifier: Apache-2.0 AND MIT
# Inference recipe adapted from Tencent-Hunyuan/AuK, Copyright (C) 2026 Tencent.
# See LICENSE for the upstream MIT permission notice.
"""AuK conditioning, sampling and decoding with optional component switching."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Mapping
from threading import Event
from typing import Protocol

import numpy as np
import torch

from sglang_omni.models.auk.flow_matching import (
    AuKFlowMatching,
    AuKSampleItem,
    fuse_hidden_states,
    request_generator,
)
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.reference_encode import AuKConditionEncoder, build_messages
from sglang_omni.models.auk.runtime import autocast
from sglang_omni.models.auk.serial_offload import AuKOffloadCoordinator
from sglang_omni.models.auk.step_cuda_graph import AuKStepCudaGraphRunner
from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


class BatchCompute(Protocol):
    def __call__(self, payloads: list[StagePayload]) -> list[StagePayload]: ...


def reference_latent(
    vae: BigVGANFlowVAE,
    device: torch.device,
    audio: np.ndarray[tuple[int, ...], np.dtype[np.float32]] | None,
    seed: int | None = None,
) -> tuple[torch.Tensor | None, int]:
    if audio is None:
        return None, 0
    else:
        pass
    waveform = torch.from_numpy(
        np.asarray(audio, dtype=np.float32).reshape(1, 1, -1)
    ).to(device)
    lengths = torch.tensor(
        [waveform.shape[-1] // vae.hop_size * vae.hop_size], device=device
    )
    latent, lengths = vae.encoding_and_normalization(
        waveform, lengths, generator=request_generator(seed, device)
    )
    return latent[0], int(lengths[0])


def condition_batch(
    payloads: list[StagePayload],
    encoder: AuKConditionEncoder,
    vae: BigVGANFlowVAE,
    fusion: tuple[torch.Tensor, ...],
    device: torch.device,
    dtype: torch.dtype,
    offload: AuKOffloadCoordinator | None = None,
) -> list[StagePayload]:
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    messages = [
        build_messages(state.instruction, state.ref_audio is not None)
        for state in states
    ]
    if offload is not None and any(state.ref_audio is not None for state in states):
        offload.switch("vae")
    else:
        pass
    for state in states:
        state.ref_latent, state.ref_length = reference_latent(
            vae, device, state.ref_audio, state.seed
        )
    if offload is not None:
        if offload.abort_event.is_set():
            raise InterruptedError("AuK conditioning cancelled")
        else:
            offload.switch("conditioning")
    else:
        pass
    with autocast(device, dtype):
        encodings = encoder.encode_batch(
            messages, [state.qwen_audio for state in states]
        )
        for state, (hidden, mask) in zip(states, encodings):
            state.conditioning = fuse_hidden_states(hidden.unsqueeze(0), *fusion)[0]
            state.text_mask = mask
            state.prompt_tokens = int(mask.sum())
    for state in states:
        state.ref_audio = state.qwen_audio = None
        state.engine_time_s += time.perf_counter() - started
    return [store_state(payload, state) for payload, state in zip(payloads, states)]


def sample_batch(
    payloads: list[StagePayload],
    flow: AuKFlowMatching,
    device: torch.device,
    dtype: torch.dtype,
    max_frames: int,
    sampling: Mapping[
        str, int | float | tuple[float, ...] | AuKStepCudaGraphRunner | Event | None
    ],
    offload: AuKOffloadCoordinator | None = None,
) -> list[StagePayload]:
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    items = [
        AuKSampleItem(
            state.conditioning.to(device),
            state.text_mask.to(device),
            min(max(state.gen_frames, 1), max_frames),
            state.ref_latent.to(device) if state.ref_latent is not None else None,
            state.seed,
            state.ref_length,
        )
        for state in states
    ]
    if offload is not None:
        offload.switch("dit")
        sampling = {**sampling, "abort_event": offload.abort_event}
    else:
        pass
    logger.info(f"AuK DiT: sampling batch of {len(items)} requests")
    with autocast(device, dtype):
        latents = flow.sample_batch(items, **sampling)
    for state, latent in zip(states, latents):
        if not torch.isfinite(latent).all():
            raise RuntimeError("AuK generated latent contains NaN/Inf")
        else:
            pass
        state.latent = latent
        state.conditioning = state.text_mask = state.ref_latent = None
        state.completion_tokens = latent.shape[0]
        state.engine_time_s += time.perf_counter() - started
    return [store_state(payload, state) for payload, state in zip(payloads, states)]


def decode_batch(
    payloads: list[StagePayload],
    vae: BigVGANFlowVAE,
    device: torch.device,
    offload: AuKOffloadCoordinator | None = None,
) -> list[StagePayload]:
    if offload is not None:
        offload.switch("vae")
    else:
        pass
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.latent.shape[0]].append(index)
    results = list(payloads)
    for indices in groups.values():
        latents = torch.stack([states[i].latent for i in indices]).to(device)
        waveforms = vae.inference_from_latents(
            vae.denormalize(latents).permute(0, 2, 1)
        )
        if not torch.isfinite(waveforms).all():
            raise RuntimeError("AuK generated audio contains NaN/Inf")
        else:
            pass
        for index, wav in zip(indices, waveforms.float().cpu()):
            state = states[index]
            state.latent = None
            state.engine_time_s += time.perf_counter() - started
            payload = store_state(payloads[index], state)
            payload.data.update(
                audio_waveform_payload(
                    wav, sample_rate=state.sample_rate, source_hint="AuK"
                )
            )
            payload.data.update(
                sample_rate=state.sample_rate,
                modality="audio",
                usage=build_usage(state),
            )
            results[index] = payload
    return results
