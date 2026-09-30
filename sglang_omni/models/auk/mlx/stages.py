# SPDX-License-Identifier: Apache-2.0
"""Native MLX stages with CPU tensor transport at pipeline boundaries."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import make_runtime_config
from sglang_omni.models.auk.mlx.conditioning import AuKMlxConditionEncoder
from sglang_omni.models.auk.mlx.flow_matching import (
    AuKFlowMatching,
    AuKSampleItem,
    fuse_hidden_states,
    request_key,
)
from sglang_omni.models.auk.mlx.loader import (
    load_flow,
    load_fusion,
    load_vae,
    resolve_dtype,
    validate_device,
)
from sglang_omni.models.auk.mlx.vae import BigVGANFlowVAE
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.reference_encode import build_messages
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint


def cpu_tensor(value: mx.array) -> torch.Tensor:
    value = value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
    return torch.from_numpy(np.array(value))


def mlx_tensor(value: torch.Tensor) -> mx.array:
    return mx.array(value.numpy())


def scheduler(
    compute_batch: Callable[[list[StagePayload]], list[StagePayload]],
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler:
    def run(payloads: list[StagePayload]) -> list[StagePayload]:
        # note (Codex): MLX stream identities belong to the executing thread.
        with mx.stream(mx.gpu):
            return compute_batch(payloads)

    return SimpleScheduler(
        lambda payload: run([payload])[0],
        batch_compute_fn=run,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
    )


def condition_batch(
    payloads: list[StagePayload],
    encoder: AuKMlxConditionEncoder,
    vae: BigVGANFlowVAE,
    fusion: tuple[mx.array, mx.array],
) -> list[StagePayload]:
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    messages = [
        build_messages(state.instruction, state.ref_audio is not None)
        for state in states
    ]
    for state in states:
        if state.ref_audio is not None:
            waveform = mx.array(np.asarray(state.ref_audio).reshape(1, 1, -1))
            lengths = mx.array([waveform.shape[-1] // vae.hop_size * vae.hop_size])
            latent, lengths = vae.encode(
                waveform, lengths, key=request_key(state.seed, stream="reference")
            )
            state.ref_latent = cpu_tensor(latent[0])
            state.ref_length = int(lengths[0].item())
        else:
            pass
    encodings = encoder.encode_batch(messages, [state.qwen_audio for state in states])
    for state, (hidden, mask) in zip(states, encodings):
        state.conditioning = cpu_tensor(fuse_hidden_states(hidden[None], *fusion)[0])
        state.text_mask = cpu_tensor(mask)
        state.prompt_tokens = int(state.text_mask.sum().item())
        state.ref_audio = state.qwen_audio = None
        state.engine_time_s += time.perf_counter() - started
    return [store_state(payload, state) for payload, state in zip(payloads, states)]


def create_conditioning_executor(
    model_path: str,
    *,
    device: str | None,
    gpu_id: int | None,
    dtype: str,
    text_encoder_path: str,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler:
    validate_device(device, gpu_id)
    compute_dtype = resolve_dtype(dtype)
    checkpoint = resolve_checkpoint(model_path)
    encoder = AuKMlxConditionEncoder(text_encoder_path, dtype=compute_dtype)
    vae = load_vae(checkpoint)
    fusion = load_fusion(checkpoint)
    return scheduler(
        lambda payloads: condition_batch(payloads, encoder, vae, fusion),
        max_batch_size,
        max_batch_wait_ms,
    )


def sample_batch(
    payloads: list[StagePayload],
    flow: AuKFlowMatching,
    max_frames: int,
    *,
    steps: int,
    cfg_strength: float,
    sway_sampling_coef: float | None,
    t_grid: Sequence[float] | None,
) -> list[StagePayload]:
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    items = [
        AuKSampleItem(
            conditioning=mlx_tensor(state.conditioning),
            text_mask=mlx_tensor(state.text_mask),
            target_frames=min(max(state.gen_frames, 1), max_frames),
            ref_latent=(
                mlx_tensor(state.ref_latent) if state.ref_latent is not None else None
            ),
            seed=state.seed,
            ref_length=state.ref_length,
        )
        for state in states
    ]
    latents = flow.sample_batch(
        items,
        steps=steps,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        t_grid=t_grid,
    )
    for state, latent in zip(states, latents):
        if not mx.all(mx.isfinite(latent)).item():
            raise RuntimeError("AuK generated latent contains NaN/Inf")
        else:
            state.latent = cpu_tensor(latent)
            state.conditioning = state.text_mask = state.ref_latent = None
            state.completion_tokens = latent.shape[0]
            state.engine_time_s += time.perf_counter() - started
    return [store_state(payload, state) for payload, state in zip(payloads, states)]


def create_auk_engine_executor(
    model_path: str,
    *,
    device: str | None,
    gpu_id: int | None,
    dtype: str,
    weight_dtype: str,
    nfe: int,
    cfg_strength: float,
    sway_sampling_coef: float | None,
    max_seconds: float,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler:
    validate_device(device, gpu_id)
    compute_dtype = resolve_dtype(dtype)
    storage_dtype = resolve_dtype(weight_dtype)
    if compute_dtype != storage_dtype:
        raise ValueError("AuK native MLX requires matching dtype and weight_dtype")
    else:
        pass
    checkpoint = resolve_checkpoint(model_path)
    config = make_runtime_config(checkpoint)
    flow = load_flow(checkpoint, storage_dtype)
    max_frames = config.seconds_to_frames(max_seconds)
    sampling = dict(
        steps=C.FLASH_NFE if config.is_flash else nfe,
        cfg_strength=C.FLASH_CFG_STRENGTH if config.is_flash else cfg_strength,
        sway_sampling_coef=None if config.is_flash else sway_sampling_coef,
        t_grid=C.FLASH_T_GRID if config.is_flash else None,
    )
    return scheduler(
        lambda payloads: sample_batch(payloads, flow, max_frames, **sampling),
        max_batch_size,
        max_batch_wait_ms,
    )


def decode_batch(
    payloads: list[StagePayload], vae: BigVGANFlowVAE
) -> list[StagePayload]:
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.latent.shape[0]].append(index)
    for indices in groups.values():
        latents = mx.stack([mlx_tensor(states[index].latent) for index in indices])
        waveforms = vae.decode(latents)
        if not mx.all(mx.isfinite(waveforms)).item():
            raise RuntimeError("AuK generated audio contains NaN/Inf")
        else:
            for index, waveform in zip(indices, np.asarray(waveforms)):
                state = states[index]
                state.latent = None
                state.engine_time_s += time.perf_counter() - started
                payload = store_state(payloads[index], state)
                payload.data.update(
                    audio_waveform_payload(
                        waveform, sample_rate=state.sample_rate, source_hint="AuK MLX"
                    ),
                    modality="audio",
                    usage=build_usage(state),
                )
    return payloads


def create_decode_executor(
    model_path: str,
    *,
    device: str | None,
    gpu_id: int | None,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler:
    validate_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    vae = load_vae(checkpoint)
    return scheduler(
        lambda payloads: decode_batch(payloads, vae), max_batch_size, max_batch_wait_ms
    )
