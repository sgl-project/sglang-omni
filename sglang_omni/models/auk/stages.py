# SPDX-License-Identifier: Apache-2.0 AND MIT
# Inference recipe adapted from Tencent-Hunyuan/AuK, Copyright (C) 2026 Tencent.
# See LICENSE for the upstream MIT permission notice.
"""Independent conditioning, DiT sampling, and audio decoding stages for AuK."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch
from safetensors import safe_open

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.dit import AuKDit, AuKDitConfig
from sglang_omni.models.auk.flow_matching import (
    AuKFlowMatching,
    AuKSampleItem,
    fuse_hidden_states,
    request_generator,
)
from sglang_omni.models.auk.hf_config import make_runtime_config
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.reference_encode import AuKConditionEncoder, build_messages
from sglang_omni.models.auk.request_builders import (
    AuKPreprocessingContext,
    preprocess_auk_payload,
    set_auk_preprocessing_context,
)
from sglang_omni.models.auk.step_cuda_graph import build_step_graph_runner
from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import (
    load_dit_weights,
    load_vae_weights,
    resolve_weight_file,
)
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(*, field: str, name: str) -> torch.dtype:
    if name not in _TORCH_DTYPES:
        raise ValueError(
            f"AuK {field} must be one of {', '.join(_TORCH_DTYPES)}, got {name!r}"
        )
    return _TORCH_DTYPES[name]


def _autocast(device, dtype):
    # fp32 means "no autocast": the weights already carry the compute dtype.
    return torch.autocast(
        device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
    )


@lru_cache(maxsize=None)
def _load_vae(checkpoint: str, device: str):
    config = make_runtime_config(checkpoint)
    vae = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
    load_vae_weights(vae, checkpoint)
    return vae.to(device=device).eval().requires_grad_(False)


# Order matches the ``fuse_hidden_states`` signature.
_FUSION_PARAMETERS = ("layer_weights", "layer_scale")


@lru_cache(maxsize=None)
def _load_fusion(checkpoint: str, device: str):
    # Read straight from the file so a conditioning-only process does not have to
    # hold the 1.5B DiT these two tensors are stored next to.
    with safe_open(str(resolve_weight_file(checkpoint)), framework="pt") as weights:
        keys = {key.rsplit(".", 1)[-1]: key for key in weights.keys()}
        return tuple(
            weights.get_tensor(keys[name]).to(device=device, dtype=torch.float32)
            for name in _FUSION_PARAMETERS
        )


@lru_cache(maxsize=None)
def _load_flow(
    checkpoint: str, device: str, backbone_dtype: torch.dtype, compile_blocks: bool
):
    config = make_runtime_config(checkpoint)
    layer_weights, _ = _load_fusion(checkpoint, device)
    dit_config = AuKDitConfig.from_dict(config.arch)
    dit = AuKDit(**{**dit_config.__dict__, "latent_dim": config.latent_dim})
    flow = AuKFlowMatching(dit, num_llm_layers=layer_weights.numel())
    load_dit_weights(flow, checkpoint)
    flow = flow.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    # Keyed by dtype and by the compile flag: both mutate the backbone, and a
    # cached one would retroactively change every executor already built on it.
    flow.transformer.to(dtype=backbone_dtype)
    if compile_blocks:
        flow.transformer.enable_compiled_blocks()
    return flow


def _warmup_flow(flow, device, dtype):
    """Pay the block compile once at startup instead of on the first request.

    The blocks compile for dynamic shapes, so a short trajectory pays most of
    the cost; a later length can still trigger a smaller recompile.
    """
    frames, ref, text = 32, 16, 8
    item = AuKSampleItem(
        torch.zeros(text, flow.transformer.txt_proj.in_features, device=device),
        torch.ones(text, dtype=torch.bool, device=device),
        frames,
        torch.zeros(ref, flow.transformer.latent_dim, device=device),
        seed=0,
        ref_length=ref,
    )
    started = time.perf_counter()
    # Under inference_mode like the request path, so dynamo compiles once.
    with torch.inference_mode(), _autocast(device, dtype):
        flow.sample(item, steps=1, cfg_strength=C.DEFAULT_CFG_STRENGTH)
    logger.info("AuK DiT: compiled the blocks in %.1fs", time.perf_counter() - started)


def _scheduler(compute_batch, device, max_batch_size, max_batch_wait_ms):
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    @torch.inference_mode()
    def run(payloads):
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            return compute_batch(payloads)

    return SimpleScheduler(
        lambda payload: run([payload])[0],
        batch_compute_fn=run,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
    default_seconds: float = C.DEFAULT_SECONDS,
    max_seconds: float = C.MAX_SECONDS,
) -> SimpleScheduler:
    config = make_runtime_config(resolve_checkpoint(model_path))
    set_auk_preprocessing_context(
        AuKPreprocessingContext(
            config=config,
            default_seconds=default_seconds,
            max_seconds=max_seconds,
        )
    )
    return SimpleScheduler(preprocess_auk_payload, max_concurrency=max_concurrency)


def _reference_latent(vae, device, audio, seed=None):
    if audio is None:
        return None, 0
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


def _condition_batch(payloads, encoder, vae, fusion, device, dtype):
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    messages = [
        build_messages(state.instruction, state.ref_audio is not None)
        for state in states
    ]
    for state in states:
        state.ref_latent, state.ref_length = _reference_latent(
            vae, device, state.ref_audio, state.seed
        )
    with _autocast(device, dtype):
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


def create_conditioning_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    text_encoder_path: str = C.DEFAULT_TEXT_ENCODER,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 10,
) -> SimpleScheduler:
    compute_dtype = _resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    encoder = AuKConditionEncoder(
        text_encoder_path, device=device, dtype=torch.bfloat16
    )
    vae = _load_vae(checkpoint, str(device))
    fusion = _load_fusion(checkpoint, str(device))
    return _scheduler(
        lambda payloads: _condition_batch(
            payloads, encoder, vae, fusion, device, compute_dtype
        ),
        device,
        max_batch_size,
        max_batch_wait_ms,
    )


def _sample_batch(payloads, flow, device, dtype, max_frames, sampling):
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
    logger.info("AuK DiT: sampling batch of %d requests", len(items))
    with _autocast(device, dtype):
        latents = flow.sample_batch(items, **sampling)
    for state, latent in zip(states, latents):
        if not torch.isfinite(latent).all():
            raise RuntimeError("AuK generated latent contains NaN/Inf")
        state.latent = latent
        state.conditioning = state.text_mask = state.ref_latent = None
        state.completion_tokens = latent.shape[0]
        state.engine_time_s += time.perf_counter() - started
    return [store_state(payload, state) for payload, state in zip(payloads, states)]


def create_auk_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    nfe: int = C.DEFAULT_NFE,
    cfg_strength: float = C.DEFAULT_CFG_STRENGTH,
    sway_sampling_coef: float | None = C.DEFAULT_SWAY_SAMPLING_COEF,
    max_seconds: float = C.MAX_SECONDS,
    max_batch_size: int = 16,
    max_batch_wait_ms: int = 10,
    weight_dtype: str = "float32",
    enable_dit_torch_compile: bool = False,
    enable_dit_cuda_graph: bool = False,
) -> SimpleScheduler:
    """Build the DiT sampling stage.

    ``weight_dtype="float32"`` keeps fp32 weights under ``dtype`` autocast, the
    upstream-exact recipe; ``"bfloat16"`` stores the backbone in bf16 and skips
    autocast. ``enable_dit_torch_compile`` fuses each block's elementwise chain,
    and ``enable_dit_cuda_graph`` replays a whole Euler step from one captured
    graph, padding the batch to shape buckets to do so (see
    docs/cookbook/auk.md, Sampling).
    """
    # Named dtypes are checked before resolve_checkpoint, which downloads.
    compute_dtype = _resolve_dtype(field="dtype", name=dtype)
    backbone_dtype = _resolve_dtype(field="weight_dtype", name=weight_dtype)
    device = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    config = make_runtime_config(checkpoint)
    # A non-fp32 backbone runs natively, and _autocast reads fp32 as "off":
    # autocast would only re-cast per op and force the norms back to fp32.
    autocast_dtype = compute_dtype if backbone_dtype == torch.float32 else torch.float32
    flow = _load_flow(checkpoint, str(device), backbone_dtype, enable_dit_torch_compile)
    sampling = dict(
        steps=C.FLASH_NFE if config.is_flash else nfe,
        cfg_strength=C.FLASH_CFG_STRENGTH if config.is_flash else cfg_strength,
        sway_sampling_coef=None if config.is_flash else sway_sampling_coef,
        t_grid=C.FLASH_T_GRID if config.is_flash else None,
    )
    if enable_dit_torch_compile:
        _warmup_flow(flow, device, autocast_dtype)
    if enable_dit_cuda_graph:
        if not flow.transformer.attn_mask_enabled:
            raise ValueError(
                "AuK enable_dit_cuda_graph needs attn_mask_enabled: without the "
                "attention bias, padded rows would reach the valid ones"
            )
        step_graph = build_step_graph_runner(device)
        if step_graph is not None:
            sampling["step_graph"] = step_graph
    return _scheduler(
        lambda payloads: _sample_batch(
            payloads,
            flow,
            device,
            autocast_dtype,
            config.seconds_to_frames(max_seconds),
            sampling,
        ),
        device,
        max_batch_size,
        max_batch_wait_ms,
    )


def _decode_batch(payloads, vae, device):
    started = time.perf_counter()
    states = [load_state(payload, AuKState) for payload in payloads]
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.latent.shape[0]].append(index)
    results = [None] * len(states)
    for indices in groups.values():
        latents = torch.stack([states[i].latent for i in indices]).to(device)
        waveforms = vae.inference_from_latents(
            vae.denormalize(latents).permute(0, 2, 1)
        )
        if not torch.isfinite(waveforms).all():
            raise RuntimeError("AuK generated audio contains NaN/Inf")
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


def create_decode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 4,
    max_batch_wait_ms: int = 10,
) -> SimpleScheduler:
    device = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    vae = _load_vae(checkpoint, str(device))
    return _scheduler(
        lambda payloads: _decode_batch(payloads, vae, device),
        device,
        max_batch_size,
        max_batch_wait_ms,
    )
