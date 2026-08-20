# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 stage factories — multi-worker AR with concurrent requests."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

from .acoustic import MiniMaxMusic3AcousticDecoder, MiniMaxMusic3AcousticScheduler
from .constants import DEFAULT_DIT_CFG_SCALE, DEFAULT_DIT_STEPS, OUTPUT_SAMPLE_RATE
from .request_builders import build_ttm_state

logger = logging.getLogger(__name__)

_DEFAULT_AR_CONCURRENCY = int(os.environ.get("MINIMAX_MUSIC3_AR_CONCURRENCY", "16"))


def create_preprocessing_executor(model_path: str, **_: Any) -> SimpleScheduler:
    del model_path

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_ttm_state(payload)
        logger.info(
            f"MiniMax Music 3 preprocessing request={payload.request_id} seed={state.seed} max_frames={state.max_audio_frames}"
        )
        return store_state(payload, state)

    return SimpleScheduler(_preprocess, max_concurrency=1)


def create_ar_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    device: str | None = None,
    max_concurrency: int = _DEFAULT_AR_CONCURRENCY,
    server_args_overrides: dict[str, Any] | None = None,
    **_: Any,
):
    if device is None:
        if gpu_id is None or not torch.cuda.is_available():
            raise RuntimeError("MiniMax Music 3 supports CUDA only right now")
        device = f"cuda:{gpu_id}"
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_cudnn_sdp(False)

    from .engine_builder import MiniMaxMusic3EngineBuilder

    overrides = dict(server_args_overrides or {})
    requested = overrides.get("max_running_requests")
    if requested is not None:
        max_concurrency = int(requested)
    builder = MiniMaxMusic3EngineBuilder(
        max_running_requests=max(int(max_concurrency), 1)
    )
    scheduler = builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype="bfloat16",
        server_args_overrides=overrides or None,
    )
    logger.info(
        f"MiniMax Music 3 AR executor ready (OmniScheduler) device={device} max_running_requests={builder.max_running_requests}"
    )
    return scheduler


def create_dit_dav_executor(
    model_path: str,
    *,
    gpu_id: int | None = 1,
    device: str | None = None,
    dtype: str = "float32",
    dit_steps: int = DEFAULT_DIT_STEPS,
    dit_cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
    attention_backend: str = "torch_sdpa",
    cache_dit: bool = False,
    compile_acoustic: bool = True,
    breakable_cuda_graph: bool = False,
    breakable_cuda_graph_min_free_gb: float | None = None,
    cache_dit_fn_compute_blocks: int = 2,
    cache_dit_bn_compute_blocks: int = 2,
    cache_dit_max_warmup_steps: int = 4,
    cache_dit_residual_diff_threshold: float = 0.08,
    cache_dit_max_continuous_cached_steps: int = 1,
    **_: Any,
) -> MiniMaxMusic3AcousticScheduler:
    if device is None:
        if gpu_id is None or not torch.cuda.is_available():
            raise RuntimeError("MiniMax Music 3 acoustic inference requires CUDA")
        device = f"cuda:{gpu_id}"
    decoder = MiniMaxMusic3AcousticDecoder(
        model_path,
        device=device,
        dtype=dtype,
        dit_steps=dit_steps,
        dit_cfg_scale=dit_cfg_scale,
        attention_backend=attention_backend,
        cache_dit=cache_dit,
        compile_acoustic=compile_acoustic,
        breakable_cuda_graph=breakable_cuda_graph,
        breakable_cuda_graph_min_free_gb=breakable_cuda_graph_min_free_gb,
        cache_dit_fn_compute_blocks=cache_dit_fn_compute_blocks,
        cache_dit_bn_compute_blocks=cache_dit_bn_compute_blocks,
        cache_dit_max_warmup_steps=cache_dit_max_warmup_steps,
        cache_dit_residual_diff_threshold=cache_dit_residual_diff_threshold,
        cache_dit_max_continuous_cached_steps=cache_dit_max_continuous_cached_steps,
    )
    logger.info(
        f"MiniMax Music 3 acoustic executor ready device={decoder.device} dtype={decoder.dtype} dit_steps={decoder.dit_steps} dit_cfg_scale={decoder.dit_cfg_scale:.3f} attention_backend={decoder.attention_backend} compile_acoustic={decoder.compile_acoustic} sample_rate={OUTPUT_SAMPLE_RATE}"
    )
    return MiniMaxMusic3AcousticScheduler(decoder)


__all__ = [
    "create_ar_executor",
    "create_dit_dav_executor",
    "create_preprocessing_executor",
]
