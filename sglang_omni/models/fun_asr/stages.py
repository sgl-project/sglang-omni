# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import torch

# note(LauraGPT): Auto* loading depends on these local registrations.
import sglang_omni.models.fun_asr.configuration_fun_asr  # noqa: F401

logger = logging.getLogger(__name__)


def _compile_fun_asr_audio_encoder(
    model: Any, *, warmup_lfr_frames: int = 128, warmup_inference_mode: bool = True
) -> None:
    """Compile the SANM encoder and adaptor with a symbolic sequence length.

    The LFR frame count varies with audio duration, so ``dynamic=True`` builds
    one symbolic-shape graph instead of specializing per length (which would
    recompile per new length until Dynamo's recompile limit silently falls
    back to eager). The bound forwards are compiled rather than wrapping the
    modules in ``OptimizedModule`` so parameter names stay stable for
    ``load_weights`` and weight updates. The warmup forward pays the one-time
    compile cost at startup instead of on the first request; Dynamo guards on
    grad mode, so the warmup must run in the same mode as the serving caller —
    ``torch.inference_mode`` for the pre-LM encoder service
    (``_batch_context``), ambient mode for inline prefill on the scheduler
    loop.
    """
    import contextlib

    from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config

    from sglang_omni.models.fun_asr.sglang_model import _sanm_mask_from_lengths

    if warmup_lfr_frames < 2:
        # Note (wilsonzheng0327) Sizes 0/1 are always shape-specialized by
        # Dynamo; warming up with them would not build the symbolic-length graph.
        raise ValueError(f"warmup_lfr_frames must be >= 2, got {warmup_lfr_frames}")
    set_torch_compile_config()
    model.audio_tower.forward = torch.compile(model.audio_tower.forward, dynamic=True)
    model.multi_modal_projector.forward = torch.compile(
        model.multi_modal_projector.forward, dynamic=True
    )
    param = next(model.audio_tower.parameters())
    warmup_ctx = (
        torch.inference_mode() if warmup_inference_mode else contextlib.nullcontext()
    )
    with warmup_ctx:
        # Note (wilsonzheng0327): tensor must be created inside the context,
        # not just passed through it; tensors allocated under inference_mode
        # lack the ADInplaceOrView dispatch key, and Dynamo guards on the key
        # set, so a normal tensor here compiles a graph the service's
        # inference-mode tensors fail, forcing a full recompile on the first
        # real request
        t = int(warmup_lfr_frames)
        feat_dim = int(model.config.encoder_config.input_size)

        # note(guozhihao-224): Dynamo specializes B=0/1 and mask=None vs tensor;
        # B1/None + B1/mask + B2/mask cover the mask branch and the B>=2 dynamic graph.
        for batch, with_mask in ((1, False), (1, True), (2, True)):
            xs = (
                torch.zeros(
                    (batch, feat_dim, t), device=param.device, dtype=param.dtype
                )
                .permute(0, 2, 1)
                .contiguous()
            )
            mask = (
                _sanm_mask_from_lengths(
                    torch.full((batch,), t, device=param.device, dtype=torch.long),
                    t,
                    dtype=param.dtype,
                    device=param.device,
                )
                if with_mask
                else None
            )
            model.multi_modal_projector(model.audio_tower(xs, mask), mask)
    logger.info(
        "Compiled Fun-ASR audio encoder + adaptor "
        "(dynamic=True, warmup_lfr_frames=%d, "
        "warmup_inference_mode=%s, "
        "signatures=B1/None+B1/mask+B2/mask)",
        warmup_lfr_frames,
        warmup_inference_mode,
    )


def create_sglang_fun_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    max_new_tokens: int = 200,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    enable_encoder_torch_compile: bool = False,
    enable_encoder_cuda_graph: bool = False,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    mm_attention_backend: str | None = None,
    enable_pre_lm_encoder: bool = True,
    pre_lm_cache_max_entries: int = 4096,
    pre_lm_cache_size_bytes: int = 2 * 1024**3,
    pre_lm_max_batch_size: int = 8,
    pre_lm_max_batch_wait_ms: int = 4,
    request_build_max_workers: int = 8,
    request_build_max_pending: int | None = 16,
    stream_emit_interval_s: float = 0.05,
    server_args_overrides: dict[str, Any] | None = None,
):
    if pre_lm_max_batch_size < 1:
        raise ValueError(
            f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
        )
    if pre_lm_max_batch_wait_ms < 0:
        raise ValueError(
            f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
        )

    from sglang_omni.models.fun_asr.engine_builder import FunASREngineBuilder

    return FunASREngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        enable_torch_compile=enable_torch_compile,
        enable_encoder_torch_compile=enable_encoder_torch_compile,
        enable_encoder_cuda_graph=enable_encoder_cuda_graph,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        mm_attention_backend=mm_attention_backend,
        enable_pre_lm_encoder=enable_pre_lm_encoder,
        pre_lm_cache_max_entries=pre_lm_cache_max_entries,
        pre_lm_cache_size_bytes=pre_lm_cache_size_bytes,
        pre_lm_max_batch_size=pre_lm_max_batch_size,
        pre_lm_max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        stream_emit_interval_s=stream_emit_interval_s,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_fun_asr_executor(*args, **kwargs):
    return create_sglang_fun_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_fun_asr_executor", "create_fun_asr_executor"]
