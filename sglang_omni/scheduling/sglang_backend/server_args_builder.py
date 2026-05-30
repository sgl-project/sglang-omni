# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines and encoders."""
from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    chunked_prefill_size: int | None = None,
    max_prefill_tokens: int = 16384,
    max_running_requests: int = 16,
    mem_fraction_static: float | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Build ServerArgs with shared defaults for all SGLang AR engines."""
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "random_seed": 123,
        "context_length": context_length,
    }
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static
    kwargs.update(overrides)
    if kwargs.get("mem_fraction_static") is None:
        kwargs.pop("mem_fraction_static", None)
    return ServerArgs(**kwargs)


_ENCODER_PROTECTED_KEYS = frozenset({
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "moe_dense_tp_size",
    "nnodes",
    "node_rank",
    "rank",
    "world_size",
    "tp_rank",
    "gpu_id",
    "base_gpu_id",
    "nccl_port",
    "dist_init_addr",
    "encoder_only",
    "language_only",
    "mm_enable_dp_encoder",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "disable_cuda_graph",
    "device",
    "mem_fraction_static",
    "max_running_requests",
    "max_prefill_tokens",
    "chunked_prefill_size",
    "context_length",
})


def build_sglang_encoder_server_args(
    model_path: str,
    *,
    tp_size: int,
    base_gpu_id: int,
    dist_init_addr: str,
    dtype: str | None = None,
    load_format: str | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Build encoder-only ServerArgs with runner-owned topology locked down."""
    bad = sorted(_ENCODER_PROTECTED_KEYS & overrides.keys())
    if bad:
        raise ValueError(
            f"server_args_overrides cannot override protected keys: {bad}. "
            "These are decided by the encoder runner / pipeline runner."
        )

    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "pp_size": 1,
        "base_gpu_id": base_gpu_id,
        "dist_init_addr": dist_init_addr,
        "encoder_only": True,
        "language_only": False,
        "mm_enable_dp_encoder": False,
        "disable_cuda_graph": True,
        "random_seed": 123,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if load_format is not None:
        kwargs["load_format"] = load_format
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def apply_encoder_mem_reserve(
    server_args: ServerArgs,
    encoder_mem_reserve: float,
) -> None:
    """Subtract Qwen external encoder headroom from an auto-selected SGLang budget."""
    if not 0.0 <= encoder_mem_reserve < 1.0:
        raise ValueError("encoder_mem_reserve must be in [0, 1)")
    if encoder_mem_reserve == 0:
        return

    current = server_args.mem_fraction_static
    if current is None:
        return

    reserved = current - encoder_mem_reserve
    if reserved < 0.1:
        raise ValueError(
            f"auto mem_fraction_static {current:.3f} minus encoder_mem_reserve "
            f"{encoder_mem_reserve:.3f} = {reserved:.3f} is below the safe "
            "floor 0.1; lower encoder_mem_reserve or pin mem_fraction_static "
            "explicitly."
        )
    server_args.mem_fraction_static = round(reserved, 3)
