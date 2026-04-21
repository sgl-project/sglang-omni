# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines."""
from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    chunked_prefill_size: int = 128,
    max_prefill_tokens: int = 4096,
    max_running_requests: int = 16,
    mem_fraction_static: float | None = None,
    omni_encoder_mem_reserve_delta: float | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Build ServerArgs with shared defaults for all SGLang AR engines."""
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "disable_cuda_graph": True,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "random_seed": 123,
        "context_length": context_length,
    }
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static
    kwargs.update(overrides)
    server_args = ServerArgs(**kwargs)
    _apply_omni_encoder_mem_fraction_clamp(
        server_args,
        enabled=omni_encoder_mem_reserve_delta is not None,
        user_mem_fraction_static=mem_fraction_static,
        reserve_delta=omni_encoder_mem_reserve_delta or 0.0,
    )
    return server_args


def _apply_omni_encoder_mem_fraction_clamp(
    server_args: ServerArgs,
    *,
    enabled: bool,
    user_mem_fraction_static: float | None,
    reserve_delta: float,
) -> None:
    """Reserve extra GPU memory for omni encoders that share the thinker GPU."""
    if not enabled or user_mem_fraction_static is not None:
        return
    if reserve_delta <= 0:
        return

    current = server_args.mem_fraction_static
    if current is None:
        return
    server_args.mem_fraction_static = round(max(0.01, current - reserve_delta), 3)
