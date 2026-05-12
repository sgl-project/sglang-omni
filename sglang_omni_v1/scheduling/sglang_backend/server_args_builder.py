# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines."""
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
