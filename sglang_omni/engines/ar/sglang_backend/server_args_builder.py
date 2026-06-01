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
    **overrides: Any,
) -> ServerArgs:
    """Build a SGLang ServerArgs with shared defaults for AR engines."""
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
    return ServerArgs(**kwargs)


def apply_encoder_mem_reserve(
    server_args: ServerArgs,
    encoder_mem_reserve: float,
) -> None:
    """Subtract encoder_mem_reserve from SGLang's auto-picked mem_fraction_static.

    # Note (Chenyang):
    Call this only when SGLang auto-selected mem_fraction_static —
    i.e. the caller did NOT pin --mem-fraction-static. When the caller
    pinned, that value is the whole budget and the reserve value is ignored.

    Raises ValueError when the result would drop below 0.1 — below
    that, SGLang's KV allocator fails deep in the scheduler with a
    confusing traceback (empirically crashes ~0.08 on H200 for
    Qwen3-Omni-30B), so surface it at build time instead.
    """
    if encoder_mem_reserve <= 0:
        return
    current = server_args.mem_fraction_static
    if current is None:
        return
    new_value = current - encoder_mem_reserve
    if new_value < 0.1:
        raise ValueError(
            f"auto mem_fraction_static {current:.3f} minus encoder_mem_reserve "
            f"{encoder_mem_reserve:.3f} = {new_value:.3f} is below the safe "
            f"floor 0.1; lower encoder_mem_reserve or pin "
            f"--mem-fraction-static explicitly."
        )
    server_args.mem_fraction_static = round(new_value, 3)
