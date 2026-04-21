# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines."""
from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs

# Note (Ratish, Chenyang):

# SGLang's VLM auto-sizing applies a dynamic 0.95 * factor reserve
# (roughly [0.8, 1.05]); Qwen3-Omni nests vision/audio configs under
# `thinker_config` so SGLang's VLM path never triggers for us. 0.05
# is a conservative linear lower-bound of that dynamic reserve; we
# subtract it after auto-sizing when the thinker GPU also hosts encoder
# stages. User-pinned mem_fraction_static bypasses this reserve.

OMNI_ENCODER_MEM_FRACTION_STATIC_RESERVE = 0.05


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    chunked_prefill_size: int = 128,
    max_prefill_tokens: int = 4096,
    max_running_requests: int = 16,
    mem_fraction_static: float | None = None,
    auto_mem_fraction_static_reserve: float | None = None,
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
    _apply_auto_mem_fraction_static_reserve(
        server_args,
        enabled=auto_mem_fraction_static_reserve is not None,
        user_mem_fraction_static=mem_fraction_static,
        reserve=auto_mem_fraction_static_reserve or 0.0,
    )
    return server_args


def _apply_auto_mem_fraction_static_reserve(
    server_args: ServerArgs,
    *,
    enabled: bool,
    user_mem_fraction_static: float | None,
    reserve: float,
) -> None:
    """Subtract a caller-requested reserve from SGLang's auto-selected value."""
    if not enabled or user_mem_fraction_static is not None:
        return
    if reserve <= 0:
        return

    current = server_args.mem_fraction_static
    if current is None:
        return
    server_args.mem_fraction_static = round(max(0.01, current - reserve), 3)
