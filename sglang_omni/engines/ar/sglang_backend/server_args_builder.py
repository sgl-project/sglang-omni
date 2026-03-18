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
    mem_fraction_static: float = 0.7,
    speculative_algorithm: str | None = None,
    speculative_draft_model_path: str | None = None,
    speculative_num_steps: int = 5,
    speculative_num_draft_tokens: int = 64,
    speculative_eagle_topk: int = 8,
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
        "mem_fraction_static": mem_fraction_static,
        "random_seed": 123,
        "context_length": context_length,
        "speculative_algorithm": speculative_algorithm,
        "speculative_draft_model_path": speculative_draft_model_path,
        "speculative_num_steps": speculative_num_steps,
        "speculative_num_draft_tokens": speculative_num_draft_tokens,
        "speculative_eagle_topk": speculative_eagle_topk,
    }
    kwargs.update(overrides)
    return ServerArgs(**kwargs)
