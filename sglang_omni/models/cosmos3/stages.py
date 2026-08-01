# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Cosmos3 pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.cosmos3.components.text_preprocessor import (
    Cosmos3TextPreprocessor,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_preprocessing_executor(
    model_path: str,
    *,
    thinker_max_seq_len: int | None = None,
) -> SimpleScheduler:
    """Create the CPU-only text preprocessing stage."""

    preprocessor = Cosmos3TextPreprocessor(
        model_path=model_path,
        max_seq_len=thinker_max_seq_len,
    )
    return SimpleScheduler(preprocessor)


def create_sglang_text_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
):
    """Create the naive single-rank Cosmos3 text AR stage."""

    if tp_size != 1 or tp_rank != 0:
        raise ValueError("Cosmos3 text MVP supports only tp_size=1")

    from sglang_omni.models.cosmos3.bootstrap import create_text_scheduler
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    overrides: dict[str, Any] = {
        "tp_size": 1,
        "enable_multimodal": False,
        "language_only": True,
        "sampling_backend": "pytorch",
    }
    overrides.update(server_args_overrides or {})
    if overrides.get("tp_size") != 1:
        raise ValueError("Cosmos3 text MVP does not support tensor parallelism")
    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    return create_text_scheduler(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )


def create_decode_executor(model_path: str):
    from sglang_omni.models.cosmos3.components.streaming_detokenizer import (
        create_streaming_detokenize_scheduler,
    )

    return create_streaming_detokenize_scheduler(model_path)


__all__ = [
    "create_decode_executor",
    "create_preprocessing_executor",
    "create_sglang_text_executor_from_config",
]
