# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 text scheduler construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def _has_transformer_weights(path: Path | None) -> bool:
    return path is not None and path.is_dir() and any(path.glob("*.safetensors"))


def resolve_transformer_weights_path(model_path: str) -> str:
    """Resolve the unified checkpoint's nested transformer component."""

    local_path = Path(model_path)
    if local_path.exists():
        transformer_path = local_path / "transformer"
    else:
        allow_patterns = [
            "transformer/*.json",
            "transformer/*.safetensors",
        ]
        try:
            snapshot_path = snapshot_download(
                model_path,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        except FileNotFoundError:
            snapshot_path = None
        transformer_path = (
            Path(snapshot_path) / "transformer" if snapshot_path is not None else None
        )
        if not _has_transformer_weights(transformer_path):
            snapshot_path = snapshot_download(
                model_path,
                allow_patterns=allow_patterns,
            )
            transformer_path = Path(snapshot_path) / "transformer"

    if transformer_path is None or not _has_transformer_weights(transformer_path):
        raise FileNotFoundError(
            f"Cosmos3 transformer weights not found under {model_path!r}"
        )
    return str(transformer_path)


def resolve_vision_encoder_path(model_path: str) -> str:
    """Resolve the standalone Qwen3-VL vision checkpoint."""

    local_path = Path(model_path)
    if local_path.exists():
        vision_path = local_path / "vision_encoder"
    else:
        allow_patterns = [
            "config.json",
            "vision_encoder/*.json",
            "vision_encoder/*.safetensors",
        ]
        try:
            snapshot_path = snapshot_download(
                model_path,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        except FileNotFoundError:
            snapshot_path = None
        vision_path = (
            Path(snapshot_path) / "vision_encoder"
            if snapshot_path is not None
            else None
        )
        if not _has_transformer_weights(vision_path):
            snapshot_path = snapshot_download(model_path, allow_patterns=allow_patterns)
            vision_path = Path(snapshot_path) / "vision_encoder"

    if vision_path is None or not _has_transformer_weights(vision_path):
        raise FileNotFoundError(
            f"Cosmos3 vision encoder weights not found under {model_path!r}"
        )
    return str(vision_path)


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
):
    """Create the SGLang-backed Cosmos3 thinker OmniScheduler."""

    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.cosmos3.model_runner import Cosmos3ThinkerModelRunner
    from sglang_omni.models.cosmos3.request_builders import (
        make_text_scheduler_adapters,
        make_text_stream_output_builder,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_manager,
        decode_manager,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="Cosmos3TextForCausalLM",
        model_weights_path=resolve_transformer_weights_path(server_args.model_path),
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    output_processor = SGLangOutputProcessor(capture_hidden=False)
    model_runner = Cosmos3ThinkerModelRunner(model_worker, output_processor)
    tokenizer = get_tokenizer(
        server_args.model_path,
        trust_remote_code=True,
    )
    request_builder, result_adapter = make_text_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        generation_config=model_config.hf_generation_config,
        thinker_config=model_config.hf_config,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=make_text_stream_output_builder(),
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )


__all__ = [
    "create_thinker_scheduler",
    "resolve_transformer_weights_path",
    "resolve_vision_encoder_path",
]
