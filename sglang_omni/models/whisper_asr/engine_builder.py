# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.scheduling.engine_factory import AsrEngineBuilder

logger = logging.getLogger(__name__)

_DEFAULT_ENCODER_GRAPH_BATCH_BUCKETS = (1, 2, 4, 8, 12, 16)


def _normalize_encoder_graph_buckets(buckets: list[int] | None) -> tuple[int, ...]:
    values = _DEFAULT_ENCODER_GRAPH_BATCH_BUCKETS if buckets is None else buckets
    normalized = {int(value) for value in values}
    return tuple(sorted(value for value in normalized if value >= 1))


def _resolve_encoder_graph_buckets(
    buckets: tuple[int, ...],
    *,
    max_prefill_tokens: int,
    encoder_token_count: int,
) -> tuple[int, ...]:
    """Filter capture buckets to batches reachable by atomic prefill."""
    if max_prefill_tokens < 1:
        raise ValueError(f"max_prefill_tokens must be >= 1, got {max_prefill_tokens}")
    if encoder_token_count < 1:
        raise ValueError(f"encoder_token_count must be >= 1, got {encoder_token_count}")
    max_encoder_batch_size = max_prefill_tokens // encoder_token_count
    return tuple(bucket for bucket in buckets if bucket <= max_encoder_batch_size)


class WhisperASREngineBuilder(AsrEngineBuilder):
    model_name = "Whisper ASR"
    model_arch_override = "WhisperForConditionalGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float,
        enable_encoder_cuda_graph: bool = False,
        encoder_graph_batch_buckets: list[int] | None = None,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.enable_encoder_cuda_graph = bool(enable_encoder_cuda_graph)
        self.encoder_graph_batch_buckets = _normalize_encoder_graph_buckets(
            encoder_graph_batch_buckets
        )
        self.processor: Any = None
        self.tokenizer: Any = None
        self.generation_config: Any = None
        self.encoder_token_count = 0
        self.context_length = 0
        self.decoder_context_len = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoConfig, AutoProcessor, GenerationConfig

        from sglang_omni.models.whisper_asr.request_builders import (
            MAX_PREV_CONTEXT_TOKENS,
        )

        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        self.tokenizer = self.processor.tokenizer
        self.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)
        self.encoder_token_count = int(
            self.processor.feature_extractor.nb_max_frames // 2
        )
        self.context_length = (
            self.encoder_token_count + MAX_PREV_CONTEXT_TOKENS + self.max_new_tokens + 8
        )
        # note (jiannan-17): prev_len + prefix_len + max_new_tokens <= decoder_context_len
        self.decoder_context_len = int(
            getattr(
                AutoConfig.from_pretrained(checkpoint_dir), "max_target_positions", 0
            )
            or 448
        )

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        if self.enable_encoder_cuda_graph and generation_cuda_graph_enabled:
            max_prefill_tokens = int(server_args.max_prefill_tokens)
            resolved_buckets = _resolve_encoder_graph_buckets(
                self.encoder_graph_batch_buckets,
                max_prefill_tokens=max_prefill_tokens,
                encoder_token_count=self.encoder_token_count,
            )
            logger.info(
                "Resolved Whisper encoder CUDA graph buckets configured=%s "
                "reachable=%s max_prefill_tokens=%d encoder_token_count=%d",
                self.encoder_graph_batch_buckets,
                resolved_buckets,
                max_prefill_tokens,
                self.encoder_token_count,
            )
            model.init_encoder_graphs(
                resolved_buckets,
                int(self.processor.feature_extractor.nb_max_frames),
            )

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("chunked_prefill_size") or 0) > 0:
            raise ValueError(
                "Whisper ASR requires chunked_prefill_size=0 because its encoder "
                "prefix must be admitted atomically"
            )
        overrides["chunked_prefill_size"] = 0

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": True,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 0,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        from sglang_omni.models.whisper_asr.request_builders import (
            make_whisper_scheduler_adapters,
        )

        return make_whisper_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            encoder_token_count=self.encoder_token_count,
            max_new_tokens=self.max_new_tokens,
            decoder_context_len=self.decoder_context_len,
        )
