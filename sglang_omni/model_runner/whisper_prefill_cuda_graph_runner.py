# SPDX-License-Identifier: Apache-2.0
"""Whisper-specific metadata adapter for breakable prefill CUDA graphs."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)


class WhisperPrefillCudaGraphRunner(PrefillCudaGraphRunner):
    """Preserve encoder-decoder metadata omitted by SGLang's generic runner."""

    def __init__(self, model_runner: Any) -> None:
        decoder = model_runner.model.model.decoder
        self_attention_layers = [layer.self_attn.attn for layer in decoder.layers]
        cross_attention_layers = [layer.encoder_attn.attn for layer in decoder.layers]
        model_runner.attention_layers = self_attention_layers + cross_attention_layers
        super().__init__(model_runner)

    def capture_prepare(self, num_tokens: int) -> tuple[ForwardBatch, Any]:
        forward_batch, attn_backend = super().capture_prepare(num_tokens)
        encoder_lens_cpu = [1] * forward_batch.batch_size
        forward_batch.encoder_lens = torch.tensor(
            encoder_lens_cpu,
            dtype=torch.int64,
            device=self.device,
        )
        forward_batch.encoder_lens_cpu = encoder_lens_cpu
        forward_batch.encoder_cached = [True] * forward_batch.batch_size
        forward_batch.encoder_out_cache_loc = None
        return forward_batch, attn_backend

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> ForwardBatch:
        static_forward_batch = super().load_batch(forward_batch, **kwargs)
        static_forward_batch.encoder_lens_cpu = forward_batch.encoder_lens_cpu
        static_forward_batch.encoder_cached = forward_batch.encoder_cached
        static_forward_batch.encoder_out_cache_loc = forward_batch.encoder_out_cache_loc
        return static_forward_batch


__all__ = ["WhisperPrefillCudaGraphRunner"]
