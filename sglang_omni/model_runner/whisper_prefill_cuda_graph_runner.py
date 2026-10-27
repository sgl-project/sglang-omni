# SPDX-License-Identifier: Apache-2.0
"""Whisper-specific metadata adapter for breakable prefill CUDA graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from typing_extensions import Unpack

from sglang_omni.models.whisper_asr.dense_prefill import WhisperDensePrefillRunner

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
else:
    pass


class PrefillReplayKwargs(TypedDict, total=False):
    pp_proxy_tensors: PPProxyTensors | None
    input_embeds: torch.Tensor
    get_embedding: bool


class WhisperPrefillCudaGraphRunner(PrefillCudaGraphRunner):
    """Preserve encoder-decoder metadata omitted by SGLang's generic runner."""

    def __init__(self, model_runner: "ModelRunner") -> None:
        self.dense_prefill_runner: WhisperDensePrefillRunner | None = None
        decoder = model_runner.model.model.decoder
        self_attention_layers = [layer.self_attn.attn for layer in decoder.layers]
        cross_attention_layers = [layer.encoder_attn.attn for layer in decoder.layers]
        model_runner.attention_layers = self_attention_layers + cross_attention_layers
        # note (Junnan Li): radix_attention indexes this by layer_id, and
        # cross-attention ids start at num_decoder_layers.
        model_runner.mha_companion_layers = [None] * len(model_runner.attention_layers)
        super().__init__(model_runner)

    def capture_prepare(self, num_tokens: int) -> tuple[ForwardBatch, AttentionBackend]:
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

    def execute(
        self,
        forward_batch: ForwardBatch,
        **kwargs: Unpack[PrefillReplayKwargs],
    ) -> LogitsProcessorOutput:
        if self.dense_prefill_runner is None or any(
            value is not None and value is not False for value in kwargs.values()
        ):
            return super().execute(forward_batch, **kwargs)
        else:
            prompt_length = self.dense_prefill_runner.prompt_length_of(forward_batch)
            if prompt_length is None:
                return super().execute(forward_batch, **kwargs)
            else:
                return self.dense_prefill_runner.run(forward_batch, prompt_length)

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs: Unpack[PrefillReplayKwargs],
    ) -> ForwardBatch:
        static_forward_batch = super().load_batch(forward_batch, **kwargs)
        static_forward_batch.encoder_lens_cpu = forward_batch.encoder_lens_cpu
        static_forward_batch.encoder_cached = forward_batch.encoder_cached
        static_forward_batch.encoder_out_cache_loc = forward_batch.encoder_out_cache_loc
        return static_forward_batch


__all__ = ["WhisperPrefillCudaGraphRunner"]
