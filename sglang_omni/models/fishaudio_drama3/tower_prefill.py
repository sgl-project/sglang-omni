# SPDX-License-Identifier: Apache-2.0
"""Paged tower attention and frame staging for backbone prefill."""

from __future__ import annotations

import dataclasses

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from torch import nn

from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder


class PagedTowerAttention:
    """DecoderAttention over the paged pool; tower layer i is RadixAttention layer i."""

    def __init__(self, layers: nn.ModuleList, forward_batch: ForwardBatch) -> None:
        self.layers: nn.ModuleList = layers
        self.forward_batch: ForwardBatch = forward_batch

    def __call__(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layers[layer_index]
        assert isinstance(layer, RadixAttention)
        return layer(query, key, value, self.forward_batch)

    def forward_mode(self) -> ForwardMode:
        return self.forward_batch.forward_mode

    def cache_slots(self) -> torch.Tensor:
        return self.forward_batch.out_cache_loc

    def key_buffer(self, layer_index: int) -> torch.Tensor:
        layer = self.layers[layer_index]
        attention_backend = get_attn_backend()
        assert isinstance(layer, RadixAttention)
        assert isinstance(attention_backend, HybridLinearAttnBackend)
        return attention_backend.token_to_kv_pool.get_key_buffer(layer.layer_id)

    def value_buffer(self, layer_index: int) -> torch.Tensor:
        layer = self.layers[layer_index]
        attention_backend = get_attn_backend()
        assert isinstance(layer, RadixAttention)
        assert isinstance(attention_backend, HybridLinearAttnBackend)
        return attention_backend.token_to_kv_pool.get_value_buffer(layer.layer_id)

    def attend(self, layer_index: int, query: torch.Tensor) -> torch.Tensor:
        layer = self.layers[layer_index]
        assert isinstance(layer, RadixAttention)
        return layer(query, None, None, self.forward_batch, save_kv_cache=False)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3TowerPrefill:
    audio_decoder: AudioDecoder
    tower_attention: nn.ModuleList
    input_frames: torch.Tensor
    next_frames: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        audio_decoder: AudioDecoder,
        tower_attention: nn.ModuleList,
        row_count: int,
        frame_width: int,
        device: torch.device,
    ) -> Drama3TowerPrefill:
        return cls(
            audio_decoder=audio_decoder,
            tower_attention=tower_attention,
            input_frames=torch.zeros(
                (row_count, frame_width), dtype=torch.int64, device=device
            ),
            next_frames=torch.zeros(
                (row_count, frame_width), dtype=torch.int64, device=device
            ),
        )

    def require_rows(self, row_count: int) -> None:
        buffer_row_count = min(self.input_frames.shape[0], self.next_frames.shape[0])
        if row_count > buffer_row_count:
            raise RuntimeError(
                f"{row_count} rows exceed the {buffer_row_count} tower prefill rows"
            )
        else:
            pass

    def stage(self, input_frames: torch.Tensor, next_frames: torch.Tensor) -> None:
        row_count = input_frames.shape[0]
        self.require_rows(row_count)
        self.input_frames[:row_count].copy_(input_frames)
        self.next_frames[:row_count].copy_(next_frames)

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        row_count = hidden_states.shape[0]
        self.require_rows(row_count)
        self.audio_decoder.prefill(
            hidden_states,
            positions,
            self.input_frames[:row_count],
            self.next_frames[:row_count],
            PagedTowerAttention(self.tower_attention, forward_batch),
        )
