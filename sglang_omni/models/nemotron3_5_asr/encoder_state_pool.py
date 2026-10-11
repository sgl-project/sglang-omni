# SPDX-License-Identifier: Apache-2.0
"""Bounded session slots for persistent streaming attention and convolution state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConv1d,
    NemotronAsrStreamingEncoderCausalConv2D,
)


@dataclass(frozen=True, kw_only=True)
class EncoderPoolLayout:
    """Per-slot attention (layers, heads, history frames, head dimension) and conv shapes."""

    attention_shape: tuple[int, int, int, int]
    convolution_shapes: dict[str, tuple[int, ...]]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_model(cls, model: Nemotron3_5AsrForRNNT) -> EncoderPoolLayout:
        config = model.config.encoder_config
        model_parameter = next(model.parameters())
        history_frames = config.sliding_window - 1
        assert history_frames > 0
        frequency_bins = config.num_mel_bins
        convolution_shapes: dict[str, tuple[int, ...]] = {}
        for convolution in model.encoder.modules():
            if isinstance(convolution, NemotronAsrStreamingEncoderCausalConv2D):
                padded_frequency_bins = frequency_bins + sum(convolution.freq_pad)
                convolution_shapes[convolution.cache_key] = (
                    convolution.in_channels,
                    convolution.left_pad,
                    padded_frequency_bins,
                )
                frequency_bins = (
                    padded_frequency_bins - convolution.kernel_size[1]
                ) // convolution.stride[1] + 1
            elif isinstance(convolution, NemotronAsrStreamingEncoderCausalConv1d):
                convolution_shapes[convolution.cache_key] = (
                    convolution.in_channels,
                    convolution.left_pad,
                )
            else:
                pass
        return cls(
            attention_shape=(
                config.num_hidden_layers,
                config.num_attention_heads,
                history_frames,
                config.hidden_size // config.num_attention_heads,
            ),
            convolution_shapes=convolution_shapes,
            dtype=model_parameter.dtype,
            device=model_parameter.device,
        )

    @property
    def slot_bytes(self) -> int:
        tensor_elements = 2 * math.prod(self.attention_shape) + sum(
            math.prod(shape) for shape in self.convolution_shapes.values()
        )
        return tensor_elements * self.dtype.itemsize + torch.int64.itemsize


@dataclass(kw_only=True, eq=False)
class EncoderStateSlot:
    pool: NemotronEncoderStatePool
    slot_id: int
    seen_frames: int = 0
    is_released: bool = False

    @property
    def nbytes(self) -> int:
        return 0 if self.is_released else self.pool.layout.slot_bytes

    def release(self) -> None:
        if self.is_released:
            return
        else:
            self.pool.active_slots.pop(self.slot_id)
            self.pool.free_slot_ids.append(self.slot_id)
            self.is_released = True


class NemotronEncoderStatePool:
    """Fixed storage whose leases are allocated and released by the model thread."""

    def __init__(self, layout: EncoderPoolLayout, capacity_slots: int) -> None:
        if capacity_slots < 1:
            raise ValueError("Nemotron state budget cannot hold one encoder slot")
        else:
            pass
        self.layout: EncoderPoolLayout = layout
        self.capacity_slots: int = capacity_slots
        self.is_closed: bool = False
        self.free_slot_ids: list[int] = list(reversed(range(capacity_slots)))
        self.active_slots: dict[int, EncoderStateSlot] = {}
        layer_count, head_count, history_frames, head_dimension = layout.attention_shape
        self.key_states: torch.Tensor = torch.empty(
            layer_count,
            capacity_slots,
            head_count,
            history_frames,
            head_dimension,
            dtype=layout.dtype,
            device=layout.device,
        )
        self.value_states: torch.Tensor = torch.empty_like(self.key_states)
        self.convolution_states: dict[str, torch.Tensor] = {
            name: torch.empty(
                (capacity_slots, *shape), dtype=layout.dtype, device=layout.device
            )
            for name, shape in layout.convolution_shapes.items()
        }
        self.seen_frames: torch.Tensor = torch.zeros(
            capacity_slots, dtype=torch.long, device=layout.device
        )

    @property
    def nbytes(self) -> int:
        return 0 if self.is_closed else self.capacity_slots * self.layout.slot_bytes

    @torch.inference_mode()
    def acquire(self) -> EncoderStateSlot:
        assert not self.is_closed and self.free_slot_ids
        slot_id = self.free_slot_ids.pop()
        # note (wirybeaver): Masked values still multiply attention weights, so unused history must be zeroed.
        self.key_states[:, slot_id].zero_()
        self.value_states[:, slot_id].zero_()
        for convolution_state in self.convolution_states.values():
            convolution_state[slot_id].zero_()
        self.seen_frames[slot_id].zero_()
        slot = EncoderStateSlot(pool=self, slot_id=slot_id)
        self.active_slots[slot_id] = slot
        return slot

    def close(self) -> None:
        if self.layout.device.type == "cuda":
            torch.cuda.synchronize(self.layout.device)
        else:
            pass
        for slot in list(self.active_slots.values()):
            slot.release()
        self.is_closed = True
        self.key_states = torch.empty(0, device=self.layout.device)
        self.value_states = torch.empty(0, device=self.layout.device)
        self.convolution_states.clear()
        self.seen_frames = torch.empty(0, dtype=torch.long, device=self.layout.device)


@dataclass(kw_only=True)
class PooledAttentionCache:
    pool: NemotronEncoderStatePool
    slot_ids: torch.Tensor

    def update(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_frames = self.pool.layout.attention_shape[2]
        key_history = self.pool.key_states[layer_idx].index_select(0, self.slot_ids)
        value_history = self.pool.value_states[layer_idx].index_select(0, self.slot_ids)
        keys = torch.cat((key_history, keys), dim=2)
        values = torch.cat((value_history, values), dim=2)
        self.pool.key_states[layer_idx].index_copy_(
            0, self.slot_ids, keys[:, :, -history_frames:]
        )
        self.pool.value_states[layer_idx].index_copy_(
            0, self.slot_ids, values[:, :, -history_frames:]
        )
        return keys, values

    def create_mask(self, chunk_frames: int, lookahead_tokens: int) -> torch.Tensor:
        history_frames = self.pool.layout.attention_shape[2]
        seen_frames = self.pool.seen_frames.index_select(0, self.slot_ids)
        key_columns = torch.arange(
            history_frames + chunk_frames, device=self.slot_ids.device
        )
        key_positions = seen_frames[:, None] - history_frames + key_columns
        query_positions = seen_frames[:, None] + key_columns[:chunk_frames]
        attention_chunk_frames = lookahead_tokens + 1
        chunk_difference = (
            torch.div(query_positions, attention_chunk_frames, rounding_mode="trunc")[
                :, :, None
            ]
            - torch.div(key_positions, attention_chunk_frames, rounding_mode="trunc")[
                :, None, :
            ]
        )
        return (
            (key_positions >= 0)[:, None, :]
            & (chunk_difference >= 0)
            & (chunk_difference <= history_frames // attention_chunk_frames)
        )[:, None]


@dataclass(kw_only=True)
class PooledPaddingCache:
    pool: NemotronEncoderStatePool
    slot_ids: torch.Tensor
    is_first_chunk: bool

    def update(
        self,
        hidden_states: torch.Tensor,
        cache_key: str,
        conv_module: (
            NemotronAsrStreamingEncoderCausalConv1d
            | NemotronAsrStreamingEncoderCausalConv2D
        ),
    ) -> torch.Tensor:
        convolution_states = self.pool.convolution_states[cache_key]
        previous_context = convolution_states.index_select(0, self.slot_ids)
        padded_hidden_states = torch.cat((previous_context, hidden_states), dim=2)
        padding_frames = conv_module.left_pad
        if padding_frames > 0:
            convolution_states.index_copy_(
                0, self.slot_ids, padded_hidden_states[:, :, -padding_frames:]
            )
        else:
            pass
        if self.is_first_chunk and isinstance(
            conv_module, NemotronAsrStreamingEncoderCausalConv2D
        ):
            return F.pad(
                padded_hidden_states,
                (0, 0, conv_module.left_pad_init - padding_frames, 0),
            )
        else:
            return padded_hidden_states
