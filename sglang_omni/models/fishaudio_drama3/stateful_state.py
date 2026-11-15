# SPDX-License-Identifier: Apache-2.0
"""Fixed-address codec state storage and windowed attention execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(kw_only=True)
class StreamState:
    decoder_identity: int
    slot_index: int
    generation: int
    processed_frames: int = 0
    batch_size: int | None = None
    is_final: bool = False
    is_failed: bool = False


@dataclass(kw_only=True)
class AttentionState:
    layer_keys: list[torch.Tensor]
    layer_values: list[torch.Tensor]


class DecoderStateArena:
    """Allocate only state belonging to the executed decode topology."""

    def __init__(
        self, codec: nn.Module, capacity: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.scratch_slot_index = capacity
        self.positions = torch.zeros(capacity + 1, device=device, dtype=torch.int64)
        self.histories: dict[str, torch.Tensor] = {}
        self.attention: dict[str, AttentionState] = {}
        self.host_generations = [0] * (capacity + 1)
        self.host_processed_frames = [0] * (capacity + 1)
        self.active_slots: set[int] = set()
        for path, module in (
            ("post", codec.quantizer.post_module),
            ("upsample", codec.quantizer.upsample),
            ("decoder", codec.decoder),
        ):
            self.allocate_module(module, path, capacity + 1, device, dtype)

    def allocate_module(
        self,
        module: nn.Module,
        path: str,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        module_name = type(module).__name__
        if module_name in ("CausalConvNet", "CausalTransConvNet"):
            convolution = module.conv
            if module_name == "CausalConvNet":
                channels = convolution.in_channels
                history_length = (
                    convolution.kernel_size[0] - 1
                ) * convolution.dilation[0]
            else:
                channels = convolution.out_channels
                history_length = convolution.kernel_size[0] - convolution.stride[0]
            self.histories[path] = torch.zeros(
                slot_count, channels, history_length, device=device, dtype=dtype
            )
        elif module_name == "WindowLimitedTransformer":
            layer_keys = []
            layer_values = []
            for layer in module.layers:
                attention = layer.attention
                shape = (
                    slot_count,
                    attention.n_local_heads,
                    module.window_size - 1,
                    attention.head_dim,
                )
                layer_keys.append(torch.zeros(shape, device=device, dtype=dtype))
                layer_values.append(torch.zeros(shape, device=device, dtype=dtype))
            self.attention[path] = AttentionState(
                layer_keys=layer_keys, layer_values=layer_values
            )
        elif module_name == "ResidualUnit":
            self.allocate_module(
                module.block, f"{path}.block", slot_count, device, dtype
            )
        elif module_name == "ConvNeXtBlock":
            self.allocate_module(
                module.dwconv, f"{path}.dwconv", slot_count, device, dtype
            )
        elif module_name in ("Decoder", "DecoderBlock"):
            self.allocate_module(
                module.model if module_name == "Decoder" else module.block,
                f"{path}.layers",
                slot_count,
                device,
                dtype,
            )
        elif isinstance(module, nn.Sequential):
            for child_index, child in enumerate(module):
                self.allocate_module(
                    child, f"{path}.{child_index}", slot_count, device, dtype
                )
        else:
            pass

    def persistent_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.positions,
            *self.histories.values(),
            *(
                tensor
                for attention in self.attention.values()
                for tensor in (*attention.layer_keys, *attention.layer_values)
            ),
        )

    @staticmethod
    def required_nbytes(codec: nn.Module, capacity: int, dtype: torch.dtype) -> int:
        return DecoderStateArena(codec, capacity, torch.device("meta"), dtype).nbytes

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.persistent_tensors()
        )

    def reset_slot(self, slot_index: int) -> None:
        for tensor in self.persistent_tensors():
            tensor[slot_index].zero_()
        self.host_processed_frames[slot_index] = 0
        self.host_generations[slot_index] += 1
        self.active_slots.discard(slot_index)


def apply_rotary(
    hidden_states: torch.Tensor, frequencies: torch.Tensor
) -> torch.Tensor:
    pairs = hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
    frequencies = frequencies.view(1, pairs.size(1), 1, pairs.size(3), 2)
    rotated = torch.stack(
        (
            pairs[..., 0] * frequencies[..., 0] - pairs[..., 1] * frequencies[..., 1],
            pairs[..., 1] * frequencies[..., 0] + pairs[..., 0] * frequencies[..., 1],
        ),
        dim=-1,
    )
    return rotated.flatten(3).type_as(hidden_states)


def decode_attention(
    transformer: nn.Module,
    hidden_states: torch.Tensor,
    arena: DecoderStateArena,
    path: str,
    slot_index: torch.Tensor,
) -> torch.Tensor:
    if transformer.channels_first:
        hidden_states = hidden_states.transpose(1, 2)
    else:
        pass
    hidden_states = transformer.input_proj(hidden_states)
    frame_count = hidden_states.shape[1]
    position = arena.positions.index_select(0, slot_index)
    positions = position + torch.arange(frame_count, device=hidden_states.device)
    frequencies = transformer.freqs_cis[positions]
    window_size = transformer.window_size
    retained_count = window_size - 1
    key_positions = position + torch.arange(
        -retained_count, frame_count, device=hidden_states.device
    )
    mask = (
        (key_positions[None, :] >= 0)
        & (key_positions[None, :] <= positions[:, None])
        & (key_positions[None, :] > positions[:, None] - window_size)
    )
    for layer_index, layer in enumerate(transformer.layers):
        attention = layer.attention
        normalized = layer.attention_norm(hidden_states)
        head_width = attention.n_local_heads * attention.head_dim
        queries, keys, values = attention.wqkv(normalized).split(
            [head_width, head_width, head_width], dim=-1
        )
        batch_size = hidden_states.shape[0]
        queries = queries.view(
            batch_size, frame_count, attention.n_head, attention.head_dim
        )
        keys = keys.view(
            batch_size, frame_count, attention.n_local_heads, attention.head_dim
        )
        values = values.view(
            batch_size, frame_count, attention.n_local_heads, attention.head_dim
        )
        queries = apply_rotary(queries, frequencies).transpose(1, 2)
        keys = apply_rotary(keys, frequencies).transpose(1, 2)
        values = values.transpose(1, 2)
        layer_keys = arena.attention[path].layer_keys[layer_index]
        layer_values = arena.attention[path].layer_values[layer_index]
        keys = torch.cat((layer_keys.index_select(0, slot_index), keys), dim=2)
        values = torch.cat((layer_values.index_select(0, slot_index), values), dim=2)
        layer_keys.index_copy_(0, slot_index, keys[:, :, frame_count:])
        layer_values.index_copy_(0, slot_index, values[:, :, frame_count:])
        attended = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, frame_count, attention.n_head * attention.head_dim)
        )
        hidden_states = hidden_states + layer.attention_layer_scale(
            attention.wo(attended)
        )
        hidden_states = hidden_states + layer.ffn_layer_scale(
            layer.feed_forward(layer.ffn_norm(hidden_states))
        )
    hidden_states = transformer.output_proj(transformer.norm(hidden_states))
    if transformer.channels_first:
        return hidden_states.transpose(1, 2)
    else:
        return hidden_states
