# SPDX-License-Identifier: Apache-2.0
"""Stateful incremental decoder for the Qwen3-TTS speech tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class Qwen3TTSIncrementalCodecState:
    frame_position: int = 0
    transformer_context_length: int = 0
    transformer_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    transformer_values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)

    def clone(self) -> Qwen3TTSIncrementalCodecState:
        return Qwen3TTSIncrementalCodecState(
            frame_position=self.frame_position,
            transformer_context_length=self.transformer_context_length,
            transformer_keys={
                key: value.clone() for key, value in self.transformer_keys.items()
            },
            transformer_values={
                key: value.clone() for key, value in self.transformer_values.items()
            },
            conv_histories={
                key: value.clone() for key, value in self.conv_histories.items()
            },
            transconv_overlaps={
                key: value.clone() for key, value in self.transconv_overlaps.items()
            },
        )


def incremental_causal_conv1d(
    module: Any,
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    conv = module.conv
    stride = int(conv.stride[0])
    if stride != 1:
        raise ValueError(f"incremental causal Conv1d requires stride=1, got {stride}")
    history_size = int(module.padding)
    history = state.conv_histories.get(key)
    if history is None:
        history = hidden_states.new_zeros(
            hidden_states.shape[0], hidden_states.shape[1], history_size
        )
    elif history.shape[:-1] != hidden_states.shape[:-1]:
        raise ValueError(f"incremental causal Conv1d state shape changed for {key}")

    combined = torch.cat((history, hidden_states), dim=-1)
    output = conv(combined).contiguous()
    if output.shape[-1] != hidden_states.shape[-1]:
        raise RuntimeError(
            f"incremental causal Conv1d changed temporal length for {key}"
        )
    state.conv_histories[key] = (
        combined[..., -history_size:].clone()
        if history_size
        else combined[..., :0].clone()
    )
    return output


def incremental_causal_transconv1d(
    module: Any,
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    conv = module.conv
    stride = int(conv.stride[0])
    right_pad = int(module.right_pad)
    output = F.conv_transpose1d(
        hidden_states,
        conv.weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        dilation=conv.dilation,
    )
    overlap = state.transconv_overlaps.get(key)
    if overlap is not None:
        if overlap.shape[:-1] != output.shape[:-1]:
            raise ValueError(
                f"incremental causal ConvTranspose1d state shape changed for {key}"
            )
        overlap_length = int(overlap.shape[-1])
        output[..., :overlap_length] += overlap

    emit_length = int(hidden_states.shape[-1]) * stride
    emitted = output[..., :emit_length]
    tail = output[..., emit_length:]
    if int(tail.shape[-1]) != right_pad:
        raise RuntimeError(
            f"incremental causal ConvTranspose1d produced the wrong overlap for {key}"
        )
    state.transconv_overlaps[key] = tail.clone()
    if conv.bias is not None:
        emitted = emitted + conv.bias.view(1, -1, 1)
    return emitted.contiguous()


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, heads, groups, length, head_dim
    )
    return hidden_states.reshape(batch, heads * groups, length, head_dim)


def _incremental_attention(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    state: Qwen3TTSIncrementalCodecState,
    layer_index: int,
    key_positions: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    head_dim = int(attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states)).view(
        *input_shape, -1, head_dim
    )
    key = attention.k_norm(attention.k_proj(hidden_states)).view(
        *input_shape, -1, head_dim
    )
    value = attention.v_proj(hidden_states).view(*input_shape, -1, head_dim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = _apply_rotary_pos_emb(
        query, key, position_embeddings[0], position_embeddings[1]
    )

    prior_key = state.transformer_keys.get(layer_index)
    prior_value = state.transformer_values.get(layer_index)
    if (prior_key is None) != (prior_value is None):
        raise RuntimeError(f"incomplete transformer state for layer {layer_index}")
    if prior_key is not None:
        key = torch.cat((prior_key, key), dim=-2)
        value = torch.cat((prior_value, value), dim=-2)

    repeated_key = _repeat_kv(key, int(attention.num_key_value_groups))
    repeated_value = _repeat_kv(value, int(attention.num_key_value_groups))
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * float(
        attention.scaling
    )
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    sliding_window = int(attention.sliding_window)
    if sliding_window > 0:
        allowed &= key_positions.unsqueeze(0) > (
            query_positions.unsqueeze(1) - sliding_window
        )
    scores = scores.masked_fill(
        ~allowed.view(1, 1, *allowed.shape),
        torch.finfo(scores.dtype).min,
    )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, repeated_value)
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return attention.o_proj(output), key, value


def _incremental_transformer(
    transformer: Any,
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
) -> torch.Tensor:
    hidden_states = transformer.input_proj(hidden_states)
    fresh_frames = int(hidden_states.shape[1])
    query_positions = torch.arange(
        state.frame_position,
        state.frame_position + fresh_frames,
        device=hidden_states.device,
        dtype=torch.long,
    )
    prior_start = state.frame_position - state.transformer_context_length
    key_positions = torch.arange(
        prior_start,
        state.frame_position + fresh_frames,
        device=hidden_states.device,
        dtype=torch.long,
    )
    position_embeddings = transformer.rotary_emb(
        hidden_states, query_positions.unsqueeze(0)
    )

    next_keys: dict[int, torch.Tensor] = {}
    next_values: dict[int, torch.Tensor] = {}
    window_size = int(transformer.window_size)
    retained_context = max(0, window_size - 1)
    for layer_index, layer in enumerate(transformer.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attended, key, value = _incremental_attention(
            layer.self_attn,
            normalized,
            position_embeddings,
            state,
            layer_index,
            key_positions,
            query_positions,
        )
        hidden_states = residual + layer.self_attn_layer_scale(attended)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + layer.mlp_layer_scale(hidden_states)
        next_keys[layer_index] = (
            key[..., -retained_context:, :].clone()
            if retained_context
            else key[..., :0, :].clone()
        )
        next_values[layer_index] = (
            value[..., -retained_context:, :].clone()
            if retained_context
            else value[..., :0, :].clone()
        )

    state.transformer_keys = next_keys
    state.transformer_values = next_values
    state.transformer_context_length = min(
        retained_context,
        state.transformer_context_length + fresh_frames,
    )
    return transformer.output_proj(transformer.norm(hidden_states))


def _incremental_convnext(
    module: Any,
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = incremental_causal_conv1d(
        module.dwconv, hidden_states, state, f"{key}.dwconv"
    )
    hidden_states = module.norm(hidden_states.permute(0, 2, 1))
    hidden_states = module.pwconv1(hidden_states)
    hidden_states = module.act(hidden_states)
    hidden_states = module.pwconv2(hidden_states)
    hidden_states = module.gamma * hidden_states
    return residual + hidden_states.permute(0, 2, 1)


def _incremental_residual_unit(
    module: Any,
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = module.act1(hidden_states)
    hidden_states = incremental_causal_conv1d(
        module.conv1, hidden_states, state, f"{key}.conv1"
    )
    hidden_states = module.act2(hidden_states)
    hidden_states = incremental_causal_conv1d(
        module.conv2, hidden_states, state, f"{key}.conv2"
    )
    return hidden_states + residual


class Qwen3TTSIncrementalDecoder:
    def __init__(self, decoder: Any) -> None:
        self._require_attrs(
            decoder,
            "decoder",
            "quantizer",
            "pre_conv",
            "pre_transformer",
            "upsample",
            "decoder",
            "total_upsample",
        )
        if len(decoder.decoder) < 3:
            raise TypeError("unsupported Qwen3-TTS decoder layout")
        self._require_attrs(decoder.pre_conv, "pre_conv", "conv", "padding")
        self._require_attrs(
            decoder.pre_transformer,
            "pre_transformer",
            "input_proj",
            "layers",
            "norm",
            "output_proj",
            "rotary_emb",
            "window_size",
        )
        for layer_index, layer in enumerate(decoder.pre_transformer.layers):
            self._require_attrs(
                layer,
                f"pre_transformer.layers.{layer_index}",
                "input_layernorm",
                "self_attn",
                "self_attn_layer_scale",
                "post_attention_layernorm",
                "mlp",
                "mlp_layer_scale",
            )
            self._require_attrs(
                layer.self_attn,
                f"pre_transformer.layers.{layer_index}.self_attn",
                "head_dim",
                "num_key_value_groups",
                "scaling",
                "sliding_window",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "k_norm",
            )
        for stage_index, blocks in enumerate(decoder.upsample):
            if len(blocks) != 2:
                raise TypeError("unsupported Qwen3-TTS upsample layout")
            self._require_attrs(
                blocks[0], f"upsample.{stage_index}.0", "conv", "right_pad"
            )
            self._require_attrs(
                blocks[1],
                f"upsample.{stage_index}.1",
                "dwconv",
                "norm",
                "pwconv1",
                "act",
                "pwconv2",
                "gamma",
            )
        self._require_attrs(decoder.decoder[0], "decoder.0", "conv", "padding")
        self._require_attrs(decoder.decoder[-1], "decoder.final", "conv", "padding")
        for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
            if not hasattr(block, "block") or len(block.block) < 2:
                raise TypeError("unsupported Qwen3-TTS decoder block layout")
            self._require_attrs(
                block.block[1],
                f"decoder.{block_index}.1",
                "conv",
                "right_pad",
            )
            for residual_index, residual in enumerate(block.block[2:]):
                self._require_attrs(
                    residual,
                    f"decoder.{block_index}.{residual_index + 2}",
                    "act1",
                    "conv1",
                    "act2",
                    "conv2",
                )
        self._decoder = decoder
        self.total_upsample = int(decoder.total_upsample)

    @staticmethod
    def _require_attrs(module: Any, path: str, *names: str) -> None:
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; "
                f"{path} is missing {', '.join(missing)}"
            )

    def decode(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
    ) -> torch.Tensor:
        if codes.ndim != 3 or int(codes.shape[0]) != 1:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires codes shaped [1, Q, T]"
            )
        fresh_frames = int(codes.shape[-1])
        if fresh_frames <= 0:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires fresh frames"
            )

        hidden_states = self._decoder.quantizer.decode(codes)
        hidden_states = incremental_causal_conv1d(
            self._decoder.pre_conv,
            hidden_states,
            state,
            "pre_conv",
        ).transpose(1, 2)
        hidden_states = _incremental_transformer(
            self._decoder.pre_transformer, hidden_states, state
        ).permute(0, 2, 1)

        for stage_index, blocks in enumerate(self._decoder.upsample):
            if len(blocks) != 2:
                raise TypeError("unsupported Qwen3-TTS upsample layout")
            hidden_states = incremental_causal_transconv1d(
                blocks[0],
                hidden_states,
                state,
                f"upsample.{stage_index}.transconv",
            )
            hidden_states = _incremental_convnext(
                blocks[1],
                hidden_states,
                state,
                f"upsample.{stage_index}.convnext",
            )

        waveform = incremental_causal_conv1d(
            self._decoder.decoder[0], hidden_states, state, "decoder.0"
        )
        for block_index, decoder_block in enumerate(
            self._decoder.decoder[1:-2], start=1
        ):
            if len(decoder_block.block) < 2:
                raise TypeError("unsupported Qwen3-TTS decoder block layout")
            waveform = decoder_block.block[0](waveform)
            waveform = incremental_causal_transconv1d(
                decoder_block.block[1],
                waveform,
                state,
                f"decoder.{block_index}.transconv",
            )
            for residual_index, residual_unit in enumerate(decoder_block.block[2:]):
                waveform = _incremental_residual_unit(
                    residual_unit,
                    waveform,
                    state,
                    f"decoder.{block_index}.residual.{residual_index}",
                )
        waveform = self._decoder.decoder[-2](waveform)
        waveform = incremental_causal_conv1d(
            self._decoder.decoder[-1], waveform, state, "decoder.final"
        ).clamp(min=-1, max=1)
        expected_samples = fresh_frames * self.total_upsample
        if int(waveform.shape[-1]) != expected_samples:
            raise RuntimeError(
                "Qwen3-TTS incremental codec decoder returned the wrong sample count"
            )
        state.frame_position += fresh_frames
        return waveform
