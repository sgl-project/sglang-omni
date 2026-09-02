# SPDX-License-Identifier: Apache-2.0
"""Stateful incremental decoder for the Qwen3-TTS speech tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Qwen3TTSIncrementalCodecStateSpec:
    """Static description of one stream's incremental state.

    ``conv_histories`` and ``transconv_overlaps`` are ``(key, channels,
    length)`` triples in the order ``Qwen3TTSIncrementalDecoder.decode``
    visits them. Zero-length entries are kept so every consumer sees the same
    key set regardless of kernel size.
    """

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int

    def bytes_per_stream(self, dtype: torch.dtype) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        elements = sum(channels * length for _, channels, length in self.conv_histories)
        elements += sum(
            channels * length for _, channels, length in self.transconv_overlaps
        )
        elements += (
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.retained_context
            * self.head_dim
        )
        return int(elements * itemsize)


@dataclass
class Qwen3TTSIncrementalCodecState:
    frame_position: int = 0
    transformer_context_length: int = 0
    transformer_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    transformer_values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)
    # Note (Qihao Liu): absolute frame position per batch row. ``None`` means
    # every row shares ``frame_position``, which is the single-request case.
    # A cohort assembled from a state arena sets this so streams at different
    # playback positions can execute in one launch.
    frame_positions: torch.Tensor | None = None

    def row_frame_positions(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.frame_positions is None:
            return torch.full(
                (batch_size,), self.frame_position, device=device, dtype=torch.long
            )
        if int(self.frame_positions.shape[0]) != batch_size:
            raise ValueError(
                "Qwen3-TTS incremental codec state carries "
                f"{int(self.frame_positions.shape[0])} row positions for a batch "
                f"of {batch_size}"
            )
        return self.frame_positions.to(device=device, dtype=torch.long)

    def advance(self, fresh_frames: int) -> None:
        self.frame_position += fresh_frames
        if self.frame_positions is not None:
            self.frame_positions = self.frame_positions + fresh_frames

    def clone(self) -> Qwen3TTSIncrementalCodecState:
        return Qwen3TTSIncrementalCodecState(
            frame_position=self.frame_position,
            transformer_context_length=self.transformer_context_length,
            frame_positions=(
                None if self.frame_positions is None else self.frame_positions.clone()
            ),
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

    if int(key.shape[-2]) != int(key_positions.shape[-1]):
        raise RuntimeError(
            f"incremental transformer state length {int(key.shape[-2])} does not "
            f"match {int(key_positions.shape[-1])} key positions"
        )
    repeated_key = _repeat_kv(key, int(attention.num_key_value_groups))
    repeated_value = _repeat_kv(value, int(attention.num_key_value_groups))
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * float(
        attention.scaling
    )
    keys_by_row = key_positions.unsqueeze(1)
    queries_by_row = query_positions.unsqueeze(2)
    allowed = keys_by_row <= queries_by_row
    sliding_window = int(attention.sliding_window)
    if sliding_window > 0:
        allowed &= keys_by_row > (queries_by_row - sliding_window)
    # Note (Qihao Liu): a right-aligned K/V buffer that a stream has not filled
    # yet holds zeros whose nominal absolute position is negative. Those slots
    # would otherwise satisfy both tests above for an early query, so mask them
    # explicitly. This is what lets a cold and a warm stream share one cohort.
    allowed &= (key_positions >= 0).unsqueeze(1)
    scores = scores.masked_fill(
        ~allowed.unsqueeze(1),
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
    batch_size = int(hidden_states.shape[0])
    fresh_frames = int(hidden_states.shape[1])
    device = hidden_states.device
    frame_positions = state.row_frame_positions(batch_size, device)
    # Note (Qihao Liu): the retained buffer is right-aligned: slot j of a buffer
    # of length P holds absolute position frame_position - P + j. Deriving P
    # from the buffer rather than from transformer_context_length keeps this
    # correct when an arena hands over a full-width buffer that a cold stream
    # has not filled.
    prior_key = state.transformer_keys.get(0)
    prior_length = 0 if prior_key is None else int(prior_key.shape[-2])
    key_offsets = torch.arange(
        -prior_length, fresh_frames, device=device, dtype=torch.long
    )
    query_offsets = torch.arange(fresh_frames, device=device, dtype=torch.long)
    key_positions = frame_positions.unsqueeze(1) + key_offsets.unsqueeze(0)
    query_positions = frame_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
    position_embeddings = transformer.rotary_emb(hidden_states, query_positions)

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
        self._state_spec: Qwen3TTSIncrementalCodecStateSpec | None = None

    @staticmethod
    def _require_attrs(module: Any, path: str, *names: str) -> None:
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; "
                f"{path} is missing {', '.join(missing)}"
            )

    def state_spec(self) -> Qwen3TTSIncrementalCodecStateSpec:
        """Describe one stream's state without running a decode.

        Note (Qihao Liu): the lazy buffers in ``decode`` only materialize once
        activations have flowed through, so a state arena cannot preallocate
        from them. This walks the validated module tree in the same order
        ``decode`` does and derives every key and shape statically.
        """
        if self._state_spec is None:
            self._state_spec = self._build_state_spec()
        return self._state_spec

    def _build_state_spec(self) -> Qwen3TTSIncrementalCodecStateSpec:
        decoder = self._decoder
        conv: list[tuple[str, int, int]] = []
        transconv: list[tuple[str, int, int]] = []

        def add_conv(module: Any, key: str) -> None:
            conv.append((key, int(module.conv.in_channels), int(module.padding)))

        def add_transconv(module: Any, key: str) -> None:
            transconv.append(
                (key, int(module.conv.out_channels), int(module.right_pad))
            )

        add_conv(decoder.pre_conv, "pre_conv")
        for stage_index, blocks in enumerate(decoder.upsample):
            add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
            add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
        add_conv(decoder.decoder[0], "decoder.0")
        for block_index, decoder_block in enumerate(decoder.decoder[1:-2], start=1):
            add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
            for residual_index, residual in enumerate(decoder_block.block[2:]):
                key = f"decoder.{block_index}.residual.{residual_index}"
                add_conv(residual.conv1, f"{key}.conv1")
                add_conv(residual.conv2, f"{key}.conv2")
        add_conv(decoder.decoder[-1], "decoder.final")

        transformer = decoder.pre_transformer
        attention = transformer.layers[0].self_attn
        head_dim = int(attention.head_dim)
        key_features = int(attention.k_proj.out_features)
        if key_features % head_dim:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; k_proj width "
                f"{key_features} is not a multiple of head_dim {head_dim}"
            )
        return Qwen3TTSIncrementalCodecStateSpec(
            conv_histories=tuple(conv),
            transconv_overlaps=tuple(transconv),
            num_layers=len(transformer.layers),
            num_key_value_heads=key_features // head_dim,
            head_dim=head_dim,
            retained_context=max(0, int(transformer.window_size) - 1),
        )

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Qwen3TTSIncrementalCodecState:
        """Allocate a zeroed state with full-width buffers.

        Note (Qihao Liu): unlike the lazily grown state, the Transformer K/V
        buffers start at their retained width. A stream that has not filled
        them yet reads zeros at negative nominal positions, which
        ``_incremental_attention`` masks out, so a cold and a warm stream share
        one execution shape.
        """
        spec = self.state_spec()
        state = Qwen3TTSIncrementalCodecState(
            frame_positions=torch.zeros(batch_size, device=device, dtype=torch.long),
            transformer_context_length=spec.retained_context,
        )
        for key, channels, length in spec.conv_histories:
            state.conv_histories[key] = torch.zeros(
                batch_size, channels, length, device=device, dtype=dtype
            )
        for key, channels, length in spec.transconv_overlaps:
            state.transconv_overlaps[key] = torch.zeros(
                batch_size, channels, length, device=device, dtype=dtype
            )
        for layer_index in range(spec.num_layers):
            shape = (
                batch_size,
                spec.num_key_value_heads,
                spec.retained_context,
                spec.head_dim,
            )
            state.transformer_keys[layer_index] = torch.zeros(
                shape, device=device, dtype=dtype
            )
            state.transformer_values[layer_index] = torch.zeros(
                shape, device=device, dtype=dtype
            )
        return state

    def state_bytes_per_stream(self, dtype: torch.dtype) -> int:
        return self.state_spec().bytes_per_stream(dtype)

    def decode(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
    ) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires codes shaped [B, Q, T]"
            )
        if int(codes.shape[0]) < 1:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires at least one row"
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
        state.advance(fresh_frames)
        return waveform
