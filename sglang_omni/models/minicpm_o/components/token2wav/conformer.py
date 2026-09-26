# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Conformer for MiniCPM-o."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.minicpm_o.components.token2wav.causal_conv import (
    CausalConv1d,
    ConvState,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer_layers import (
    ConformerEncoderLayer,
    EspnetRelPositionalEncoding,
    LinearNoSubsampling,
    PositionwiseFeedForward,
    RelPositionMultiHeadedAttention,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer_state import (
    AttentionState,
    ConformerState,
)


class Upsample1D(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int,
        stride: int = 2,
        scale_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = CausalConv1d(
            self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0
        )
        self.scale_factor = (
            float(self.stride) if scale_factor is None else float(scale_factor)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        state: ConvState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ConvState | None]:
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        outputs, next_state = self.conv(outputs, state)
        return outputs, input_lengths * self.stride, next_state


class PreLookaheadLayer(nn.Module):

    def __init__(self, channels: int, pre_lookahead_len: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size=3, stride=1, padding=0
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: ConvState | None = None,
        last_chunk: bool = True,
    ) -> tuple[torch.Tensor, ConvState | None]:
        outputs = inputs.transpose(1, 2)
        if state is None:
            outputs = outputs.contiguous()
        else:
            pass
        if last_chunk:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len))
        else:
            pass
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs, next_state = self.conv2(outputs, state)
        outputs = outputs.transpose(1, 2)
        if state is None:
            outputs = outputs.contiguous()
        else:
            pass
        return outputs + inputs[:, : outputs.shape[1]], next_state


class UpsampleConformerEncoderV2(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        input_layer: Literal["linear"] = "linear",
        pre_lookahead_len: int = 3,
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        up_stride: int = 2,
        up_scale_factor: float = 2,
        attention_heads: int = 4,
        pos_enc_layer_type: Literal["rel_pos_espnet"] = "rel_pos_espnet",
        selfattention_layer_type: Literal["rel_selfattn"] = "rel_selfattn",
        key_bias: bool = True,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        activation_type: Literal["swish"] = "swish",
    ) -> None:
        super().__init__()
        if (
            input_layer,
            pos_enc_layer_type,
            selfattention_layer_type,
            activation_type,
        ) != ("linear", "rel_pos_espnet", "rel_selfattn", "swish"):
            raise ValueError("Unsupported MiniCPM-o flow encoder configuration")
        else:
            pass
        self.output_dim = output_size
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-05)
        activation = nn.SiLU()
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)
        self.pre_lookahead_layer = PreLookaheadLayer(
            channels=output_size, pre_lookahead_len=pre_lookahead_len
        )
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(
            channels=output_size,
            out_channels=output_size,
            stride=up_stride,
            scale_factor=up_scale_factor,
        )
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_up_blocks)
            ]
        )

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        last_chunk: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, masks, _ = self.run_stages(xs, xs_lens, None, last_chunk)
        return xs, masks

    def run_stages(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        state: ConformerState | None,
        last_chunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, ConformerState | None]:
        attention = state.attention if state is not None else ()
        history_length = (
            attention[0].history.shape[2]
            if attention and attention[0].history is not None
            else 0
        )
        lookahead_state = state.lookahead if state is not None else None
        upsample_state = state.upsample if state is not None else None
        attention_states: list[AttentionState] = []
        for stage, (embedding, layers) in enumerate(
            ((self.embed, self.encoders), (self.up_embed, self.up_encoders))
        ):
            masks = (
                ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
                if state is None
                else xs.new_empty((0, 0, 0))
            )
            xs, positions, masks = embedding(xs, masks)
            if state is None:
                xs = xs * masks.transpose(1, 2).to(xs)
            else:
                pass
            if stage == 0:
                xs, lookahead_state = self.pre_lookahead_layer(
                    xs, lookahead_state, last_chunk
                )
                if state is None:
                    xs = xs * masks.transpose(1, 2).to(xs)
                else:
                    pass
            else:
                pass
            if state is not None:
                positions = embedding.pos_enc.position_embedding(xs, history_length)
            else:
                pass
            previous_states = (
                (state.attention if stage == 0 else state.up_attention)
                if state is not None
                else ()
            )
            assert not previous_states or len(previous_states) == len(layers)
            for index, layer in enumerate(layers):
                if state is None:
                    layer_state = None
                else:
                    layer_state = (
                        previous_states[index] if previous_states else AttentionState()
                    )
                xs, next_layer_state = layer(xs, masks, positions, layer_state)
                if next_layer_state is not None:
                    attention_states.append(next_layer_state)
                else:
                    pass
            if stage == 0:
                xs = xs.transpose(1, 2)
                if state is None:
                    xs = xs.contiguous()
                else:
                    pass
                xs, xs_lens, upsample_state = self.up_layer(
                    xs,
                    xs_lens,
                    upsample_state,
                )
                xs = xs.transpose(1, 2)
                if state is None:
                    xs = xs.contiguous()
                else:
                    pass
                history_length *= self.up_layer.stride
            else:
                pass
        if self.normalize_before:
            xs = self.after_norm(xs)
        else:
            pass
        if state is None:
            next_state = None
        else:
            assert lookahead_state is not None and upsample_state is not None
            next_state = ConformerState(
                lookahead=lookahead_state,
                upsample=upsample_state,
                attention=tuple(attention_states[: len(self.encoders)]),
                up_attention=tuple(attention_states[len(self.encoders) :]),
            )
        return xs, masks, next_state

    def forward_chunk(
        self,
        xs: torch.Tensor,
        last_chunk: bool = False,
        state: ConformerState | None = None,
    ) -> tuple[torch.Tensor, ConformerState]:
        state = ConformerState() if state is None else state
        lengths = torch.full((xs.shape[0],), xs.shape[1], device=xs.device)
        xs, _, next_state = self.run_stages(xs, lengths, state, last_chunk)
        assert next_state is not None
        return xs, next_state


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
