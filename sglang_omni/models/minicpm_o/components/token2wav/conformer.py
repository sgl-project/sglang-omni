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

from sglang_omni.models.minicpm_o.components.token2wav.conformer_layers import (
    ConformerEncoderLayer,
    EspnetRelPositionalEncoding,
    LinearNoSubsampling,
    PositionwiseFeedForward,
    RelPositionMultiHeadedAttention,
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
        self.conv = nn.Conv1d(
            self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0
        )
        self.scale_factor = (
            float(self.stride) if scale_factor is None else float(scale_factor)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        cache: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        if cache:
            outputs = torch.cat((cache[0], outputs), dim=2)
        else:
            outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        if cache is not None:
            cache[:] = [outputs[:, :, -self.stride * 2 :].clone()]
        else:
            pass
        outputs = self.conv(outputs)
        return (outputs, input_lengths * self.stride)


class PreLookaheadLayer(nn.Module):

    def __init__(self, channels: int, pre_lookahead_len: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: list[torch.Tensor] | None = None,
        last_chunk: bool = True,
    ) -> torch.Tensor:
        outputs = inputs.transpose(1, 2)
        if cache is None:
            outputs = outputs.contiguous()
        else:
            pass
        if last_chunk:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len))
        else:
            pass
        outputs = F.leaky_relu(self.conv1(outputs))
        if cache:
            outputs = torch.cat((cache[0], outputs), dim=2)
        else:
            outputs = F.pad(outputs, (2, 0))
        if cache is not None:
            cache[:] = [outputs[:, :, -2:].clone()]
        else:
            pass
        outputs = self.conv2(outputs).transpose(1, 2)
        if cache is None:
            outputs = outputs.contiguous()
        else:
            pass
        return outputs + inputs[:, : outputs.shape[1]]


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
        cache: dict[str, torch.Tensor] | None = None,
        last_chunk: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = cache.get("attention") if cache is not None else None
        history_length = (
            attention.shape[3] // self.up_layer.stride if attention is not None else 0
        )
        lookahead_cache = (
            [cache["cnn"][:, :, :2]] if cache is not None and cache else []
        )
        upsample_cache = [cache["cnn"][:, :, 2:]] if cache is not None and cache else []
        attention_caches = []
        for stage, (embedding, layers) in enumerate(
            ((self.embed, self.encoders), (self.up_embed, self.up_encoders))
        ):
            masks = (
                ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
                if cache is None
                else xs.new_empty((0, 0, 0))
            )
            xs, positions, masks = embedding(xs, masks)
            if cache is None:
                xs = xs * masks.transpose(1, 2).to(xs)
            else:
                pass
            if stage == 0:
                xs = self.pre_lookahead_layer(
                    xs, lookahead_cache if cache is not None else None, last_chunk
                )
                if cache is None:
                    xs = xs * masks.transpose(1, 2).to(xs)
                else:
                    pass
            else:
                pass
            if cache is not None:
                positions = embedding.pos_enc.position_embedding(xs, history_length)
            else:
                pass
            cache_offset = 0 if stage == 0 else len(self.encoders)
            for index, layer in enumerate(layers):
                layer_cache = (
                    [attention[cache_offset + index, :, :, :history_length]]
                    if attention is not None
                    else []
                )
                xs = layer(
                    xs, masks, positions, layer_cache if cache is not None else None
                )
                if cache is not None:
                    attention_caches.append(layer_cache[0])
                else:
                    pass
            if stage == 0:
                xs = xs.transpose(1, 2)
                if cache is None:
                    xs = xs.contiguous()
                else:
                    pass
                xs, xs_lens = self.up_layer(
                    xs,
                    xs_lens,
                    upsample_cache if cache is not None else None,
                )
                xs = xs.transpose(1, 2)
                if cache is None:
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
        if cache is not None:
            cache["cnn"] = torch.cat((lookahead_cache[0], upsample_cache[0]), dim=2)
            first = torch.stack(attention_caches[: len(self.encoders)]).repeat(
                1, 1, 1, self.up_layer.stride, 1
            )
            second = torch.stack(attention_caches[len(self.encoders) :])
            cache["attention"] = torch.cat((first, second))
        else:
            pass
        return xs, masks

    def forward_chunk(
        self,
        xs: torch.Tensor,
        last_chunk: bool = False,
        cnn_cache: torch.Tensor | None = None,
        att_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = (
            {"cnn": cnn_cache, "attention": att_cache} if att_cache is not None else {}
        )
        lengths = torch.full((xs.shape[0],), xs.shape[1], device=xs.device)
        xs, _ = self.forward(xs, lengths, cache, last_chunk)
        return xs, cache["cnn"], cache["attention"]


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
