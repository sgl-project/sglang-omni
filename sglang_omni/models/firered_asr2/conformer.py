# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)
"""FireRedASR2 Conformer, adapted from FireRedTeam/FireRedASR2S.

Source: fireredasr2s/fireredasr2/models/module/conformer_encoder.py.
Keep checkpoint parameter names and relative-position attention unchanged.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers_enc: int,
        n_head: int,
        d_model: int,
        residual_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ) -> None:
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model, pe_maxlen)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList(
            RelPosEmbConformerBlock(
                d_model, n_head, residual_dropout, dropout_rate, kernel_size
            )
            for _ in range(n_layers_enc)
        )

    def forward(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        else:
            pass
        padding_mask = (
            torch.arange(padded_input.size(1), device=padded_input.device)[None, :]
            < input_lengths.to(padded_input.device)[:, None]
        ).unsqueeze(1)

        embeddings, input_lengths, padding_mask = self.input_preprocessor(
            padded_input, padding_mask
        )
        encoder_output = self.dropout(embeddings)

        position_embeddings = self.dropout(self.positional_encoding(embeddings))

        for encoder_layer in self.layer_stack:
            encoder_output = encoder_layer(
                encoder_output,
                position_embeddings,
                slf_attn_mask=padding_mask,
                pad_mask=padding_mask,
            )

        return encoder_output, input_lengths, padding_mask


class RelPosEmbConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        residual_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_size: int = 33,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model, residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size, dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        slf_attn_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, position_embeddings, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)

        self.context = 7

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        batch_size, channels, frames, features = x.size()
        x = self.out(
            x.transpose(1, 2).contiguous().view(batch_size, frames, channels * features)
        )
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)).item() / d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_positions, frames = self.pe.size(1), x.size(1)
        position_embeddings = (
            self.pe[:, max_positions // 2 - frames + 1 : max_positions // 2 + frames]
            .clone()
            .detach()
        )
        return position_embeddings


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model * 4)
        nonlinear = Swish()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model * 4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(
            pre_layer_norm,
            linear_expand,
            nonlinear,
            dropout_pre,
            linear_project,
            dropout_post,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.net(x)
        output = output + residual
        return output


class ConformerConvolution(nn.Module):
    def __init__(
        self, d_model: int, kernel_size: int = 33, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        else:
            pass
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)

        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)

        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        else:
            pass
        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(
        self, n_head: int, d_model: int, residual_dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    def forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, query_length, key_length, value_length = (
            q.size(0),
            q.size(1),
            k.size(1),
            v.size(1),
        )

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(batch_size, query_length, n_head, d_k)
        k = self.w_ks(k).view(batch_size, key_length, n_head, d_k)
        v = self.w_vs(v).view(batch_size, value_length, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(
        self,
        output: torch.Tensor,
        residual: torch.Tensor,
        batch_size: int,
        query_length: int,
    ) -> torch.Tensor:
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        fc_out = self.fc(output)
        output = self.dropout(fc_out)
        output = output + residual
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.0)
        self.INF = float("inf")

    def forward_attention(
        self,
        attention_weights: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attention_weights = attention_weights.masked_fill(mask, -self.INF)
            attention_weights = torch.softmax(attention_weights, dim=-1).masked_fill(
                mask, 0.0
            )
        else:
            attention_weights = torch.softmax(attention_weights, dim=-1)

        dropped_attention_weights = self.dropout(attention_weights)
        output = torch.matmul(dropped_attention_weights, v)

        return output, attention_weights


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(
        self, n_head: int, d_model: int, residual_dropout: float = 0.1
    ) -> None:
        super().__init__(n_head, d_model, residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        # note (cocoa00): empty honors the loader device and dtype; FloatTensor does not.
        self.pos_bias_u = nn.Parameter(torch.empty(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def relative_shift(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, heads, query_length, key_length = x.size()
        zero_pad = torch.zeros(
            (batch_size, heads, query_length, 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(batch_size, heads, key_length + 1, query_length)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        position_batch_size = position_embeddings.size(0)
        p = self.linear_pos(position_embeddings).view(
            position_batch_size, -1, self.n_head, self.d_k
        )
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.relative_shift(matrix_bd)

        attention_scores = matrix_ac + matrix_bd
        attention_scores.mul_(self.scale)

        output, attention_weights = self.attention.forward_attention(
            attention_scores, v, mask=mask
        )

        output = self.forward_output(output, residual, batch_size, query_length)
        return output, attention_weights
