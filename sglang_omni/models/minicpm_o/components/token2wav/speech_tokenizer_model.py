# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c)  (Mddct: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Speech tokenizer model for MiniCPM-o."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class LayerNorm(nn.LayerNorm):

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


@dataclass(kw_only=True)
class ModelConfig:
    n_mels: int = 128
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8
    use_sdpa: bool = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    real = torch.view_as_real(freqs_cis)
    cos, sin = (real[:, :, 0], real[:, :, 1])
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    D = xq.shape[-1]
    half_l, half_r = (xq[:, :, :, : D // 2], xq[:, :, :, D // 2 :])
    xq_r = torch.cat((-half_r, half_l), dim=-1)
    D = xk.shape[-1]
    half_l, half_r = (xk[:, :, :, : D // 2], xk[:, :, :, D // 2 :])
    xk_r = torch.cat((-half_r, half_l), dim=-1)
    return (xq * cos + xq_r * sin, xk * cos + xk_r * sin)


class FSQCodebook(torch.nn.Module):

    def __init__(self, dim: int, level: int = 3) -> None:
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        h = self.project_down(x).float()
        L = int(self.level)
        eps = 1e-06
        h = torch.tanh(h)
        h = torch.clamp(h, -1 + eps, 1 - eps)
        h = ((h + 1.0) * (L - 1) / 2.0).round().to(torch.int64)
        D = h.size(-1)
        powers = L ** torch.arange(D, device=h.device, dtype=torch.int64)
        idx = (h * powers.unsqueeze(0)).sum(dim=-1)
        idx = idx.reshape(x_shape[0], x_shape[1]).int()
        return idx


class FSQVectorQuantization(torch.nn.Module):

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        assert 3**8 == codebook_size
        self.codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.codebook.encode(x)


class FSMNMultiHeadAttention(nn.Module):

    def __init__(
        self, n_state: int, n_head: int, kernel_size: int = 31, use_sdpa: bool = False
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.fsmn_block = torch.nn.Conv1d(
            n_state,
            n_state,
            kernel_size,
            stride=1,
            padding=0,
            groups=n_state,
            bias=False,
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = torch.nn.ConstantPad1d(
            (self.left_padding, self.right_padding), 0.0
        )
        self.use_sdpa = use_sdpa

    def forward_fsmn(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_pad: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        _, _, D = q.shape
        scale = (D // self.n_head) ** (-0.25)
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        fsm_memory = self.forward_fsmn(v, mask_pad)
        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)
        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (
                (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2),
                qk.detach(),
                fsm_memory,
            )
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0
            )
            output = output.transpose(1, 2).contiguous().view(q.size(0), -1, D)
            return (output, None, fsm_memory)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_pad: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        return (self.out(wv) + fsm_memory, qk)


class ResidualAttentionBlock(torch.nn.Module):

    def __init__(
        self, n_state: int, n_head: int, kernel_size: int = 31, use_sdpa: bool = False
    ) -> None:
        super().__init__()
        self.attn = FSMNMultiHeadAttention(
            n_state, n_head, kernel_size, use_sdpa=use_sdpa
        )
        self.attn_ln = LayerNorm(n_state, eps=1e-06)
        n_mlp = n_state * 4
        self.mlp = torch.nn.Sequential(
            Linear(n_state, n_mlp), torch.nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_pad: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = (
            x
            + self.attn(
                self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis
            )[0]
        )
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(torch.nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.freqs_cis = precompute_freqs_cis(64, 1024 * 2)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa)
                for _ in range(n_layer)
            ]
        )

    def forward(
        self, x: torch.Tensor, x_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)
        freqs_cis = self.freqs_cis.to(x.device)
        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype)
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[: x.size(1)])
        return (x, x_len)


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -10000000000.0
    return mask
