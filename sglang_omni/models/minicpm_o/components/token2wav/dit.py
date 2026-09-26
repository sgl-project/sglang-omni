# SPDX-License-Identifier: Apache-2.0
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Dit for MiniCPM-o."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, repeat
from torch.nn.attention.varlen import varlen_attn

from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    pack_fixed_capacity,
    unpack_fixed_capacity,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
    PackedDiTCudaGraphRunner,
)

TIMESTEP_MAX_PERIOD = 10000


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, ts: torch.Tensor) -> torch.Tensor:
        b, t, c = ts.shape
        ts = ts.reshape(b, t, self.num_heads, c // self.num_heads)
        ts = ts.transpose(1, 2)
        return ts

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_heads(q)
        k = self.to_heads(k)
        v = self.to_heads(v)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_packed(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        q = self.to_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x).view(-1, self.num_heads, self.head_dim)
        attention_dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.to(attention_dtype)
        k = k.to(attention_dtype)
        v = v.to(attention_dtype)
        x = varlen_attn(q, k, v, cu_seqlens, cu_seqlens, max_length, max_length)
        x = self.proj(x.reshape(-1, self.inner_dim))
        return self.proj_drop(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.scale = 1000
        half = frequency_embedding_size // 2
        # note (MayDomine): autocast timesteps can remain FP32 with FP16 weights.
        self.frequencies = torch.exp(
            -math.log(TIMESTEP_MAX_PERIOD) * torch.arange(half) / half
        )
        self.frequency_cache: torch.Tensor | None = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if (
            self.frequency_cache is None
            or self.frequency_cache.device != t.device
            or self.frequency_cache.dtype != t.dtype
        ):
            frequencies = self.frequencies.to(t)
            self.frequency_cache = frequencies
        else:
            frequencies = self.frequency_cache
        args = (t * self.scale)[:, None] * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        else:
            pass
        return self.mlp(embedding)


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x


class CausalConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = torch.nn.Sequential(
            Transpose(1, 2),
            CausalConv1d(in_channels, out_channels, kernel_size),
            Transpose(1, 2),
            nn.LayerNorm(out_channels),
            nn.Mish(),
            Transpose(1, 2),
            CausalConv1d(out_channels, out_channels, kernel_size),
            Transpose(1, 2),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        else:
            pass
        x = self.block(x)
        if mask is not None:
            x = x * mask
        else:
            pass
        return x

    def forward_packed(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        guarded_valid: torch.Tensor,
    ) -> torch.Tensor:
        guarded_length = guarded_valid.shape[0]
        guarded = x.new_zeros(guarded_length, x.shape[1])
        guarded[positions] = x

        guarded = self.block[1](guarded.transpose(0, 1).unsqueeze(0))
        guarded = guarded.squeeze(0).transpose(0, 1)
        guarded = self.block[3](guarded)
        guarded = self.block[4](guarded)
        guarded = guarded * guarded_valid.unsqueeze(1)
        guarded = self.block[6](guarded.transpose(0, 1).unsqueeze(0))
        guarded = guarded.squeeze(0).transpose(0, 1)
        return guarded[positions]


class DiTBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=True,
            qk_norm=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.conv = CausalConvBlock(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask
        )
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward_packed(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        sequence_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_length: int,
        conv_positions: torch.Tensor,
        conv_valid: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(c)[sequence_ids].chunk(9, dim=-1)
        x = x + gate_msa * self.attn.forward_packed(
            modulate(self.norm1(x), shift_msa, scale_msa),
            cu_seqlens,
            max_length,
        )
        x = x + gate_conv * self.conv.forward_packed(
            modulate(self.norm3(x), shift_conv, scale_conv),
            conv_positions,
            conv_valid,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def forward_packed(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        sequence_ids: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning)[sequence_ids].chunk(
            2, dim=-1
        )
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
        enable_variable_length: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enable_variable_length = enable_variable_length
        self.packed_graph_runner: PackedDiTCudaGraphRunner | None = None
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:

        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                else:
                    pass
            else:
                pass

        self.apply(initialize_linear)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        *,
        packed_layout: FixedPackedLayout | None = None,
        packed_graph: CapturedPackedDiTGraph | None = None,
    ) -> torch.Tensor:
        t = self.t_embedder(t).unsqueeze(1)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        else:
            pass
        if cond is not None:
            x = pack([x, cond], "b * t")[0]
        else:
            pass
        x = x.transpose(1, 2)
        if self.enable_variable_length and x.shape[0] > 2:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                packed_input = self.in_proj(x).to(torch.bfloat16)
                conditioning = t.to(torch.bfloat16)
                if packed_layout is not None:
                    return self.forward_packed_fixed(
                        packed_input, conditioning, packed_layout, packed_graph
                    )
                else:
                    lengths = mask.bool().squeeze(1).sum(dim=1, dtype=torch.int32)
                    return self.forward_packed(packed_input, conditioning, lengths)
        else:
            pass
        attn_mask = mask.bool()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, t, attn_mask)
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x

    def forward_packed(
        self, x: torch.Tensor, conditioning: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch_size, padded_length, _ = x.shape
        valid = torch.arange(padded_length, device=x.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        x = x[valid]
        conditioning = conditioning.squeeze(1)
        cu_seqlens = torch.nn.functional.pad(
            lengths.cumsum(0, dtype=torch.int32), (1, 0)
        )
        max_length = padded_length
        guard_width = self.blocks[0].conv.kernel_size - 1
        sequence_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=x.device), lengths
        )
        conv_positions = (
            torch.arange(x.shape[0], device=x.device) + (sequence_ids + 1) * guard_width
        )
        conv_valid = torch.zeros(
            x.shape[0] + batch_size * guard_width,
            device=x.device,
            dtype=torch.bool,
        )
        conv_valid[conv_positions] = True
        x = self.run_packed_blocks(
            x,
            conditioning,
            sequence_ids,
            cu_seqlens,
            max_length,
            conv_positions,
            conv_valid,
        )
        dense = x.new_zeros(batch_size, padded_length, self.out_channels)
        dense[valid] = x
        return dense.transpose(1, 2)

    def forward_packed_fixed(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        layout: FixedPackedLayout,
        captured_graph: CapturedPackedDiTGraph | None = None,
    ) -> torch.Tensor:
        """Run packed DiT blocks with static tensor shapes for graph replay."""
        padded_length = x.shape[1]
        packed = pack_fixed_capacity(
            x,
            layout,
            out=captured_graph.inputs[0] if captured_graph is not None else None,
        )
        if captured_graph is not None:
            assert self.packed_graph_runner is not None
            packed = self.packed_graph_runner.replay(
                captured_graph, conditioning.squeeze(1)
            )
        else:
            conditioning_rows = torch.cat(
                (
                    conditioning.squeeze(1),
                    conditioning.new_zeros((1, conditioning.shape[-1])),
                ),
                dim=0,
            )
            packed = self.run_packed_blocks(
                packed,
                conditioning_rows,
                layout.row_ids,
                layout.cu_seqlens,
                padded_length,
                layout.conv_positions,
                layout.conv_valid,
            )
        return unpack_fixed_capacity(packed, layout).transpose(1, 2)

    def run_packed_blocks(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        sequence_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_length: int,
        conv_positions: torch.Tensor,
        conv_valid: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block.forward_packed(
                x,
                conditioning,
                sequence_ids,
                cu_seqlens,
                max_length,
                conv_positions,
                conv_valid,
            )
        x = self.final_layer.forward_packed(x, conditioning, sequence_ids)
        return x
