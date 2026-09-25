# SPDX-License-Identifier: Apache-2.0
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Dit for MiniCPM-o."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, repeat
from torch.nn.attention.varlen import varlen_attn

from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    build_fixed_packed_layout,
    pack_fixed_capacity,
    unpack_fixed_capacity,
)

TIMESTEP_MAX_PERIOD = 10000
# note (wirybeaver): SeedTTS EN packed batches stay below 784 mel frames; longer runs use eager.
PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH = 1024
logger = logging.getLogger(__name__)


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

    def enable_compiled_blocks(self) -> None:
        """Compile the shared block forward for inference."""
        compiled = torch.compile(DiTBlock.forward, dynamic=True)
        for block in self.blocks:
            block.forward = MethodType(compiled, block)

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
        packed_capacity: int | None = None,
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
        attn_mask = mask.bool()
        lengths = attn_mask.squeeze(1).sum(dim=1, dtype=torch.int32)
        if self.enable_variable_length and x.shape[0] > 2:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                packed_input = self.in_proj(x).to(torch.bfloat16)
                conditioning = t.to(torch.bfloat16)
                if packed_capacity is not None:
                    return self.forward_packed_fixed(
                        packed_input, conditioning, lengths, packed_capacity
                    )
                else:
                    pass
                return self.forward_packed(packed_input, conditioning, lengths)
        else:
            pass
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
        lengths: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Run packed DiT blocks with static tensor shapes for graph replay."""
        padded_length = x.shape[1]
        layout = build_fixed_packed_layout(
            lengths, padded_length, capacity, self.blocks[0].conv.kernel_size - 1
        )
        packed = pack_fixed_capacity(x, layout)
        conditioning_rows = torch.cat(
            (
                conditioning.squeeze(1),
                conditioning.new_zeros((1, conditioning.shape[-1])),
            ),
            dim=0,
        )
        runner = self.packed_graph_runner
        packed_output = (
            runner.replay(capacity, packed, conditioning_rows, layout)
            if runner is not None
            else None
        )
        packed = (
            self.run_packed_blocks(
                packed,
                conditioning_rows,
                layout.row_ids,
                layout.cu_seqlens,
                padded_length,
                layout.conv_positions,
                layout.conv_valid,
            )
            if packed_output is None
            else packed_output
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


@dataclass
class CapturedPackedDiTGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class PackedDiTCudaGraphRunner:
    """Replay fixed-capacity DiT blocks without capturing mel-width-dependent work."""

    def __init__(
        self, estimator: DiT, *, device: torch.device, min_free_gb: float = 3.0
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Packed DiT CUDA graphs require a CUDA device")
        else:
            pass
        self.estimator = estimator
        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self.graphs: dict[tuple[int, int], CapturedPackedDiTGraph] = {}
        self.pool: tuple[int, int] | None = None
        self.lock = Lock()
        self.graph_replays = 0
        self.graph_misses = 0

    @torch.inference_mode()
    def capture(self, shapes: tuple[tuple[int, int], ...]) -> None:
        if not shapes or len(set(shapes)) != len(shapes):
            raise ValueError("Packed DiT CUDA graph shapes must be nonempty and unique")
        else:
            pass
        if any(batch <= 1 or capacity <= 2 * batch for batch, capacity in shapes):
            raise ValueError("Packed DiT graph capacities must exceed the row count")
        else:
            pass
        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        graphs: dict[tuple[int, int], CapturedPackedDiTGraph] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            for batch, capacity in sorted(
                shapes, key=lambda shape: shape[1], reverse=True
            ):
                free, _ = torch.cuda.mem_get_info(self.device)
                if free < self.min_free_bytes:
                    logger.warning(
                        f"MiniCPM-o packed DiT graph skipped batch={batch} "
                        f"capacity={capacity}: free VRAM {free / 1024**3:.1f} GB "
                        f"is below {self.min_free_bytes / 1024**3:.1f} GB headroom"
                    )
                    continue
                else:
                    pass
                try:
                    hidden_size = self.estimator.in_proj.out_features
                    row_length = capacity // (2 * batch + 1)
                    lengths = torch.full(
                        (2 * batch,),
                        row_length,
                        device=self.device,
                        dtype=torch.int32,
                    )
                    layout = build_fixed_packed_layout(
                        lengths,
                        row_length,
                        capacity,
                        self.estimator.blocks[0].conv.kernel_size - 1,
                    )
                    inputs = (
                        torch.zeros(
                            capacity,
                            hidden_size,
                            device=self.device,
                            dtype=torch.bfloat16,
                        ),
                        torch.zeros(
                            2 * batch + 1,
                            hidden_size,
                            device=self.device,
                            dtype=torch.bfloat16,
                        ),
                        layout.row_ids,
                        layout.cu_seqlens,
                        layout.conv_positions,
                        layout.conv_valid,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        max_length = min(capacity, PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH)
                        for _ in range(3):
                            self.estimator.run_packed_blocks(
                                inputs[0],
                                inputs[1],
                                *inputs[2:4],
                                max_length,
                                *inputs[4:],
                            )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            cuda_graph=graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ):
                            output = self.estimator.run_packed_blocks(
                                inputs[0],
                                inputs[1],
                                *inputs[2:4],
                                max_length,
                                *inputs[4:],
                            )
                    graphs[(batch, capacity)] = CapturedPackedDiTGraph(
                        graph, inputs, output
                    )
                except Exception as exc:
                    logger.warning(
                        f"MiniCPM-o packed DiT graph capture failed for "
                        f"batch={batch} capacity={capacity}: {exc}; using eager"
                    )
        current_stream.wait_stream(stream)
        self.graphs = graphs
        logger.info(f"Captured {len(graphs)} MiniCPM-o packed DiT CUDA graph shapes")

    def fit(
        self, batch: int, mel_frames: int, packed_valid_frames: int | None
    ) -> int | None:
        capacity = min(
            (
                shape[1]
                for shape in self.graphs
                if shape[0] == batch
                and packed_valid_frames is not None
                and mel_frames <= min(shape[1], PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH)
                and shape[1] > packed_valid_frames
                and shape[1] - packed_valid_frames <= mel_frames
            ),
            default=None,
        )
        if capacity is None:
            self.graph_misses += 1
        else:
            pass
        return capacity

    def replay(
        self,
        capacity: int,
        packed: torch.Tensor,
        conditioning: torch.Tensor,
        layout: FixedPackedLayout,
    ) -> torch.Tensor | None:
        with self.lock:
            captured = self.graphs.get((layout.valid.shape[0] // 2, capacity))
            if captured is None:
                self.graph_misses += 1
                return None
            else:
                pass
            for target, source in zip(
                captured.inputs,
                (
                    packed,
                    conditioning,
                    layout.row_ids,
                    layout.cu_seqlens,
                    layout.conv_positions,
                    layout.conv_valid,
                ),
                strict=True,
            ):
                target.copy_(source)
            captured.graph.replay()
            self.graph_replays += 1
            return captured.output.clone()
