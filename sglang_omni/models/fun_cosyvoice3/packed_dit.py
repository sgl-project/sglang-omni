# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 DiT on a packed sequence: the rows of a Flow batch concatenated
along the sequence for every per token module, attention within each row."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from sglang.kernels.ops.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

logger = logging.getLogger(__name__)

# Note (Jiaxin Deng): each positional conv has kernel 31, so it reads the 30
# frames before its input frame.
CONV_CONTEXT_FRAMES = 30
# note (ratish, chenyang): a row's chunks share a key prefix, so FA3 pages are one frame.
FA3_PAGE_SIZE = 1
# note(ratish): the fused positional conv's tl.dot reduces at least 16 channels of a power
# of two per group.
GROUP_CONV_KERNEL_CHANNELS = (16, 32, 64, 128)
FA3_DTYPES = (torch.float16, torch.bfloat16)
# note(ratish): the first call benchmark runs at a warmup shape, not a serving one,
# so its pick can change between boots; the heuristic config is the same on every boot.
DIT_INDUCTOR_OPTIONS: dict[str, bool] = {"triton.autotune_pointwise": False}
PACKED_INDUCTOR_OPTIONS: dict[str, bool] = {
    **DIT_INDUCTOR_OPTIONS,
    "emulate_precision_casts": True,
}


def ragged_fa3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_splits: int = 0,
) -> torch.Tensor:
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        causal=False,
        num_splits=num_splits,
    )


# note(ratish): the compiled forward calls FA3 through this alias-free op;
# eager calls ragged_fa3 directly and skips the custom op dispatch per block.
packed_fa3 = torch.library.custom_op(
    "sglang_omni_fun_cosyvoice3::packed_fa3", mutates_args=(), device_types="cuda"
)(ragged_fa3)


@packed_fa3.register_fake
def fake_packed_fa3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_splits: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


def whole_row_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )


packed_whole_row_fa3 = torch.library.custom_op(
    "sglang_omni_fun_cosyvoice3::whole_row_fa3", mutates_args=(), device_types="cuda"
)(whole_row_fa3)


@packed_whole_row_fa3.register_fake
def fake_whole_row_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    return torch.empty_like(q)


@dataclass(frozen=True)
class PackedRows:
    lengths: tuple[int, ...]
    starts_host: torch.Tensor
    starts: torch.Tensor
    row_ids: torch.Tensor
    positions: torch.Tensor
    # note (ratish): the positional convs read the packed frames with CONV_CONTEXT_FRAMES
    # zero frames before every row. Index 0 is the zero frame, 1 + i packed frame i.
    conv_input_index: torch.Tensor
    conv_output_index: torch.Tensor
    width: int

    @property
    def total(self) -> int:
        return sum(self.lengths)


def pack_rows(lengths: Sequence[int], device: torch.device) -> PackedRows:
    lengths = tuple(int(length) for length in lengths)
    starts_host = F.pad(torch.tensor(lengths, dtype=torch.int64).cumsum(0), (1, 0))
    starts = starts_host.to(device)
    total = int(starts_host[-1])
    row_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
        output_size=total,
    )
    frames = torch.arange(total, device=device)
    conv_output_index = frames + CONV_CONTEXT_FRAMES * row_ids
    conv_input_index = frames.new_zeros(total + CONV_CONTEXT_FRAMES * len(lengths))
    conv_input_index[conv_output_index + CONV_CONTEXT_FRAMES] = frames + 1
    return PackedRows(
        lengths=lengths,
        starts_host=starts_host.to(torch.int32),
        starts=starts.to(torch.int32),
        row_ids=row_ids,
        positions=frames - starts[row_ids],
        conv_input_index=conv_input_index,
        conv_output_index=conv_output_index,
        width=max(lengths),
    )


def gather_rows(padded: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    """(rows, width, channels) -> (1, total, channels), each row's first
    length frames in row order."""
    width = padded.shape[1]
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)


def scatter_rows(packed: torch.Tensor, rows: PackedRows, width: int) -> torch.Tensor:
    """(1, total, channels) -> (rows, width, channels), zero past each row's
    length."""
    row_count, channels = len(rows.lengths), packed.shape[2]
    flat = packed.new_zeros(row_count * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(row_count, width, channels)


def chunk_causal_mask(
    length: int, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """(length, length) bool: a frame attends every frame of its chunk and of
    the chunks before it, CosyVoice's subsequent_chunk_mask."""
    position = torch.arange(length, device=device)
    chunk_end = (position // chunk_size + 1) * chunk_size
    return position.unsqueeze(0) < chunk_end.unsqueeze(1)


def chunk_segments(
    lengths: Sequence[int], chunk_size: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """One query segment per (row, chunk), reading that row's frames
    [0, chunk end)."""
    segment_rows: list[int] = []
    segment_ends: list[int] = []
    offsets: list[int] = [0]
    for row, length in enumerate(lengths):
        frame = 0
        while frame < length:
            end = min((frame // chunk_size + 1) * chunk_size, length)
            segment_rows.append(row)
            segment_ends.append(end)
            offsets.append(offsets[-1] + end - frame)
            frame = end
    return tuple(segment_rows), tuple(segment_ends), tuple(offsets)


class RowAttention:
    """Attention within each row of a packed sequence, computed as the padded
    DiT computes it: one SDPA call over the rows scattered to the padded layout,
    under the row's key mask and, for hops, the chunk causal mask, built once
    per Flow call.
    """

    def __init__(self, rows: PackedRows, *, chunk_size: int | None, heads: int) -> None:
        self.rows = rows
        self.heads = heads
        width = rows.width
        device = rows.row_ids.device
        lengths = torch.tensor(rows.lengths, device=device)
        keys = torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        if chunk_size is None:
            mask = keys.unsqueeze(1).expand(-1, width, -1)
        else:
            mask = keys.unsqueeze(1) & chunk_causal_mask(width, chunk_size, device)
        self.mask = mask.unsqueeze(1)

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, total, heads * head_dim). Returns the same
        shape."""
        row_count, width = len(self.rows.lengths), self.rows.width
        padded = scatter_rows(torch.cat((query, key, value), dim=-1), self.rows, width)
        query, key, value = (
            part.view(row_count, width, self.heads, -1).transpose(1, 2)
            for part in padded.chunk(3, dim=-1)
        )
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask)
        return gather_rows(out.transpose(1, 2).reshape(row_count, width, -1), self.rows)


class RaggedRowAttention:
    """Chunk causal row attention on the packed sequence via FA3 paged KV, no
    pad-to-widest."""

    def __init__(
        self,
        rows: PackedRows,
        *,
        chunk_size: int,
        heads: int,
        head_dim: int,
    ) -> None:
        self.heads = heads
        self.head_dim = head_dim
        device = rows.row_ids.device
        segment_rows, segment_ends, offsets = chunk_segments(rows.lengths, chunk_size)
        self.cache_seqlens = torch.tensor(
            segment_ends, dtype=torch.int32, device=device
        )
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.max_seqlen_q = max(end - start for start, end in pairwise(offsets))
        starts = rows.starts_host[list(segment_rows)].to(device)
        # note (ratish): FA3 page ids must land inside the packed keys; pad with page 0.
        page = torch.arange(max(segment_ends), dtype=torch.int32, device=device)
        self.page_table = torch.where(
            page.unsqueeze(0) < self.cache_seqlens.unsqueeze(1),
            starts.unsqueeze(1) + page.unsqueeze(0),
            0,
        )

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, total, heads * head_dim). Returns the same
        shape."""
        page_shape = (-1, FA3_PAGE_SIZE, self.heads, self.head_dim)
        if torch.compiler.is_compiling():
            fa3 = packed_fa3
        else:
            fa3 = ragged_fa3
        out = fa3(
            query[0].reshape(-1, self.heads, self.head_dim),
            key[0].reshape(page_shape),
            value[0].reshape(page_shape),
            self.cache_seqlens,
            self.page_table,
            self.cu_seqlens_q,
            self.max_seqlen_q,
        )
        return out.reshape(1, -1, self.heads * self.head_dim)


class WholeRowAttention:
    """Bidirectional attention within each whole row of the packed sequence via
    FA3 varlen. max_frames bounds the widest row."""

    def __init__(
        self, rows: PackedRows, *, heads: int, head_dim: int, max_frames: int
    ) -> None:
        self.heads = heads
        self.head_dim = head_dim
        self.cu_seqlens = rows.starts
        self.max_frames = max_frames

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, total, heads * head_dim). Returns the same
        shape."""
        if torch.compiler.is_compiling():
            fa3 = packed_whole_row_fa3
        else:
            fa3 = whole_row_fa3
        head_shape = (-1, self.heads, self.head_dim)
        out = fa3(
            query[0].reshape(head_shape),
            key[0].reshape(head_shape),
            value[0].reshape(head_shape),
            self.cu_seqlens,
            self.max_frames,
        )
        return out.reshape(1, -1, self.heads * self.head_dim)


PackedRowAttention = RowAttention | RaggedRowAttention | WholeRowAttention


class PackedDiT:
    """DiT.forward over a packed sequence with the same modules in the same
    order. Attention is ragged on FA3 half-precision CUDA and padded elsewhere.
    """

    def __init__(self, dit: torch.nn.Module, *, device: str | torch.device) -> None:
        self.dit = dit
        device = torch.device(device)
        self.is_ragged = device.type == "cuda" and _is_fa3_supported()
        self.is_compiled = False
        # note (ratish): Parameters, whose shapes Dynamo keeps static under the dynamic
        # prefix compile. to_q, to_k and to_v become row views of them, so the DiT
        # must already hold its serving dtype.
        qkv_weights: list[torch.nn.Parameter] = []
        qkv_biases: list[torch.nn.Parameter] = []
        with torch.no_grad():
            for block in dit.transformer_blocks:
                attention = block.attn
                projections = (attention.to_q, attention.to_k, attention.to_v)
                qkv_weight = torch.nn.Parameter(
                    torch.cat([projection.weight for projection in projections]),
                    requires_grad=False,
                )
                qkv_bias = torch.nn.Parameter(
                    torch.cat([projection.bias for projection in projections]),
                    requires_grad=False,
                )
                for index, projection in enumerate(projections):
                    rows = slice(
                        index * attention.inner_dim, (index + 1) * attention.inner_dim
                    )
                    projection.weight = torch.nn.Parameter(
                        qkv_weight[rows], requires_grad=False
                    )
                    projection.bias = torch.nn.Parameter(
                        qkv_bias[rows], requires_grad=False
                    )
                qkv_weights.append(qkv_weight)
                qkv_biases.append(qkv_bias)
        self.qkv_weights = tuple(qkv_weights)
        self.qkv_biases = tuple(qkv_biases)
        self.positional_conv_weights: tuple[torch.Tensor, ...] | None = None
        if device.type == "cuda":
            conv_pos_embed = dit.input_embed.conv_pos_embed
            convs = (conv_pos_embed.conv1[0], conv_pos_embed.conv2[0])
            group_channels = convs[0].in_channels // convs[0].groups
            if (
                convs[0].weight.dtype in FA3_DTYPES
                and group_channels in GROUP_CONV_KERNEL_CHANNELS
            ):
                # note(ratish): Triton ships only with CUDA builds, so the kernel's
                # module is imported here.
                from sglang_omni.models.fun_cosyvoice3.causal_conv import (
                    pack_group_conv_weight,
                )

                self.positional_conv_weights = tuple(
                    pack_group_conv_weight(conv) for conv in convs
                )
                conv_pos_embed.forward = self.native_conv_pos_embed
            else:
                pass
        else:
            pass
        logger.info(
            "Fun-CosyVoice3 Flow row attention on %s: %s",
            device,
            "ragged FA3" if self.is_ragged else "padded SDPA",
        )

    @property
    def chunk_size(self) -> int:
        return int(self.dit.static_chunk_size)

    def row_attention(
        self, rows: PackedRows, *, streaming: bool, dtype: torch.dtype
    ) -> PackedRowAttention:
        attention = self.dit.transformer_blocks[0].attn
        heads = attention.heads
        head_dim = attention.inner_dim // heads
        row_attention: RaggedRowAttention | WholeRowAttention
        if not self.is_ragged or dtype not in FA3_DTYPES:
            return RowAttention(
                rows, chunk_size=self.chunk_size if streaming else None, heads=heads
            )
        elif streaming:
            row_attention = RaggedRowAttention(
                rows, chunk_size=self.chunk_size, heads=heads, head_dim=head_dim
            )
            dynamic_dims = [
                (row_attention.page_table, (0, 1)),
                (row_attention.cu_seqlens_q, 0),
                (row_attention.cache_seqlens, 0),
            ]
        else:
            row_attention = WholeRowAttention(
                rows, heads=heads, head_dim=head_dim, max_frames=rows.width
            )
            dynamic_dims = [(row_attention.cu_seqlens, 0)]
        if self.is_compiled:
            # note(ratish): hints, not constraints; they share x's total frames,
            # which the first call specializes, so mark_dynamic would fail.
            for tensor, dims in dynamic_dims + [
                (rows.row_ids, 0),
                (rows.positions, 0),
                (rows.conv_input_index, 0),
                (rows.conv_output_index, 0),
            ]:
                dynamo.maybe_mark_dynamic(tensor, dims)
        else:
            pass
        return row_attention

    def compile(self, dtype: torch.dtype | None) -> bool:
        if not self.is_ragged or dtype not in FA3_DTYPES:
            logger.debug(
                f"Skipping PackedDiT torch.compile (ragged={self.is_ragged}, dtype={dtype})"
            )
            return False
        else:
            pass
        # note(ratish): not dynamic=True, which makes the head count and size symbolic;
        # the reshape into FA3's layout then copies query and key in every block.
        self.forward = torch.compile(
            self.forward,
            backend="inductor",
            fullgraph=True,
            options=dict(PACKED_INDUCTOR_OPTIONS),
        )
        self.is_compiled = True
        logger.info(
            "Compiled the Fun-CosyVoice3 PackedDiT forward "
            f"(fullgraph=True, emulate_precision_casts=True, dtype={dtype})"
        )
        return True

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: PackedRowAttention,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """x, mu, cond, spks: (1, total, channels); t: (1,); rope: rope(rows).
        Returns (1, total, out_channels)."""
        dit = self.dit
        t = dit.time_embed(t)
        h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
        h = self.conv_pos_embed(h, rows) + h
        residual = h
        for block_index, block in enumerate(dit.transformer_blocks):
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(h, emb=t)
            h = h + gate_msa.unsqueeze(1) * self.attend(
                block.attn,
                norm,
                rope,
                attention,
                self.qkv_weights[block_index],
                self.qkv_biases[block_index],
            )
            ff_norm = block.ff_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
        if dit.long_skip_connection is not None:
            h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
        else:
            pass
        h = dit.norm_out(h, t)
        return dit.proj_out(h)

    def conv_pos_embed(self, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        # note (ratish): the padded call zero pads conv2's input, not conv1's output,
        # so conv2's gaps are gathered again.
        first_input = torch.cat((h.new_zeros(1, h.shape[2]), h[0]))[
            rows.conv_input_index
        ]
        first_output = self.positional_conv(first_input, 0)
        second_input = torch.cat(
            (
                first_output.new_zeros(1, first_output.shape[1]),
                first_output[rows.conv_output_index],
            )
        )[rows.conv_input_index]
        second_output = self.positional_conv(second_input, 1)
        return second_output[rows.conv_output_index].unsqueeze(0)

    def positional_conv(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Positional conv index with its Mish over x: (frames, channels), each
        output frame reading the CONV_CONTEXT_FRAMES frames before it."""
        conv_pos_embed = self.dit.input_embed.conv_pos_embed
        conv = (conv_pos_embed.conv1, conv_pos_embed.conv2)[index]
        weights = self.positional_conv_weights
        if weights is not None and x.dtype == weights[index].dtype:
            return torch.ops.sglang_omni_fun_cosyvoice3.group_conv_mish(
                x.unsqueeze(0), weights[index], conv[0].bias
            )[0]
        else:
            return conv(x.T.unsqueeze(0))[0].T

    def native_conv_pos_embed(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The padded DiT's positional convs on the fused kernel. x: (batch,
        frames, channels), mask: (batch, frames)."""
        conv_pos_embed = self.dit.input_embed.conv_pos_embed
        weights = self.positional_conv_weights
        assert weights is not None
        if x.dtype != weights[0].dtype:
            return type(conv_pos_embed).forward(conv_pos_embed, x, mask)
        elif mask is not None:
            x = x.masked_fill(~mask[..., None], 0.0)
        else:
            pass
        padding = (0, 0, CONV_CONTEXT_FRAMES, 0)
        x = torch.ops.sglang_omni_fun_cosyvoice3.group_conv_mish(
            F.pad(x, padding).contiguous(), weights[0], conv_pos_embed.conv1[0].bias
        )
        out = torch.ops.sglang_omni_fun_cosyvoice3.group_conv_mish(
            F.pad(x, padding), weights[1], conv_pos_embed.conv2[0].bias
        )
        if mask is not None:
            out = out.masked_fill(~mask[..., None], 0.0)
        else:
            pass
        return out

    def rope_angles(self, frame_count: int) -> torch.Tensor:
        """RoPE angles of positions [0, frame_count), (1, frame_count, rotary
        dims), in float32."""
        angles, scale = self.dit.rotary_embed.forward_from_seq_len(frame_count)
        assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
        return angles

    @staticmethod
    def attend(
        attn: torch.nn.Module,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention: PackedRowAttention,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = F.linear(x, qkv_weight, qkv_bias).chunk(3, dim=-1)
        if torch.compiler.is_compiling():
            query = rotated(query, *rope)
            key = rotated(key, *rope)
        else:
            rotate_in_place(query, *rope)
            rotate_in_place(key, *rope)
        out = attention(query, key, value).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](out))


def rotate_in_place(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """x: (1, total, heads * head_dim). Interleaved RoPE in float32 on the
    rotary dims, rounded back into x."""
    # note (ratish): the DiT rotates only the first rotary dims of the
    # flattened heads, so the rest of x is never copied.
    rotary = x[..., : cos.shape[-1]]
    half = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), dim=-1).flatten(-2)
    rotary.copy_(rotary * cos + half * sin)


def rotated(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # note(ratish): the same values as rotate_in_place; compiled, its in-place write
    # becomes a full copy of x before FA3, while this where is one kernel.
    rotary_dims, width = cos.shape[-1], x.shape[-1]
    cos = F.pad(cos, (0, width - rotary_dims))
    sin = F.pad(sin, (0, width - rotary_dims))
    half = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    turned = (x * cos + half * sin).to(x.dtype)
    is_rotary = torch.arange(width, device=x.device) < rotary_dims
    return torch.where(is_rotary, turned, x)


def solve_flow_euler_packed(
    estimator: PackedDiT,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    twin_rows: PackedRows,
    attention: PackedRowAttention,
    angles: torch.Tensor,
    *,
    cfg_rate: float,
) -> torch.Tensor:
    """Euler steps with classifier free guidance, the rows and their twins in one
    DiT call. noise, mu, cond: (1, total, channels), spks: (rows, channels),
    twin_rows the rows then their twins, angles covering the widest row."""
    total = noise.shape[1]
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    spks_cfg = spks_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=noise.device, dtype=spks.dtype)
    # note(ratish): once per solve and outside the compiled forward,
    # whose graph would otherwise hold RoPE's autocast region and miss the AOT cache.
    angles = angles[:, twin_rows.positions]
    rope = (angles.cos(), angles.sin())
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(1, len(time_span)):
        flow_time[:] = t
        vector_field = estimator.forward(
            torch.cat((x, x), dim=1),
            mu_cfg,
            spks_cfg,
            cond_cfg,
            flow_time,
            twin_rows,
            attention,
            rope,
        )
        conditional = vector_field[:, :total]
        unconditional = vector_field[:, total:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if step < len(time_span) - 1:
            dt = time_span[step + 1] - t
        else:
            pass
    return x.float()
