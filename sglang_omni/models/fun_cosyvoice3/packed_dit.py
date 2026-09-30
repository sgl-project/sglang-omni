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
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

logger = logging.getLogger(__name__)

# note (ratish, chenyang): a row's chunks share a key prefix, so FA3 pages are one frame.
FA3_PAGE_SIZE = 1
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
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "sglang_omni_fun_cosyvoice3::native_mish",
    mutates_args=(),
    device_types="cuda",
)
def native_mish(x: torch.Tensor) -> torch.Tensor:
    """Preserve eager CUDA Mish arithmetic across the Inductor boundary."""
    return F.mish(x)


@native_mish.register_fake
def fake_native_mish(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@dataclass(frozen=True)
class PackedRows:
    lengths: tuple[int, ...]
    starts_host: torch.Tensor
    row_ids: torch.Tensor
    positions: torch.Tensor
    # note(ratish): the compiled forward reads these instead of lengths,
    # which it would guard on for every row count.
    row_count: int
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
    positions = torch.arange(total, device=device) - starts[row_ids]
    return PackedRows(
        lengths=lengths,
        starts_host=starts_host.to(torch.int32),
        row_ids=row_ids,
        positions=positions,
        row_count=len(lengths),
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
    channels = packed.shape[2]
    flat = packed.new_zeros(rows.row_count * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(rows.row_count, width, channels)


def chunk_causal_mask(
    length: int, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """(length, length) bool: a frame attends every frame of its chunk and of
    the chunks before it, CosyVoice's subsequent_chunk_mask."""
    position = torch.arange(length, device=device)
    chunk_end = (position // chunk_size + 1) * chunk_size
    return position.unsqueeze(0) < chunk_end.unsqueeze(1)


def chunk_segments(
    lengths: Sequence[int], chunk_size: int | None
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """One query segment per (row, chunk), reading that row's frames
    [0, chunk end). Without a chunk size a row is one segment."""
    segment_rows: list[int] = []
    segment_ends: list[int] = []
    offsets: list[int] = [0]
    for row, length in enumerate(lengths):
        span = length if chunk_size is None else chunk_size
        frame = 0
        while frame < length:
            end = min((frame // span + 1) * span, length)
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
    """Row attention on the packed sequence via FA3 paged KV, no pad-to-widest."""

    def __init__(
        self,
        rows: PackedRows,
        *,
        chunk_size: int | None,
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


PackedRowAttention = RowAttention | RaggedRowAttention


class PackedDiT:
    """DiT.forward over a packed sequence with the same modules in the same
    order; conv pos-emb stays padded, attention is ragged on FA3 half-precision
    CUDA and padded elsewhere.
    """

    def __init__(self, dit: torch.nn.Module, *, device: str | torch.device) -> None:
        self.dit = dit
        device = torch.device(device)
        self.is_ragged = device.type == "cuda" and _is_fa3_supported()
        self.is_compiled = False
        blocks = len(self.dit.transformer_blocks)
        self.qkv_weights: tuple[torch.Tensor | None, ...] = (None,) * blocks
        self.qkv_biases: tuple[torch.Tensor | None, ...] = (None,) * blocks
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
        chunk_size = self.chunk_size if streaming else None
        if self.is_ragged and dtype in FA3_DTYPES:
            attention = RaggedRowAttention(
                rows,
                chunk_size=chunk_size,
                heads=attention.heads,
                head_dim=attention.inner_dim // attention.heads,
            )
            if self.is_compiled:
                # note(ratish): hints, not constraints; they share x's total frames,
                # which the first call specializes, so mark_dynamic would fail.
                dynamo.maybe_mark_dynamic(attention.page_table, (0, 1))
                dynamo.maybe_mark_dynamic(attention.cu_seqlens_q, 0)
                dynamo.maybe_mark_dynamic(attention.cache_seqlens, 0)
                dynamo.maybe_mark_dynamic(rows.row_ids, 0)
                dynamo.maybe_mark_dynamic(rows.positions, 0)
            else:
                pass
            return attention
        else:
            pass
        return RowAttention(rows, chunk_size=chunk_size, heads=attention.heads)

    def compile(self, dtype: torch.dtype | None) -> bool:
        if not self.is_ragged or dtype not in FA3_DTYPES:
            logger.debug(
                f"Skipping PackedDiT torch.compile (ragged={self.is_ragged}, dtype={dtype})"
            )
            return False
        else:
            pass

        fused_weights: list[torch.Tensor] = []
        fused_biases: list[torch.Tensor | None] = []
        fused_bytes = 0
        with torch.no_grad():
            for block in self.dit.transformer_blocks:
                attn = block.attn

                qkv_weight = torch.cat(
                    (
                        attn.to_q.weight,
                        attn.to_k.weight,
                        attn.to_v.weight,
                    ),
                    dim=0,
                )
                if attn.to_q.bias is None:
                    qkv_bias = None
                else:
                    assert attn.to_k.bias is not None
                    assert attn.to_v.bias is not None
                    qkv_bias = torch.cat(
                        (
                            attn.to_q.bias,
                            attn.to_k.bias,
                            attn.to_v.bias,
                        ),
                        dim=0,
                    )

                fused_weights.append(qkv_weight)
                fused_biases.append(qkv_bias)
                fused_bytes += qkv_weight.nbytes
                if qkv_bias is not None:
                    fused_bytes += qkv_bias.nbytes
                else:
                    pass

        self.qkv_weights = tuple(fused_weights)
        self.qkv_biases = tuple(fused_biases)
        logger.info(
            "Materialized PackedDiT fused QKV weights+biases for %d blocks "
            "(%.1f MiB)",
            len(self.qkv_weights),
            fused_bytes / (1024 * 1024),
        )

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
        module = self.dit.input_embed.conv_pos_embed
        x = scatter_rows(h, rows, rows.width).permute(0, 2, 1)
        x = mish(module.conv1[0](F.pad(x, (module.kernel_size - 1, 0, 0, 0))))
        x = mish(module.conv2[0](F.pad(x, (module.kernel_size - 1, 0, 0, 0))))
        return gather_rows(x.permute(0, 2, 1), rows)

    def rope(self, rows: PackedRows) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin, (1, total, rotary dims) each, in float32."""
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(rows.width)
        assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
        freqs = freqs[:, rows.positions]
        return freqs.cos(), freqs.sin()

    @staticmethod
    def attend(
        attn: torch.nn.Module,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention: PackedRowAttention,
        qkv_weight: torch.Tensor | None,
        qkv_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if qkv_weight is None:
            # note (ratish): under autocast to_q, to_k and to_v would each cast the
            # float32 norm output again.
            x = x.to(attn.to_q.weight.dtype)
            query = attn.to_q(x)
            key = attn.to_k(x)
            value = attn.to_v(x)
        else:
            # note(chenye): one fused projection removes two GEMM launches per
            # transformer block while leaving the model's original Q/K/V modules
            # intact.
            x = x.to(qkv_weight.dtype)
            query, key, value = F.linear(x, qkv_weight, qkv_bias).chunk(3, dim=-1)
        if torch.compiler.is_compiling():
            query = rotated(query, *rope)
            key = rotated(key, *rope)
        else:
            rotate_in_place(query, *rope)
            rotate_in_place(key, *rope)
        out = attention(query, key, value).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](out))


def mish(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return native_mish(x)
    else:
        return F.mish(x)


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
    rows: PackedRows,
    *,
    cfg_rate: float,
    streaming: bool,
) -> torch.Tensor:
    """Euler steps over a packed sequence with classifier free guidance: the
    conditional rows and their unconditional twins share one DiT call."""
    total = noise.shape[1]
    twin_rows = pack_rows(rows.lengths * 2, noise.device)
    attention = estimator.row_attention(
        twin_rows, streaming=streaming, dtype=spks.dtype
    )
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    spks_cfg = spks_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=noise.device, dtype=spks.dtype)
    # note(ratish): once per solve and outside the compiled forward,
    # whose graph would otherwise hold RoPE's autocast region and miss the AOT cache.
    rope = estimator.rope(twin_rows)
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
