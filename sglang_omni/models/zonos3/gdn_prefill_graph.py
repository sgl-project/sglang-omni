# SPDX-License-Identifier: Apache-2.0
"""Gated delta net prefill on per-bucket static tables, so a breakable prefill CUDA graph captures it instead of breaking at every layer.

The eager prefill derives its launch grids from host-side sequence lengths. Here each captured token bucket owns tables sized by its bounds (sequences, delta-rule chunks, convolution blocks); they are refilled before every replay, and padded rows point past every sequence or at the padding state slot, so their programs exit without touching any state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_kernel_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    CAUSAL_CONV1D_FWD_BLOCK_M,
    PAD_SLOT_ID,
    causal_conv1d_fn,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import is_cuda

# Chunk index of padded chunk-table rows: past the end of any sequence, so every varlen chunk kernel masks the row out.
PADDED_CHUNK_INDEX = 1 << 20
# note (zhaochenyang20): the tiles chunk_fwd_o launches with; the same tiles keep the output bit-identical to the eager prefill.
OUTPUT_KEY_BLOCK = 128
OUTPUT_VALUE_BLOCK = 64
OUTPUT_WARP_COUNT = 4
OUTPUT_STAGE_COUNT = 2


def padded_prefix_sums(counts: np.ndarray, length: int) -> np.ndarray:
    """[length] prefix sums of counts starting at 0, the total repeated past the counts."""
    sums = np.full(length, counts.sum(), dtype=np.int64)
    sums[0] = 0
    np.cumsum(counts, out=sums[1 : counts.shape[0] + 1])
    return sums


def sequence_pair_table(
    counts: np.ndarray, row_count: int, padding_row: tuple[int, int]
) -> np.ndarray:
    """[row_count, 2] sequence-major (sequence, index) rows, counts[sequence] per sequence, then padding_row."""
    table = np.empty((row_count, 2), dtype=np.int64)
    table[:] = padding_row
    pair_count = int(counts.sum())
    table[:pair_count, 0] = np.repeat(np.arange(counts.shape[0]), counts)
    table[:pair_count, 1] = np.arange(pair_count) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return table


@dataclass(frozen=True, kw_only=True)
class GDNPrefillGraphTables:
    """Static tables one captured token bucket reads; fill rewrites them for each replayed batch."""

    token_count: int
    max_sequence_count: int
    # One int32 device buffer, refreshed by a single host copy, and its views.
    packed_tables: torch.Tensor
    cumulative_sequence_lengths: torch.Tensor
    # [rows, 2] (sequence, chunk); the chunk kernels read per-chunk states at chunk_offsets[sequence] + chunk, so rows are sequence-major.
    chunk_table: torch.Tensor
    sequence_chunk_offsets: torch.Tensor
    # [rows, 2] (sequence, block) of the convolution's CAUSAL_CONV1D_FWD_BLOCK_M-token blocks.
    convolution_block_table: torch.Tensor
    state_slots: torch.Tensor
    has_initial_state: torch.Tensor

    @classmethod
    def allocate(
        cls, *, token_count: int, max_sequence_count: int, device: torch.device
    ) -> GDNPrefillGraphTables:
        # Each sequence adds at most one partial chunk and one partial block.
        chunk_row_count = triton.cdiv(token_count, CHUNK_SIZE) + max_sequence_count
        block_row_count = (
            triton.cdiv(token_count, CAUSAL_CONV1D_FWD_BLOCK_M) + max_sequence_count
        )
        sizes = [
            max_sequence_count + 1,
            2 * chunk_row_count,
            max_sequence_count + 1,
            2 * block_row_count,
        ]
        packed_tables = torch.zeros(sum(sizes), dtype=torch.int32, device=device)
        views = torch.split(packed_tables, sizes)
        return cls(
            token_count=token_count,
            max_sequence_count=max_sequence_count,
            packed_tables=packed_tables,
            cumulative_sequence_lengths=views[0],
            chunk_table=views[1].view(chunk_row_count, 2),
            sequence_chunk_offsets=views[2],
            convolution_block_table=views[3].view(block_row_count, 2),
            state_slots=torch.full(
                (max_sequence_count,), PAD_SLOT_ID, dtype=torch.int32, device=device
            ),
            has_initial_state=torch.zeros(
                max_sequence_count, dtype=torch.bool, device=device
            ),
        )

    def fill(
        self,
        *,
        extend_sequence_lengths: list[int],
        state_slots: torch.Tensor,
        extend_prefix_lengths: torch.Tensor,
    ) -> None:
        """Rewrite the tables for one batch; state_slots and extend_prefix_lengths are its device tensors."""
        sequence_count = len(extend_sequence_lengths)
        lengths = np.asarray(extend_sequence_lengths, dtype=np.int64)
        assert (
            sequence_count <= self.max_sequence_count
            and lengths.sum() <= self.token_count
        ), (
            f"{sequence_count} sequences of {lengths.sum()} tokens exceed the bucket "
            f"bounds ({self.max_sequence_count} sequences, {self.token_count} tokens)"
        )
        chunk_counts = -(-lengths // CHUNK_SIZE)
        block_counts = -(-lengths // CAUSAL_CONV1D_FWD_BLOCK_M)
        host_tables = np.concatenate(
            [
                padded_prefix_sums(lengths, self.max_sequence_count + 1),
                sequence_pair_table(
                    chunk_counts, self.chunk_table.shape[0], (0, PADDED_CHUNK_INDEX)
                ).ravel(),
                padded_prefix_sums(chunk_counts, self.max_sequence_count + 1),
                sequence_pair_table(
                    block_counts,
                    self.convolution_block_table.shape[0],
                    (PAD_SLOT_ID, 0),
                ).ravel(),
            ]
        ).astype(np.int32)
        # A pinned block from the caching host allocator keeps the copy asynchronous and is not reused before it completes.
        self.packed_tables.copy_(
            torch.from_numpy(host_tables).pin_memory(), non_blocking=True
        )
        self.state_slots[:sequence_count].copy_(state_slots[:sequence_count])
        self.state_slots[sequence_count:].fill_(PAD_SLOT_ID)
        torch.gt(
            extend_prefix_lengths[:sequence_count],
            0,
            out=self.has_initial_state[:sequence_count],
        )
        self.has_initial_state[sequence_count:].fill_(False)


def gdn_extend_in_graph(
    *,
    layer: RadixLinearAttention,
    tables: GDNPrefillGraphTables,
    convolution_states: torch.Tensor,
    recurrent_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """The eager prefill of one layer with grids fixed by the bucket; [1, tokens, heads, value] with zero rows past the batch."""
    convolved_qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        conv_states=convolution_states,
        has_initial_state=tables.has_initial_state,
        cache_indices=tables.state_slots,
        query_start_loc=tables.cumulative_sequence_lengths,
        seq_lens_cpu=None,
        block_table=tables.convolution_block_table,
    ).transpose(0, 1)
    q, k, v = fused_qkv_split_gdn_prefill(
        convolved_qkv,
        layer.num_q_heads,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_q_dim,
        layer.head_k_dim,
        layer.head_v_dim,
    )
    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    g = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=tables.cumulative_sequence_lengths,
        chunk_indices=tables.chunk_table,
    )
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=tables.cumulative_sequence_lengths,
        chunk_indices=tables.chunk_table,
    )
    chunk_states, new_values = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=recurrent_states,
        initial_state_indices=tables.state_slots,
        cu_seqlens=tables.cumulative_sequence_lengths,
        chunk_indices=tables.chunk_table,
        chunk_offsets=tables.sequence_chunk_offsets,
    )
    _, token_count, head_count, value_dimension = new_values.shape
    key_head_count, key_dimension = k.shape[-2:]
    output = torch.zeros_like(new_values)
    chunk_fwd_kernel_o[
        (
            triton.cdiv(value_dimension, OUTPUT_VALUE_BLOCK),
            tables.chunk_table.shape[0],
            head_count,
        )
    ](
        q,
        k,
        new_values,
        chunk_states,
        g,
        output,
        tables.cumulative_sequence_lengths,
        tables.chunk_table,
        key_dimension**-0.5,
        T=token_count,
        H=head_count,
        Hg=key_head_count,
        K=key_dimension,
        V=value_dimension,
        BT=CHUNK_SIZE,
        BK=OUTPUT_KEY_BLOCK,
        BV=OUTPUT_VALUE_BLOCK,
        USE_G=True,
        IS_VARLEN=True,
        num_warps=OUTPUT_WARP_COUNT,
        num_stages=OUTPUT_STAGE_COUNT,
    )
    return output


class Zonos3GDNAttnBackend(GDNAttnBackend):
    """Runs the extend of a captured breakable prefill bucket on that bucket's static tables."""

    # Set while a breakable prefill CUDA graph captures or replays a bucket.
    prefill_graph_tables: GDNPrefillGraphTables | None = None

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        # A prefill batch holds at most one sequence per running request, so every batch fits its bucket's tables.
        self.prefill_graph_max_seqs = get_schedule().max_running_requests

    def supports_prefill_graph_extend(self) -> bool:
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        # The chunk kernels write the final states back assuming contiguous slots.
        return (
            is_cuda()
            and isinstance(self.kernel_dispatcher.extend_kernel, TritonGDNKernel)
            and mamba_cache.temporal.is_contiguous()
            and all(convolution.is_contiguous() for convolution in mamba_cache.conv)
        )

    def prefill_graph_extend_active(self) -> bool:
        return self.prefill_graph_tables is not None

    def init_prefill_graph_metadata(
        self, forward_batch: ForwardBatch
    ) -> GDNPrefillGraphTables:
        token_count = forward_batch.positions.shape[0]
        tables = GDNPrefillGraphTables.allocate(
            token_count=token_count,
            max_sequence_count=min(token_count, self.prefill_graph_max_seqs),
            device=self.device,
        )
        self.refresh_prefill_graph_metadata(tables, forward_batch)
        return tables

    def refresh_prefill_graph_metadata(
        self, tables: GDNPrefillGraphTables, forward_batch: ForwardBatch
    ) -> None:
        # Zonos3 runs without the radix cache, so no row snapshots its state for a prefix.
        assert forward_batch.mamba_track_mask is None
        tables.fill(
            extend_sequence_lengths=forward_batch.extend_seq_lens_cpu,
            state_slots=self.req_to_token_pool.translate_mamba_indices(
                self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
            ),
            extend_prefix_lengths=forward_batch.extend_prefix_lens,
        )
        self.prefill_graph_tables = tables

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        # An eager forward never runs on a bucket's tables.
        self.prefill_graph_tables = None
        super().init_forward_metadata(forward_batch)

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: torch.Tensor | bool | int | None,
    ) -> torch.Tensor:
        """kwargs is the hybrid backend's pass-through for the eager extend."""
        tables = self.prefill_graph_tables
        if tables is None:
            return super().forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        else:
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            return gdn_extend_in_graph(
                layer=layer,
                tables=tables,
                convolution_states=layer_cache.conv[0],
                recurrent_states=layer_cache.temporal,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )


def install_gdn_prefill_graph() -> None:
    """Make SGLang build Zonos3GDNAttnBackend, the backend it looks up by this name, for the gated delta net layers."""
    gdn_backend.GDNAttnBackend = Zonos3GDNAttnBackend
