# SPDX-License-Identifier: Apache-2.0
"""GDN extend captured inside breakable prefill CUDA graphs.

The eager Triton chain runs on fixed grids over per-bucket tables, which
GDNPrefillGraphMetadata.fill rebuilds before every replay without a host sync.
A batch of more than max_sequences sequences runs eagerly through the runner's
replay vote on prefill_cuda_graph_max_batch_size. Padding falls outside every
kernel's bounds, so a real row's output and state keep the eager bits.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import triton
from sglang.kernels.ops.attention.fla.chunk import CHUNK_SIZE
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
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
from sglang.srt.layers.attention.linear.gdn_backend import (
    MAX_FUSED_QKV_SPLIT_DIM,
    GDNAttnBackend,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_exec

__all__ = [
    "GDNPrefillGraphMetadata",
    "GDNPrefillGraphAttnBackend",
    "gdn_extend_in_graph",
    "install_gdn_prefill_graph",
]

# note (Wenyao Gao): past the end of any sequence, so every chunk kernel's block
# pointers for a padded chunk_indices row fall outside their boundary checks.
PAD_CHUNK = 1 << 20


@dataclasses.dataclass(kw_only=True)
class GDNPrefillGraphMetadata:
    """Static per-bucket tables read by the captured GDN extend."""

    num_tokens: int
    max_sequences: int
    max_chunks: int
    max_conv_blocks: int
    # note (Wenyao Gao): one int32 device buffer, so a single copy refreshes the
    # four tables viewed from it.
    packed: torch.Tensor
    cumulative_sequence_lengths: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    conv_block_table: torch.Tensor
    host: np.ndarray
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        max_sequences_cap: int,
        cache_index_dtype: torch.dtype,
        device: torch.device,
    ) -> GDNPrefillGraphMetadata:
        max_sequences = max(1, min(num_tokens, max_sequences_cap))
        max_chunks = triton.cdiv(num_tokens, CHUNK_SIZE) + max_sequences
        max_conv_blocks = (
            triton.cdiv(num_tokens, CAUSAL_CONV1D_FWD_BLOCK_M) + max_sequences
        )
        sizes = [
            max_sequences + 1,
            2 * max_chunks,
            max_sequences + 1,
            2 * max_conv_blocks,
        ]
        packed = torch.zeros(sum(sizes), dtype=torch.int32, device=device)
        views = torch.split(packed, sizes)
        return cls(
            num_tokens=num_tokens,
            max_sequences=max_sequences,
            max_chunks=max_chunks,
            max_conv_blocks=max_conv_blocks,
            packed=packed,
            cumulative_sequence_lengths=views[0],
            chunk_indices=views[1].view(max_chunks, 2),
            chunk_offsets=views[2],
            conv_block_table=views[3].view(max_conv_blocks, 2),
            host=np.zeros(sum(sizes), dtype=np.int32),
            cache_indices=torch.full(
                (max_sequences,), PAD_SLOT_ID, dtype=cache_index_dtype, device=device
            ),
            has_initial_state=torch.zeros(
                max_sequences, dtype=torch.bool, device=device
            ),
        )

    def fill(
        self,
        *,
        sequence_lengths: list[int],
        cache_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> None:
        assert (
            len(sequence_lengths) <= self.max_sequences
            and sum(sequence_lengths) <= self.num_tokens
        ), (
            f"a batch of {len(sequence_lengths)} sequences and {sum(sequence_lengths)} "
            f"tokens exceeds the captured bounds of {self.max_sequences} sequences and "
            f"{self.num_tokens} tokens"
        )
        batch_size = len(sequence_lengths)
        self.write_host_tables(np.asarray(sequence_lengths, dtype=np.int64))
        # note (Wenyao Gao): a pinned block from the caching host allocator keeps
        # the copy asynchronous and is not reused before the copy completes.
        self.packed.copy_(torch.from_numpy(self.host).pin_memory(), non_blocking=True)
        self.cache_indices[:batch_size].copy_(cache_indices[:batch_size])
        self.cache_indices[batch_size:].fill_(PAD_SLOT_ID)
        torch.gt(
            prefix_lengths[:batch_size], 0, out=self.has_initial_state[:batch_size]
        )
        self.has_initial_state[batch_size:].fill_(False)

    def write_host_tables(self, sequence_lengths: np.ndarray) -> None:
        batch_size = sequence_lengths.shape[0]
        token_count = int(sequence_lengths.sum())
        cumulative_sequence_lengths = np.empty(self.max_sequences + 1, dtype=np.int64)
        cumulative_sequence_lengths[0] = 0
        np.cumsum(sequence_lengths, out=cumulative_sequence_lengths[1 : batch_size + 1])
        cumulative_sequence_lengths[batch_size + 1 :] = token_count
        offset = self.write_values(0, cumulative_sequence_lengths)
        # note (Wenyao Gao): sequence-major (sequence, chunk) rows, because
        # chunk_fwd_kernel_o reads the per-chunk states by row and
        # chunk_gated_delta_rule_fwd_h writes them at chunk_offsets[sequence] + chunk.
        chunk_counts = (sequence_lengths + CHUNK_SIZE - 1) // CHUNK_SIZE
        offset = self.write_pairs(
            offset, chunk_counts, self.max_chunks, pad=(0, PAD_CHUNK)
        )
        chunk_offsets = np.empty(self.max_sequences + 1, dtype=np.int64)
        chunk_offsets[0] = 0
        np.cumsum(chunk_counts, out=chunk_offsets[1 : batch_size + 1])
        chunk_offsets[batch_size + 1 :] = chunk_offsets[batch_size]
        offset = self.write_values(offset, chunk_offsets)
        block_counts = (
            sequence_lengths + CAUSAL_CONV1D_FWD_BLOCK_M - 1
        ) // CAUSAL_CONV1D_FWD_BLOCK_M
        self.write_pairs(
            offset, block_counts, self.max_conv_blocks, pad=(PAD_SLOT_ID, 0)
        )

    def write_values(self, offset: int, values: np.ndarray) -> int:
        self.host[offset : offset + values.shape[0]] = values
        return offset + values.shape[0]

    def write_pairs(
        self, offset: int, counts: np.ndarray, row_count: int, pad: tuple[int, int]
    ) -> int:
        """(sequence, index) rows, counts[sequence] of them per sequence, then the pad."""
        filled_row_count = int(counts.sum())
        table = self.host[offset : offset + 2 * row_count].reshape(row_count, 2)
        if filled_row_count:
            table[:filled_row_count, 0] = np.repeat(np.arange(counts.shape[0]), counts)
            table[:filled_row_count, 1] = np.arange(filled_row_count) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
        else:
            pass
        table[filled_row_count:, 0] = pad[0]
        table[filled_row_count:, 1] = pad[1]
        return offset + 2 * row_count


def chunk_fwd_o_on_tables(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_states: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    metadata: GDNPrefillGraphMetadata,
) -> torch.Tensor:
    """chunk_fwd_o on the bucket's chunk table instead of one prepared on the host."""
    batch_size, token_count, key_head_count, key_head_dim, value_head_dim = (
        *query.shape,
        value.shape[-1],
    )
    value_head_count = value.shape[-2]
    chunk_block_size = min(CHUNK_SIZE, max(16, triton.next_power_of_2(token_count)))
    output = torch.zeros_like(value)
    chunk_fwd_kernel_o[
        (
            triton.cdiv(value_head_dim, 64),
            metadata.max_chunks,
            batch_size * value_head_count,
        )
    ](
        query,
        key,
        value,
        chunk_states,
        g,
        output,
        metadata.cumulative_sequence_lengths,
        metadata.chunk_indices,
        scale,
        T=token_count,
        H=value_head_count,
        Hg=key_head_count,
        K=key_head_dim,
        V=value_head_dim,
        BT=chunk_block_size,
        BK=128,
        BV=64,
        USE_G=True,
        IS_VARLEN=True,
        num_warps=4,
        num_stages=2,
    )
    return output


def chunk_gated_delta_rule_on_tables(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    metadata: GDNPrefillGraphMetadata,
) -> torch.Tensor:
    """chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True, inplace_update=True) on
    the bucket's tables: the same kernels and configs on fixed grids, with no
    host-side chunk preparation."""
    scale = key.shape[-1] ** -0.5
    query, key, value, g, beta = (
        tensor.contiguous() for tensor in (query, key, value, g, beta)
    )
    query = l2norm_fwd(query)
    key = l2norm_fwd(key)
    g = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=metadata.cumulative_sequence_lengths,
        chunk_indices=metadata.chunk_indices,
    )
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=key,
        v=value,
        g=g,
        beta=beta,
        cu_seqlens=metadata.cumulative_sequence_lengths,
        chunk_indices=metadata.chunk_indices,
    )
    chunk_states, new_value = chunk_gated_delta_rule_fwd_h(
        k=key,
        w=w,
        u=u,
        g=g,
        initial_state=ssm_states,
        initial_state_indices=metadata.cache_indices,
        cu_seqlens=metadata.cumulative_sequence_lengths,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
        inplace_update=True,
    )
    return chunk_fwd_o_on_tables(
        query, key, new_value, chunk_states, g, scale, metadata
    )


def gdn_extend_in_graph(
    *,
    layer: RadixLinearAttention,
    metadata: GDNPrefillGraphMetadata,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """GDNAttnBackend.forward_extend of a plain extend on grids that depend only on
    the bucket. Updates the batch's slots of conv_states and ssm_states in place and
    returns [1, num_tokens, heads, V], zero past the batch's tokens."""
    convolved = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        conv_states=conv_states,
        has_initial_state=metadata.has_initial_state,
        cache_indices=metadata.cache_indices,
        query_start_loc=metadata.cumulative_sequence_lengths,
        seq_lens_cpu=None,
        block_table=metadata.conv_block_table,
    ).transpose(0, 1)
    if layer.q_dim + layer.k_dim + layer.v_dim <= MAX_FUSED_QKV_SPLIT_DIM:
        query, key, value = fused_qkv_split_gdn_prefill(
            convolved,
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_k_dim,
            layer.head_v_dim,
        )
    else:
        row_count = convolved.shape[0]
        query, key, value = torch.split(
            convolved, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        query = query.view(1, row_count, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, row_count, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, row_count, layer.num_v_heads, layer.head_v_dim)
    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
    return chunk_gated_delta_rule_on_tables(
        query, key, value, g, beta, ssm_states, metadata
    )


MAX_SEQUENCES_BY_ARCHITECTURE: dict[str, int] = {}


class GDNPrefillGraphAttnBackend(GDNAttnBackend):
    """GDNAttnBackend whose plain extend the breakable prefill graph captures on
    static per-bucket tables, for the architectures that opted in; the stock
    backend otherwise."""

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        self.prefill_graph_metadata: GDNPrefillGraphMetadata | None = None
        architectures = model_runner.model_config.hf_config.architectures
        self.prefill_graph_max_seqs: int = max(
            (
                MAX_SEQUENCES_BY_ARCHITECTURE.get(name, 0)
                for name in architectures or []
            ),
            default=0,
        )

    def prefill_graph_unsupported_reason(self) -> str | None:
        extend_kernel = self.kernel_dispatcher.extend_kernel
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        if self.prefill_graph_max_seqs <= 0:
            return "the model did not opt in"
        elif not isinstance(extend_kernel, TritonGDNKernel):
            return f"the {type(extend_kernel).__name__} extend kernel is not the Triton chain"
        elif self.enable_mis:
            return "multi-item scoring is on"
        elif get_exec().mamba.enable_mamba_extra_buffer:
            return "the prefix-cache track buffer is on"
        elif not (
            mamba_cache.conv[0].is_contiguous() and mamba_cache.temporal.is_contiguous()
        ):
            return "the state pool is strided"
        else:
            return None

    def supports_prefill_graph_extend(self) -> bool:
        return self.prefill_graph_unsupported_reason() is None

    def prefill_graph_extend_active(self) -> bool:
        return self.prefill_graph_metadata is not None

    def init_prefill_graph_metadata(
        self, forward_batch: ForwardBatch
    ) -> GDNPrefillGraphMetadata:
        cache_indices = self.prefill_graph_cache_indices(forward_batch)
        metadata = GDNPrefillGraphMetadata.allocate(
            num_tokens=forward_batch.extend_num_tokens,
            max_sequences_cap=self.prefill_graph_max_seqs,
            cache_index_dtype=cache_indices.dtype,
            device=self.device,
        )
        self.fill_prefill_graph_metadata(metadata, forward_batch, cache_indices)
        return metadata

    def refresh_prefill_graph_metadata(
        self, metadata: GDNPrefillGraphMetadata, forward_batch: ForwardBatch
    ) -> None:
        self.fill_prefill_graph_metadata(
            metadata, forward_batch, self.prefill_graph_cache_indices(forward_batch)
        )

    def prefill_graph_cache_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        # note (Wenyao Gao): the slots MambaAttnBackendBase._forward_metadata
        # resolves for the same batch in an eager forward.
        cache_indices = self._translate_mamba_indices(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
        )
        real_batch_size = (
            forward_batch._original_batch_size
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        if real_batch_size is not None and real_batch_size < cache_indices.shape[0]:
            cache_indices = cache_indices.clone()
            cache_indices[real_batch_size:] = PAD_SLOT_ID
        else:
            pass
        return cache_indices

    def fill_prefill_graph_metadata(
        self,
        metadata: GDNPrefillGraphMetadata,
        forward_batch: ForwardBatch,
        cache_indices: torch.Tensor,
    ) -> None:
        metadata.fill(
            sequence_lengths=list(forward_batch.extend_seq_lens_cpu),
            cache_indices=cache_indices,
            prefix_lengths=forward_batch.extend_prefix_lens,
        )
        self.prefill_graph_metadata = metadata

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        # note (Wenyao Gao): an eager forward never runs on a graph bucket's tables.
        self.prefill_graph_metadata = None
        super().init_forward_metadata(forward_batch)

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor | tuple[torch.Tensor, ...],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        metadata = self.prefill_graph_metadata
        if metadata is None or forward_batch.forward_mode.is_target_verify():
            return super().forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        else:
            assert isinstance(mixed_qkv, torch.Tensor)
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            return gdn_extend_in_graph(
                layer=layer,
                metadata=metadata,
                conv_states=layer_cache.conv[0],
                ssm_states=layer_cache.temporal,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )


def install_gdn_prefill_graph(architecture: str, *, max_sequences: int) -> None:
    """Capture the GDN extend in this architecture's breakable prefill graphs for
    batches of up to max_sequences requests; 0 keeps the eager breaks. Runners built
    afterwards get the subclass, which acts as the stock backend elsewhere."""
    MAX_SEQUENCES_BY_ARCHITECTURE[architecture] = max_sequences
    gdn_backend.GDNAttnBackend = GDNPrefillGraphAttnBackend
