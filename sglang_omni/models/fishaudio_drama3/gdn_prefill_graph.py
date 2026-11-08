# SPDX-License-Identifier: Apache-2.0
"""GDN extend captured inside breakable prefill CUDA graphs.

The eager GDN extend sizes its launch grids and chunk tables from host-side
sequence lengths (prepare_chunk_indices, causal_conv1d_fn's
seq_lens_cpu), so a breakable prefill graph breaks at every linear-attention
layer and launches the whole chain, about a dozen kernels per layer plus their
Python, eagerly. sglang captures the KDA extend on static per-bucket tables
through the runner's captured-metadata contract
(supports_prefill_graph_extend / init_prefill_graph_metadata /
refresh_prefill_graph_metadata, see
sglang/srt/layers/attention/linear/kda_prefill_graph.py); this module gives
the Triton GDN chain the same treatment for the architectures that opt in
through install_gdn_prefill_graph.

Per bucket of num_tokens: max_seqs = min(num_tokens, cap) sequences
(larger batches run eagerly through the runner's replay vote on
prefill_cuda_graph_max_batch_size), cdiv(num_tokens, 64) + max_seqs
delta-rule chunks and cdiv(num_tokens, 8) + max_seqs conv blocks. Before
every replay GDNPrefillGraphMetadata.fill rebuilds the tables from the
batch's CPU lengths: one pinned host-to-device copy and two small device
copies, no synchronization. The captured kernels launch fixed grids over these
bounds. Padded sequences are empty and carry the PAD_SLOT_ID state slot the
conv and state kernels skip; padded chunk rows point past any sequence, so the
chunk kernels mask them out. The kernels, configs and per-sequence arithmetic
are those of the eager chain, so a real row's output and state are bitwise the
eager ones.
"""

from __future__ import annotations

import dataclasses
from threading import Lock

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

# Chunk index of padded chunk_indices rows: past the end of any sequence, so every
# chunk kernel's block pointers for them fall outside their boundary checks.
PAD_CHUNK = 1 << 20


@dataclasses.dataclass(kw_only=True)
class GDNPrefillGraphMetadata:
    """Static per-bucket tables read by the captured GDN extend."""

    num_tokens: int
    max_seqs: int
    max_chunks: int
    max_conv_blocks: int
    # One int32 device buffer, refreshed by a single copy, and its views.
    packed: torch.Tensor
    cu_seqlens: torch.Tensor  # [max_seqs + 1]
    chunk_indices: torch.Tensor  # [max_chunks, 2] (seq, chunk)
    chunk_offsets: torch.Tensor  # [max_seqs + 1]
    conv_block_table: torch.Tensor  # [max_conv_blocks, 2] (seq, block)
    host: np.ndarray  # int32 staging for packed
    cache_indices: torch.Tensor  # [max_seqs], PAD_SLOT_ID past the batch
    has_initial_state: torch.Tensor  # [max_seqs] bool

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        max_seqs_cap: int,
        cache_index_dtype: torch.dtype,
        device: torch.device,
    ) -> GDNPrefillGraphMetadata:
        max_seqs = max(1, min(num_tokens, max_seqs_cap))
        max_chunks = triton.cdiv(num_tokens, CHUNK_SIZE) + max_seqs
        max_conv_blocks = triton.cdiv(num_tokens, CAUSAL_CONV1D_FWD_BLOCK_M) + max_seqs
        sizes = [max_seqs + 1, 2 * max_chunks, max_seqs + 1, 2 * max_conv_blocks]
        packed = torch.zeros(sum(sizes), dtype=torch.int32, device=device)
        views = torch.split(packed, sizes)
        return cls(
            num_tokens=num_tokens,
            max_seqs=max_seqs,
            max_chunks=max_chunks,
            max_conv_blocks=max_conv_blocks,
            packed=packed,
            cu_seqlens=views[0],
            chunk_indices=views[1].view(max_chunks, 2),
            chunk_offsets=views[2],
            conv_block_table=views[3].view(max_conv_blocks, 2),
            host=np.zeros(sum(sizes), dtype=np.int32),
            cache_indices=torch.full(
                (max_seqs,), PAD_SLOT_ID, dtype=cache_index_dtype, device=device
            ),
            has_initial_state=torch.zeros(max_seqs, dtype=torch.bool, device=device),
        )

    def can_fit(self, extend_seq_lens: list[int]) -> bool:
        return (
            len(extend_seq_lens) <= self.max_seqs
            and sum(extend_seq_lens) <= self.num_tokens
        )

    def fill(
        self,
        *,
        extend_seq_lens: list[int],
        cache_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        """Rebuild the tables for this batch; cache_indices and
        extend_prefix_lens are the batch's device tensors."""
        bs = len(extend_seq_lens)
        if not self.can_fit(extend_seq_lens):
            raise RuntimeError(
                f"a batch of {bs} sequences / {sum(extend_seq_lens)} tokens exceeds "
                f"the captured bounds ({self.max_seqs} / {self.num_tokens})"
            )
        else:
            pass
        self.write_host_tables(np.asarray(extend_seq_lens, dtype=np.int64))
        # A pinned block from the caching host allocator keeps the copy
        # asynchronous and is not reused before the copy completes.
        self.packed.copy_(torch.from_numpy(self.host).pin_memory(), non_blocking=True)
        self.cache_indices[:bs].copy_(cache_indices[:bs])
        self.cache_indices[bs:].fill_(PAD_SLOT_ID)
        torch.gt(extend_prefix_lens[:bs], 0, out=self.has_initial_state[:bs])
        self.has_initial_state[bs:].fill_(False)

    def write_host_tables(self, lens: np.ndarray) -> None:
        bs = lens.shape[0]
        total = int(lens.sum())
        # Padded sequences are empty and sit at the end.
        cu = np.empty(self.max_seqs + 1, dtype=np.int64)
        cu[0] = 0
        np.cumsum(lens, out=cu[1 : bs + 1])
        cu[bs + 1 :] = total
        offset = self.write_values(0, cu)
        # Sequence-major (seq, chunk) rows: chunk_fwd_kernel_o reads the per-chunk
        # states by row and chunk_gated_delta_rule_fwd_h writes them at
        # chunk_offsets[seq] + chunk.
        chunk_counts = (lens + CHUNK_SIZE - 1) // CHUNK_SIZE
        offset = self.write_pairs(
            offset, chunk_counts, self.max_chunks, pad=(0, PAD_CHUNK)
        )
        offsets = np.empty(self.max_seqs + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(chunk_counts, out=offsets[1 : bs + 1])
        offsets[bs + 1 :] = offsets[bs]
        offset = self.write_values(offset, offsets)
        block_counts = (
            lens + CAUSAL_CONV1D_FWD_BLOCK_M - 1
        ) // CAUSAL_CONV1D_FWD_BLOCK_M
        offset = self.write_pairs(
            offset, block_counts, self.max_conv_blocks, pad=(PAD_SLOT_ID, 0)
        )
        assert offset == self.host.shape[0]

    def write_values(self, offset: int, values: np.ndarray) -> int:
        self.host[offset : offset + values.shape[0]] = values
        return offset + values.shape[0]

    def write_pairs(
        self, offset: int, counts: np.ndarray, rows: int, pad: tuple[int, int]
    ) -> int:
        """(seq, index) rows, counts[seq] of them per sequence, then the pad."""
        n = int(counts.sum())
        table = self.host[offset : offset + 2 * rows].reshape(rows, 2)
        if n:
            table[:n, 0] = np.repeat(np.arange(counts.shape[0]), counts)
            table[:n, 1] = np.arange(n) - np.repeat(np.cumsum(counts) - counts, counts)
        else:
            pass
        table[n:, 0] = pad[0]
        table[n:, 1] = pad[1]
        return offset + 2 * rows


def split_qkv(
    layer: RadixLinearAttention, mixed_qkv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDNAttnBackend.forward_extend's split of the convolved [T, qkv] rows into
    [1, T, heads, head_dim] query, key and value."""
    row_count = mixed_qkv.shape[0]
    if layer.q_dim + layer.k_dim + layer.v_dim <= MAX_FUSED_QKV_SPLIT_DIM:
        return fused_qkv_split_gdn_prefill(
            mixed_qkv,
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_k_dim,
            layer.head_v_dim,
        )
    else:
        query, key, value = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        return (
            query.view(1, row_count, layer.num_q_heads, layer.head_q_dim),
            key.view(1, row_count, layer.num_k_heads, layer.head_k_dim),
            value.view(1, row_count, layer.num_v_heads, layer.head_v_dim),
        )


def chunk_fwd_o_on_tables(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    meta: GDNPrefillGraphMetadata,
) -> torch.Tensor:
    """fla's chunk_fwd_o with the bucket's chunk table instead of one prepared
    from cu_seqlens on the host."""
    B, T, Hg, K, V = *query.shape, value.shape[-1]
    H = value.shape[-2]
    BT = min(CHUNK_SIZE, max(16, triton.next_power_of_2(T)))
    output = torch.zeros_like(value)
    chunk_fwd_kernel_o[(triton.cdiv(V, 64), meta.max_chunks, B * H)](
        query,
        key,
        value,
        h,
        g,
        output,
        meta.cu_seqlens,
        meta.chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
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
    meta: GDNPrefillGraphMetadata,
) -> torch.Tensor:
    """fla's chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True,
    inplace_update=True) with the bucket's tables: same kernels and configs,
    fixed grids, no host-side chunk preparation."""
    scale = key.shape[-1] ** -0.5
    query, key, value, g, beta = (
        tensor.contiguous() for tensor in (query, key, value, g, beta)
    )
    query = l2norm_fwd(query)
    key = l2norm_fwd(key)
    g = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=meta.cu_seqlens,
        chunk_indices=meta.chunk_indices,
    )
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=key,
        v=value,
        g=g,
        beta=beta,
        cu_seqlens=meta.cu_seqlens,
        chunk_indices=meta.chunk_indices,
    )
    h, new_value = chunk_gated_delta_rule_fwd_h(
        k=key,
        w=w,
        u=u,
        g=g,
        initial_state=ssm_states,
        initial_state_indices=meta.cache_indices,
        cu_seqlens=meta.cu_seqlens,
        chunk_indices=meta.chunk_indices,
        chunk_offsets=meta.chunk_offsets,
        inplace_update=True,
    )
    return chunk_fwd_o_on_tables(query, key, new_value, h, g, scale, meta)


def gdn_extend_in_graph(
    *,
    layer: RadixLinearAttention,
    meta: GDNPrefillGraphMetadata,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """GDNAttnBackend.forward_extend of a plain extend with launch grids that
    depend only on the bucket. conv_states [slots, dim, width - 1] and
    ssm_states [slots, heads, K, V] are the layer's caches, updated in place
    for the batch's slots. Returns [1, num_tokens, heads, V]; rows past the
    batch's tokens are zero."""
    if mixed_qkv.shape[0] != meta.num_tokens:
        raise RuntimeError(
            f"{mixed_qkv.shape[0]} rows reached the tables of a "
            f"{meta.num_tokens}-token bucket"
        )
    else:
        pass
    convolved = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        conv_states=conv_states,
        has_initial_state=meta.has_initial_state,
        cache_indices=meta.cache_indices,
        query_start_loc=meta.cu_seqlens,
        seq_lens_cpu=None,
        block_table=meta.conv_block_table,
    ).transpose(0, 1)
    query, key, value = split_qkv(layer, convolved)
    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
    return chunk_gated_delta_rule_on_tables(
        query, key, value, g, beta, ssm_states, meta
    )


_MAX_SEQS_BY_ARCHITECTURE: dict[str, int] = {}
_INSTALL_LOCK = Lock()


class GDNPrefillGraphAttnBackend(GDNAttnBackend):
    """GDNAttnBackend whose plain extend the breakable prefill graph captures on
    static per-bucket tables, for the architectures that opted in; the stock
    backend otherwise."""

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        self.prefill_graph_metadata: GDNPrefillGraphMetadata | None = None
        architectures = model_runner.model_config.hf_config.architectures
        self.prefill_graph_max_seqs: int = max(
            (_MAX_SEQS_BY_ARCHITECTURE.get(name, 0) for name in architectures or []),
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
        meta = GDNPrefillGraphMetadata.allocate(
            num_tokens=forward_batch.extend_num_tokens,
            max_seqs_cap=self.prefill_graph_max_seqs,
            cache_index_dtype=cache_indices.dtype,
            device=self.device,
        )
        self.fill_prefill_graph_metadata(meta, forward_batch, cache_indices)
        return meta

    def refresh_prefill_graph_metadata(
        self, meta: GDNPrefillGraphMetadata, forward_batch: ForwardBatch
    ) -> None:
        self.fill_prefill_graph_metadata(
            meta, forward_batch, self.prefill_graph_cache_indices(forward_batch)
        )

    def prefill_graph_cache_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        # The slot resolution of MambaAttnBackendBase._forward_metadata.
        cache_indices = self._translate_mamba_indices(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
        )
        real_bs = (
            forward_batch._original_batch_size
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        if real_bs is not None and real_bs < cache_indices.shape[0]:
            cache_indices = cache_indices.clone()
            cache_indices[real_bs:] = PAD_SLOT_ID
        else:
            pass
        return cache_indices

    def fill_prefill_graph_metadata(
        self,
        meta: GDNPrefillGraphMetadata,
        forward_batch: ForwardBatch,
        cache_indices: torch.Tensor,
    ) -> None:
        if forward_batch.mamba_track_mask is not None:
            raise RuntimeError(
                "a prefix-cache track batch reached a GDN prefill graph bucket"
            )
        else:
            pass
        meta.fill(
            extend_seq_lens=list(forward_batch.extend_seq_lens_cpu),
            cache_indices=cache_indices,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
        )
        self.prefill_graph_metadata = meta

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        # An eager forward never runs on a graph bucket's tables.
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
        meta = self.prefill_graph_metadata
        if meta is None:
            return super().forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        elif (
            forward_batch.forward_mode.is_target_verify()
            or self.mis_metadata is not None
        ):
            raise RuntimeError(
                "GDN prefill graph tables serve plain extends only; got "
                f"{forward_batch.forward_mode} with mis={self.mis_metadata is not None}"
            )
        else:
            assert isinstance(mixed_qkv, torch.Tensor)
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            conv_states = layer_cache.conv[0]
            ssm_states = layer_cache.temporal
            if not (conv_states.is_contiguous() and ssm_states.is_contiguous()):
                raise RuntimeError(
                    f"layer {layer.layer_id} keeps a strided state pool; the "
                    "captured GDN extend writes contiguous slots"
                )
            else:
                pass
            return gdn_extend_in_graph(
                layer=layer,
                meta=meta,
                conv_states=conv_states,
                ssm_states=ssm_states,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )


def install_gdn_prefill_graph(architecture: str, *, max_seqs: int) -> None:
    """Capture the GDN extend inside this architecture's breakable prefill
    graphs for batches of up to max_seqs requests; 0 keeps the eager
    breaks. Runners built afterwards get the backend subclass, which behaves as
    the stock backend for every other architecture."""
    with _INSTALL_LOCK:
        _MAX_SEQS_BY_ARCHITECTURE[architecture] = max_seqs
        gdn_backend.GDNAttnBackend = GDNPrefillGraphAttnBackend
