# SPDX-License-Identifier: Apache-2.0
"""FlashInfer attention with grouped LLaDA2 CFG left-padding correctness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
    _safe_merge_state,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc


@dataclass(frozen=True)
class CFGAttentionLengths:
    local_left_pad_lengths: tuple[int, ...]
    cached_left_pad_lengths: tuple[int, ...]
    paged_kernel_num_tokens: int


class LLaDA2CFGFlashInferAttnBackend(FlashInferAttnBackend):
    """Stock FlashInfer plus opt-in grouped CFG padding support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )
        self._cfg_runtime_forward_batch = None

    def set_cfg_runtime_forward_batch(self, forward_batch: Any | None) -> None:
        """Set the real, unpadded batch used by public CUDA-graph replay APIs."""
        self._cfg_runtime_forward_batch = forward_batch

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: torch.Tensor | None = None,
    ) -> None:
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)
        if self.skip_prefill or not self.is_dllm_model:
            return
        self._require_single_ragged_wrapper()
        device = self.workspace_buffer.device
        self._cfg_cuda_graph_prefix_lens = torch.empty(
            max_bs, dtype=torch.int32, device=device
        )
        self._cfg_cuda_graph_cached_left_pad_lens = torch.empty(
            max_bs, dtype=torch.int32, device=device
        )
        self._cfg_cuda_graph_paged_kernel_lens = torch.empty(
            max_bs, dtype=torch.int32, device=device
        )

    def _require_single_ragged_wrapper(self) -> None:
        if self.num_wrappers != 1 or self.use_paged:
            raise RuntimeError(
                "LLaDA2 grouped CFG padding requires one ragged FlashInfer wrapper"
            )

    @staticmethod
    def _clear_stale_ragged_custom_mask(wrapper: Any) -> None:
        if getattr(wrapper, "is_cuda_graph_enabled", False):
            return
        for attribute in ("_custom_mask_buf", "_mask_indptr_buf"):
            if not hasattr(wrapper, attribute):
                raise RuntimeError(
                    f"Unsupported FlashInfer ragged wrapper: missing {attribute}"
                )
            setattr(wrapper, attribute, None)

    @staticmethod
    def _get_cfg_attention_lengths(
        forward_batch: Any,
    ) -> CFGAttentionLengths | None:
        if forward_batch is None:
            return None
        left_pad_values = getattr(forward_batch, "dllm_left_pad_lens_cpu", None)
        if left_pad_values is None:
            return None

        left_pad_lengths = torch.as_tensor(
            left_pad_values, dtype=torch.int64, device="cpu"
        )
        prefix_lengths = torch.as_tensor(
            forward_batch.extend_prefix_lens_cpu,
            dtype=torch.int64,
            device="cpu",
        )
        query_lengths = torch.as_tensor(
            forward_batch.extend_seq_lens_cpu,
            dtype=torch.int64,
            device="cpu",
        )
        expected_shape = (int(forward_batch.batch_size),)
        if not (
            tuple(left_pad_lengths.shape)
            == tuple(prefix_lengths.shape)
            == tuple(query_lengths.shape)
            == expected_shape
        ):
            raise RuntimeError("CFG padding metadata must match batch size")
        if bool(torch.any(query_lengths <= 0)):
            raise RuntimeError("DLLM CFG batch contains an empty active block")
        if not bool(torch.any(left_pad_lengths > 0)):
            return None

        cached = torch.minimum(left_pad_lengths, prefix_lengths)
        local = torch.minimum(
            torch.clamp(left_pad_lengths - prefix_lengths, min=0), query_lengths
        )
        paged_kernel_num_tokens = int(torch.sum(prefix_lengths - cached))
        return CFGAttentionLengths(
            local_left_pad_lengths=tuple(int(value) for value in local.tolist()),
            cached_left_pad_lengths=tuple(int(value) for value in cached.tolist()),
            paged_kernel_num_tokens=paged_kernel_num_tokens,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: Any,
        in_capture: bool = False,
    ) -> None:
        """Plan corrected cached-prefix geometry before public graph replay."""
        runtime_batch = self._cfg_runtime_forward_batch
        forward_mode = forward_batch.forward_mode
        if in_capture or runtime_batch is None or not forward_mode.is_dllm_extend():
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

        lengths = self._get_cfg_attention_lengths(runtime_batch)
        if lengths is None:
            return super().init_forward_metadata_out_graph(forward_batch)
        if any(lengths.local_left_pad_lengths):
            raise RuntimeError(
                "DLLM CFG CUDA-graph replay reached an in-query left pad; "
                "this block must run eagerly"
            )
        self._require_single_ragged_wrapper()

        capture_batch_size = int(forward_batch.batch_size)
        raw_batch_size = int(runtime_batch.batch_size)
        if raw_batch_size > capture_batch_size:
            raise RuntimeError(
                f"CFG CUDA-graph raw batch {raw_batch_size} exceeds "
                f"capture batch {capture_batch_size}"
            )

        prefix_lengths = self._cfg_cuda_graph_prefix_lens[:capture_batch_size]
        prefix_lengths.zero_()
        prefix_lengths[:raw_batch_size].copy_(
            runtime_batch.extend_prefix_lens.to(
                device=prefix_lengths.device, dtype=prefix_lengths.dtype
            )
        )
        real_left_pad_lengths = runtime_batch.dllm_left_pad_lens.to(
            device=prefix_lengths.device, dtype=prefix_lengths.dtype
        )
        cached_left_pad_lengths = self._cfg_cuda_graph_cached_left_pad_lens[
            :capture_batch_size
        ]
        cached_left_pad_lengths.zero_()
        torch.minimum(
            real_left_pad_lengths,
            prefix_lengths[:raw_batch_size],
            out=cached_left_pad_lengths[:raw_batch_size],
        )
        paged_kernel_lengths = self._cfg_cuda_graph_paged_kernel_lens[
            :capture_batch_size
        ]
        torch.sub(prefix_lengths, cached_left_pad_lengths, out=paged_kernel_lengths)

        self._cfg_local_left_pad_active = False
        updater = self.indices_updater_prefill
        capture_wrapper = self.prefill_cuda_graph_metadata[capture_batch_size][0]
        updater.call_begin_forward(
            updater.prefill_wrapper_ragged,
            capture_wrapper,
            forward_batch.req_pool_indices[:capture_batch_size],
            paged_kernel_lengths,
            lengths.paged_kernel_num_tokens,
            forward_batch.seq_lens[:capture_batch_size],
            prefix_lengths,
            cached_left_pad_lengths,
            updater.kv_indptr[0],
            updater.qo_indptr[0],
            True,
            None,
            fixed_split_size=self.prefill_split_tile_size,
        )
        self.forward_metadata = PrefillMetadata(
            [capture_wrapper], use_ragged=True, extend_no_prefix=False
        )

    def init_forward_metadata(self, forward_batch: Any) -> None:
        self._cfg_local_left_pad_active = False
        lengths = self._get_cfg_attention_lengths(forward_batch)
        if lengths is None or not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)
        self._require_single_ragged_wrapper()
        self._clear_stale_ragged_custom_mask(self._cfg_prefill_wrapper_ragged)

        query_lengths_cpu = torch.as_tensor(
            forward_batch.extend_seq_lens_cpu,
            dtype=torch.int64,
            device="cpu",
        )
        if bool(torch.any(query_lengths_cpu != query_lengths_cpu[0])):
            raise RuntimeError("CFG physical rows must have equal active-block lengths")

        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        left_pad_lens = forward_batch.dllm_left_pad_lens.to(
            device=seq_lens.device, dtype=seq_lens.dtype
        )
        cached_left_pad_lens = torch.minimum(left_pad_lens, prefix_lens)
        paged_kernel_lens = prefix_lens - cached_left_pad_lens
        updater = self.indices_updater_prefill
        has_local_pad = any(lengths.local_left_pad_lengths)
        updater.call_begin_forward(
            (
                self._cfg_prefill_wrapper_ragged
                if has_local_pad
                else updater.prefill_wrapper_ragged
            ),
            self.prefill_wrappers_paged[0],
            forward_batch.req_pool_indices,
            paged_kernel_lens,
            lengths.paged_kernel_num_tokens,
            seq_lens,
            prefix_lens,
            cached_left_pad_lens,
            updater.kv_indptr[0],
            updater.qo_indptr[0],
            not has_local_pad,
            None,
            fixed_split_size=self.prefill_split_tile_size,
        )

        if has_local_pad:
            query_length = int(query_lengths_cpu[0])
            batch_size = int(forward_batch.batch_size)
            device = seq_lens.device
            local_pad_lens = torch.as_tensor(
                lengths.local_left_pad_lengths,
                dtype=torch.int64,
                device=device,
            )
            indices = torch.arange(query_length, device=device)
            valid_keys = indices.unsqueeze(0) >= local_pad_lens.unsqueeze(1)
            custom_mask = (
                valid_keys.unsqueeze(1)
                .expand(batch_size, query_length, query_length)
                .clone()
            )
            pad_queries = indices.unsqueeze(0) < local_pad_lens.unsqueeze(1)
            diagonal = torch.eye(query_length, dtype=torch.bool, device=device)
            custom_mask.logical_or_(pad_queries.unsqueeze(2) & diagonal.unsqueeze(0))
            qo_indptr = torch.arange(
                0,
                (batch_size + 1) * query_length,
                query_length,
                dtype=torch.int32,
                device=device,
            )
            self._cfg_prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                updater.num_qo_heads,
                updater.num_kv_heads,
                updater.head_dim,
                custom_mask=custom_mask.reshape(-1),
                causal=False,
                q_data_type=updater.q_data_type,
                kv_data_type=updater.data_type,
                non_blocking=True,
                fixed_split_size=self.prefill_split_tile_size,
            )
            self._cfg_local_left_pad_active = True
            self._cfg_has_cached_prefix = lengths.paged_kernel_num_tokens > 0

        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_paged, use_ragged=True, extend_no_prefix=False
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        if not getattr(self, "_cfg_local_left_pad_active", False):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        if k is None or v is None:
            raise RuntimeError("CFG active-block attention requires explicit K/V")

        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_view = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_view = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if self._cfg_has_cached_prefix:
            current_output, current_lse = (
                self._cfg_prefill_wrapper_ragged.forward_return_lse(
                    q_view,
                    k_view,
                    v_view,
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )
            )
            cached_output, cached_lse = self.prefill_wrappers_paged[
                0
            ].forward_return_lse(
                q_view,
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )
            attention_output, _ = _safe_merge_state(
                current_output, current_lse, cached_output, cached_lse
            )
        else:
            attention_output = self._cfg_prefill_wrapper_ragged.forward(
                q_view,
                k_view,
                v_view,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )
        if save_kv_cache:
            cache_location = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(cache_location, self.forward_metadata.swa_out_cache_loc),
                k,
                v,
                *self._kv_write_scales(layer),
            )
        return attention_output.view(-1, layer.tp_q_head_num * layer.head_dim)


def register_llada2_cfg_flashinfer_backend() -> None:
    """Install the LLaDA2-specific FlashInfer factory before runner creation."""

    def create_backend(runner):
        if runner.use_mla_backend:
            raise ValueError("LLaDA2 grouped CFG does not use an MLA backend")
        return LLaDA2CFGFlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )

    ATTENTION_BACKENDS["flashinfer"] = create_backend
