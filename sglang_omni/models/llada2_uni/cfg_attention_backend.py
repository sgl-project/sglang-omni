# SPDX-License-Identifier: Apache-2.0
"""FlashInfer backend for LLaDA2 image-edit CFG padding."""

from __future__ import annotations

import torch
from flashinfer.cascade import merge_state
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
from sglang.srt import server_args
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

CFG_ATTENTION_BACKEND = "llada2_uni_cfg_flashinfer"


class LLaDA2CFGFlashInferAttnBackend(FlashInferAttnBackend):
    """Stock FlashInfer plus masking of the padding in CFG companion rows."""

    def __init__(
        self, model_runner: ModelRunner, *, init_new_workspace: bool = False
    ) -> None:
        super().__init__(model_runner, init_new_workspace=init_new_workspace)
        if self.num_wrappers != 1:
            raise ValueError("DLLM CFG requires one attention wrapper")
        else:
            pass
        if self.flashinfer_kv_cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("DLLM CFG requires FP16/BF16 KV cache")
        else:
            pass
        self.cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )
        self.cfg_local_pad_active = False
        self.cfg_has_cached_prefix = False
        self.cfg_replay_batch: ForwardBatch | None = None

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        self.cfg_local_left_pad_active = False
        batch = self.cfg_replay_batch
        if in_capture or batch is None or not any(batch.dllm_left_pad_lens_cpu):
            return super().init_forward_metadata_out_graph(forward_batch, in_capture)
        else:
            pass

        bs = forward_batch.batch_size
        prefix_lens = forward_batch.seq_lens - self.dllm_config.block_size
        pad_lens = batch.dllm_left_pad_lens_cpu + [0] * (bs - batch.batch_size)
        cached_pad_lens = torch.tensor(
            pad_lens, dtype=prefix_lens.dtype, device=prefix_lens.device
        )
        paged_lens = prefix_lens - cached_pad_lens
        num_cached_tokens = sum(
            prefix - pad
            for prefix, pad in zip(
                batch.extend_prefix_lens_cpu,
                batch.dllm_left_pad_lens_cpu,
                strict=True,
            )
        )
        updater = self.indices_updater_prefill
        updater.call_begin_forward(
            updater.prefill_wrapper_ragged,
            self.prefill_cuda_graph_metadata[bs][0],
            forward_batch.req_pool_indices,
            paged_lens,
            num_cached_tokens,
            forward_batch.seq_lens,
            prefix_lens,
            cached_pad_lens,
            updater.kv_indptr[0],
            updater.qo_indptr[0],
            not self.use_paged,
            None,
            fixed_split_size=self.prefill_split_tile_size,
            plan=forward_batch.kv_loc_plan,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        self.cfg_local_pad_active = False
        # note (Anmuliar): FlashInfer retains custom masks between wrapper plans.
        self.cfg_prefill_wrapper_ragged._custom_mask_buf = (
            None  # noqa: leading-underscore  # FlashInfer API
        )
        self.cfg_prefill_wrapper_ragged._mask_indptr_buf = (
            None  # noqa: leading-underscore  # FlashInfer API
        )
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)
        else:
            pass
        pad_lens = tuple(forward_batch.dllm_pad_lens_cpu)
        if not any(pad_lens):
            return super().init_forward_metadata(forward_batch)
        else:
            pass
        pad_starts = tuple(forward_batch.dllm_pad_starts_cpu)
        prefix_lens = tuple(forward_batch.extend_prefix_lens_cpu)
        query_lens = tuple(forward_batch.extend_seq_lens_cpu)
        assert len(pad_lens) == len(pad_starts) == len(prefix_lens) == len(query_lens)
        # Each span is the row's padding in cached positions and in this block's columns.
        cached_pad_spans = tuple(
            (min(start, prefix), min(start + pad, prefix))
            for start, pad, prefix in zip(
                pad_starts, pad_lens, prefix_lens, strict=True
            )
        )
        local_pad_spans = tuple(
            (
                min(max(start - prefix, 0), query),
                min(max(start + pad - prefix, 0), query),
            )
            for start, pad, prefix, query in zip(
                pad_starts, pad_lens, prefix_lens, query_lens, strict=True
            )
        )
        self.plan_cfg_attention(
            forward_batch,
            query_lens,
            cached_pad_spans,
            local_pad_spans,
        )

    def plan_cfg_attention(
        self,
        forward_batch: ForwardBatch,
        query_lens: tuple[int, ...],
        cached_pad_spans: tuple[tuple[int, int], ...],
        local_pad_spans: tuple[tuple[int, int], ...],
    ) -> None:
        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        wrappers = self.prefill_wrappers_paged
        cached_pad_lens = torch.tensor(
            [end - start for start, end in cached_pad_spans],
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
        paged_kernel_lens = prefix_lens - cached_pad_lens
        paged_lens = paged_kernel_lens.tolist()
        # A left pad is skipped by start offset, any other needs explicit KV indices.
        if any(start > 0 and end > start for start, end in cached_pad_spans):
            kv_indptr = torch.zeros(
                prefix_lens.numel() + 1, dtype=torch.int32, device=prefix_lens.device
            )
            kv_indptr[1:] = torch.cumsum(prefix_lens, dim=0)
            kv_indices = torch.empty(
                int(kv_indptr[-1]), dtype=torch.int32, device=prefix_lens.device
            )
            self.kv_index_translator.pack_read_stream(
                forward_batch.kv_loc_plan,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=prefix_lens,
                indptr=kv_indptr,
                out=kv_indices,
            )
            keep = torch.ones_like(kv_indices, dtype=torch.bool)
            for row_start, (pad_start, pad_end) in zip(
                kv_indptr[:-1].tolist(), cached_pad_spans, strict=True
            ):
                keep[row_start + pad_start : row_start + pad_end] = False
            custom_kv_indices = kv_indices[keep]
        else:
            custom_kv_indices = None
        local_pad_active = any(end > start for start, end in local_pad_spans)
        cfg_prefill_wrapper = self.cfg_prefill_wrapper_ragged
        prefill_indices_updater = self.indices_updater_prefill
        if local_pad_active:
            flattened_request_masks = []
            for query_length, (local_pad_start, local_pad_end) in zip(
                query_lens, local_pad_spans, strict=True
            ):
                request_attention_mask = torch.ones(
                    (query_length, query_length),
                    dtype=torch.bool,
                    device=seq_lens.device,
                )
                if local_pad_end > local_pad_start:
                    request_attention_mask[:, local_pad_start:local_pad_end] = False
                    # note (Anmuliar): discarded pad queries still need a valid softmax.
                    pad_indices = torch.arange(
                        local_pad_start, local_pad_end, device=seq_lens.device
                    )
                    request_attention_mask[pad_indices, pad_indices] = True
                else:
                    pass
                flattened_request_masks.append(request_attention_mask.flatten())
            custom_mask = torch.cat(flattened_request_masks)
            qo_indptr = torch.zeros(
                seq_lens.numel() + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            # note (Anmuliar): planning cached KV must preserve the local padding mask.
            prefill_indices_updater.call_begin_forward(
                cfg_prefill_wrapper,
                wrappers[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                sum(paged_lens),
                seq_lens,
                prefix_lens,
                cached_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                False,
                None,
                fixed_split_size=self.prefill_split_tile_size,
                custom_kv_indices=custom_kv_indices,
                plan=forward_batch.kv_loc_plan,
            )
            cfg_prefill_wrapper.begin_forward(
                qo_indptr,
                qo_indptr,
                prefill_indices_updater.num_qo_heads,
                prefill_indices_updater.num_kv_heads,
                prefill_indices_updater.head_dim,
                custom_mask=custom_mask,
                causal=False,
                q_data_type=prefill_indices_updater.q_data_type,
                kv_data_type=prefill_indices_updater.data_type,
                non_blocking=True,
                fixed_split_size=self.prefill_split_tile_size,
            )
            self.cfg_local_pad_active = True
            self.cfg_has_cached_prefix = any(paged_lens)
        else:
            self.cfg_local_pad_active = False
            prefill_indices_updater.call_begin_forward(
                prefill_indices_updater.prefill_wrapper_ragged,
                wrappers[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                sum(paged_lens),
                seq_lens,
                prefix_lens,
                cached_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                True,
                None,
                fixed_split_size=self.prefill_split_tile_size,
                custom_kv_indices=custom_kv_indices,
                plan=forward_batch.kv_loc_plan,
            )

        self.forward_metadata = PrefillMetadata(
            wrappers,
            use_ragged=True,
            extend_no_prefix=not any(paged_lens),
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if not self.cfg_local_pad_active:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        else:
            pass

        if k is None or v is None:
            raise RuntimeError("CFG first-block attention requires explicit K/V")
        else:
            pass

        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_view = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_view = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if self.cfg_has_cached_prefix:
            current_output, current_lse = (
                self.cfg_prefill_wrapper_ragged.forward_return_lse(
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
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            attention_output, _ = merge_state(
                current_output, current_lse, cached_output, cached_lse
            )
        else:
            attention_output = self.cfg_prefill_wrapper_ragged.forward(
                q_view,
                k_view,
                v_view,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )
        if save_kv_cache:
            kv_cache_location = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_kv_buffer(  # noqa: leading-underscore  # SGLang API
                layer,
                KVWriteLoc(kv_cache_location, self.forward_metadata.swa_out_cache_loc),
                k,
                v,
                *self._kv_write_scales(layer),
            )
        else:
            pass
        return attention_output.view(-1, layer.tp_q_head_num * layer.head_dim)


def register_llada2_cfg_flashinfer_backend() -> None:
    """Register DLLM pad masking without changing stock/text-condition backends."""

    def _create_backend(runner: ModelRunner) -> LLaDA2CFGFlashInferAttnBackend:
        if runner.use_mla_backend:
            raise ValueError("LLaDA2 CFG attention does not use an MLA backend")
        else:
            pass
        return LLaDA2CFGFlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )

    ATTENTION_BACKENDS[CFG_ATTENTION_BACKEND] = _create_backend
    if CFG_ATTENTION_BACKEND not in server_args.ATTENTION_BACKEND_CHOICES:
        server_args.add_attention_backend_choices([CFG_ATTENTION_BACKEND])
    else:
        pass
