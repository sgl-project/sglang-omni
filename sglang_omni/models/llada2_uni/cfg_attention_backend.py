# SPDX-License-Identifier: Apache-2.0
"""FlashInfer backend for LLaDA2 image-edit CFG padding."""

from __future__ import annotations

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
    merge_state,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.server_args import (
    ATTENTION_BACKEND_CHOICES,
    add_attention_backend_choices,
)

CFG_ATTENTION_BACKEND = "llada2_uni_cfg_flashinfer"


class LLaDA2CFGFlashInferAttnBackend(FlashInferAttnBackend):
    """Stock FlashInfer plus opt-in image-edit CFG left-pad masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_wrappers != 1:
            raise ValueError("DLLM CFG requires one attention wrapper")
        if self.flashinfer_kv_cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("DLLM CFG requires FP16/BF16 KV cache")
        self._cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )
        self._cfg_local_left_pad_active = False
        self._cfg_has_cached_prefix = False

    def init_forward_metadata(self, forward_batch):
        self._cfg_local_left_pad_active = False
        # FlashInfer keeps the previous custom mask on its reusable wrapper.
        for attr in ("_custom_mask_buf", "_mask_indptr_buf"):
            setattr(self._cfg_prefill_wrapper_ragged, attr, None)
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)
        left_pad_lens = tuple(forward_batch.dllm_left_pad_lens_cpu)
        if not any(left_pad_lens):
            return super().init_forward_metadata(forward_batch)
        prefix_lens = tuple(forward_batch.extend_prefix_lens_cpu)
        query_lens = tuple(forward_batch.extend_seq_lens_cpu)
        assert len(left_pad_lens) == len(prefix_lens) == len(query_lens)
        cached_left_pad_lens = tuple(
            min(pad, prefix)
            for pad, prefix in zip(left_pad_lens, prefix_lens, strict=True)
        )
        local_left_pad_lens = tuple(
            min(max(pad - prefix, 0), query)
            for pad, prefix, query in zip(
                left_pad_lens, prefix_lens, query_lens, strict=True
            )
        )
        paged_lens = tuple(
            max(prefix - pad, 0)
            for prefix, pad in zip(prefix_lens, left_pad_lens, strict=True)
        )
        self.plan_cfg_attention(
            forward_batch,
            query_lens,
            cached_left_pad_lens,
            local_left_pad_lens,
            paged_lens,
        )

    def plan_cfg_attention(
        self,
        forward_batch,
        query_lens: tuple[int, ...],
        cached_pad_lens: tuple[int, ...],
        local_pad_lens: tuple[int, ...],
        paged_lens: tuple[int, ...],
    ) -> None:
        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        wrappers = self.prefill_wrappers_paged
        cached_left_pad_lens = torch.tensor(
            cached_pad_lens, dtype=seq_lens.dtype, device=seq_lens.device
        )
        paged_kernel_lens = torch.tensor(
            paged_lens, dtype=seq_lens.dtype, device=seq_lens.device
        )
        local_pad_active = any(local_pad_lens)
        cfg_prefill_wrapper = self._cfg_prefill_wrapper_ragged
        prefill_indices_updater = self.indices_updater_prefill
        if local_pad_active:
            flattened_request_masks = []
            for query_length, local_left_pad_length in zip(
                query_lens, local_pad_lens, strict=True
            ):
                request_attention_mask = torch.ones(
                    (query_length, query_length),
                    dtype=torch.bool,
                    device=seq_lens.device,
                )
                if local_left_pad_length:
                    request_attention_mask[:, :local_left_pad_length] = False
                    # Give discarded pad queries a diagonal key to avoid invalid softmax.
                    pad_indices = torch.arange(
                        local_left_pad_length, device=seq_lens.device
                    )
                    request_attention_mask[pad_indices, pad_indices] = True
                flattened_request_masks.append(request_attention_mask.flatten())
            custom_mask = torch.cat(flattened_request_masks)
            qo_indptr = torch.zeros(
                seq_lens.numel() + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            # Exclude edit pads from cached-prefix attention without replacing the custom mask.
            prefill_indices_updater.call_begin_forward(
                cfg_prefill_wrapper,
                wrappers[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                sum(paged_lens),
                seq_lens,
                prefix_lens,
                cached_left_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                False,
                None,
                fixed_split_size=self.prefill_split_tile_size,
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
            self._cfg_local_left_pad_active = True
            self._cfg_has_cached_prefix = any(paged_lens)
        else:
            self._cfg_local_left_pad_active = False
            prefill_indices_updater.call_begin_forward(
                prefill_indices_updater.prefill_wrapper_ragged,
                wrappers[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                sum(paged_lens),
                seq_lens,
                prefix_lens,
                cached_left_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                True,
                None,
                fixed_split_size=self.prefill_split_tile_size,
            )

        self.forward_metadata = PrefillMetadata(
            wrappers,
            use_ragged=True,
            extend_no_prefix=False,
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
        if not self._cfg_local_left_pad_active:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        if k is None or v is None:
            raise RuntimeError("CFG first-block attention requires explicit K/V")

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
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            attention_output, _ = merge_state(
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
            kv_cache_location = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(kv_cache_location, self.forward_metadata.swa_out_cache_loc),
                k,
                v,
                *self._kv_write_scales(layer),
            )
        return attention_output.view(-1, layer.tp_q_head_num * layer.head_dim)


def register_llada2_cfg_flashinfer_backend() -> None:
    """Register DLLM pad masking without changing stock/text-condition backends."""

    def _create_backend(runner):
        if runner.use_mla_backend:
            raise ValueError("LLaDA2 CFG attention does not use an MLA backend")
        return LLaDA2CFGFlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )

    ATTENTION_BACKENDS[CFG_ATTENTION_BACKEND] = _create_backend
    if CFG_ATTENTION_BACKEND not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([CFG_ATTENTION_BACKEND])
