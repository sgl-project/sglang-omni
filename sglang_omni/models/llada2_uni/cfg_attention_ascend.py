# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for LLaDA2 image-edit CFG padding."""

from __future__ import annotations

from itertools import accumulate

import torch
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang_omni.models.llada2_uni.bootstrap import ASCEND_ATTENTION_BACKEND
from sglang_omni.models.llada2_uni.cfg_padding import build_cfg_kv_indices


class LLaDA2CFGAscendAttnBackend(AscendAttnBackend):
    """Stock ascend attention plus opt-in image-edit CFG left-pad hiding.

    ``npu_fused_infer_attention_score`` only accepts block sizes that are a
    multiple of 16, so a sub-page left padding cannot be dropped by shortening
    ``actual_seq_lengths_kv``. The padding tokens are dropped from a gathered
    dense KV view instead, which keeps the page table untouched.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        self.cfg_kv_indices: torch.Tensor | None = None
        self.cfg_kv_lens_cumsum: list[int] = []

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        self.cfg_kv_indices = None
        self.cfg_kv_lens_cumsum = []
        super().init_forward_metadata(forward_batch)
        if not self.is_dllm_model:
            return
        else:
            pass
        if not forward_batch.forward_mode.is_dllm_extend():
            return
        else:
            pass
        left_pad_lens = getattr(forward_batch, "dllm_left_pad_lens_cpu", None)
        if not left_pad_lens or not any(left_pad_lens):
            return
        else:
            pass

        seq_lens = forward_batch.seq_lens_cpu
        if seq_lens is None:
            seq_lens = forward_batch.seq_lens
        else:
            pass
        self.cfg_kv_indices, kv_lens = build_cfg_kv_indices(
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices.tolist(),
            seq_lens.tolist(),
            list(left_pad_lens),
        )
        self.cfg_kv_lens_cumsum = list(accumulate(kv_lens))

    def forward_dllm(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: torch.Tensor | None = None,
        k_rope: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
    ):
        if self.cfg_kv_indices is None:
            return super().forward_dllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                topk_indices=topk_indices,
            )
        else:
            pass

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                ),
                k,
                v,
            )
        else:
            pass

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, layer.tp_k_head_num * layer.qk_head_dim
        )
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1, layer.tp_v_head_num * layer.v_head_dim
        )
        indices = self.cfg_kv_indices
        if indices.device != k_cache.device:
            indices = indices.to(k_cache.device)
            self.cfg_kv_indices = indices
        else:
            pass
        keys = k_cache.index_select(0, indices).view(
            -1, layer.tp_k_head_num, layer.qk_head_dim
        )
        values = v_cache.index_select(0, indices).view(
            -1, layer.tp_v_head_num, layer.v_head_dim
        )

        if self.forward_metadata.extend_seq_lens_cpu_int is None:
            actual_seq_lengths = self.forward_metadata.seq_lens_list_cumsum
        else:
            actual_seq_lengths = (
                torch.cumsum(self.forward_metadata.extend_seq_lens_cpu_int, dim=0)
                .int()
                .tolist()
            )

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            keys,
            values,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=None,
            scale=layer.scaling,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=self.cfg_kv_lens_cumsum,
        )
        return attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def register_llada2_cfg_ascend_backend() -> None:
    """Route the ascend backend to the CFG variant for this algorithm only.

    NPU server-argument resolution pins the attention backend to ``ascend``
    before the model overrides run, so a dedicated registry name can never be
    selected. Wrapping the factory keeps the stock backend for other models.
    """
    stock_factory = ATTENTION_BACKENDS.get(ASCEND_ATTENTION_BACKEND)
    if stock_factory is None:
        raise ValueError("The ascend attention backend is not registered")
    else:
        pass
    if getattr(stock_factory, "_llada2_cfg_wrapped", False):  # noqa: leading-underscore
        return
    else:
        pass

    def _create_backend(runner: ModelRunner) -> AscendAttnBackend:
        if runner.server_args.dllm_algorithm != "LowConfidenceCFG":
            return stock_factory(runner)
        else:
            pass
        if runner.use_mla_backend:
            raise ValueError("LLaDA2 CFG attention does not use an MLA backend")
        else:
            pass
        return LLaDA2CFGAscendAttnBackend(runner)

    _create_backend._llada2_cfg_wrapped = True  # noqa: leading-underscore
    ATTENTION_BACKENDS[ASCEND_ATTENTION_BACKEND] = _create_backend
