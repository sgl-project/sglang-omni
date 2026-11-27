# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for LLaDA2 image-edit CFG padding."""

from __future__ import annotations

import types
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

    Under a decode graph the padded tokens occupy whole pages (the scheduler
    rounds the left pad up to a page), so replay can start the page table at
    the first real token and shorten the KV lengths by the same amount.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        self.cfg_kv_indices: torch.Tensor | None = None
        self.cfg_kv_lens_cumsum: list[int] = []
        self.cfg_replay_batch: ForwardBatch | None = None
        self.cfg_replay_untrimmed_seq_lens_cpu: list[int] | None = None

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

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ) -> None:
        """Plan graph replay metadata with the CFG left pad excluded.

        The scheduler rounds a companion's left pad up to a page, so the real
        tokens of every branch start on a page boundary. Replay can therefore
        drop the leading ``pad / page_size`` pages from the page table and
        shorten ``actual_seq_lengths_kv`` by the same token count (the runner
        does the latter before computing the graph update payload).
        """
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        if in_capture:
            return
        else:
            pass
        batch = self.cfg_replay_batch
        if batch is None:
            return
        else:
            pass
        left_pad_lens = getattr(batch, "dllm_left_pad_lens_cpu", None)
        if not left_pad_lens or not any(left_pad_lens):
            return
        else:
            pass

        num_rows = len(left_pad_lens)
        page_size = self.page_size
        block_tables = self.forward_metadata.block_tables
        req_pool_indices = forward_batch.req_pool_indices[:num_rows].tolist()
        # The replay hook trims ``seq_lens`` before the runner stages it, so the
        # lengths reaching this view already exclude the pad. Use the untrimmed
        # snapshot the hook kept, or the pad would be subtracted twice.
        if self.cfg_replay_untrimmed_seq_lens_cpu is not None:
            seq_lens = self.cfg_replay_untrimmed_seq_lens_cpu[:num_rows]
        else:
            seq_lens = forward_batch.seq_lens_cpu[:num_rows].tolist()
        for row, req_index, seq_len, left_pad in zip(
            block_tables[:num_rows],
            req_pool_indices,
            seq_lens,
            left_pad_lens,
            strict=True,
        ):
            left_pad = int(left_pad)
            if left_pad <= 0:
                continue
            else:
                pass
            real_len = int(seq_len) - left_pad
            num_pages = (real_len + page_size - 1) // page_size
            row.fill_(0)
            if num_pages <= 0:
                continue
            else:
                pass
            row[:num_pages].copy_(
                self.req_to_token[
                    int(req_index), left_pad : left_pad + real_len : page_size
                ]
                .to(row.dtype)
                // page_size
            )

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
        if self.graph_mode or self.cfg_kv_indices is None:
            # Graph capture/replay reads the page table planned by
            # ``init_forward_metadata_out_graph``, which already excludes the
            # aligned left pad; only eager steps need the dense gather below.
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


def install_cfg_npu_replay_hooks(model_runner: ModelRunner) -> None:
    """Make NPU decode-graph replay hide the CFG left pad.

    Ascend FIA receives ``actual_seq_lengths_kv`` as a host list that the NPU
    graph runner rebinds from ``forward_batch.seq_lens`` on every replay, so a
    companion's pad cannot be dropped by attention metadata alone. Shorten the
    live batch's lengths for the duration of the replay and hand the untrimmed
    batch to the backend for the page-table rewrite.
    """
    runner = getattr(model_runner, "decode_cuda_graph_runner", None)
    if runner is None or getattr(runner, "llada2_cfg_replay_hook", False):
        return
    else:
        pass
    original_execute = runner.execute

    def execute(self, forward_batch, pp_proxy_tensors=None):
        left_pad_lens = getattr(forward_batch, "dllm_left_pad_lens_cpu", None)
        seq_lens = forward_batch.seq_lens
        if (
            not left_pad_lens
            or not any(left_pad_lens)
            or len(left_pad_lens) != seq_lens.numel()
        ):
            return original_execute(forward_batch, pp_proxy_tensors)
        else:
            pass
        backend = self._replay_attn_backend()  # noqa: leading-underscore - sglang private API
        if not isinstance(backend, LLaDA2CFGAscendAttnBackend):
            return original_execute(forward_batch, pp_proxy_tensors)
        else:
            pass
        left_pad = torch.tensor(
            left_pad_lens, dtype=seq_lens.dtype, device=seq_lens.device
        )
        backend.cfg_replay_batch = forward_batch
        if forward_batch.needs_forward_metadata_init():
            # Only the re-planning step reads these lengths; the replay fast path
            # must stay free of a device-to-host sync.
            backend.cfg_replay_untrimmed_seq_lens_cpu = seq_lens.cpu().tolist()
        else:
            pass
        seq_lens.sub_(left_pad)
        try:
            return original_execute(forward_batch, pp_proxy_tensors)
        finally:
            seq_lens.add_(left_pad)
            backend.cfg_replay_batch = None
            backend.cfg_replay_untrimmed_seq_lens_cpu = None

    runner.execute = types.MethodType(execute, runner)
    runner.llada2_cfg_replay_hook = True
    install_inline_graph_update(runner)


class CompletedUpdateFuture:
    def result(self) -> None:
        return None


class InlineUpdateExecutor:
    """Run ``graph.update`` inline.

    SGLang hands ``graph.update`` to a background thread so it can overlap with
    ``graph.replay``. On Ascend the two calls deadlock once a decode batch holds
    more than a few rows, even with per-stream Host task queues enabled, so the
    CFG decode path runs the update before the replay instead.
    """

    def submit(self, fn, /, *args, **kwargs):
        fn(*args, **kwargs)
        return CompletedUpdateFuture()

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


def install_inline_graph_update(runner) -> None:
    backend = getattr(runner, "backend", None)
    executor = getattr(backend, "_update_executor", None)  # noqa: leading-underscore - sglang private API
    if executor is None or isinstance(executor, InlineUpdateExecutor):
        return
    else:
        pass
    if hasattr(executor, "shutdown"):
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        pass
    backend._update_executor = InlineUpdateExecutor()  # noqa: leading-underscore - sglang private API
