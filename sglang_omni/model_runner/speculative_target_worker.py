# SPDX-License-Identifier: Apache-2.0
"""Serve an Omni model worker as the target of SGLang's speculative workers."""

from __future__ import annotations

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.scheduling.types import SchedulerRequest


class SpeculativeTargetWorker:
    """Prefill through the Omni model runner, verify through the SGLang forward."""

    def __init__(self, model_worker: ModelWorker) -> None:
        self.model_worker: ModelWorker = model_worker
        self.model_runner: SGLModelRunner = model_worker.model_runner
        self.model_config: ModelConfig = model_worker.model_config
        self.device: str = model_worker.device
        self.random_seed: int = model_worker.random_seed
        # note (wilsonzheng0327): drafts read a tokenizer only to resolve a mask token
        # their config does not name, and every supported draft names it.
        self.tokenizer: None = None
        self.omni_runner: ModelRunner | None = None

    def attach_runner(self, omni_runner: ModelRunner) -> None:
        if omni_runner.tp_worker is not self.model_worker:
            raise ValueError("the speculative target runner must use the target worker")
        else:
            omni_runner.retain_prefill_input_embeds = SpeculativeAlgorithm.from_string(
                get_spec().speculative_algorithm
            ).is_eagle()
            self.omni_runner = omni_runner

    def get_memory_pool(self) -> tuple[ReqToTokenPool, BaseTokenToKVPoolAllocator]:
        return self.model_worker.get_memory_pool()

    def forward_batch_generation(
        self,
        batch: ScheduleBatch | None,
        forward_batch: ForwardBatch | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        is_verify: bool = False,
        skip_attn_backend_init: bool | None = None,
        *,
        capture_hidden_mode: CaptureHiddenMode | None = None,
        return_kv_loc_plan: bool = False,
    ) -> GenerationBatchResult:
        if pp_proxy_tensors is not None:
            raise ValueError("speculative decoding does not support pipeline parallel")
        elif forward_batch is not None:
            assert batch is None and is_verify
            forward_batch.apply_deprecated_skip_attn_backend_init(
                skip_attn_backend_init
            )
            output = self.model_runner.forward(forward_batch=forward_batch)
            return GenerationBatchResult(
                logits_output=output.logits_output,
                can_run_cuda_graph=output.can_run_graph,
                expert_distribution_metrics=output.expert_distribution_metrics,
                routed_experts_output=output.routed_experts_output,
                indexer_topk_output=output.indexer_topk_output,
                kv_loc_plan=forward_batch.kv_loc_plan if return_kv_loc_plan else None,
            )
        elif self.omni_runner is None:
            raise RuntimeError("the speculative target runner is not attached")
        else:
            # Prefill runs the model's own hooks and samples with its own sampler.
            assert batch is not None and not is_verify
            omni_runner = self.omni_runner
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                capture_hidden_mode=capture_hidden_mode,
                return_hidden_states_before_norm=False,
            )
            requests = [
                SchedulerRequest(request_id=request.rid, data=request.omni_data)
                for request in batch.reqs
            ]
            batch_result = omni_runner.prepare_and_forward(
                forward_batch, batch, requests, is_prefill=True
            )
            if omni_runner.retain_prefill_input_embeds:
                batch_result.logits_output.mm_input_embeds = (
                    omni_runner.retained_prefill_input_embeds
                )
                omni_runner.retained_prefill_input_embeds = None
            else:
                pass
            if batch.is_prefill_only:
                batch_result.next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
            elif batch_result.next_token_ids is None:
                batch_result.next_token_ids = omni_runner.sample_next_token_ids(
                    batch_result.logits_output, forward_batch, batch, requests
                )
            else:
                pass
            # note (wilsonzheng0327): The draft writes its KV into the slots this
            # plan resolved for the target.
            batch_result.kv_loc_plan = (
                forward_batch.kv_loc_plan if return_kv_loc_plan else None
            )
            return batch_result
