# SPDX-License-Identifier: Apache-2.0
"""Adapt the loaded Omni thinker to native DFlash target calls."""

from __future__ import annotations

from typing import Protocol

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.scheduling.types import PrefillTokenDrainError, SchedulerRequest


class PrefillTokenPublisher(Protocol):
    def __call__(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None: ...


class SpeculativeTargetWorker:
    """Preserve Omni prefill hooks while native DFlash owns verification."""

    def __init__(self, worker: ModelWorker) -> None:
        self.worker = worker
        self.model_runner: SGLModelRunner = worker.model_runner
        self.model_config: ModelConfig = self.model_runner.model_config
        self.device: str = worker.device
        self.thinker_runner: ThinkerModelRunner | None = None
        self.auxiliary_hidden_width: int = 0
        self.prefill_token_publisher: PrefillTokenPublisher | None = None

    def attach_runner(self, thinker_runner: ThinkerModelRunner) -> None:
        if thinker_runner.tp_worker is not self.worker:
            raise ValueError("DFlash target runner must use the loaded target worker")
        elif not thinker_runner.outer_model.capture_aux_hidden_states:
            raise ValueError("DFlash target auxiliary capture must be configured first")
        else:
            self.auxiliary_hidden_width = (
                len(thinker_runner.text_model.layers_to_capture)
                * self.model_runner.model_config.hidden_size
            )
            assert self.auxiliary_hidden_width > 0
            self.thinker_runner = thinker_runner

    def forward_batch_generation(
        self,
        batch: ScheduleBatch | None,
        forward_batch: ForwardBatch | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        is_verify: bool = False,
        skip_attn_backend_init: bool | None = None,
        *,
        capture_hidden_mode: CaptureHiddenMode | None = None,
    ) -> GenerationBatchResult:
        if self.thinker_runner is None:
            raise RuntimeError("DFlash target thinker runner is not attached")
        elif pp_proxy_tensors is not None:
            raise ValueError("Omni DFlash does not support pipeline parallel targets")
        else:
            thinker_runner = self.thinker_runner

        if batch is not None:
            if forward_batch is not None or is_verify:
                raise ValueError("DFlash prefill requires only a ScheduleBatch")
            elif capture_hidden_mode != CaptureHiddenMode.FULL:
                raise ValueError("DFlash prefill requires FULL auxiliary capture")
            elif not (batch.forward_mode.is_extend() or batch.forward_mode.is_idle()):
                raise ValueError("DFlash target accepts prefill or native verification")
            else:
                forward_batch = ForwardBatch.init_new(
                    batch,
                    self.model_runner,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    return_hidden_states_before_norm=False,
                )
        elif forward_batch is None or not is_verify:
            raise ValueError("DFlash verification requires a native ForwardBatch")
        elif capture_hidden_mode is not None:
            raise ValueError("DFlash verification carries its own capture mode")
        else:
            pass

        assert forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
        if is_verify and not (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_idle()
        ):
            raise ValueError("DFlash verification must use TARGET_VERIFY or IDLE mode")
        else:
            pass
        if forward_batch.forward_mode.is_idle():
            pass
        elif forward_batch.sampling_info is None:
            raise ValueError("DFlash target sampling metadata is missing")
        elif not forward_batch.sampling_info.is_all_greedy:
            raise ValueError("Omni DFlash currently supports only greedy requests")
        else:
            pass
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        if is_verify:
            native_output = self.model_runner.forward(forward_batch=forward_batch)
            batch_result = GenerationBatchResult(
                logits_output=native_output.logits_output,
                can_run_cuda_graph=native_output.can_run_graph,
                expert_distribution_metrics=native_output.expert_distribution_metrics,
                routed_experts_output=native_output.routed_experts_output,
                indexer_topk_output=native_output.indexer_topk_output,
            )
        else:
            assert batch is not None
            requests = [
                SchedulerRequest(request_id=request.rid, data=request.omni_data)
                for request in batch.reqs
            ]
            batch_result = thinker_runner.prepare_and_forward(
                forward_batch, batch, requests, is_prefill=True
            )
            if forward_batch.is_prefill_only:
                batch_result.next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
            elif batch_result.next_token_ids is None:
                batch_result.next_token_ids = thinker_runner.sample_next_token_ids(
                    batch_result.logits_output, forward_batch, batch, requests
                )
            else:
                pass

        if forward_batch.forward_mode.is_idle():
            pass
        else:
            assert batch_result.logits_output is not None
            auxiliary_hidden = batch_result.logits_output.hidden_states
            expected_shape = (
                forward_batch.input_ids.numel(),
                self.auxiliary_hidden_width,
            )
            assert (
                auxiliary_hidden is not None
            ), "DFlash target returned no auxiliary rows"
            assert auxiliary_hidden.shape == expected_shape, (
                f"DFlash target auxiliary shape {tuple(auxiliary_hidden.shape)} "
                f"does not match {expected_shape}"
            )
        if batch is not None and self.prefill_token_publisher is not None:
            try:
                self.prefill_token_publisher(batch, batch_result)
            except Exception:
                try:
                    torch.cuda.current_stream(
                        forward_batch.input_ids.device
                    ).synchronize()
                except Exception as drain_error:
                    raise PrefillTokenDrainError(
                        "Cannot drain failed first-token publication; stop scheduling "
                        "without releasing request storage"
                    ) from drain_error
                raise
        else:
            pass
        return batch_result
