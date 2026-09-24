# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

if TYPE_CHECKING:
    import torch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
    from sglang_omni.scheduling.types import (
        RequestOutput,
        SchedulerOutput,
        SchedulerRequest,
    )


class MiniCPMOThinkerModelRunner(ThinkerModelRunner):
    """Run the thinker and accumulate hidden states for speech conditioning."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            get_server_return_hidden_states_mode,
        )

        # note (MayDomine): the thinker initializer requires a nested Qwen config.
        ModelRunner.__init__(self, tp_worker, output_processor)

        # note (MayDomine): parent embedding injection reads these names.
        model = self.model
        self.outer_model = model.thinker
        self.text_model = self.outer_model.model
        self.embed_tokens = self.text_model.embed_tokens
        self.th_host_bufs = None
        self.th_slot = 0
        # note (MayDomine): bound-based injection needs no modality token ids.
        self.image_token_id = -1
        self.video_token_id = -1
        self.audio_token_id = -1

        self.capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if output_processor.capture_hidden
            else get_server_return_hidden_states_mode()
        )
        self.pending_hidden: dict[str, list[torch.Tensor]] = {}

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Use deployment-wide capture; batch arguments follow the runner interface."""
        return self.capture_hidden_mode

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Use deployment-wide capture; batch arguments follow the runner interface."""
        return self.capture_hidden_mode

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Collect request outputs; the raw result is part of the runner interface."""
        for sched_req in scheduler_output.requests:
            req_output = outputs.get(sched_req.request_id)
            if req_output is None or req_output.extra is None:
                continue
            hidden = req_output.extra.pop("hidden_states", None)
            if hidden is None:
                continue
            if sched_req.data.req.inflight_middle_chunks > 0:
                continue
            hidden = hidden.reshape(-1, hidden.shape[-1])[-1]
            seq = self.pending_hidden.setdefault(sched_req.request_id, [])
            # note (MayDomine): CUDA graph replay overwrites the original hidden buffer.
            seq.append(hidden.detach().clone())

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        """Do not advance generation state for non-final prefill chunks."""
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if sched_req.data.req.inflight_middle_chunks > 0
        }

    def on_request_finished(
        self, request_id: str, req_data: SGLangARRequestData
    ) -> None:
        """Flush the request's hidden accumulator with a single D2H copy."""
        import torch

        seq = self.pending_hidden.pop(request_id, None)
        if not seq:
            return
        stacked = torch.stack(seq).to("cpu")
        req_data.extra_model_outputs["hidden_states_seq"] = list(stacked.unbind(0))

    def reset_request(self, request_id: str) -> None:
        """Drop accumulated hidden states on abort (no terminal flush runs)."""
        self.pending_hidden.pop(request_id, None)
