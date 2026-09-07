# SPDX-License-Identifier: Apache-2.0
"""Omni scheduler bridge for the MLX Qwen3-TTS talker.

Two jobs the generic MLX bridge cannot do:

*Register prompts.*  SGLang's ``prefill_start`` only sees its own ``Req``, so
the assembled prompt embeddings and per-request sampling settings are pushed
into the MLX runner here, before the launch.

*Collect frames.*  ``finalize_mlx_result`` returns only group-0 token ids, so
the rest of each codec frame is drained out of the MLX runner and appended to
the request state the vocoder stage reads.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.scheduling.types import RequestOutput

logger = logging.getLogger(__name__)


class Qwen3TTSMlxSchedulerModelRunner(MlxSchedulerModelRunner):
    """Drives Qwen3-TTS frames through Omni's MLX scheduler path."""

    @property
    def _mlx_runner(self):
        return self.tp_worker._mlx_runner

    # -- prompt registration --------------------------------------------

    def execute_launch(self, scheduler_output: Any):
        self._register_new_requests(scheduler_output)
        return super().execute_launch(scheduler_output)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        self._register_requests(requests)
        return super().custom_prefill_forward(forward_batch, schedule_batch, requests)

    def _register_new_requests(self, scheduler_output: Any) -> None:
        requests = getattr(scheduler_output, "requests", None)
        if requests:
            self._register_requests(requests)

    def _register_requests(self, requests: list[Any]) -> None:
        """Hand the runner any prompt it has not seen yet."""
        from sglang_omni.models.qwen3_tts.mlx.request_spec import build_request_spec

        runner = self._mlx_runner
        for sched_req in requests:
            req_id = getattr(sched_req, "request_id", None)
            data = getattr(sched_req, "data", None)
            if req_id is None or data is None:
                continue
            if req_id in runner._tts_specs or runner.has_request(req_id):
                continue
            spec = build_request_spec(data)
            if spec is not None:
                runner.register_request(req_id, spec)

    # -- frame collection ------------------------------------------------

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Append this step's codec frames to each request's output.

        A request whose reported token is the codec EOS produced no audio for
        this step, matching the CUDA path, so its frame is dropped.
        """
        runner = self._mlx_runner
        eos_id = int(runner._talker_config.codec_eos_token_id)

        for sched_req in scheduler_output.requests:
            frames = runner.drain_frames(sched_req.request_id)
            if not frames:
                continue
            req_output = outputs.get(sched_req.request_id)
            if req_output is None:
                continue
            if req_output.data is not None and int(req_output.data) == eos_id:
                continue
            for frame in frames:
                sched_req.data.output_codes.append(frame)
            sched_req.data.latest_stream_code_chunk = frames[-1]
