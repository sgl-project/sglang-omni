# SPDX-License-Identifier: Apache-2.0
"""The Decode stage emits the Prefill sample before terminal output."""
import queue

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req

from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.pipeline.test_pd_utils import prefill_req


@pytest.mark.parametrize(
    "mode",
    [
        "stream",
        "aborted",
        "abort_reason",
        "nonstream",
        "skip",
        "disabled",
        "max_tokens",
        "stop_token",
    ],
)
def test_first_sample_streams_once_before_terminal(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler.emit_transferred_token = mode != "disabled"
    scheduler.outbox = queue.Queue()
    scheduler.first_emit_done = set()
    request = prefill_req()
    request.inflight_middle_chunks = 0
    if mode == "abort_reason":
        request.finished_reason = FINISH_ABORT("cancelled")
    elif mode == "max_tokens":
        request.sampling_params.max_new_tokens = 1
        request.check_finished()
        assert request.finished()
    elif mode == "stop_token":
        request.sampling_params.stop_token_ids = {42}
        request.check_finished()
        assert request.finished()
    request.omni_data.stage_payload.request.params.update(
        stream=mode != "nonstream", modalities=["text"]
    )
    scheduler.aborted_request_ids = {request.rid} if mode == "aborted" else set()

    def build_stream(
        request_id: str,
        request_data: SGLangARRequestData,
        output: RequestOutput,
    ) -> list[OutgoingMessage]:
        if not request_data.stage_payload.request.params["stream"]:
            return []
        else:
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=torch.tensor([output.data]),
                    target="decode",
                )
            ]

    scheduler.stream_output_builder = build_stream
    counts_at_terminal = []

    def terminal(
        self: OmniScheduler,
        reqs: list[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        counts_at_terminal.append(self.outbox.qsize())

    monkeypatch.setattr(OmniScheduler, "stream_output", terminal)
    scheduler.stream_output([request], skip_req=request if mode == "skip" else None)
    scheduler.stream_output([request], skip_req=request if mode == "skip" else None)
    if mode not in ("stream", "max_tokens", "stop_token"):
        assert counts_at_terminal == [0, 0]
        assert scheduler.outbox.empty()
    else:
        assert counts_at_terminal == [1, 1]
        message = scheduler.outbox.get_nowait()
        assert message.target == "decode"
        assert message.data.tolist() == [42]
        assert scheduler.outbox.empty()
        if mode == "stream":
            request.output_ids.append(43)
            scheduler.put_stream_messages(
                request.rid,
                scheduler.stream_output_builder(
                    request.rid,
                    request.omni_data,
                    RequestOutput(request_id=request.rid, data=43),
                ),
            )
            scheduler.stream_output([request])
            next_message = scheduler.outbox.get_nowait()
            assert next_message.target == message.target
            assert next_message.data.tolist() == [43]
            assert scheduler.outbox.empty()
