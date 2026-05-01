# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni_v1.models.fishaudio_s2_pro.fish_scheduler import FishScheduler
from sglang_omni_v1.models.fishaudio_s2_pro.streaming_vocoder import (
    S2ProVocoderScheduler,
)
from sglang_omni_v1.pipeline.stage.stream_queue import StreamItem
from sglang_omni_v1.proto import OmniRequest, StagePayload
from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)


class _FakeCodec:
    sample_rate = 44100
    frame_length = 4
    delay = 0

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(indices.shape))
        batch = indices.shape[0]
        tokens = indices.shape[-1]
        rows = []
        for row in range(batch):
            values = torch.arange(tokens * self.frame_length, dtype=torch.float32)
            rows.append(values.unsqueeze(0) + float(row + 1))
        return torch.stack(rows, dim=0)


def _payload(
    request_id: str,
    *,
    stream: bool = True,
    code_len: int = 4,
    usage: dict | None = None,
) -> StagePayload:
    output_codes = torch.arange(11 * code_len, dtype=torch.long).reshape(11, code_len)
    data = {
        "output_codes": output_codes.tolist(),
        "sample_rate": 123,
    }
    if usage is not None:
        data["usage"] = usage
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=data,
    )


def _code(value: int = 1) -> torch.Tensor:
    return torch.full((11, 1), value, dtype=torch.long)


def _chunk(value: int = 1) -> StreamItem:
    return StreamItem(chunk_id=value, data=_code(value), from_stage="tts_engine")


def _start_scheduler(
    *,
    stream_stride: int = 3,
    stream_followup_stride: int = 5,
    stream_crossfade_samples: int = 0,
) -> tuple[S2ProVocoderScheduler, threading.Thread]:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=1,
        stream_crossfade_samples=stream_crossfade_samples,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return scheduler, thread


def _stop_scheduler(scheduler: S2ProVocoderScheduler, thread: threading.Thread) -> None:
    scheduler.stop()
    thread.join(timeout=2.0)


def test_streaming_vocoder_chunk_cadence() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.1)

        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(3)))
        first = scheduler.outbox.get(timeout=2.0)
        assert first.type == "stream"
        assert first.data["modality"] == "audio"
        assert len(first.data["audio_data"]) == 12

        for value in range(4, 8):
            scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(value)))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.1)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_final_flush_emits_tail_before_result() -> None:
    scheduler, thread = _start_scheduler(stream_stride=2, stream_followup_stride=10)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        first = scheduler.outbox.get(timeout=2.0)
        assert first.type == "stream"

        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(3)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        flush = scheduler.outbox.get(timeout=2.0)
        final = scheduler.outbox.get(timeout=2.0)
        assert flush.type == "stream"
        assert final.type == "result"
        assert final.data.data["modality"] == "audio"
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_done_before_payload_finalizes_after_new_request() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        stream = scheduler.outbox.get(timeout=2.0)
        final = scheduler.outbox.get(timeout=2.0)
        assert stream.type == "stream"
        assert final.type == "result"
        assert final.request_id == "req"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_final_payload_preserves_usage_and_authoritative_audio() -> (
    None
):
    scheduler, thread = _start_scheduler()
    usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    try:
        scheduler.inbox.put(
            IncomingMessage(
                "req",
                "new_request",
                _payload("req", stream=True, code_len=4, usage=usage),
            )
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        data = final.data.data
        assert data["usage"] == usage
        assert data["sample_rate"] == 44100
        assert data["modality"] == "audio"
        assert len(data["audio_data"]) == 16
    finally:
        _stop_scheduler(scheduler, thread)


def test_non_streaming_vocoder_clears_static_stream_done_signal() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=False))
        )
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=True))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(3)))
        stream = scheduler.outbox.get(timeout=2.0)
        assert stream.type == "stream"
    finally:
        _stop_scheduler(scheduler, thread)


def test_non_streaming_vocoder_clears_prefetched_stream_done_signal() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=False))
        )
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_abort_cleans_state_and_suppresses_final() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.abort("req")
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_fish_scheduler_emits_code_chunks_only_for_streaming_requests() -> None:
    class _IterationController:
        def update_request(self, request, output_token_id) -> None:
            pass

        def is_finished(self, request, output_token_id) -> bool:
            return False

    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = set()
    scheduler.iteration_controller = _IterationController()

    stream_codes = _code(7)
    stream_req = SchedulerRequest(
        request_id="stream",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(
            stage_payload=_payload("stream", stream=True),
            latest_stream_code_chunk=stream_codes,
        ),
    )
    non_stream_req = SchedulerRequest(
        request_id="non-stream",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(
            stage_payload=_payload("non-stream", stream=False),
            latest_stream_code_chunk=_code(8),
        ),
    )

    finished = scheduler.update(
        SchedulerOutput(
            requests=[stream_req, non_stream_req],
            batch_data=None,
        ),
        ModelRunnerOutput(
            outputs={
                "stream": RequestOutput("stream", data=1),
                "non-stream": RequestOutput("non-stream", data=1),
            }
        ),
    )

    assert finished == []
    out = scheduler.outbox.get_nowait()
    assert out.request_id == "stream"
    assert out.type == "stream"
    assert out.target == "vocoder"
    assert out.data is stream_codes
    assert stream_req.data.latest_stream_code_chunk is None
    assert scheduler.outbox.empty()
