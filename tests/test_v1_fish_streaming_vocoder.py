# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni_v1.models.fishaudio_s2_pro.fish_scheduler import FishScheduler
from sglang_omni_v1.models.fishaudio_s2_pro.streaming_vocoder import (
    S2ProVocoderScheduler,
    _StreamVocoderState,
    build_stream_vocoder_chunk,
    flush_stream_vocoder_chunk,
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


class _ContextCodec:
    sample_rate = 44100
    frame_length = 3
    delay = 3

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(
            1,
            indices.shape[1] + 1,
            dtype=torch.float32,
            device=indices.device,
        ).view(1, -1, 1)
        token_values = (indices.float() * weights).sum(dim=1)
        prev_values = torch.nn.functional.pad(token_values[:, :-1], (1, 0))
        frames = token_values + 0.25 * prev_values
        frame_offsets = torch.arange(
            self.frame_length,
            dtype=torch.float32,
            device=indices.device,
        ).view(1, 1, -1)
        audio = frames.unsqueeze(-1) + frame_offsets
        return audio.reshape(indices.shape[0], 1, -1)


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


def _empty_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": False}),
        data={},
    )


def _zero_length_payload(request_id: str) -> StagePayload:
    output_codes = torch.empty((11, 0), dtype=torch.long)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": False}),
        data={"output_codes": output_codes.tolist()},
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


def test_streaming_vocoder_sample_level_matches_contextual_full_decode() -> None:
    codec = _ContextCodec()
    state = _StreamVocoderState()
    full_codes = torch.arange(11 * 7, dtype=torch.long).reshape(11, 7)
    chunks = []

    for idx in range(full_codes.shape[1]):
        output = build_stream_vocoder_chunk(
            state,
            full_codes[:, idx : idx + 1],
            codec=codec,
            device=torch.device("cpu"),
            stream_stride=3,
            stream_followup_stride=2,
            stream_overlap_tokens=1,
            stream_crossfade_samples=0,
        )
        if output is not None:
            chunks.append(torch.tensor(output["audio_data"]))

    output = flush_stream_vocoder_chunk(
        state,
        codec=codec,
        device=torch.device("cpu"),
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    if output is not None:
        chunks.append(torch.tensor(output["audio_data"]))

    streaming_audio = torch.cat(chunks)
    full_audio = codec.from_indices(full_codes[1:][None])[0, 0]

    torch.testing.assert_close(streaming_audio, full_audio)


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
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_stride=3,
        stream_followup_stride=5,
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    try:
        scheduler._payloads["req"] = _payload("req")
        scheduler._pending_done.add("req")
        scheduler._on_chunk("req", _chunk(1))
        scheduler._pending_messages.append(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        scheduler.abort("req")
        thread.start()

        assert "req" not in scheduler._payloads
        assert "req" not in scheduler._stream_states
        assert "req" not in scheduler._pending_done
        assert "req" in scheduler._aborted_request_ids
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_abort_does_not_block_other_request() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(
            IncomingMessage("aborted", "new_request", _payload("aborted"))
        )
        scheduler.inbox.put(IncomingMessage("aborted", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("other", "new_request", _payload("other")))
        scheduler.inbox.put(IncomingMessage("other", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("other", "stream_chunk", _chunk(2)))
        scheduler.inbox.put(IncomingMessage("other", "stream_chunk", _chunk(3)))

        scheduler.abort("aborted")

        out = scheduler.outbox.get(timeout=2.0)
        assert out.request_id == "other"
        assert out.type == "stream"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_chunk_failure_emits_one_error_and_no_success() -> None:
    scheduler, thread = _start_scheduler()

    def _raise_on_chunk(request_id, chunk) -> None:
        del request_id, chunk
        raise RuntimeError("chunk failed")

    scheduler._on_chunk = _raise_on_chunk
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        error = scheduler.outbox.get(timeout=2.0)

        assert error.request_id == "req"
        assert error.type == "error"
        assert isinstance(error.data, RuntimeError)
        assert "req" in scheduler._aborted_request_ids

        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_abort_during_final_vocode_suppresses_result() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    scheduler._on_streaming_new_request("req", _payload("req"))

    def _abort_during_vocode(payload):
        scheduler.abort(payload.request_id)
        return payload

    scheduler._vocode_payload = _abort_during_vocode

    scheduler._on_done("req")

    assert "req" in scheduler._aborted_request_ids
    assert scheduler.outbox.empty()


def test_non_streaming_vocoder_rejects_missing_output_codes() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(
            IncomingMessage("req-empty", "new_request", _empty_payload("req-empty"))
        )
        output = scheduler.outbox.get(timeout=2.0)
    finally:
        _stop_scheduler(scheduler, thread)

    assert output.request_id == "req-empty"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert "req-empty" in str(output.data)
    assert scheduler._codec.calls == []


def test_non_streaming_vocoder_batch_rejects_zero_length_before_decode() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    with pytest.raises(ValueError, match="req-zero"):
        scheduler._vocode_payloads(
            [
                _payload("req-good", stream=False),
                _zero_length_payload("req-zero"),
            ]
        )
    assert scheduler._codec.calls == []


def test_non_streaming_vocoder_batch_skips_aborted_request() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
        max_batch_wait_ms=0,
    )
    scheduler.abort("aborted")
    first = IncomingMessage("other", "new_request", _payload("other", stream=False))
    scheduler.inbox.put(
        IncomingMessage("aborted", "new_request", _payload("aborted", stream=False))
    )

    batch = scheduler._collect_new_request_batch(first)
    scheduler._handle_new_request_batch(batch)

    out = scheduler.outbox.get_nowait()
    assert out.request_id == "other"
    assert out.type == "result"
    assert scheduler.outbox.empty()


def test_non_streaming_vocoder_abort_during_batch_decode_suppresses_result() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    messages = [
        IncomingMessage("other", "new_request", _payload("other", stream=False)),
        IncomingMessage("aborted", "new_request", _payload("aborted", stream=False)),
    ]

    def _abort_during_decode(payloads):
        scheduler.abort("aborted")
        return payloads

    scheduler._vocode_payloads = _abort_during_decode

    scheduler._vocode_non_streaming_batch(messages)

    out = scheduler.outbox.get_nowait()
    assert out.request_id == "other"
    assert out.type == "result"
    assert scheduler.outbox.empty()


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


def test_fish_scheduler_abort_during_update_suppresses_stream_chunk() -> None:
    freed = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = ["req"]
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = set()
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )

    class _IterationController:
        def update_request(self, request, output_token_id) -> None:
            del request, output_token_id
            scheduler.abort("req")

        def is_finished(self, request, output_token_id) -> bool:
            del request, output_token_id
            return True

    scheduler.iteration_controller = _IterationController()
    request = SchedulerRequest(
        request_id="req",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(
            stage_payload=_payload("req", stream=True),
            latest_stream_code_chunk=_code(9),
        ),
    )
    scheduler._requests["req"] = request

    finished = scheduler.update(
        SchedulerOutput(requests=[request], batch_data=None),
        ModelRunnerOutput(outputs={"req": RequestOutput("req", data=1)}),
    )

    assert finished == []
    assert freed == ["req"]
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._running_ids


def test_fish_scheduler_emit_finished_suppresses_aborted_result() -> None:
    adapted = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = {"req"}
    scheduler._requests = {}
    scheduler._submit_times = {"req": 1.0}

    def _result_adapter(data):
        adapted.append(data)
        return _payload("req")

    scheduler._result_adapter = _result_adapter
    request = SchedulerRequest(
        request_id="req",
        status=SchedulerStatus.FINISHED,
        data=SimpleNamespace(req=SimpleNamespace(output_ids=[1])),
    )
    scheduler._requests["req"] = request

    scheduler.emit_finished([request])

    assert adapted == []
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_finish_preserves_abort_marker_for_emit_suppression() -> None:
    freed = []
    adapted = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = {"req"}
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = ["req"]
    scheduler._submit_times = {"req": 1.0}
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )
    scheduler._result_adapter = lambda data: adapted.append(data) or _payload("req")
    request = SchedulerRequest(
        request_id="req",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(req=SimpleNamespace(output_ids=[1])),
    )
    scheduler._requests["req"] = request

    scheduler._finish_request(request)
    scheduler.emit_finished([request])

    assert freed == ["req"]
    assert adapted == []
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_abort_cleanup_frees_waiting_request_resources() -> None:
    freed = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque(["req"])
    scheduler._running_ids = []
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = set()
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )
    request = SchedulerRequest("req", data=SimpleNamespace())
    scheduler._requests["req"] = request

    scheduler.abort("req")

    assert freed == []
    assert request.status == SchedulerStatus.ABORTED
    assert "req" in scheduler._requests

    scheduler._cleanup_aborted_requests()

    assert freed == ["req"]
    assert request.status == SchedulerStatus.ABORTED
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._waiting
    assert "req" in scheduler._aborted_request_ids
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_abort_defers_inflight_resource_free_until_update() -> None:
    freed = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = ["req"]
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = {"req"}
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )
    request = SchedulerRequest(
        "req",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(),
    )
    scheduler._requests["req"] = request

    scheduler.abort("req")

    assert freed == []
    assert request.status == SchedulerStatus.ABORTED
    assert "req" in scheduler._requests

    scheduler._cleanup_aborted_requests()
    assert freed == []
    assert "req" in scheduler._requests

    finished = scheduler.update(
        SchedulerOutput(requests=[request], batch_data=None),
        ModelRunnerOutput(outputs={}),
    )

    assert finished == []
    assert freed == ["req"]
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._running_ids
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_batch_exception_cleans_finished_request() -> None:
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = []
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = set()
    scheduler.resource_manager = SimpleNamespace(free=lambda request: None)

    request = SchedulerRequest(
        "req",
        status=SchedulerStatus.FINISHED,
        data=SimpleNamespace(),
    )
    scheduler._requests["req"] = request
    error = RuntimeError("adapter failed")

    scheduler._handle_batch_exception(
        SchedulerOutput(requests=[request], batch_data=None),
        error,
    )

    out = scheduler.outbox.get_nowait()
    assert out.request_id == "req"
    assert out.type == "error"
    assert out.data is error
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._submit_times
    assert "req" not in scheduler._aborted_request_ids
