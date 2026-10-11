# SPDX-License-Identifier: Apache-2.0
"""Lifecycle tests without model downloads or a CUDA device."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.models.voxtream2.backend import VoXtreamBackend
from sglang_omni.models.voxtream2.scheduler import VoXtreamScheduler
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.serve.speech_errors import speech_generation_error


def message(rid="a", stream=True, inputs="Hello", **params):
    payload = StagePayload(rid, OmniRequest(inputs, {"stream": stream, **params}), None)
    return IncomingMessage(rid, "new_request", payload)


def audio(data):
    return np.frombuffer(data["audio_waveform"], dtype=np.float32)


class Backend:
    sample_rate = 24000
    max_samples = 100

    def __init__(self):
        self.started = []
        self.closed = []

    def iter_audio(self, payload):
        rid = payload.request_id
        self.started.append(rid)
        try:
            buffer = np.empty(2, dtype=np.float32)
            for value in range(3):
                buffer.fill(value)
                yield buffer
        finally:
            self.closed.append(rid)


@pytest.fixture
def run_scheduler():
    running = []

    def run(backend=None, **kwargs):
        scheduler = VoXtreamScheduler(backend or Backend(), **kwargs)
        worker = threading.Thread(target=scheduler.start)
        worker.start()
        running.append((scheduler, worker))
        return scheduler

    yield run
    for scheduler, worker in running:
        scheduler.stop()
        worker.join(3)
        assert not worker.is_alive()
        assert not scheduler.request_cancellations


def receive(scheduler, rid="a"):
    messages = []
    while True:
        item = scheduler.outbox.get(timeout=3)
        assert item.request_id == rid
        messages.append(item)
        if item.type in ("result", "error"):
            return messages


@pytest.mark.parametrize("stream", [False, True])
def test_audio_order_and_terminal_payload(run_scheduler, stream):
    scheduler = run_scheduler()
    scheduler.enqueue(message(stream=stream))
    messages = receive(scheduler)
    assert messages[-1].type == "result"
    assert scheduler.backend.closed == ["a"]
    result = messages[-1].data.data
    assert result["sample_rate"] == 24000
    if stream:
        assert [m.type for m in messages] == ["stream"] * 3 + ["result"]
        assert "audio_waveform" not in result
        assert all(m.metadata == {"modality": "audio"} for m in messages[:-1])
        output = np.concatenate([audio(m.data) for m in messages[:-1]])
    else:
        assert len(messages) == 1
        output = audio(result)
    np.testing.assert_array_equal(output, [0, 0, 1, 1, 2, 2])


def test_abort_before_admission_and_queued_request(run_scheduler):
    scheduler = VoXtreamScheduler(Backend())
    scheduler.abort("early")
    scheduler.enqueue(message("early"))
    scheduler.enqueue(message("queued"))
    scheduler.abort("queued")
    scheduler.enqueue(message("normal"))
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        receive(scheduler, "normal")
        assert scheduler.backend.started == ["normal"]
        assert scheduler.backend.closed == ["normal"]
    finally:
        scheduler.stop()
        worker.join(3)
        assert not worker.is_alive()


def test_abort_during_frame_suppresses_late_audio_and_cleans_before_next(run_scheduler):
    entered, release = threading.Event(), threading.Event()

    class Blocking(Backend):
        def iter_audio(self, payload):
            if payload.request_id == "a":
                self.started.append("a")
                try:
                    entered.set()
                    assert release.wait(3)
                    yield np.ones(2)
                finally:
                    self.closed.append("a")
            else:
                assert self.closed == ["a"]
                yield from super().iter_audio(payload)

    scheduler = run_scheduler(Blocking())
    scheduler.enqueue(message())
    assert entered.wait(3)
    scheduler.enqueue(message("next"))
    scheduler.abort("a")
    release.set()
    receive(scheduler, "next")
    assert scheduler.backend.started == ["a", "next"]
    assert scheduler.backend.closed == ["a", "next"]


def test_stop_unblocks_full_output_queue(run_scheduler):
    scheduler = run_scheduler(output_buffer_chunks=1)
    scheduler.enqueue(message())
    # The first output proves the worker is active; leave it in the queue.
    first = scheduler.outbox.get(timeout=3)
    assert first.type == "stream"
    scheduler.stop()


def test_cancel_with_full_output_queue_releases_next_request(run_scheduler):
    scheduler = run_scheduler(output_buffer_chunks=1)
    scheduler.enqueue(message())
    assert scheduler.outbox.get(timeout=3).type == "stream"
    scheduler.abort("a")
    # Discard anything enqueued before the abort, just as Stage does.
    while True:
        try:
            scheduler.outbox.get_nowait()
        except queue.Empty:
            break
    scheduler.enqueue(message("next"))
    receive(scheduler, "next")
    assert scheduler.backend.closed == ["a", "next"]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.zeros((2, 2)),
        np.array([]),
        np.zeros(101),
    ],
)
def test_invalid_audio_fails_only_its_request(run_scheduler, bad):
    class Invalid(Backend):
        def iter_audio(self, payload):
            if payload.request_id == "a":
                try:
                    yield bad
                finally:
                    self.closed.append("a")
            else:
                yield from super().iter_audio(payload)

    scheduler = run_scheduler(Invalid())
    scheduler.enqueue(message())
    scheduler.enqueue(message("next"))
    assert receive(scheduler)[-1].type == "error"
    assert receive(scheduler, "next")[-1].type == "result"
    assert scheduler.backend.closed == ["a", "next"]


def test_generator_exception_after_audio_propagates(run_scheduler):
    class Failing(Backend):
        def iter_audio(self, payload):
            yield np.zeros(2)
            raise RuntimeError("decode failed")

    scheduler = run_scheduler(Failing())
    scheduler.enqueue(message())
    output = receive(scheduler)
    assert [m.type for m in output] == ["stream", "error"]
    assert str(output[-1].data) == "decode failed"


def test_empty_generation_is_error(run_scheduler):
    class Empty(Backend):
        def iter_audio(self, payload):
            yield from ()

    scheduler = run_scheduler(Empty())
    scheduler.enqueue(message())
    assert receive(scheduler)[-1].type == "error"


def test_admission_limits_and_duplicate_ids():
    scheduler = VoXtreamScheduler(Backend(), max_pending=1)
    scheduler.enqueue(message())
    with pytest.raises(ValueError, match="already"):
        scheduler.enqueue(message())
    scheduler.enqueue(message("b"))
    rejected = scheduler.outbox.get_nowait()
    assert rejected.request_id == "b" and rejected.type == "error"
    assert isinstance(rejected.data, QueueFullError)
    scheduler.abort("a")
    for i in range(1100):
        scheduler.abort(f"unknown-{i}")
    assert scheduler.request_cancellations["a"].is_set()
    assert len(scheduler.early_aborts) == 1024
    scheduler.stop()
    with pytest.raises(RuntimeError):
        scheduler.enqueue(message("c"))


@pytest.mark.parametrize("kind", ["stream_chunk", "stream_done"])
def test_incremental_input_not_silently_accepted(kind):
    scheduler = VoXtreamScheduler(Backend())
    with pytest.raises(ValueError, match="complete text"):
        scheduler.enqueue(IncomingMessage("a", kind))


def author_backend(tmp_path, *, failure=False):
    reference = tmp_path / "reference.wav"
    reference.touch()

    def generate(*args, **kwargs):
        assert args == (reference, "Hello")
        assert next(kwargs["speaking_rate"]) == 3.0
        yield np.zeros(2), 0.01
        if failure:
            raise RuntimeError("model failed")

    generator = SimpleNamespace(
        config=SimpleNamespace(mimi_sr=24000, max_audio_length_ms=20000),
        generate_stream=generate,
        model=Mock(),
        mimi=Mock(),
        _mimi_streaming_started=True,
    )
    return VoXtreamBackend(generator, reference)


@pytest.mark.parametrize("mode", ["finish", "close", "error"])
def test_backend_resets_author_state_for_every_exit(tmp_path, mode):
    backend = author_backend(tmp_path, failure=mode == "error")
    iterator = backend.iter_audio(message().data)
    next(iterator)
    if mode == "error":
        with pytest.raises(RuntimeError, match="model failed"):
            next(iterator)
    elif mode == "close":
        iterator.close()
    else:
        assert list(iterator) == []
    backend.generator.model.reset_caches.assert_called_once()
    backend.generator.mimi.reset_streaming.assert_called_once()


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan"), True, "3"])
def test_invalid_rate_rejected_before_generation(tmp_path, rate):
    backend = author_backend(tmp_path)
    with pytest.raises(ValueError, match="speaking_rate_sps"):
        list(backend.iter_audio(message(speaking_rate_sps=rate).data))
    backend.generator.model.reset_caches.assert_not_called()


@pytest.mark.parametrize("text", ["", "  ", {}, [], None])
def test_invalid_text_rejected(tmp_path, text):
    backend = author_backend(tmp_path)
    with pytest.raises(ValueError, match="complete text"):
        list(backend.iter_audio(message(inputs=text).data))


def test_cleanup_can_modify_real_inference_tensors(tmp_path):
    backend = author_backend(tmp_path)
    with torch.inference_mode():
        state = torch.ones(2)
    backend.generator.mimi.reset_streaming.side_effect = lambda: state.zero_()
    iterator = backend.iter_audio(message().data)
    next(iterator)
    iterator.close()
    assert torch.equal(state, torch.zeros(2))
    assert not torch.is_inference_mode_enabled()


def test_failed_cleanup_blocks_model_reuse(tmp_path):
    backend = author_backend(tmp_path)
    backend.generator.mimi.reset_streaming.side_effect = RuntimeError("reset failed")
    with pytest.raises(RuntimeError, match="reset failed"):
        list(backend.iter_audio(message().data))
    backend.generator.generate_stream = Mock()
    with pytest.raises(RuntimeError, match="recreate the backend"):
        list(backend.iter_audio(message("next").data))
    backend.generator.generate_stream.assert_not_called()


def test_stop_while_output_is_provably_full(run_scheduler):
    second_frame = threading.Event()

    class Backpressure(Backend):
        def iter_audio(self, payload):
            try:
                yield np.zeros(2)
                second_frame.set()
                yield np.ones(2)
                pytest.fail("Worker advanced despite its full output queue")
            finally:
                self.closed.append(payload.request_id)

    scheduler = run_scheduler(Backpressure(), output_buffer_chunks=1)
    scheduler.enqueue(message())
    assert second_frame.wait(3)
    assert scheduler.outbox.qsize() == 1
    scheduler.stop()


def test_request_seed_is_repeatable_and_restores_cpu_rng(tmp_path, monkeypatch):
    backend = author_backend(tmp_path)
    backend.generator.ctx = SimpleNamespace(device="cpu")

    def generate(*args, **kwargs):
        yield torch.rand(8).numpy(), 0.01

    backend.generator.generate_stream = generate
    seed_all_cuda_devices = Mock()
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_all_cuda_devices)
    before = torch.random.get_rng_state().clone()
    first = list(backend.iter_audio(message(seed=17).data))[0]
    second = list(backend.iter_audio(message(seed=17).data))[0]
    np.testing.assert_array_equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())
    seed_all_cuda_devices.assert_not_called()


@pytest.mark.parametrize("name", ["temperature", "max_new_tokens", "top_p"])
def test_unsupported_explicit_sampling_is_not_silently_ignored(tmp_path, name):
    backend = author_backend(tmp_path)
    payload = message().data
    payload.request.metadata = {"tts_params": {"explicit_generation_params": [name]}}
    with pytest.raises(ValueError, match="does not support"):
        list(backend.iter_audio(payload))


def test_stage_rate_parameter_reaches_generator(tmp_path):
    backend = author_backend(tmp_path)

    def generate(*args, **kwargs):
        assert next(kwargs["speaking_rate"]) == 4.0
        yield np.zeros(2), 0.01

    backend.generator.generate_stream = generate
    payload = message(stage_params={"tts_engine": {"speaking_rate_sps": 4.0}}).data
    assert len(list(backend.iter_audio(payload))) == 1


def test_overload_rejection_does_not_block_on_full_audio_outbox():

    scheduler = VoXtreamScheduler(Backend(), max_pending=1, output_buffer_chunks=1)
    scheduler.enqueue(message())
    scheduler.outbox.put_nowait(OutgoingMessage("a", "stream", {}))
    scheduler.enqueue(message("rejected"))
    assert scheduler.outbox.get_nowait().type == "stream"
    error = scheduler.outbox.get_nowait()
    assert error.request_id == "rejected" and error.type == "error"
    assert list(scheduler.request_cancellations) == ["a"]


@pytest.mark.parametrize(
    "params,metadata,inputs",
    [
        ({}, {}, " "),
        ({"speaking_rate_sps": -1}, {}, "Hello"),
        ({"seed": -1}, {}, "Hello"),
        ({}, {"instructions": "whisper"}, "Hello"),
        ({}, {"language": "French"}, "Hello"),
        ({}, {"explicit_generation_params": ["temperature"]}, "Hello"),
    ],
)
def test_bad_request_survives_scheduler_error_serialization(
    tmp_path, run_scheduler, params, metadata, inputs
):

    backend = author_backend(tmp_path)
    scheduler = run_scheduler(backend)
    msg = message(inputs=inputs, **params)
    msg.data.request.metadata = {"tts_params": metadata}
    scheduler.enqueue(msg)
    rejected = receive(scheduler)[-1]
    assert rejected.type == "error"
    # Stage transports error text and Coordinator reconstructs an exception.
    error = speech_generation_error(RuntimeError(str(rejected.data)))
    assert error.status_code == 400
    assert error.error_type == "BadRequestError"
    scheduler.enqueue(message("next"))
    assert receive(scheduler, "next")[-1].type == "result"


def test_overload_survives_error_serialization():

    scheduler = VoXtreamScheduler(Backend(), max_pending=1)
    scheduler.enqueue(message())
    scheduler.enqueue(message("rejected"))
    rejected = scheduler.outbox.get_nowait()
    assert speech_generation_error(RuntimeError(str(rejected.data))).status_code == 503
    scheduler.stop()


@pytest.mark.parametrize(
    "message", ["decode failed", "VoXtream2 state reset failed; recreate the backend"]
)
def test_internal_model_failure_remains_server_error(message):

    assert speech_generation_error(RuntimeError(message)).status_code == 500
