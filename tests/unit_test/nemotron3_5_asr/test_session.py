# SPDX-License-Identifier: Apache-2.0
"""Nemotron session batching, lifetime, and error isolation contracts."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator

import pytest

from sglang_omni.models.nemotron3_5_asr.batch_engine import NemotronBatchEngine
from sglang_omni.models.nemotron3_5_asr.streaming import AppendResult
from sglang_omni.models.nemotron3_5_asr.session import NemotronSessionScheduler
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SESSION_METADATA_KEY, SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.nemotron3_5_asr.test_streaming import FakeRunner


def make_engine(runner: FakeRunner, *, batch_size: int = 8, wait_ms: float = 0,
                max_pcm_bytes: int = 1024 * 1024) -> NemotronBatchEngine:
    return NemotronBatchEngine(
        runner, max_batch_size=batch_size, max_batch_wait_ms=wait_ms,
        max_pending_tasks=128, max_open_sessions=64, max_state_bytes=1 << 30,
        max_pcm_bytes=max_pcm_bytes, max_history_tokens=16384, max_text_bytes=1 << 20,
    )


def make_chunk(sequence: int, samples: int = 4040, *, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", sequence * 20.0, samples / 16, sequence,
                      b"\0\0" * samples, format="pcm16", eos=eos)


def payload_for(identity: SessionIdentity, operation: str, sequence: int = 0,
                samples: int = 4040, *, eos: bool = False) -> StagePayload:
    operation_fields = SessionOperation(
        operation, identity, ("asr",),
        make_chunk(sequence, samples, eos=eos) if operation == "append" else None,
    )
    return StagePayload(
        request_id=f"{identity.id}:{identity.open_index}:{operation}:{sequence}",
        request=OmniRequest(None, {"language": "auto"}, metadata={SESSION_METADATA_KEY: operation_fields.to_dict()}),
        data=None,
    )


@contextmanager
def running_scheduler(engine: NemotronBatchEngine, concurrency: int = 8) -> Iterator[NemotronSessionScheduler]:
    scheduler = NemotronSessionScheduler(engine, max_concurrency=concurrency,
                                         max_open_sessions=64, max_state_bytes=1 << 30)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler
    finally:
        scheduler.stop()
        thread.join(5)
        assert not thread.is_alive()
        assert not engine.thread.is_alive()
        assert not engine.states and not engine.tasks
        assert not scheduler.open_sessions


def submit(scheduler: NemotronSessionScheduler, payload: StagePayload) -> None:
    scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))


def test_real_scheduler_forms_one_eight_session_runner_call() -> None:
    runner = FakeRunner()
    engine = make_engine(runner, wait_ms=10000)
    with running_scheduler(engine) as scheduler:
        identities = [SessionIdentity(str(index)) for index in range(8)]
        for identity in identities:
            submit(scheduler, payload_for(identity, "open"))
        assert all(scheduler.outbox.get(timeout=5).type == "result" for _ in identities)
        for identity in identities:
            submit(scheduler, payload_for(identity, "append"))
        messages = [scheduler.outbox.get(timeout=5) for _ in range(16)]
        assert [len(batch) for batch in runner.batches] == [8]
        assert len({id(state) for state in runner.batches[0]}) == 8
        for identity in identities:
            assert [message.type for message in messages if message.request_id.startswith(identity.id + ":")] == ["stream", "result"]
    assert runner.is_closed


@pytest.mark.parametrize("concurrency,wait_ms", [(1, 0), (1, 2), (4, 2)])
def test_short_inputs_long_append_and_empty_eos(concurrency: int, wait_ms: float) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, wait_ms=wait_ms)
    identity = SessionIdentity("audio")
    with running_scheduler(engine, concurrency) as scheduler:
        submit(scheduler, payload_for(identity, "open"))
        assert scheduler.outbox.get(timeout=5).type == "result"
        for sequence in range(12):
            submit(scheduler, payload_for(identity, "append", sequence, 320))
            assert scheduler.outbox.get(timeout=5).type == "result"
        assert not runner.batches
        submit(scheduler, payload_for(identity, "append", 12, 20000))
        partial = scheduler.outbox.get(timeout=5)
        assert partial.type == "stream"
        assert scheduler.outbox.get(timeout=5).type == "result"
        assert len(runner.batches) >= 4
        assert engine.usage(identity).slots["pcm_bytes"] < 2 * engine.spec.subsequent_samples
        submit(scheduler, payload_for(identity, "append", 13, 0, eos=True))
        final_chunk = scheduler.outbox.get(timeout=5)
        final = scheduler.outbox.get(timeout=5)
        assert final_chunk.data["eos"]
        assert final.data.data["duration_s"] == pytest.approx(23840 / 16000)
        assert final.data.data["text"] == final_chunk.data["payload"]["full_text"]


def test_empty_stream_and_exact_window_eos_do_not_repeat_inference() -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    try:
        for index, samples in enumerate([0, 4040]):
            identity = SessionIdentity("reopen", index + 1)
            engine.open(identity, OmniRequest(None)).result(5)
            payload = payload_for(identity, "append")
            first = engine.append(identity, make_chunk(0, samples), payload, threading.Event()).result(5)
            before_eos = len(runner.batches)
            final = engine.append(identity, make_chunk(1, 0, eos=True), payload, threading.Event()).result(5)
            assert isinstance(first, AppendResult) and isinstance(final, AppendResult)
            assert final.is_final
            assert len(runner.batches) == before_eos
            with pytest.raises(ValueError, match="ended"):
                engine.append(identity, make_chunk(2), payload, threading.Event()).result(5)
            engine.close(identity).result(5)
            engine.close(identity).result(5)
        assert len(runner.batches) == 1
    finally:
        engine.shutdown()


def test_prefix_failure_and_inflight_cancel_leave_other_lane_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)
    identities = [SessionIdentity("bad"), SessionIdentity("good")]
    entered, proceed = threading.Event(), threading.Event()
    original = runner.run_streaming_batch

    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert proceed.wait(5)
        result.clean_texts[0] = "changed"
        return result

    try:
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        first = [engine.append(identity, make_chunk(0), payload_for(identity, "append"), threading.Event()) for identity in identities]
        for future in first:
            future.result(5)
        monkeypatch.setattr(runner, "run_streaming_batch", blocked)
        second = [engine.append(identity, make_chunk(1, 5224), payload_for(identity, "append"), threading.Event()) for identity in identities]
        assert entered.wait(5)
        proceed.set()
        with pytest.raises(RuntimeError, match="prefix"):
            second[0].result(5)
        assert second[1].result(5).full_text == "word more"
        assert identities[0] not in engine.states
        monkeypatch.setattr(runner, "run_streaming_batch", original)
        engine.close(identities[1]).result(5)
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        entered.clear()
        proceed.clear()
        monkeypatch.setattr(runner, "run_streaming_batch", blocked)
        cancelled = threading.Event()
        futures = [engine.append(identities[0], make_chunk(0), payload_for(identities[0], "append"), cancelled),
                   engine.append(identities[1], make_chunk(0), payload_for(identities[1], "append"), threading.Event())]
        assert entered.wait(5)
        cancelled.set()
        proceed.set()
        with pytest.raises(RuntimeError, match="closed|cancelled"):
            futures[0].result(5)
        assert futures[1].result(5).full_text == "word"
    finally:
        proceed.set()
        engine.shutdown()


def test_shutdown_releases_waiting_tickets_before_forward_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    entered, proceed = threading.Event(), threading.Event()
    original = runner.run_streaming_batch

    def blocked(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_streaming_batch", blocked)
    identity = SessionIdentity("stop")
    engine.open(identity, OmniRequest(None)).result(5)
    future = engine.append(identity, make_chunk(0), payload_for(identity, "append"), threading.Event())
    assert entered.wait(5)
    engine.begin_shutdown()
    with pytest.raises(RuntimeError, match="stopping"):
        future.result(1)
    proceed.set()
    engine.shutdown()
    assert not engine.thread.is_alive() and not engine.tasks and not engine.states


def test_budget_failure_and_decode_limit_do_not_accumulate_pcm() -> None:
    runner = FakeRunner()
    engine = make_engine(runner, max_pcm_bytes=9000)
    identity = SessionIdentity("limited")
    try:
        engine.open(identity, OmniRequest(None, {"max_new_tokens": 1})).result(5)
        payload = payload_for(identity, "append")
        engine.append(identity, make_chunk(0), payload, threading.Event()).result(5)
        for sequence in range(1, 10):
            engine.append(identity, make_chunk(sequence, 10000), payload, threading.Event()).result(5)
        assert engine.usage(identity).slots["pcm_bytes"] == 0
        assert len(runner.batches) == 1
        result = engine.append(identity, make_chunk(10, 0, eos=True), payload, threading.Event()).result(5)
        assert result.final_payload.data["duration_s"] == pytest.approx(94040 / 16000)
        other = SessionIdentity("overflow")
        engine.open(other, OmniRequest(None)).result(5)
        with pytest.raises(RuntimeError, match="PCM budget"):
            engine.append(other, make_chunk(0, 5000), payload, threading.Event()).result(5)
        assert other not in engine.states
    finally:
        engine.shutdown()


def test_offline_batch_and_stream_use_one_owner_and_cancel_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)
    calls = []
    owner_threads = []
    original = runner.run_streaming_batch

    def stream(*args, **kwargs):
        owner_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    def offline(requests):
        owner_threads.append(threading.get_ident())
        calls.append([request.stage_payload.request_id for request in requests])
        return [request.stage_payload for request in requests]

    monkeypatch.setattr(runner, "run_streaming_batch", stream)
    monkeypatch.setattr(runner, "run_batch", offline)
    engine.build_request = lambda payload: SimpleNamespace(stage_payload=payload)
    try:
        identities = [SessionIdentity("a"), SessionIdentity("b")]
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        cancelled = threading.Event()
        cancelled.set()
        rejected = engine.submit_offline(payload_for(SessionIdentity("cancelled"), "open"), cancelled)
        with pytest.raises(RuntimeError, match="cancelled"):
            rejected.result(5)
        with engine.condition:
            jobs = [engine.append(identity, make_chunk(0), payload_for(identity, "append"), threading.Event()) for identity in identities]
            jobs += [engine.submit_offline(payload_for(identity, "open"), threading.Event()) for identity in identities]
        for future in jobs:
            future.result(5)
        assert calls == [["a:1:open:0", "b:1:open:0"]]
        assert [len(batch) for batch in runner.batches] == [2]
        assert set(owner_threads) == {engine.thread.ident}
    finally:
        engine.shutdown()


def test_fatal_owner_error_settles_every_future(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)

    def fatal(*args, **kwargs):
        raise SystemExit("fatal model error")

    monkeypatch.setattr(runner, "run_streaming_batch", fatal)
    try:
        identities = [SessionIdentity("a"), SessionIdentity("b")]
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        futures = [engine.append(identity, make_chunk(0), payload_for(identity, "append"), threading.Event()) for identity in identities]
        for future in futures:
            with pytest.raises(RuntimeError, match="fatal model error"):
                future.result(5)
        with pytest.raises(RuntimeError, match="failed"):
            engine.open(SessionIdentity("later"), OmniRequest(None)).result(5)
    finally:
        engine.shutdown()
    assert not engine.tasks and not engine.thread.is_alive()
