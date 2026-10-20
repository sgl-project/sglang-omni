# SPDX-License-Identifier: Apache-2.0
"""Nemotron session batching, lifetime, and error isolation contracts."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pytest

from sglang_omni.models.nemotron3_5_asr.batch_engine import NemotronBatchEngine
from sglang_omni.models.nemotron3_5_asr.decoder import Nemotron3_5ASRDecodeState
from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRPreparedChunk,
    Nemotron3_5ASRStreamingBatchResult,
)
from sglang_omni.models.nemotron3_5_asr.request_builders import Nemotron3_5ASRRequest
from sglang_omni.models.nemotron3_5_asr.session import NemotronSessionScheduler
from sglang_omni.models.nemotron3_5_asr.streaming import AppendResult
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.nemotron3_5_asr.test_streaming import FakeRunner


def make_engine(
    runner: FakeRunner,
    *,
    batch_size: int = 8,
    wait_ms: float = 0,
    max_pcm_bytes: int = 1024 * 1024,
) -> NemotronBatchEngine:
    return NemotronBatchEngine(
        runner,
        max_batch_size=batch_size,
        max_batch_wait_ms=wait_ms,
        max_pending_tasks=128,
        max_open_sessions=64,
        max_state_bytes=1 << 30,
        max_pcm_bytes=max_pcm_bytes,
        max_history_tokens=16384,
        max_text_bytes=1 << 20,
    )


def make_chunk(sequence: int, samples: int = 4040, *, eos: bool = False) -> TimedChunk:
    return TimedChunk(
        "audio",
        sequence * 20.0,
        samples / 16,
        sequence,
        b"\0\0" * samples,
        format="pcm16",
        eos=eos,
    )


def test_pool_capacity_follows_session_reservation_budget() -> None:
    runner = FakeRunner()
    max_pcm_bytes, max_history_tokens, max_text_bytes = 8, 16, 8
    reservation = (
        runner.streaming_state_budget_bytes
        + max_pcm_bytes
        + 96 * max_history_tokens
        + 12 * max_text_bytes
    )
    engine = NemotronBatchEngine(
        runner,
        max_batch_size=8,
        max_batch_wait_ms=0,
        max_pending_tasks=8,
        max_open_sessions=4,
        max_state_bytes=2 * reservation,
        max_pcm_bytes=max_pcm_bytes,
        max_history_tokens=max_history_tokens,
        max_text_bytes=max_text_bytes,
    )
    try:
        assert runner.encoder_state_pool.capacity_slots == 2
        identities = [SessionIdentity(str(index)) for index in range(3)]
        for identity in identities[:2]:
            engine.open(identity, OmniRequest(None)).result(5)
        with pytest.raises(RuntimeError, match="reservation exhausted"):
            engine.open(identities[2], OmniRequest(None)).result(5)
        engine.close(identities[0]).result(5)
        engine.open(identities[2], OmniRequest(None)).result(5)
        assert len(runner.encoder_state_pool.active_slots) == 2
    finally:
        engine.shutdown()


def payload_for(
    identity: SessionIdentity,
    operation: Literal["open", "append", "close"],
    sequence: int = 0,
    samples: int = 4040,
    *,
    eos: bool = False,
) -> StagePayload:
    operation_fields = SessionOperation(
        operation,
        identity,
        ("asr",),
        make_chunk(sequence, samples, eos=eos) if operation == "append" else None,
    )
    return StagePayload(
        request_id=f"{identity.id}:{identity.open_index}:{operation}:{sequence}",
        request=OmniRequest(
            None,
            {"language": "auto"},
            metadata={SESSION_METADATA_KEY: operation_fields.to_dict()},
        ),
        data=None,
    )


@contextmanager
def running_scheduler(
    engine: NemotronBatchEngine, concurrency: int = 8
) -> Iterator[NemotronSessionScheduler]:
    scheduler = NemotronSessionScheduler(
        engine,
        max_concurrency=concurrency,
        max_open_sessions=64,
        max_state_bytes=1 << 30,
    )
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


def test_inbox_transfer_is_bounded_and_preserves_fifo() -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    engine.max_pending_tasks = 3
    scheduler = NemotronSessionScheduler(
        engine, max_concurrency=8, max_open_sessions=64, max_state_bytes=1 << 30
    )
    messages = [
        IncomingMessage("first", "stream_chunk"),
        IncomingMessage("second", "abort"),
        IncomingMessage("third", "stream_done"),
        IncomingMessage("fourth", "stream_chunk"),
    ]
    try:
        for message in messages:
            scheduler.inbox.put(message)
        assert scheduler.collect_inbox_messages() == messages[:3]
        assert scheduler.collect_inbox_messages() == messages[3:]
        with pytest.raises(queue.Empty):
            scheduler.collect_inbox_messages()
    finally:
        scheduler.stop()
    assert runner.is_closed


def test_shutdown_during_inbox_transfer_settles_the_whole_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    scheduler = NemotronSessionScheduler(
        engine, max_concurrency=8, max_open_sessions=64, max_state_bytes=1 << 30
    )
    identity = SessionIdentity("transfer-stop")
    payloads = [payload_for(identity, "open")]
    payloads.extend(payload_for(identity, "append", sequence) for sequence in range(16))
    payloads.append(payload_for(identity, "close"))
    for payload in payloads:
        submit(scheduler, payload)
    entered, proceed = threading.Event(), threading.Event()
    collect = scheduler.collect_inbox_messages

    def paused_collect() -> list[IncomingMessage]:
        messages = collect()
        entered.set()
        assert proceed.wait(5)
        return messages

    monkeypatch.setattr(scheduler, "collect_inbox_messages", paused_collect)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        assert entered.wait(5)
        scheduler.stop()
    finally:
        proceed.set()
        thread.join(5)
    assert not thread.is_alive() and not engine.thread.is_alive()
    assert engine.usage(identity) == ResourceUsage()
    assert runner.is_closed


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
            assert [
                message.type
                for message in messages
                if message.request_id.startswith(identity.id + ":")
            ] == ["stream", "result"]
    assert runner.is_closed


def test_scheduler_batches_sixteen_sessions_with_eight_workers() -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=16, wait_ms=10000)
    with running_scheduler(engine) as scheduler:
        identities = [SessionIdentity(str(index)) for index in range(16)]
        for identity in identities:
            submit(scheduler, payload_for(identity, "open"))
        assert all(scheduler.outbox.get(timeout=5).type == "result" for _ in identities)
        for identity in identities:
            submit(scheduler, payload_for(identity, "append"))
        messages = [scheduler.outbox.get(timeout=5) for _ in range(32)]
        assert [len(batch) for batch in runner.batches] == [16]
        assert sum(message.type == "result" for message in messages) == 16


def test_queued_operations_preserve_order_through_close_and_reopen() -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    first, second = SessionIdentity("ordered", 1), SessionIdentity("ordered", 2)
    payloads = [payload_for(first, "open")]
    payloads.extend(
        payload_for(first, "append", sequence, 320) for sequence in range(20)
    )
    payloads.extend(
        [payload_for(first, "append", 20, 0, eos=True), payload_for(first, "close")]
    )
    with running_scheduler(engine) as scheduler:
        for payload in payloads:
            submit(scheduler, payload)
        results, streams = [], []
        while len(results) < len(payloads):
            message = scheduler.outbox.get(timeout=5)
            assert message.type != "error"
            if message.type == "result":
                results.append(message)
            else:
                streams.append(message)
        reopened_payloads = [
            payload_for(second, "open"),
            payload_for(second, "append", 0, 4040, eos=True),
            payload_for(second, "close"),
        ]
        for payload in reopened_payloads:
            submit(scheduler, payload)
        payloads.extend(reopened_payloads)
        while len(results) < len(payloads):
            message = scheduler.outbox.get(timeout=5)
            assert message.type != "error"
            if message.type == "result":
                results.append(message)
            else:
                streams.append(message)
        assert [message.request_id for message in results] == [
            payload.request_id for payload in payloads
        ]
        final_results = [
            message.data.data
            for message in results
            if "duration_s" in message.data.data
        ]
        assert [result["duration_s"] for result in final_results] == pytest.approx(
            [0.4, 0.2525]
        )
        assert sum(message.data["eos"] for message in streams) == 2
        assert engine.usage(first) == ResourceUsage()
        assert engine.usage(second) == ResourceUsage()


def test_registered_arrival_order_survives_concurrent_inbox_puts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    identity = SessionIdentity("arrival")
    entered, proceed = threading.Event(), threading.Event()
    with running_scheduler(engine) as scheduler:
        submit(scheduler, payload_for(identity, "open"))
        assert scheduler.outbox.get(timeout=5).type == "result"
        earlier = payload_for(identity, "append", 0, 320)
        later = payload_for(identity, "append", 1, 0, eos=True)
        register = scheduler.inbox.register

        def delayed_register(message: IncomingMessage) -> None:
            register(message)
            if message.request_id == earlier.request_id:
                entered.set()
                assert proceed.wait(5)
            else:
                pass

        monkeypatch.setattr(scheduler.inbox, "register", delayed_register)
        sender = threading.Thread(target=submit, args=(scheduler, earlier))
        sender.start()
        try:
            assert entered.wait(5)
            submit(scheduler, later)
        finally:
            proceed.set()
            sender.join(5)
        messages = [scheduler.outbox.get(timeout=5) for _ in range(3)]
        assert [
            message.request_id for message in messages if message.type == "result"
        ] == [earlier.request_id, later.request_id]
        assert all(message.type != "error" for message in messages)


class PausedRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.entered: threading.Event = threading.Event()
        self.proceed: threading.Event = threading.Event()

    def run_streaming_batch(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        self.entered.set()
        assert self.proceed.wait(5)
        return super().run_streaming_batch(
            states,
            chunks,
            requested_languages=requested_languages,
            max_new_tokens=max_new_tokens,
        )


def test_scheduler_cancelled_append_and_queued_close_do_not_leak() -> None:
    runner = PausedRunner()
    engine = make_engine(runner, batch_size=1)
    first, healthy = SessionIdentity("cancelled"), SessionIdentity("healthy")
    with running_scheduler(engine) as scheduler:
        for identity in [first, healthy]:
            submit(scheduler, payload_for(identity, "open"))
        assert all(scheduler.outbox.get(timeout=5).type == "result" for _ in range(2))
        blocked = payload_for(first, "append")
        queued = payload_for(first, "append", 1)
        close = payload_for(first, "close")
        try:
            submit(scheduler, blocked)
            assert runner.entered.wait(5)
            submit(scheduler, queued)
            submit(scheduler, close)
            submit(scheduler, payload_for(healthy, "append", eos=True))
            for payload in [blocked, queued, close]:
                scheduler.abort(payload.request_id)
        finally:
            runner.proceed.set()
        messages = [scheduler.outbox.get(timeout=5) for _ in range(3)]
        assert all(message.type != "error" for message in messages)
        assert close.request_id in [message.request_id for message in messages]
        assert not {blocked.request_id, queued.request_id} & {
            message.request_id for message in messages
        }
        assert engine.usage(first) == ResourceUsage()
        reopened = SessionIdentity(first.id, first.open_index + 1)
        submit(scheduler, payload_for(reopened, "open"))
        assert scheduler.outbox.get(timeout=5).type == "result"
        submit(scheduler, payload_for(reopened, "append", eos=True))
        assert [scheduler.outbox.get(timeout=5).type for _ in range(2)] == [
            "stream",
            "result",
        ]


def test_scheduler_shutdown_waits_for_owner_before_releasing_busy_session() -> None:
    runner = PausedRunner()
    engine = make_engine(runner)
    scheduler = NemotronSessionScheduler(
        engine, max_concurrency=8, max_open_sessions=64, max_state_bytes=1 << 30
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    identity = SessionIdentity("shutdown")
    try:
        submit(scheduler, payload_for(identity, "open"))
        assert scheduler.outbox.get(timeout=5).type == "result"
        submit(scheduler, payload_for(identity, "append"))
        assert runner.entered.wait(5)
        submit(scheduler, payload_for(identity, "append", 1))
        scheduler.stop()
        assert thread.is_alive()
        assert engine.usage(identity).bytes > 0
    finally:
        runner.proceed.set()
        scheduler.stop()
        thread.join(5)
    assert not thread.is_alive() and not engine.thread.is_alive()
    assert engine.usage(identity) == ResourceUsage()
    assert runner.is_closed


def test_scheduler_budget_failure_settles_following_operations() -> None:
    runner = FakeRunner()
    engine = make_engine(runner, max_pcm_bytes=9000)
    bad, good = SessionIdentity("overflow"), SessionIdentity("good")
    with running_scheduler(engine) as scheduler:
        for identity in [bad, good]:
            submit(scheduler, payload_for(identity, "open"))
        assert all(scheduler.outbox.get(timeout=5).type == "result" for _ in range(2))
        submit(scheduler, payload_for(bad, "append", samples=5000))
        submit(scheduler, payload_for(bad, "append", 1))
        submit(scheduler, payload_for(bad, "close"))
        submit(scheduler, payload_for(good, "append", eos=True))
        messages = [scheduler.outbox.get(timeout=5) for _ in range(5)]
        errors = [message.data for message in messages if message.type == "error"]
        assert len(errors) == 2
        assert any("PCM budget" in str(error) for error in errors)
        assert any("unknown session" in str(error) for error in errors)
        assert engine.usage(bad) == ResourceUsage()
        submit(scheduler, payload_for(bad, "open"))
        assert scheduler.outbox.get(timeout=5).type == "result"
        submit(scheduler, payload_for(bad, "append", samples=4040, eos=True))
        assert [scheduler.outbox.get(timeout=5).type for _ in range(2)] == [
            "stream",
            "result",
        ]


def test_scheduler_mixes_offline_requests_and_streaming_on_one_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)
    owner_threads: list[int] = []

    def offline(requests: Sequence[Nemotron3_5ASRRequest]) -> list[StagePayload]:
        owner_threads.append(threading.get_ident())
        return [request.stage_payload for request in requests]

    def build_request(payload: StagePayload) -> Nemotron3_5ASRRequest:
        return Nemotron3_5ASRRequest(
            waveform=np.zeros(0, dtype=np.float32),
            duration_s=0,
            language="auto",
            stage_payload=payload,
        )

    monkeypatch.setattr(runner, "run_batch", offline)
    engine.build_request = build_request
    with running_scheduler(engine, concurrency=2) as scheduler:
        identities = [SessionIdentity("mixed-a"), SessionIdentity("mixed-b")]
        for identity in identities:
            submit(scheduler, payload_for(identity, "open"))
        assert all(scheduler.outbox.get(timeout=5).type == "result" for _ in identities)
        for identity in identities:
            submit(scheduler, payload_for(identity, "append", eos=True))
            submit(
                scheduler,
                StagePayload(identity.id + "-offline", OmniRequest(None), data=None),
            )
        messages = [scheduler.outbox.get(timeout=5) for _ in range(6)]
        assert sum(message.type == "result" for message in messages) == 4
        assert sum(message.type == "stream" for message in messages) == 2
        assert set(owner_threads) == {engine.thread.ident}
        for identity in identities:
            submit(
                scheduler,
                StagePayload(identity.id + "-offline", OmniRequest(None), data=None),
            )
        repeated_results = [scheduler.outbox.get(timeout=5) for _ in identities]
        assert all(message.type == "result" for message in repeated_results)
        assert {message.request_id for message in repeated_results} == {
            identity.id + "-offline" for identity in identities
        }


@pytest.mark.parametrize("concurrency,wait_ms", [(1, 0), (1, 2), (4, 2)])
def test_short_inputs_long_append_and_empty_eos(
    concurrency: int, wait_ms: float
) -> None:
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
        assert (
            engine.usage(identity).slots["pcm_bytes"]
            < 2 * engine.spec.subsequent_samples
        )
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
            first = engine.append(
                identity, make_chunk(0, samples), payload, threading.Event()
            ).result(5)
            before_eos = len(runner.batches)
            final = engine.append(
                identity, make_chunk(1, 0, eos=True), payload, threading.Event()
            ).result(5)
            assert isinstance(first, AppendResult) and isinstance(final, AppendResult)
            assert final.is_final
            assert len(runner.batches) == before_eos
            assert not runner.encoder_state_pool.active_slots
            assert engine.usage(identity).slots["cache_bytes"] == 0
            with pytest.raises(ValueError, match="ended"):
                engine.append(
                    identity, make_chunk(2), payload, threading.Event()
                ).result(5)
            engine.close(identity).result(5)
            engine.close(identity).result(5)
        assert len(runner.batches) == 1
    finally:
        engine.shutdown()


def test_prefix_failure_and_inflight_cancel_leave_other_lane_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)
    identities = [SessionIdentity("bad"), SessionIdentity("good")]
    entered, proceed = threading.Event(), threading.Event()
    original = runner.run_streaming_batch

    def blocked(
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        result = original(
            states,
            chunks,
            requested_languages=requested_languages,
            max_new_tokens=max_new_tokens,
        )
        entered.set()
        assert proceed.wait(5)
        result.clean_texts[0] = "changed"
        return result

    try:
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        first = [
            engine.append(
                identity,
                make_chunk(0),
                payload_for(identity, "append"),
                threading.Event(),
            )
            for identity in identities
        ]
        for future in first:
            future.result(5)
        monkeypatch.setattr(runner, "run_streaming_batch", blocked)
        second = [
            engine.append(
                identity,
                make_chunk(1, 5224),
                payload_for(identity, "append"),
                threading.Event(),
            )
            for identity in identities
        ]
        assert entered.wait(5)
        proceed.set()
        with pytest.raises(RuntimeError, match="prefix"):
            second[0].result(5)
        assert second[1].result(5).full_text == "word more"
        assert identities[0] not in engine.states
        assert len(runner.encoder_state_pool.active_slots) == 1
        monkeypatch.setattr(runner, "run_streaming_batch", original)
        engine.close(identities[1]).result(5)
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        entered.clear()
        proceed.clear()
        monkeypatch.setattr(runner, "run_streaming_batch", blocked)
        cancelled = threading.Event()
        futures = [
            engine.append(
                identities[0],
                make_chunk(0),
                payload_for(identities[0], "append"),
                cancelled,
            ),
            engine.append(
                identities[1],
                make_chunk(0),
                payload_for(identities[1], "append"),
                threading.Event(),
            ),
        ]
        assert entered.wait(5)
        cancelled.set()
        assert len(runner.encoder_state_pool.active_slots) == 2
        proceed.set()
        with pytest.raises(RuntimeError, match="closed|cancelled"):
            futures[0].result(5)
        assert futures[1].result(5).full_text == "word"
        assert len(runner.encoder_state_pool.active_slots) == 1
    finally:
        proceed.set()
        engine.shutdown()


def test_shutdown_releases_waiting_tickets_before_forward_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner)
    entered, proceed = threading.Event(), threading.Event()
    original = runner.run_streaming_batch

    def blocked(
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        entered.set()
        assert proceed.wait(5)
        return original(
            states,
            chunks,
            requested_languages=requested_languages,
            max_new_tokens=max_new_tokens,
        )

    monkeypatch.setattr(runner, "run_streaming_batch", blocked)
    identity = SessionIdentity("stop")
    engine.open(identity, OmniRequest(None)).result(5)
    future = engine.append(
        identity, make_chunk(0), payload_for(identity, "append"), threading.Event()
    )
    assert entered.wait(5)
    engine.begin_shutdown()
    with pytest.raises(RuntimeError, match="stopping"):
        future.result(1)
    proceed.set()
    engine.shutdown()
    assert not engine.thread.is_alive() and not engine.tasks and not engine.states
    assert not runner.encoder_state_pool.active_slots
    assert runner.encoder_state_pool.nbytes == 0


def test_budget_failure_and_decode_limit_do_not_accumulate_pcm() -> None:
    runner = FakeRunner()
    engine = make_engine(runner, max_pcm_bytes=9000)
    identity = SessionIdentity("limited")
    try:
        engine.open(identity, OmniRequest(None, {"max_new_tokens": 1})).result(5)
        payload = payload_for(identity, "append")
        engine.append(identity, make_chunk(0), payload, threading.Event()).result(5)
        for sequence in range(1, 10):
            engine.append(
                identity, make_chunk(sequence, 10000), payload, threading.Event()
            ).result(5)
        assert engine.usage(identity).slots["pcm_bytes"] == 0
        assert len(runner.batches) == 1
        assert not runner.encoder_state_pool.active_slots
        result = engine.append(
            identity, make_chunk(10, 0, eos=True), payload, threading.Event()
        ).result(5)
        assert result.final_payload.data["duration_s"] == pytest.approx(94040 / 16000)
        other = SessionIdentity("overflow")
        engine.open(other, OmniRequest(None)).result(5)
        with pytest.raises(RuntimeError, match="PCM budget"):
            engine.append(
                other, make_chunk(0, 5000), payload, threading.Event()
            ).result(5)
        assert other not in engine.states
        assert not runner.encoder_state_pool.active_slots
    finally:
        engine.shutdown()


def test_offline_batch_and_stream_use_one_owner_and_cancel_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)
    calls = []
    owner_threads = []
    original = runner.run_streaming_batch

    def stream(
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        owner_threads.append(threading.get_ident())
        return original(
            states,
            chunks,
            requested_languages=requested_languages,
            max_new_tokens=max_new_tokens,
        )

    def offline(requests: Sequence[Nemotron3_5ASRRequest]) -> list[StagePayload]:
        owner_threads.append(threading.get_ident())
        calls.append([request.stage_payload.request_id for request in requests])
        return [request.stage_payload for request in requests]

    monkeypatch.setattr(runner, "run_streaming_batch", stream)
    monkeypatch.setattr(runner, "run_batch", offline)

    def build_request(payload: StagePayload) -> Nemotron3_5ASRRequest:
        return Nemotron3_5ASRRequest(
            waveform=np.zeros(0, dtype=np.float32),
            duration_s=0,
            language="auto",
            stage_payload=payload,
        )

    engine.build_request = build_request
    try:
        identities = [SessionIdentity("a"), SessionIdentity("b")]
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        cancelled = threading.Event()
        cancelled.set()
        rejected = engine.submit_offline(
            payload_for(SessionIdentity("cancelled"), "open"), cancelled
        )
        with pytest.raises(RuntimeError, match="cancelled"):
            rejected.result(5)
        with engine.condition:
            jobs = [
                engine.append(
                    identity,
                    make_chunk(0),
                    payload_for(identity, "append"),
                    threading.Event(),
                )
                for identity in identities
            ]
            jobs += [
                engine.submit_offline(payload_for(identity, "open"), threading.Event())
                for identity in identities
            ]
        for future in jobs:
            future.result(5)
        assert calls == [["a:1:open:0", "b:1:open:0"]]
        assert [len(batch) for batch in runner.batches] == [2]
        assert set(owner_threads) == {engine.thread.ident}
    finally:
        engine.shutdown()


def test_fatal_owner_error_settles_every_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    engine = make_engine(runner, batch_size=2, wait_ms=10000)

    def fatal(
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        raise SystemExit("fatal model error")

    monkeypatch.setattr(runner, "run_streaming_batch", fatal)
    try:
        identities = [SessionIdentity("a"), SessionIdentity("b")]
        for identity in identities:
            engine.open(identity, OmniRequest(None)).result(5)
        futures = [
            engine.append(
                identity,
                make_chunk(0),
                payload_for(identity, "append"),
                threading.Event(),
            )
            for identity in identities
        ]
        for future in futures:
            with pytest.raises(RuntimeError, match="fatal model error"):
                future.result(5)
        with pytest.raises(RuntimeError, match="failed"):
            engine.open(SessionIdentity("later"), OmniRequest(None)).result(5)
    finally:
        engine.shutdown()
    assert not engine.tasks and not engine.thread.is_alive()
    assert not runner.encoder_state_pool.active_slots
    assert runner.encoder_state_pool.nbytes == 0
