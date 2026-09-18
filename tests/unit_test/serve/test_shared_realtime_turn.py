# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextvars
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
from sglang_omni.serve.realtime.control import Cancelled, Closed, Failure, UnitCompleted
from sglang_omni.serve.realtime.output import TextDelta
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    RuntimeLimits,
    SessionRuntime,
)
from sglang_omni.serve.realtime.vad import VADEvent
from tests.unit_test.fixtures.realtime_websocket import collect_events, next_event


@pytest.mark.asyncio
async def test_shared_vad_server_cancel_uses_runtime_epoch(monkeypatch):

    class Detector:
        def process(self, pcm):
            return [
                SimpleNamespace(event_type=VADEvent.SPEECH_STARTED, sample_offset=0)
            ]

        def reset(self):
            pass

    monkeypatch.setattr(
        "sglang_omni.serve.realtime.session.build_turn_detector",
        lambda config, model: SimpleNamespace(
            detector=Detector(),
            effective_config={"type": "server_vad", "interrupt_response": True},
        ),
    )
    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(object(), "mock"),
        RuntimeLimits(),
    )
    await runtime.update(
        {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}, "open"
    )
    engine = runtime.adapter.engine
    engine.active_response_has_audio = True
    engine.cancel_active_response = AsyncMock()
    await runtime.append(b"\0\0" * 320, 0, 0, "append")
    outputs = runtime.outputs()
    event = (await next_event(outputs, Cancelled)).event
    assert event.client_event_id is None and runtime.epoch == 1
    engine.cancel_active_response.assert_not_awaited()
    await outputs.aclose()
    await runtime.close("client_closed")


@pytest.mark.asyncio
async def test_queued_turn_starts_in_new_epoch_after_cancel():

    class Client:
        async def completion_stream(self, request, **kwargs):
            yield SimpleNamespace(
                modality="text", text="new turn", finish_reason="stop", usage=None
            )

        async def abort(self, request_id):
            pass

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    engine = runtime.adapter.engine
    engine.audio_buffer.append_bytes(b"\0\0" * 80)
    payload = engine.audio_buffer.to_full_wav_data_uri()
    engine.speech_idle.clear()
    engine.queued_audio_bytes = len(payload)
    await engine.response_queue.put(("item", payload, contextvars.copy_context()))
    engine.queue_drainer = asyncio.create_task(engine.drain_queue())
    await runtime.cancel("cancel")
    engine.speech_idle.set()
    outputs = runtime.outputs()
    assert (await next_event(outputs, TextDelta)).epoch == 1
    await outputs.aclose()
    await runtime.close("client_closed")


@pytest.mark.asyncio
async def test_turn_text_limit_closes_before_unit_completion():

    class Client:
        calls = 0

        async def completion_stream(self, request, **kwargs):
            self.calls += 1
            text = "x" * 17 if self.calls == 1 else "ok"
            yield SimpleNamespace(
                modality="text", text=text, finish_reason="stop", usage=None
            )

        async def abort(self, *args):
            pass

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(max_history_chars=16, max_responses=1, cleanup_timeout_s=0.1),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    await runtime.append(b"\0\0" * 80, 0, 0, "append")
    await runtime.end("end")
    events = await collect_events(runtime)
    assert any(
        isinstance(e, Failure) and e.code == "context_limit" and e.fatal for e in events
    )
    assert any(isinstance(e, Closed) and e.reason == "context_limit" for e in events)
    assert not any(isinstance(e, UnitCompleted) for e in events)
    assert runtime.worker.done()
    engine = runtime.adapter.engine
    assert engine.queue_drainer is None or engine.queue_drainer.done()
    await runtime.close("client_closed")


@pytest.mark.asyncio
async def test_turn_abort_failure_stops_drainer_response_and_runtime_tasks():
    entered = asyncio.Event()

    class Client:
        async def completion_stream(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
            if False:
                yield None

        async def abort(self, *args):
            raise RuntimeError("engine release unacknowledged")

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(cleanup_timeout_s=0.1),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    await runtime.append(b"\0\0" * 80, 0, 0, "append")
    await runtime.end("end")
    await entered.wait()
    engine = runtime.adapter.engine
    tasks = [
        runtime.worker,
        engine.queue_drainer,
        engine.active_task,
        engine.active_response_task,
    ]
    await runtime.close("client_closed")
    assert all(task.done() for task in tasks if task is not None)
    events = await collect_events(runtime)
    assert any(isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events)
    assert not any(isinstance(e, Closed) for e in events)


@pytest.mark.asyncio
async def test_turn_adapter_cancel_waits_for_history_without_aborting():
    release = asyncio.Event()
    history = []

    async def turn():
        await release.wait()
        history.append("completed reply")

    adapter = TurnBasedAdapterFactory(object(), "mock")()
    active = asyncio.create_task(turn())
    abort = AsyncMock()
    adapter.engine = SimpleNamespace(active_task=active, cancel_active_response=abort)
    cancel = asyncio.create_task(adapter.cancel())
    for _ in range(10):
        await asyncio.sleep(0)
    assert adapter.output_epoch == 1 and not cancel.done()
    assert not active.cancelled()
    release.set()
    await cancel
    assert history == ["completed reply"]
    abort.assert_not_awaited()
