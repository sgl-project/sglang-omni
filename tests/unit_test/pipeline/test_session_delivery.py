# SPDX-License-Identifier: Apache-2.0
"""A Session delivers reply audio to its client paced in real time from the reply's start."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

import pytest

from sglang_omni.models.qwen3_omni.session_node import SESSION_PORTS as QWEN3_OMNI_PORTS
from sglang_omni.pipeline.session_delivery import DeliveryPacer
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    ControlEdge,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionNode,
    TimedChunk,
)
from sglang_omni.proto.streams import PCM_F32, control_type
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.session_bridge import SessionBridge
from sglang_omni.serve.realtime.types import RuntimeLimits
from tests.unit_test.fixtures.session_client import FakeSessionClient
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import (
    chunk,
    event_log,
    pipeline,
    wait_until,
)

LEAD_MS = 250
OUTPUT_TIMEOUT_S = 10
IDENTITY = SessionIdentity("session", 1)


def output(
    t_start_ms: float,
    duration_ms: float,
    payload: object,
    format: str,
    *,
    modality: str = "audio",
    turn: int | None = 1,
) -> OutputChunk:
    return OutputChunk(
        IDENTITY, 0, 0, modality, t_start_ms, duration_ms, payload, format, turn=turn
    )


def control(kind: str, t_start_ms: float = 0, turn: int | None = 1) -> OutputChunk:
    return output(
        t_start_ms, 0, {"type": kind}, "session", modality="control", turn=turn
    )


def audio(t_start_ms: float, turn: int = 1) -> OutputChunk:
    return output(t_start_ms, 100, bytes(3200), "pcm16", turn=turn)


def labels(released: list[tuple[OutputChunk, int]]) -> list[str]:
    return [control_type(item) or item.format for item, _ in released]


def confirmation(seq: int, t_start_ms: float) -> OutputChunk:
    """The input_done of a 200 ms pcm16 input append, as a client route emits it."""
    return replace(
        output(t_start_ms, 200, None, "pcm16", turn=None),
        input_seq=seq,
        kind="input_done",
    )


@pytest.fixture
def pacer() -> DeliveryPacer:
    return DeliveryPacer(LEAD_MS, asyncio.Event())


def offer_reply(
    pacer: DeliveryPacer, count: int, *, first: int = 0, turn: int = 1, done=True
) -> None:
    """turn.started, `count` 100 ms audio chunks from chunk `first`, then turn.done when `done`."""
    pacer.offer(control("turn.started", first * 100, turn=turn), 10)
    for index in range(first, first + count):
        pacer.offer(audio(index * 100, turn=turn), 3300)
    if done:
        pacer.offer(control("turn.done", (first + count) * 100, turn=turn), 10)
    else:
        pass


def test_audio_leaves_within_the_lead_of_the_wall_clock_since_the_reply_started(pacer):
    offer_reply(pacer, 20)
    pacer.offer(output(0, 0, {"text": "hi"}, "text_delta", modality="text"), 20)
    first = pacer.release(100.0)
    assert labels(first) == ["turn.started", "pcm16", "text_delta"]
    head = first[1][0]
    assert (head.t_start_ms, head.duration_ms, len(head.payload)) == (0, 250, 8000)
    second = pacer.release(100.5)
    assert second[0][0].t_start_ms == 250
    assert second[-1][0].t_start_ms + second[-1][0].duration_ms == 750
    assert "turn.done" not in labels(second)
    last = pacer.release(102.0)
    assert labels(last)[-1] == "turn.done"
    released = first + second + last
    assert not pacer.items
    assert (
        sum(item.duration_ms for item, _ in released if item.modality == "audio")
        == 2000
    )
    assert sum(size for _, size in released) == 10 + 20 * 3300 + 10 + 20


def test_audio_that_comes_after_its_turn_started_leaves_within_the_lead_of_its_first_release(
    pacer,
):
    pacer.offer(control("turn.started"), 10)
    assert labels(pacer.release(100.0)) == ["turn.started"]
    for index in range(20):
        pacer.offer(audio(index * 100), 3300)
    first = pacer.release(100.6)
    assert sum(item.duration_ms for item, _ in first) == LEAD_MS
    later = pacer.release(101.1)
    assert later[-1][0].t_start_ms + later[-1][0].duration_ms == 500 + LEAD_MS
    pacer.release(103.0)
    pacer.offer(control("turn.started", 2000, turn=2), 10)
    pacer.release(103.0)
    pacer.offer(audio(2000, turn=2), 3300)
    pacer.offer(audio(2100, turn=2), 3300)
    pacer.offer(audio(2200, turn=2), 3300)
    second = pacer.release(103.4)
    assert sum(item.duration_ms for item, _ in second) == LEAD_MS


def test_a_turns_done_waits_until_the_client_has_played_its_audio(pacer):
    """A turn.done left with the turn's last audio, so a reply the client was still playing could no longer be interrupted."""
    offer_reply(pacer, 5)
    assert "turn.done" not in labels(pacer.release(100.0))
    assert labels(pacer.release(100.25)) == ["pcm16"]
    assert labels(list(pacer.items)) == ["turn.done"]
    assert pacer.release(100.4) == []
    assert labels(pacer.release(100.5)) == ["turn.done"]


def test_an_interruption_drops_what_its_turn_still_holds(pacer):
    offer_reply(pacer, 10)
    pacer.release(100.0)
    held = sum(size for _, size in pacer.items)
    assert pacer.purge(None) == 0
    assert pacer.offer(control("turn.interrupted"), 10) == held
    assert labels(pacer.release(100.0)) == ["turn.interrupted"]
    offer_reply(pacer, 10, first=10, turn=2, done=False)
    pacer.release(101.0)
    pacer.unsent.extend(pacer.release(101.5))
    held = sum(size for _, size in pacer.items + pacer.unsent)
    assert pacer.unsent and held > 0 and pacer.purge(2) == held
    assert not pacer.items and not pacer.unsent


def test_an_input_confirmation_passes_the_held_audio_and_the_input_end_releases_it(
    pacer,
):
    offer_reply(pacer, 20)
    pacer.offer(confirmation(7, 2400), 30)
    released = pacer.release(100.0)
    assert released[-1] == (confirmation(7, 2400), 30)
    assert "turn.done" not in labels(released)
    assert [item.kind for item, _ in pacer.items] == ["data"] * 19
    pacer.end_input()
    assert labels(pacer.release(100.0))[-1] == "turn.done"
    assert not pacer.items


def test_a_horizon_below_one_sample_holds_the_whole_chunk(pacer):
    pacer.offer(control("turn.started"), 10)
    pacer.offer(output(0, LEAD_MS, bytes(LEAD_MS * 32), "pcm16"), 8300)
    pacer.offer(audio(LEAD_MS), 3300)
    assert labels(pacer.release(100.0)) == ["turn.started", "pcm16"]
    assert pacer.release(100.00001) == []
    assert [item.t_start_ms for item, _ in pacer.items] == [LEAD_MS]
    ((head, _),) = pacer.release(100.001)
    assert (head.duration_ms, len(head.payload)) == (1, 32)


QWEN_CHUNK_BYTES = 6400
QWEN_CHUNK_MS = QWEN_CHUNK_BYTES / 4 / 24
QWEN_LEAD_MS = 1000


def held_qwen_reply(count: int) -> DeliveryPacer:
    """A pacer holding a Qwen3-Omni reply of `count` 67 ms float chunks behind the first lead, its turn.done after them."""
    pacer = DeliveryPacer(QWEN_LEAD_MS, asyncio.Event())
    pacer.offer(control("turn.started"), 10)
    for index in range(count):
        pacer.offer(
            output(
                index * QWEN_CHUNK_MS,
                QWEN_CHUNK_MS,
                bytes([index % 255]) * QWEN_CHUNK_BYTES,
                PCM_F32,
            ),
            3300,
        )
    pacer.offer(control("turn.done", count * QWEN_CHUNK_MS), 10)
    return pacer


@pytest.mark.parametrize("release_before_eos", [False, True])
def test_input_end_coalesces_audio_within_realtime_budgets(release_before_eos):
    pacer = held_qwen_reply(600)
    released = pacer.release(100.0) if release_before_eos else []
    pacer.end_input()
    released += pacer.release(100.0)
    audio = [item for item, _ in released if item.modality == "audio"]
    assert len(audio) <= 600 * QWEN_CHUNK_MS / QWEN_LEAD_MS + 2
    assert all(item.duration_ms <= QWEN_LEAD_MS for item in audio)
    assert all(item.turn == 1 for item in audio)
    assert all(
        previous.t_start_ms + previous.duration_ms == pytest.approx(item.t_start_ms)
        for previous, item in zip(audio, audio[1:])
    )
    assert sum(item.duration_ms for item in audio) == pytest.approx(600 * QWEN_CHUNK_MS)
    assert b"".join(item.payload for item in audio) == b"".join(
        bytes([index % 255]) * QWEN_CHUNK_BYTES for index in range(600)
    )
    assert labels(released)[-1] == "turn.done"
    assert sum(size for _, size in released) == 10 + 600 * 3300 + 10
    assert not pacer.items

    bridge = SessionBridge(FakeSessionClient(), QWEN3_OMNI_PORTS)
    limits = RuntimeLimits()
    buffer = OutputBuffer(limits)
    for item, _ in released:
        for event in bridge.project(item):
            buffer.emit(event, None, ("text", "audio"))
    assert 600 * QWEN_CHUNK_MS == pytest.approx(40000)
    assert len(buffer.queued_envelopes) <= 40 + 4
    assert buffer.queued_bytes < limits.max_output_bytes


@pytest.mark.asyncio
async def test_the_pacer_wakes_when_the_held_audio_may_leave():
    pacer = DeliveryPacer(50, asyncio.Event())
    loop = asyncio.get_running_loop()
    pacer.offer(control("turn.started"), 10)
    pacer.offer(output(0, 200, bytes(6400), "pcm16"), 6600)
    started_at = loop.time()
    assert labels(pacer.release(started_at)) == ["turn.started", "pcm16"]
    await asyncio.wait_for(pacer.wait(), 1)
    waited_s = loop.time() - started_at
    assert 0.03 <= waited_s < 0.5
    (slice_,) = pacer.release(loop.time())
    assert slice_[0].t_start_ms == 50 and slice_[0].duration_ms >= 40


async def receive_reply(outputs) -> list[tuple[OutputChunk, float]]:
    """Data outputs with their arrival time, through the reply's turn.done or turn.interrupted."""
    received: list[tuple[OutputChunk, float]] = []
    while not received or control_type(received[-1][0]) not in (
        "turn.done",
        "turn.interrupted",
    ):
        item = await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
        if item.kind == "data":
            received.append((item, time.monotonic()))
        else:
            pass
    return received


async def receive_until(
    outputs: AsyncIterator[OutputChunk], matches: Callable[[OutputChunk], bool]
) -> list[OutputChunk]:
    """Every output through the first one that `matches`."""
    received = [await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)]
    while not matches(received[-1]):
        received.append(await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S))
    return received


def is_confirmation_of(seq: int) -> Callable[[OutputChunk], bool]:
    return lambda item: item.kind == "input_done" and item.input_seq == seq


def is_audio(item: OutputChunk) -> bool:
    return item.kind == "data" and item.modality == "audio"


def assert_paced(received: list[tuple[OutputChunk, float]], audio_ms: float) -> None:
    """The client never holds more than the lead of the reply, counted from its first audio, when it starts playing."""
    audio = [(item, at) for item, at in received if item.modality == "audio"]
    first, first_at = audio[0]
    assert sum(item.duration_ms for item, _ in audio) == pytest.approx(audio_ms)
    assert all(
        item.t_start_ms + item.duration_ms - first.t_start_ms
        <= (at - first_at) * 1000 + LEAD_MS + 50
        for item, at in audio
    )
    assert audio[-1][1] - first_at >= (audio_ms - LEAD_MS) / 1000 - 0.1
    assert control_type(received[-1][0]) == "turn.done"


async def wait_for_turn_done_returned(events) -> None:
    """Both stages of the client route get the delivered reply's turn.done back."""
    log: list = []

    def returned() -> bool:
        log.extend(event_log(events))
        return {(event[1], event[3]) for event in log if event[0] == "control"} == {
            ("source", "turn.done"),
            ("sink", "turn.done"),
        }

    await wait_until(returned)


def input_chunk(seq: int, eos: bool = False) -> TimedChunk:
    """A 20 ms pcm16 input, as the realtime bridge appends it."""
    return TimedChunk("audio", seq * 20, 20, seq, bytes(640), format="pcm16", eos=eos)


async def open_client_route(coordinator, reply_ms: float):
    coordinator.max_output_lead_ms = LEAD_MS
    identity = await coordinator.open_session(
        OmniRequest(None, {"emit_reply_ms": reply_ms})
    )
    return identity, coordinator.session_outputs(identity)


@pytest.mark.asyncio
async def test_a_client_route_paces_the_reply_without_appends_and_returns_its_turn_done(
    tmp_path,
):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity, outputs = await open_client_route(coordinator, 1000)
        await coordinator.append_session(identity, chunk(0))
        assert_paced(await receive_reply(outputs), 1000)
        await wait_for_turn_done_returned(events)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions)


@pytest.mark.asyncio
async def test_a_returned_turn_done_does_not_wait_behind_the_append_in_flight(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, _):
        coordinator.max_output_lead_ms = LEAD_MS
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_reply_ms": 2000, "delay": 3})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await receive_reply(outputs)
        assert control_type(received[-1][0]) == "turn.done"
        done_at = time.monotonic()
        await wait_for_turn_done_returned(events)
        assert time.monotonic() - done_at < 1.0
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions)


@pytest.mark.asyncio
async def test_a_client_route_confirms_input_while_audio_is_held_and_flushes_at_input_end(
    tmp_path,
):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity, outputs = await open_client_route(coordinator, 3000)
        await coordinator.append_session(identity, input_chunk(0))
        held = await receive_until(outputs, is_audio)
        appended_at = time.monotonic()
        await coordinator.append_session(identity, input_chunk(1))
        held += await receive_until(outputs, is_confirmation_of(1))
        assert time.monotonic() - appended_at < 0.5
        await coordinator.append_session(identity, input_chunk(2, eos=True))
        ended_at = time.monotonic()
        received = await receive_reply(outputs)
        assert time.monotonic() - ended_at < 1.0
        assert control_type(received[-1][0]) == "turn.done"
        delivered = held + [item for item, _ in received]
        assert sum(
            item.duration_ms for item in delivered if is_audio(item)
        ) == pytest.approx(3000)
        await receive_until(outputs, is_confirmation_of(2))
        await wait_for_turn_done_returned(events)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions)


async def open_reply_graph(
    coordinator, reply_ms: float, reply_after_s: float = 0
) -> SessionIdentity:
    """A graph of one node over the route that answers its first input with a whole reply at once, like the Qwen3-Omni node."""
    coordinator.session_route = None
    coordinator.session_graph = SessionGraph(
        nodes={
            "model": SessionNode(
                stages=("source", "sink"),
                inputs=("input",),
                controls=(ControlEdge("input", "interrupt"),),
            )
        },
        publish={"model": ("pcm16", "session")},
    )
    coordinator.max_output_lead_ms = LEAD_MS
    return await coordinator.open_session(
        OmniRequest(
            None, {"emit_reply_ms": reply_ms, "emit_reply_after": reply_after_s}
        )
    )


@pytest.mark.asyncio
async def test_a_graph_node_publishing_audio_is_paced_and_gets_its_turn_done_back(
    tmp_path,
):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity = await open_reply_graph(coordinator, 1000)
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        assert_paced(await receive_reply(outputs), 1000)
        await wait_for_turn_done_returned(events)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_a_graph_reply_whose_audio_comes_late_is_paced_from_its_first_audio(
    tmp_path,
):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity = await open_reply_graph(coordinator, 1500, reply_after_s=0.5)
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        assert_paced(await receive_reply(outputs), 1500)
        await wait_for_turn_done_returned(events)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_an_interrupt_drops_the_held_audio_of_its_turn(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity = await open_reply_graph(coordinator, 3000)
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        held = await receive_until(outputs, is_audio)
        await coordinator.control_session(identity, control_chunk("interrupt", turn=1))
        received = await receive_reply(outputs)
        assert control_type(received[-1][0]) == "turn.interrupted"
        assert "turn.done" not in [control_type(item) for item, _ in received]
        delivered = held + [item for item, _ in received]
        assert sum(item.duration_ms for item in delivered if is_audio(item)) < 1000
        controls: list = []

        def interrupted() -> bool:
            controls.extend(
                event for event in event_log(events) if event[0] == "control"
            )
            return len(controls) >= 2

        await wait_until(interrupted)
        assert [(event[1], event[3]) for event in controls] == [
            ("source", "interrupt"),
            ("sink", "interrupt"),
        ]
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
