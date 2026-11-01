# SPDX-License-Identifier: Apache-2.0
"""Real coordinator and SessionScheduler routes: the dialog's turns and a synthetic model node that answers them."""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import AsyncIterator
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Protocol

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ControlEdge,
    DataEdge,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionLimits,
    SessionNode,
    TimedChunk,
    wire_size,
)
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    CONTROL,
    INTERRUPT,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    TEXT,
    TEXT_DELTA,
    TURN_DONE,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from tests.unit_test.fixtures.session_pipeline import StageEvent, pipeline, wait_until


class SyntheticModel(SessionHooks):
    """Answers a reply unit with turn.started, a text delta naming the turns it asks again and the configured sampling, audio after audio_after_s and turn.done, as the route's terminal owner does."""

    def __init__(self) -> None:
        self.audio_after_s: dict[SessionIdentity, float] = {}
        self.sampling: dict[SessionIdentity, dict[str, object]] = {}

    def open(self, session_identity, request) -> None:
        self.audio_after_s[session_identity] = float(
            request.params.get("audio_after_s", 0)
        )

    def close(self, session_identity) -> None:
        self.audio_after_s.pop(session_identity)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        if control_type(event) == CONFIGURE:
            self.sampling[session_identity] = dict(event.payload["sampling"])
        else:
            pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        payload.data = {}
        if chunk.format != "qwen.unit" or context.cancelled.wait(0.02):
            return payload
        else:
            pass
        settings = "".join(
            f" {name}={value}"
            for name, value in self.sampling.get(context.session_identity, {}).items()
        )
        carried_turns = chunk.payload["carried_turns"]
        text = (
            f"reply {chunk.turn}"
            + (f" after {carried_turns}" if carried_turns else "")
            + settings
        )
        for event in (
            TimedChunk(
                CONTROL,
                0,
                0,
                chunk.seq,
                {"type": TURN_STARTED},
                format=SESSION_CONTROL,
                turn=chunk.turn,
            ),
            TimedChunk(
                TEXT,
                0,
                0,
                chunk.seq,
                {"text": text},
                format=TEXT_DELTA,
                turn=chunk.turn,
            ),
        ):
            context.emit(event)
        if context.cancelled.wait(self.audio_after_s[context.session_identity]):
            return payload
        else:
            context.emit(
                TimedChunk(
                    AUDIO, 0, 10, chunk.seq, bytes(960), format=PCM_F32, turn=chunk.turn
                )
            )
        # The reply's audio keeps playing for a while; an interrupt in that window cancels its turn.done.
        if context.cancelled.wait(0.5):
            return payload
        else:
            context.emit(
                TimedChunk(
                    CONTROL,
                    0,
                    0,
                    chunk.seq,
                    {"type": TURN_DONE},
                    format=SESSION_CONTROL,
                    turn=chunk.turn,
                )
            )
        return payload


class UtteranceDigestModel(SessionHooks):
    """Answers a reply unit with the length and digest of the utterance audio it received."""

    def open(self, session_identity, request) -> None:
        pass

    def close(self, session_identity) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        payload.data = {}
        for event in (
            TimedChunk(
                CONTROL,
                0,
                0,
                chunk.seq,
                {"type": TURN_STARTED},
                format=SESSION_CONTROL,
                turn=chunk.turn,
            ),
            TimedChunk(
                TEXT,
                0,
                0,
                chunk.seq,
                {"text": digest(chunk.payload["pcm"])},
                format=TEXT_DELTA,
                turn=chunk.turn,
            ),
            TimedChunk(
                CONTROL,
                0,
                0,
                chunk.seq,
                {"type": TURN_DONE},
                format=SESSION_CONTROL,
                turn=chunk.turn,
            ),
        ):
            context.emit(event)
        return payload


def create_long_utterance_stage(
    name: str, events: Queue[StageEvent]
) -> SessionScheduler:
    if name == "dialog":
        return SessionScheduler(
            QwenInputHooks(
                max_utterance_seconds=120, max_queued_turns=2, max_utterance_frames=4
            )
        )
    else:
        return SessionScheduler(UtteranceDigestModel())


def create_graph_stage(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    if name == "dialog":
        return SessionScheduler(
            QwenInputHooks(
                max_utterance_seconds=1, max_queued_turns=2, max_utterance_frames=4
            )
        )
    else:
        return SessionScheduler(SyntheticModel())


def session_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "qwen_omni": SessionNode(
                ("model",),
                ("dialog:qwen.unit",),
                controls=(
                    ControlEdge("input", CONFIGURE),
                    ControlEdge("dialog", CONFIGURE),
                    ControlEdge("dialog", INTERRUPT, should_preempt=True),
                    ControlEdge("dialog", "qwen.truncate"),
                ),
            ),
            "dialog": SessionNode(
                ("dialog",),
                (
                    "input:pcm16",
                    "input:session",
                    DataEdge(
                        "qwen_omni:session", feedback=True, overflow="drop_oldest"
                    ),
                ),
                controls=(ControlEdge("input", INTERRUPT, should_preempt=True),),
            ),
        },
        publish={
            "dialog": (SESSION_CONTROL,),
            "qwen_omni": (PCM_F32, SESSION_CONTROL, TEXT_DELTA),
        },
    )


def audio_chunk(sequence: int, *, eos: bool = False) -> TimedChunk:
    return TimedChunk(
        "audio", sequence * 20, 20, sequence, b"\x01\x00" * 320, format=PCM16, eos=eos
    )


def control_chunk(
    event: str, *, turn: int | None = None, sequence: int = 0, **fields: object
) -> TimedChunk:
    return TimedChunk(
        "control",
        0,
        0,
        sequence,
        {"type": event, **fields},
        format=SESSION_CONTROL,
        turn=turn,
    )


def digest(pcm: bytes) -> str:
    return f"{len(pcm)} {hashlib.sha256(pcm).hexdigest()}"


def utterance(size: int) -> bytes:
    return random.Random(size).randbytes(size)


# note (luojiaxuan): 16 kHz PCM16 is 32 bytes per ms; the client sends an utterance in pieces of at most this many bytes.
PCM16_BYTES_PER_MS = 32
CLIENT_CHUNK_BYTES = 65536


async def append_when_taken(
    coordinator, identity: SessionIdentity, chunk: TimedChunk
) -> None:
    """Append as a client does: a full edge keeps the chunk's seq, so retry it."""
    while True:
        try:
            await coordinator.append_session(identity, chunk)
            return
        except QueueFullError:
            await asyncio.sleep(0.01)


async def say(
    coordinator,
    identity: SessionIdentity,
    pcm: bytes,
    *,
    sequence: int,
    start_ms: float,
) -> tuple[int, float]:
    """Append `pcm` as client audio and commit it as one turn; returns the next input seq and the audio end."""
    for offset in range(0, len(pcm), CLIENT_CHUNK_BYTES):
        piece = pcm[offset : offset + CLIENT_CHUNK_BYTES]
        duration_ms = len(piece) / PCM16_BYTES_PER_MS
        await append_when_taken(
            coordinator,
            identity,
            TimedChunk(AUDIO, start_ms, duration_ms, sequence, piece, format=PCM16),
        )
        sequence += 1
        start_ms += duration_ms
    await append_when_taken(
        coordinator, identity, control_chunk(COMMIT, sequence=sequence)
    )
    return sequence + 1, start_ms


def unit_bytes(pcm: bytes, end_ms: float) -> int:
    """The wire size of the dialog's unit for a first turn of `pcm` ending at `end_ms`, as the model node admits it."""
    unit = TimedChunk(
        CONTROL,
        end_ms,
        0.0,
        0,
        {"pcm": pcm, "images": [], "carried_turns": []},
        format=UNIT_FORMAT,
        turn=1,
    )
    return wire_size(unit.to_dict())


def reply_text(received: list[OutputChunk], turn: int) -> str:
    return "".join(
        output.payload["text"]
        for output in received
        if output.format == TEXT_DELTA and output.turn == turn
    )


class OutputPredicate(Protocol):
    def __call__(self, output: OutputChunk) -> bool: ...


async def read_until(
    outputs: AsyncIterator[OutputChunk], predicate: OutputPredicate
) -> list[OutputChunk]:
    received: list[OutputChunk] = []
    while True:
        output = await asyncio.wait_for(anext(outputs), 10)
        received.append(output)
        if predicate(output):
            return received
        else:
            pass


def texts_of(received: list[OutputChunk], turn: int) -> list[str]:
    return [
        output.payload["text"]
        for output in received
        if output.format == TEXT_DELTA and output.turn == turn
    ]


def is_turn_done(output: OutputChunk, turn: int) -> bool:
    return (
        output.kind == "data"
        and control_type(output) == TURN_DONE
        and output.turn == turn
    )


@pytest.mark.asyncio
async def test_queued_turns_follow_each_reply_without_more_microphone_input(
    tmp_path: Path,
) -> None:
    factory = f"{__name__}.create_graph_stage"
    async with pipeline(
        tmp_path,
        names=["dialog", "model"],
        factories={"dialog": factory, "model": factory},
    ) as (coordinator, events, processes):
        coordinator.session_graph = session_graph()
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, audio_chunk(0))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=1))
        await coordinator.append_session(identity, audio_chunk(2))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=3))
        received = await read_until(outputs, lambda output: is_turn_done(output, 2))
        texts = [
            output.payload["text"] for output in received if output.format == TEXT_DELTA
        ]
        assert texts == ["reply 1", "reply 2"]
        audio = [output for output in received if output.format == PCM_F32]
        assert [output.turn for output in audio] == [1, 2]
        assert received.index(audio[0]) < [
            output.turn == 1 and control_type(output) == TURN_DONE
            for output in received
        ].index(True)
        await coordinator.append_session(identity, audio_chunk(4, eos=True))
        await read_until(
            outputs, lambda output: output.kind == "input_done" and output.eos
        )
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_an_appended_configure_reaches_the_model_node_and_not_the_client(
    tmp_path: Path,
) -> None:
    """A realtime client appends configure with its audio; the model node took it only as a control frame, so the next reply kept the old sampling."""
    factory = f"{__name__}.create_graph_stage"
    async with pipeline(
        tmp_path,
        names=["dialog", "model"],
        factories={"dialog": factory, "model": factory},
    ) as (coordinator, events, processes):
        coordinator.session_graph = session_graph()
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, audio_chunk(0))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=1))
        await read_until(outputs, lambda output: is_turn_done(output, 1))
        await coordinator.append_session(
            identity,
            control_chunk(CONFIGURE, sequence=2, sampling={"max_new_tokens": 16}),
        )
        await coordinator.append_session(identity, audio_chunk(3))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=4))
        received = await read_until(outputs, lambda output: is_turn_done(output, 2))
        texts = [
            output.payload["text"] for output in received if output.format == TEXT_DELTA
        ]
        assert texts == ["reply 2 max_new_tokens=16"]
        assert not any(control_type(output) == CONFIGURE for output in received)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_an_interrupt_fences_the_old_reply_and_the_next_turn_is_answered(
    tmp_path: Path,
) -> None:
    factory = f"{__name__}.create_graph_stage"
    async with pipeline(
        tmp_path,
        names=["dialog", "model"],
        factories={"dialog": factory, "model": factory},
    ) as (coordinator, events, processes):
        coordinator.session_graph = session_graph()
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, audio_chunk(0))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=1))
        await read_until(outputs, lambda output: output.format == PCM_F32)
        await coordinator.control_session(identity, control_chunk(INTERRUPT))
        await coordinator.append_session(identity, audio_chunk(2))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=3))
        received = await read_until(outputs, lambda output: is_turn_done(output, 2))
        assert any(output.format == PCM_F32 and output.turn == 2 for output in received)
        assert not any(is_turn_done(output, 1) for output in received)
        assert texts_of(received, 2) == ["reply 2"]
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_turn", [None, 1], ids=["open-reply", "named-turn"])
async def test_a_reply_interrupted_before_any_of_its_audio_left_is_asked_again(
    tmp_path: Path, interrupt_turn: int | None
) -> None:
    """A reply whose text but none of whose audio reached the client stayed in the history when it was interrupted; a client interrupt naming the turn also fenced the dialog's own interrupt and the delivered report, so no later reply started."""
    factory = f"{__name__}.create_graph_stage"
    async with pipeline(
        tmp_path,
        names=["dialog", "model"],
        factories={"dialog": factory, "model": factory},
    ) as (coordinator, events, processes):
        coordinator.session_graph = session_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"audio_after_s": 1})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, audio_chunk(0))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=1))
        await read_until(outputs, lambda output: output.format == TEXT_DELTA)
        await coordinator.control_session(
            identity, control_chunk(INTERRUPT, turn=interrupt_turn)
        )
        await coordinator.append_session(identity, audio_chunk(2))
        await coordinator.append_session(identity, control_chunk(COMMIT, sequence=3))
        received = await read_until(outputs, lambda output: is_turn_done(output, 2))
        assert texts_of(received, 2) == ["reply 2 after [1]"]
        assert not any(
            output.format == PCM_F32 and output.turn == 1 for output in received
        )
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_an_utterance_over_max_chunk_bytes_reaches_the_model_whole(
    tmp_path: Path,
) -> None:
    factory = f"{__name__}.create_long_utterance_stage"
    async with pipeline(
        tmp_path,
        names=["dialog", "model"],
        factories={"dialog": factory, "model": factory},
    ) as (coordinator, events, processes):
        coordinator.session_graph = session_graph()
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        # note (luojiaxuan): 41.6 s of speech makes a unit of 1331333 bytes, over the default 1 MiB bound.
        first = utterance(1_331_200)
        second = utterance(1_600_000)
        sequence, end_ms = await say(
            coordinator, identity, first, sequence=0, start_ms=0.0
        )
        sequence, end_ms = await say(
            coordinator, identity, second, sequence=sequence, start_ms=end_ms
        )
        received = await read_until(outputs, lambda output: is_turn_done(output, 2))
        assert [reply_text(received, 1), reply_text(received, 2)] == [
            digest(first),
            digest(second),
        ]
        third = utterance(6400)
        await say(coordinator, identity, third, sequence=sequence, start_ms=end_ms)
        received = await read_until(outputs, lambda output: is_turn_done(output, 3))
        assert reply_text(received, 3) == digest(third)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_a_unit_enters_whole_up_to_max_chunk_bytes_and_in_fragments_beyond(
    tmp_path: Path,
) -> None:
    factory = f"{__name__}.create_long_utterance_stage"
    pcm = utterance(262_144)
    size = unit_bytes(pcm, len(pcm) / PCM16_BYTES_PER_MS)
    async with pipeline(
        tmp_path,
        names=["dialog", "model"],
        factories={"dialog": factory, "model": factory},
    ) as (coordinator, events, processes):
        coordinator.session_graph = session_graph()
        for max_chunk_bytes, is_whole in (
            (size + 1, True),
            (size, True),
            (size - 1, False),
            (size // 3, False),
        ):
            identity = await coordinator.open_session(
                OmniRequest(None),
                limits=SessionLimits(max_chunk_bytes=max_chunk_bytes),
            )
            outputs = coordinator.session_outputs(identity)
            await say(coordinator, identity, pcm, sequence=0, start_ms=0.0)
            received = await read_until(outputs, lambda output: is_turn_done(output, 1))
            assert reply_text(received, 1) == digest(pcm)
            model_inputs = coordinator.sessions[f"{identity.id}:qwen_omni"].next_input
            assert (model_inputs == 1) == is_whole
            await outputs.aclose()
            await wait_until(
                lambda: not coordinator.sessions and not coordinator.graphs
            )
