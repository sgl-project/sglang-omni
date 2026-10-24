# SPDX-License-Identifier: Apache-2.0
"""Real coordinator and SessionScheduler routes: the dialog's turns and a synthetic model node that answers them."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Protocol

import pytest

from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ControlEdge,
    DataEdge,
    OutputChunk,
    SessionGraph,
    SessionNode,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
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
    """Answers a reply unit with turn.started, a text delta, audio and turn.done, as the route's terminal owner does."""

    def open(self, session_identity, request) -> None:
        pass

    def close(self, session_identity) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        payload.data = {}
        if chunk.format != "qwen.unit" or context.cancelled.wait(0.02):
            return payload
        else:
            pass
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
                {"text": f"reply {chunk.turn}"},
                format=TEXT_DELTA,
                turn=chunk.turn,
            ),
            TimedChunk(
                AUDIO, 0, 10, chunk.seq, bytes(960), format=PCM_F32, turn=chunk.turn
            ),
        ):
            context.emit(event)
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
    event: str, *, turn: int | None = None, sequence: int = 0
) -> TimedChunk:
    return TimedChunk(
        "control", 0, 0, sequence, {"type": event}, format=SESSION_CONTROL, turn=turn
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
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
