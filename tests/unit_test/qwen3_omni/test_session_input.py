# SPDX-License-Identifier: Apache-2.0
"""The dialog stage turns utterances into one reply unit per turn and forwards interruption and truncation."""

import pytest

from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.qwen3_omni.thinker_session import TRUNCATE_FORMAT
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    IMAGE,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    REPLY_DELIVERED,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.qwen_fakes import make_dialog
from tests.unit_test.fixtures.session_outputs import control_chunk

IDENTITY = SessionIdentity("dialog")


@pytest.fixture
def dialog() -> tuple[QwenInputHooks, list[TimedChunk], SessionContext]:
    return make_dialog(IDENTITY)


def audio(t_ms: float) -> TimedChunk:
    return TimedChunk("audio", t_ms, 20, 0, b"\x01\x00" * 320, format=PCM16)


def append(hooks: QwenInputHooks, context: SessionContext, chunk: TimedChunk) -> None:
    hooks.append(chunk, StagePayload("in", OmniRequest(None), {}), context)


def events(outputs: list[TimedChunk]) -> list[tuple[str | None, int | None]]:
    return [(control_type(chunk) or chunk.format, chunk.turn) for chunk in outputs]


def test_a_turn_waits_for_the_previous_reply_to_be_done(dialog) -> None:
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, audio(20))
    append(hooks, context, control_chunk(COMMIT))
    assert events(outputs) == [(TURN_QUEUED, 1), ("qwen.unit", 1), (TURN_QUEUED, 2)]
    unit = outputs[1]
    assert unit.format == UNIT_FORMAT and len(unit.payload["pcm"]) == 640
    append(hooks, context, control_chunk(TURN_DONE, turn=1))
    assert events(outputs)[3:] == [("qwen.unit", 2)]


def test_speech_during_a_reply_interrupts_it_and_the_next_turn_follows(dialog) -> None:
    """A truncate refers to the reply whose output reached the client, not the queued turn started after it."""
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, audio(20))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, control_chunk(TURN_STARTED, turn=1))
    append(hooks, context, control_chunk(SPEECH_STARTED, t_ms=40))
    assert events(outputs)[3:] == [(INTERRUPT, 1), (TURN_INTERRUPTED, 1)]
    append(hooks, context, control_chunk(REPLY_DELIVERED, turn=1, audio_ms=320.0))
    assert events(outputs)[5:] == [(TRUNCATE_FORMAT, None), ("qwen.unit", 2)]
    assert outputs[5].payload == {"turn": 1, "t_ms": 320.0}
    assert outputs[-1].payload["carried_turns"] == []
    append(hooks, context, control_chunk(TRUNCATE, t_ms=250))
    truncate = outputs[-1]
    assert truncate.format == TRUNCATE_FORMAT
    assert truncate.turn is None and truncate.payload == {"turn": 1, "t_ms": 250.0}


def test_the_delivered_audio_of_an_interrupted_reply_reaches_the_model_as_a_truncate(
    dialog,
) -> None:
    """A client that sends no truncate used to leave the whole generated reply in history; the graph's report of the audio it delivered bounds it instead."""
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, control_chunk(TURN_STARTED, turn=1))
    hooks.control(IDENTITY, control_chunk(INTERRUPT))
    append(hooks, context, control_chunk(REPLY_DELIVERED, turn=1, audio_ms=500.0))
    delivered = outputs[-1]
    assert delivered.format == TRUNCATE_FORMAT
    assert delivered.turn is None and delivered.payload == {"turn": 1, "t_ms": 500.0}


def test_a_reply_none_of_whose_audio_reached_the_client_is_asked_again(dialog) -> None:
    """A reply interrupted after its turn.started but before any of its audio reached the client stayed in the history and its utterance was not asked again."""
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, audio(20))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, control_chunk(TURN_STARTED, turn=1))
    append(hooks, context, control_chunk(SPEECH_STARTED, t_ms=40))
    append(hooks, context, control_chunk(REPLY_DELIVERED, turn=1, audio_ms=0.0))
    assert events(outputs)[3:] == [
        (INTERRUPT, 1),
        (TURN_INTERRUPTED, 1),
        ("qwen.unit", 2),
    ]
    unit = outputs[-1]
    assert unit.payload["carried_turns"] == [1]
    assert len(unit.payload["pcm"]) == 2 * 640


def test_two_unheard_interruptions_ask_both_utterances_again(dialog) -> None:
    """A unit that asked a carried utterance again kept only its newest utterance when it was interrupted unheard too, so the next unit lost the first."""
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, audio(20))
    append(hooks, context, control_chunk(COMMIT))
    append(hooks, context, audio(40))
    append(hooks, context, control_chunk(COMMIT))
    for turn, t_ms in ((1, 60), (2, 80)):
        append(hooks, context, control_chunk(TURN_STARTED, turn=turn))
        append(hooks, context, control_chunk(SPEECH_STARTED, t_ms=t_ms))
        append(hooks, context, control_chunk(REPLY_DELIVERED, turn=turn, audio_ms=0.0))
    unit = outputs[-1]
    assert unit.format == UNIT_FORMAT and unit.turn == 3
    assert unit.payload["carried_turns"] == [2]
    assert len(unit.payload["pcm"]) == 3 * 640


def test_a_client_interrupt_targets_the_turn_being_answered(dialog) -> None:
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    hooks.control(IDENTITY, control_chunk(INTERRUPT, turn=7))
    assert events(outputs) == [(TURN_QUEUED, 1), ("qwen.unit", 1)]
    append(hooks, context, control_chunk(TURN_STARTED, turn=1))
    hooks.control(IDENTITY, control_chunk(INTERRUPT))
    assert events(outputs)[2:] == [(INTERRUPT, 1), (TURN_INTERRUPTED, 1)]
    append(hooks, context, control_chunk(TRUNCATE, t_ms=100))
    assert outputs[-1].format == TRUNCATE_FORMAT and outputs[-1].payload["turn"] == 1


def test_an_empty_commit_and_an_overlong_utterance(dialog) -> None:
    hooks, outputs, context = dialog
    append(hooks, context, control_chunk(COMMIT))
    assert events(outputs) == [(TURN_QUEUED, 1), (TURN_EMPTY, 1)]
    with pytest.raises(ValueError, match="utterance"):
        append(
            hooks,
            context,
            TimedChunk("audio", 0, 1001, 0, bytes(32002), format=PCM16),
        )


def test_camera_frames_between_utterances_are_kept_only_within_reach(dialog) -> None:
    """While no one speaks only the frames the segmenter's held audio could still open an utterance over remain."""
    hooks, outputs, context = dialog
    append(hooks, context, control_chunk(SPEECH_STARTED, t_ms=0))
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(SPEECH_STOPPED, t_ms=20))
    append(hooks, context, control_chunk(TURN_DONE, turn=1))
    for t_ms in range(20, 6000, 20):
        append(hooks, context, audio(t_ms))
        if t_ms % 200 == 0:
            frame = f"frame {t_ms}".encode()
            append(
                hooks, context, TimedChunk(IMAGE, t_ms, 0, 0, frame, format=IMAGE_FRAME)
            )
        else:
            pass
    assert hooks.usage(IDENTITY).bytes <= 11 * len(b"frame 5800")
    append(hooks, context, control_chunk(SPEECH_STARTED, t_ms=5500))
    append(hooks, context, control_chunk(SPEECH_STOPPED, t_ms=6000))
    unit = outputs[-1]
    assert unit.format == "qwen.unit" and unit.turn == 2
    assert unit.payload["images"] == [b"frame 5600", b"frame 5800"]


def test_an_utterance_interrupted_before_its_reply_is_asked_again_with_the_next(
    dialog,
) -> None:
    hooks, outputs, context = dialog
    append(hooks, context, audio(0))
    append(hooks, context, control_chunk(COMMIT))
    hooks.control(IDENTITY, control_chunk(INTERRUPT))
    append(hooks, context, control_chunk(REPLY_DELIVERED, turn=1, audio_ms=0.0))
    append(hooks, context, audio(20))
    append(hooks, context, control_chunk(COMMIT))
    unit = outputs[-1]
    assert unit.format == "qwen.unit" and unit.turn == 2
    assert unit.payload["carried_turns"] == [1]
    assert len(unit.payload["pcm"]) == 2 * 640


def test_an_utterance_takes_only_its_newest_frames(dialog) -> None:
    hooks, outputs, context = dialog
    append(hooks, context, control_chunk(SPEECH_STARTED, t_ms=0))
    for t_ms in range(0, 1000, 20):
        append(hooks, context, audio(t_ms))
        if t_ms % 200 == 0:
            frame = f"frame {t_ms}".encode()
            append(
                hooks, context, TimedChunk(IMAGE, t_ms, 0, 0, frame, format=IMAGE_FRAME)
            )
        else:
            pass
    append(hooks, context, control_chunk(SPEECH_STOPPED, t_ms=1000))
    unit = outputs[-1]
    assert unit.format == UNIT_FORMAT
    assert unit.payload["images"] == [
        b"frame 200",
        b"frame 400",
        b"frame 600",
        b"frame 800",
    ]
