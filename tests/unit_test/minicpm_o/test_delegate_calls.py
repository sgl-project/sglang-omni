# SPDX-License-Identifier: Apache-2.0
"""A task the model hands off reaches the client as a function call, which the client runs while the reply goes on."""

import json

import pytest

from sglang_omni.models.minicpm_o.session_adapters import OutputConverter
from sglang_omni.proto.session import OutputChunk, SessionIdentity
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    FunctionCallFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.projection import project_output

IDENTITY = SessionIdentity("session", 0)
TASK = "weather in Nanjing"
OTHER_TASK = "book a table for two"
PCM = b"\x00\x00"


def voice(
    unit_index: int,
    *,
    text: str = "",
    pcm: bytes = b"",
    delegates: tuple[str, ...] = (),
    end_of_turn: bool = False,
    turn: int = 0,
) -> OutputChunk:
    return OutputChunk(
        IDENTITY,
        unit_index,
        unit_index,
        "voice",
        0,
        1000,
        dict(
            text=text,
            pcm=pcm,
            end_of_turn=end_of_turn,
            is_listen=False,
            model_end_of_turn=end_of_turn,
            is_speech_cut=False,
            delegates=list(delegates),
        ),
        None,
        turn=turn,
    )


def call(
    response: int, index: int, call_serial: int, task: str
) -> FunctionCallFinished:
    return FunctionCallFinished(
        f"session-response-{response}",
        item_id=f"session-response-{response}-function-call-{index}",
        output_index=index,
        call_id=f"session-call-{call_serial}",
        name="delegate",
        arguments=json.dumps({"task": task}),
    )


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        (
            [voice(0, text="Sure.", pcm=PCM, delegates=(TASK,))],
            [
                ResponseStarted("session-response-0"),
                TextDelta("session-response-0", "session-response-0-message", "Sure."),
                AudioDelta("session-response-0", "session-response-0-message", PCM),
                call(0, 1, 0, TASK),
            ],
        ),
        (
            [voice(0, delegates=(TASK,))],
            [ResponseStarted("session-response-0"), call(0, 1, 0, TASK)],
        ),
        (
            [
                voice(0, text="Sure.", pcm=PCM),
                voice(1, delegates=(TASK, OTHER_TASK)),
            ],
            [
                ResponseStarted("session-response-0"),
                TextDelta("session-response-0", "session-response-0-message", "Sure."),
                AudioDelta("session-response-0", "session-response-0-message", PCM),
                call(0, 1, 0, TASK),
                call(0, 2, 1, OTHER_TASK),
            ],
        ),
        (
            [
                voice(0, delegates=(TASK,), end_of_turn=True),
                voice(1, delegates=(OTHER_TASK,), turn=1),
            ],
            [
                ResponseStarted("session-response-0"),
                call(0, 1, 0, TASK),
                TextFinished("session-response-0", "session-response-0-message", ""),
                ResponseFinished(
                    "session-response-0",
                    "session-response-0-message",
                    "",
                    False,
                    "completed",
                    "stop",
                ),
                ResponseStarted("session-response-1"),
                call(1, 1, 1, OTHER_TASK),
            ],
        ),
    ],
    ids=[
        "a_spoken_unit_calls_after_its_words",
        "a_silent_unit_still_opens_a_reply_for_its_call",
        "each_task_is_its_own_call",
        "a_later_reply_numbers_its_calls_on_from_the_session",
    ],
)
def test_each_task_the_model_hands_off_reaches_the_client_once_as_a_call(
    units: list[OutputChunk], expected: list[OutputEvent]
) -> None:
    """The model wrote a task inside a reply, or in a unit with nothing to say, and the client never got a call to run it."""
    converter = OutputConverter()

    events = [event for unit in units for event in converter(unit)]

    assert events == expected


def test_a_task_from_a_reply_the_client_stopped_is_never_called() -> None:
    """The client stopped the reply, the thinker forgot what came after it, and the client still ran the task written there."""
    converter = OutputConverter()
    converter(voice(0, text="Sure.", pcm=PCM))
    converter.cancel()

    events = converter(voice(1, delegates=(TASK,)))

    assert [event for event in events if isinstance(event, FunctionCallFinished)] == []


def test_a_stop_keeps_the_unit_whose_call_the_client_received() -> None:
    """The client stopped the reply after it got a call from a unit with no speech, and the thinker forgot the task the client was running."""
    converter = OutputConverter()
    converter(voice(0, text="Sure.", pcm=PCM))
    converter(voice(1, delegates=(TASK,)))

    kept_reply = converter.cancel()

    assert kept_reply is not None
    assert kept_reply.keep_through_unit == 1


def test_a_call_reaches_the_wire_as_its_finished_arguments() -> None:
    """A client watching for function calls never saw the call the model made."""
    assert project_output(call(0, 1, 0, TASK)) == dict(
        type="response.function_call_arguments.done",
        response_id="session-response-0",
        item_id="session-response-0-function-call-1",
        output_index=1,
        call_id="session-call-0",
        name="delegate",
        arguments=json.dumps({"task": TASK}),
    )
