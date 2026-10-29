# SPDX-License-Identifier: Apache-2.0
"""A reply whose turn ends before the talker reaches its codec EOS ends incomplete."""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from sglang_omni.models.minicpm_o.session_adapters import OutputConverter
from sglang_omni.models.minicpm_o.talker_session import TalkerAdapter
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.serve.realtime.output import ResponseFinished

IDENTITY = SessionIdentity("session", 0)
# note (luojiaxuan): SGLang's finish reason types for a request that used up max_new_tokens and one that hit a stop token.
FINISH_LENGTH, FINISH_STOP = "length", "stop"


# note (luojiaxuan): A talker unit's window is CODEC_CHUNK_SIZE (25) codec slots and one more for the EOS. On an H200, a turn's last unit that reached the end of its window returned 25 tokens and finished on its stop token, as a unit mid-reply returns 25 finishing on length; a last unit that ended its speech inside the window returned fewer.
WINDOW_END = 25


class Unit(NamedTuple):
    """One thinker unit as the talker gets it, and the talker's finish reason and output length when it generates."""

    is_speaking: bool
    is_input_end: bool
    finish_reason: str | None
    is_model_turn_end: bool = False
    tokens: int = WINDOW_END


@pytest.mark.parametrize(
    ("units", "expected_cuts"),
    [
        (
            [Unit(True, False, FINISH_LENGTH), Unit(True, True, FINISH_LENGTH)],
            [False, True],
        ),
        (
            [
                Unit(True, False, FINISH_LENGTH),
                Unit(True, False, FINISH_LENGTH, is_model_turn_end=True),
            ],
            [False, True],
        ),
        (
            [Unit(True, False, FINISH_LENGTH), Unit(False, True, None)],
            [False, True],
        ),
        (
            [Unit(True, False, FINISH_LENGTH), Unit(True, True, FINISH_STOP)],
            [False, True],
        ),
        (
            [
                Unit(True, False, FINISH_LENGTH),
                Unit(True, False, FINISH_STOP, is_model_turn_end=True, tokens=11),
                Unit(False, False, None),
                Unit(False, True, None),
            ],
            [False, False, False, False],
        ),
        (
            [
                Unit(True, False, FINISH_LENGTH),
                Unit(True, True, FINISH_STOP, tokens=WINDOW_END - 1),
            ],
            [False, False],
        ),
    ],
    ids=[
        "input-end-on-the-talkers-budget",
        "model-turn-end-on-the-talkers-budget",
        "input-end-on-a-unit-boundary-mid-reply",
        "input-end-with-the-eos-in-the-windows-last-slot",
        "reply-ends-at-the-codec-eos-before-the-input-end",
        "input-end-after-speech-that-ended-inside-the-window",
    ],
)
def test_a_turn_that_ends_before_the_codec_eos_cuts_the_speech(
    units: list[Unit], expected_cuts: list[bool]
) -> None:
    """A reply whose turn ended before the talker reached its codec EOS, on the input end or on the model's own turn end, was reported as finished."""
    adapter = TalkerAdapter(SimpleNamespace(codec_eos_id=0))
    adapter.open(IDENTITY, None)
    cuts = []
    for unit in units:
        # note (luojiaxuan): A unit without speech is a listening unit, or the thinker's result for an input end on a unit boundary.
        payload = SimpleNamespace(
            data=dict(
                text="hi" if unit.is_speaking else "",
                talker_conditions=(
                    [(1, torch.zeros(1), unit.is_model_turn_end)]
                    if unit.is_speaking
                    else []
                ),
                is_listen=not unit.is_speaking,
                end_of_turn=unit.is_model_turn_end
                or (unit.is_input_end and not unit.is_speaking),
            )
        )
        chunk = TimedChunk("text", 0, 1000, 0, None, eos=unit.is_input_end)
        if adapter.prepare_unit(IDENTITY, chunk, payload).bypass_generation:
            cuts.append(payload.data["is_speech_cut"])
        else:
            request_data = SimpleNamespace(
                stage_payload=payload,
                output_ids=list(range(unit.tokens)),
                finish_reason=unit.finish_reason,
            )
            cuts.append(adapter.result(IDENTITY, request_data).data["is_speech_cut"])
    assert cuts == expected_cuts


def voice(text: str, *, end_of_turn: bool, is_speech_cut: bool) -> OutputChunk:
    return OutputChunk(
        IDENTITY,
        0,
        0,
        "voice",
        0,
        1000,
        dict(
            text=text,
            pcm=b"\x00\x00",
            end_of_turn=end_of_turn,
            is_listen=False,
            model_end_of_turn=end_of_turn,
            is_speech_cut=is_speech_cut,
        ),
        None,
        turn=1,
    )


@pytest.mark.parametrize(
    ("is_cancelled", "is_speech_cut", "status", "reason", "text"),
    [
        (False, True, "incomplete", "max_output_tokens", " Hello there."),
        (False, False, "completed", "stop", " Hello there."),
        (True, True, "cancelled", "interrupted", " Hello"),
    ],
    ids=["cut", "finished", "cancelled-then-cut"],
)
def test_the_reply_status_follows_the_last_units_speech(
    is_cancelled: bool, is_speech_cut: bool, status: str, reason: str, text: str
) -> None:
    """A reply whose last unit's speech was cut ended completed with a transcript the audio never covered; a client stop still ends it cancelled."""
    converter = OutputConverter()
    converter(voice(" Hello", end_of_turn=False, is_speech_cut=False))
    if is_cancelled:
        converter.cancel()
    else:
        pass
    events = converter(voice(" there.", end_of_turn=True, is_speech_cut=is_speech_cut))
    (finished,) = [event for event in events if isinstance(event, ResponseFinished)]
    assert (finished.status, finished.reason, finished.text) == (status, reason, text)
