# SPDX-License-Identifier: Apache-2.0
"""A reply the model ends keeps speaking the text the talker still owes; a client stop, the user's voice or a new reply ends it at once, and a reply cut short ends incomplete."""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from sglang_omni.models.minicpm_o.components.tts_runtime import CODEC_CHUNK_SIZE
from sglang_omni.models.minicpm_o.session_adapters import OutputConverter
from sglang_omni.models.minicpm_o.talker_session import TalkerAdapter
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.serve.realtime.output import ResponseFinished

IDENTITY = SessionIdentity("session", 0)
HIDDEN_SIZE = 4
CODEC_VOCAB_SIZE = 16


class Unit(NamedTuple):
    """One thinker unit as the talker gets it, and the codec tokens the talker commits if it speaks the unit."""

    is_speaking: bool
    committed_tokens: int = 0
    is_model_turn_end: bool = False
    is_input_end: bool = False
    is_interrupted: bool = False
    hears_user: bool = False


class Spoken(NamedTuple):
    """What the talker did with a unit: whether it spoke, how many units of speech its request could hold, and what it reported downstream; whether it reset its history, and how early its request could stop at a codec EOS."""

    speaks: bool
    speech_units: int
    end_of_turn: bool
    is_speech_cut: bool
    ends_previous_turn: bool = False
    reset_history: bool = False
    masks_chunk_boundary_eos: bool = False
    min_new_tokens: int = 0


FULL = CODEC_CHUNK_SIZE
REPLY = Unit(True, FULL)
OWING_TURN_END = Unit(True, FULL, is_model_turn_end=True)
SILENT_RESET = Spoken(False, 0, False, False, reset_history=True)


def talker_model() -> SimpleNamespace:
    return SimpleNamespace(
        codec_eos_id=CODEC_VOCAB_SIZE - 1,
        num_audio_tokens=CODEC_VOCAB_SIZE,
        emb_text=torch.nn.Embedding(8, HIDDEN_SIZE),
        projector_semantic=torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        audio_bos_token_id=1,
        normalize_projected_hidden=True,
    )


def run_talker(units: list[Unit]) -> list[Spoken]:
    adapter = TalkerAdapter(talker_model())
    adapter.open(IDENTITY, None)
    spoken = []
    for unit in units:
        # note (luojiaxuan): A unit without speech is a listening unit, or the thinker's result for an input end on a unit boundary.
        payload = SimpleNamespace(
            request_id="unit",
            request=SimpleNamespace(
                params=dict(talker_temperature=0.8, talker_repetition_penalty=1.05)
            ),
            data=dict(
                text="hi" if unit.is_speaking else "",
                talker_conditions=(
                    [(2, torch.zeros(HIDDEN_SIZE), unit.is_model_turn_end)]
                    if unit.is_speaking
                    else []
                ),
                is_listen=not unit.is_speaking,
                end_of_turn=unit.is_model_turn_end
                or unit.is_interrupted
                or (unit.is_input_end and not unit.is_speaking),
                is_interrupted=unit.is_interrupted,
                hears_user=unit.hears_user,
            ),
        )
        chunk = TimedChunk("text", 0, 1000, 0, None, eos=unit.is_input_end)
        preparation = adapter.prepare_unit(IDENTITY, chunk, payload)
        if preparation.bypass_generation:
            speech_units = 0
            masks_chunk_boundary_eos = False
            min_new_tokens = 0
        else:
            request_data = adapter.build(IDENTITY, chunk, payload)
            speech_units = request_data.max_new_tokens // CODEC_CHUNK_SIZE
            masks_chunk_boundary_eos = request_data.talker_model_inputs[
                "masks_chunk_boundary_eos"
            ]
            min_new_tokens = request_data.req.sampling_params.min_new_tokens
            request_data.output_ids = list(range(unit.committed_tokens))
            adapter.result(IDENTITY, request_data)
        spoken.append(
            Spoken(
                speech_units > 0,
                speech_units,
                payload.data["end_of_turn"],
                payload.data["is_speech_cut"],
                payload.data["ends_previous_turn"],
                preparation.reset_history,
                masks_chunk_boundary_eos,
                min_new_tokens,
            )
        )
    return spoken


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        (
            [REPLY, OWING_TURN_END, Unit(False, FULL), Unit(False, 10), Unit(False)],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 1, True, False),
                SILENT_RESET,
            ],
        ),
        (
            [
                REPLY,
                OWING_TURN_END,
                Unit(False, FULL),
                Unit(False, FULL),
                Unit(False, FULL),
                Unit(False),
            ],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 1, True, True),
                SILENT_RESET,
            ],
        ),
        (
            [REPLY, OWING_TURN_END, Unit(False, hears_user=True), Unit(False)],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(False, 0, True, True),
                SILENT_RESET,
            ],
        ),
        (
            [REPLY, OWING_TURN_END, Unit(False, is_interrupted=True)],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(False, 0, True, True),
            ],
        ),
        (
            [
                REPLY,
                OWING_TURN_END,
                Unit(False, is_input_end=True, is_interrupted=True),
            ],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(False, 0, True, True),
            ],
        ),
        (
            [REPLY, Unit(False, is_interrupted=True)],
            [Spoken(True, 1, False, False), Spoken(False, 0, True, True)],
        ),
        (
            [REPLY, OWING_TURN_END, Unit(True, FULL)],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(
                    True, 1, False, False, ends_previous_turn=True, reset_history=True
                ),
            ],
        ),
        (
            [
                REPLY,
                Unit(True, 10, is_model_turn_end=True),
                Unit(False),
                Unit(False, is_input_end=True),
            ],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, True, False),
                SILENT_RESET,
                Spoken(False, 0, True, False),
            ],
        ),
        (
            [REPLY, Unit(False, 60, is_input_end=True)],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 4, True, False, masks_chunk_boundary_eos=True),
            ],
        ),
        (
            [REPLY, Unit(False, 4 * FULL + 1, is_input_end=True)],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 4, True, True, masks_chunk_boundary_eos=True),
            ],
        ),
        (
            [
                REPLY,
                OWING_TURN_END,
                Unit(False, FULL),
                Unit(False, 50, is_input_end=True),
            ],
            [
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 1, False, False),
                Spoken(True, 2, True, True, masks_chunk_boundary_eos=True),
            ],
        ),
    ],
    ids=[
        "model-turn-end-speaks-the-owed-text-to-the-codec-eos",
        "owed-text-past-the-drain-budget-ends-incomplete",
        "the-users-voice-stops-the-owed-speech",
        "a-client-stop-stops-the-owed-speech",
        "a-stop-at-the-input-end-stops-the-owed-speech",
        "a-client-stop-mid-reply-speaks-nothing-more",
        "a-new-reply-ends-the-owed-speech-first",
        "a-turn-spoken-to-the-codec-eos-owes-nothing",
        "input-end-mid-reply-speaks-what-is-owed-in-one-request",
        "a-codec-token-in-the-windows-extra-slot-still-owes-speech",
        "input-end-while-owing-speaks-the-rest-of-the-budget",
    ],
)
def test_a_reply_the_model_ends_speaks_the_text_the_talker_still_owes(
    units: list[Unit], expected: list[Spoken]
) -> None:
    """When the model ended its turn, or the input ended, while the talker still owed text, the talker dropped that text and the reply ended mid-sentence."""
    assert run_talker(units) == expected


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
