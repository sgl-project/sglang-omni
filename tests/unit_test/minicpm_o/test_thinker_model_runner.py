# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker runners: offline length penalty and the duplex unit sampler."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.audio.vad import StatelessVAD
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    SAMPLE_RATE,
    UNIT_SAMPLES,
)
from sglang_omni.models.minicpm_o.duplex_sampler import filter_top_k_top_p
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_WINDOW_SETTINGS,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.native_thinker_model_runner import (
    MiniCPMOThinkerModelRunner as DuplexThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.models.minicpm_o.thinker_state import (
    DuplexUnitRequestData,
    MiniCPMOThinkerSessionState,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    CONTROL,
    INTERRUPT,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_CONTROL,
    TRUNCATE,
)
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionPreparation
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)

EOS_TOKEN_IDS = [1, 3]
VOCAB_SIZE = 128
SPECIAL_TOKEN_ID_START = 100
BAD_TOKEN_IDS = (7, 8, 94)
PLAIN_TOKEN_ID = 42
OTHER_TOKEN_ID = 55
THIRD_TOKEN_ID = 60
SESSION = SessionIdentity("duplex")
# The native session's prompt lengths, retained turns plus the unit's own ids, of three units in a row.
FIRST_UNIT_PROMPT_LENGTH = 7
SECOND_UNIT_PROMPT_LENGTH = 12
THIRD_UNIT_PROMPT_LENGTH = 18
# The native session's prompt lengths of thirteen units in a row, the first three as above.
UNIT_PROMPT_LENGTHS = (
    FIRST_UNIT_PROMPT_LENGTH,
    SECOND_UNIT_PROMPT_LENGTH,
    THIRD_UNIT_PROMPT_LENGTH,
    25,
    33,
    42,
    52,
    63,
    75,
    88,
    102,
    117,
    133,
)
# A unit's perception ids are one audio placeholder, this id plus the unit index, so a re-fed unit is told apart by its id.
AUDIO_PLACEHOLDER_ID_BASE = 80
# The listen units after a reply's last spoken unit within which the thinker still cuts the reply.
TRUNCATE_WINDOW_UNITS = 10
# note (luojiaxuan): 16 kHz mono PCM16 speech, voiced throughout from 3 s to 4 s.
SPEECH_FIXTURE = Path(__file__).resolve().parents[2] / "data" / "query_to_draw.wav"
VOICED_START_SAMPLE = 3 * SAMPLE_RATE
# note (luojiaxuan): Input levels as RMS in dB of full scale: a voice under an auto-gain microphone, a voice from a quiet microphone with no gain control, and a loud room's steady noise.
VOICE_DBFS = -20
QUIET_VOICE_DBFS = -45
LOUD_ROOM_DBFS = -20
SHORT_WORD_SAMPLES = SAMPLE_RATE * 300 // 1000
SILENT_UNIT_PCM = bytes(UNIT_SAMPLES * PCM16_BYTES_PER_SAMPLE)


def session_control(payload: dict[str, object]) -> TimedChunk:
    return TimedChunk(CONTROL, 0, 0, 0, payload, format=SESSION_CONTROL, turn=0)


def interrupt_after(keep_through_unit: int) -> TimedChunk:
    return session_control({"type": INTERRUPT, "keep_through_unit": keep_through_unit})


def truncate_through(keep_through_unit: int) -> TimedChunk:
    return session_control(
        {"type": TRUNCATE, "t_ms": 0.0, "keep_through_unit": keep_through_unit}
    )


def audio_unit(input_seq: int, payload: bytes | dict[str, object] = b"") -> TimedChunk:
    return TimedChunk("audio", input_seq * 1000, 1000, input_seq, payload)


def speech(sample_count: int) -> np.ndarray:
    """sample_count voiced samples of the speech fixture, in full-scale units."""
    with wave.open(str(SPEECH_FIXTURE)) as speech_file:
        speech_file.setpos(VOICED_START_SAMPLE)
        return np.frombuffer(speech_file.readframes(sample_count), dtype="<i2") / 32768


def unit_pcm(signal: np.ndarray, dbfs: float) -> bytes:
    """One unit of 16 kHz PCM16 input, silent but for signal, scaled to an RMS of dbfs, over its last samples."""
    samples = np.zeros(UNIT_SAMPLES)
    samples[UNIT_SAMPLES - len(signal) :] = (
        signal * 10 ** (dbfs / 20) / np.sqrt(np.mean(signal**2))
    )
    return np.round(samples * 32768).astype("<i2").tobytes()


def bare_runner() -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.eos_token_ids = EOS_TOKEN_IDS
    runner.eos_token_id_cache = None
    return runner


def scheduler_request(params: dict[str, Any]) -> SimpleNamespace:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=None,
            params=params,
            metadata={"output_modalities": ["text"]},
        ),
        data=None,
    )
    return SimpleNamespace(data=SimpleNamespace(stage_payload=payload))


def penalized_request(length_penalty: float) -> SimpleNamespace:
    return scheduler_request(
        {"stage_params": {"thinker": {"length_penalty": length_penalty}}}
    )


def test_length_penalty_scales_eos_logits_per_request() -> None:
    original = torch.tensor(
        [
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
        ]
    )
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(
        logits_output,
        [
            penalized_request(1.0),
            penalized_request(2.0),
            penalized_request(1.0),
        ],
    )

    torch.testing.assert_close(logits_output.next_token_logits[0], original[0])
    torch.testing.assert_close(logits_output.next_token_logits[2], original[2])
    expected = original[1].clone()
    expected[EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_output.next_token_logits[1], expected)


def test_lookahead_samples_from_penalized_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())
    logits_seen_by_parent = []

    def parent_sample_lookahead(self, logits_output, forward_batch, requests):
        logits_seen_by_parent.append(logits_output.next_token_logits.clone())
        return torch.tensor([0])

    monkeypatch.setattr(ThinkerModelRunner, "sample_lookahead", parent_sample_lookahead)

    bare_runner().sample_lookahead(
        logits_output, forward_batch=None, requests=[penalized_request(2.0)]
    )

    expected = original.clone()
    expected[0, EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_seen_by_parent[0], expected)


def test_missing_length_penalty_leaves_logits_unchanged() -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(logits_output, [scheduler_request({})])

    torch.testing.assert_close(logits_output.next_token_logits, original)


def duplex_tokenizer() -> Mock:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=list(BAD_TOKEN_IDS))
    special_token_ids = range(
        SPECIAL_TOKEN_ID_START, SPECIAL_TOKEN_ID_START + len(REQUIRED_SPECIAL_TOKENS)
    )
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, special_token_ids)
    ).__getitem__
    return tokenizer


SPECIAL_TOKENS = resolve_special_token_ids(
    duplex_tokenizer(), bad_token_ids=BAD_TOKEN_IDS
)


def duplex_runner(
    special_tokens: MiniCPMOSpecialTokenIds | None,
) -> DuplexThinkerModelRunner:
    runner = DuplexThinkerModelRunner.__new__(DuplexThinkerModelRunner)
    runner.special_tokens = special_tokens
    runner.forbidden_token_index = None
    runner.eos_token_ids = []
    return runner


def session_state(
    **sampling_overrides: float | int | bool,
) -> MiniCPMOThinkerSessionState:
    sampling = {"greedy": True, "repetition_penalty": 1.0, **sampling_overrides}
    return MiniCPMOThinkerSessionState(
        sampling=MiniCPMODuplexSampling(**sampling),
        speech_detector=Mock(spec=StatelessVAD),
    )


def unit_request(
    state: MiniCPMOThinkerSessionState,
    *,
    generation_step: int = 1,
    is_listen_forced: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            thinker_state=state,
            generation_steps=generation_step,
            is_listen_forced=is_listen_forced,
        )
    )


def unit_logits(*rows: dict[int, float]) -> torch.Tensor:
    logits = torch.full((len(rows), VOCAB_SIZE), -torch.inf)
    for row_index, finite_logits in enumerate(rows):
        for token_id, value in finite_logits.items():
            logits[row_index, token_id] = value
    return logits


def sample(
    runner: DuplexThinkerModelRunner,
    logits: torch.Tensor,
    requests: list[SimpleNamespace],
) -> list[int]:
    return runner.sample_next_token_ids(
        SimpleNamespace(next_token_logits=logits), None, None, requests
    ).tolist()


def make_thinker_adapter(tokenizer: Mock) -> ThinkerAdapter:
    return ThinkerAdapter(
        tokenizer,
        VOCAB_SIZE,
        window=DEFAULT_WINDOW_SETTINGS,
        rotary_embedding=SimpleNamespace(
            head_size=64, rotary_dim=64, is_neox_style=True, cos_sin_cache=None
        ),
    )


@pytest.mark.parametrize(
    ("use_runner", "max_new_tokens", "generation_step", "closes"),
    [
        (True, 20, 0, False),
        (False, 20, 0, False),
        (False, 5, 4, True),
        (False, 6, 4, False),
    ],
)
def test_duplex_sample_masks_bad_tokens_and_closes_at_budget(
    use_runner: bool, max_new_tokens: int, generation_step: int, closes: bool
) -> None:
    """Both init paths keep the checkpoint mask; the token budget closes the chunk."""
    tokenizer = duplex_tokenizer()
    if use_runner:
        runner = duplex_runner(None)
        special_tokens = runner.resolve_special_tokens(
            SimpleNamespace(req=Mock(tokenizer=tokenizer))
        )
    else:
        special_tokens = make_thinker_adapter(tokenizer).special
        runner = duplex_runner(special_tokens)
    logits = unit_logits(
        {BAD_TOKEN_IDS[0]: 100.0, special_tokens.tts_pad: 99.0, PLAIN_TOKEN_ID: 0.0}
    )
    state = session_state(max_new_tokens_per_unit=max_new_tokens)

    token_ids = sample(
        runner, logits, [unit_request(state, generation_step=generation_step)]
    )

    assert token_ids == [special_tokens.chunk_eos if closes else PLAIN_TOKEN_ID]


def test_forced_listen_applies_to_the_first_step_only() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    first_step = session_state()
    later_step = session_state()
    logits = unit_logits({PLAIN_TOKEN_ID: 1.0}, {PLAIN_TOKEN_ID: 1.0})

    token_ids = sample(
        runner,
        logits,
        [
            unit_request(first_step, generation_step=0, is_listen_forced=True),
            unit_request(later_step, generation_step=1, is_listen_forced=True),
        ],
    )

    assert token_ids == [SPECIAL_TOKENS.listen, PLAIN_TOKEN_ID]
    assert (first_step.force_listen_counter, later_step.force_listen_counter) == (1, 0)
    assert (first_step.generated_history, later_step.generated_history) == (
        [],
        [PLAIN_TOKEN_ID],
    )


# note (0xtoward): A unit's own ids as streaming perception plans them, <unit> first.
UNIT_INPUT_IDS = [SPECIAL_TOKENS.unit, PLAIN_TOKEN_ID]


def step_plan_payload(request: OmniRequest) -> StagePayload:
    return StagePayload(
        "unit",
        request,
        {"token_ids": UNIT_INPUT_IDS, "input_embeds": None, "embedding_spans": []},
    )


def complete_unit(
    adapter: ThinkerAdapter,
    data: SGLangARRequestData,
    *,
    prompt_length: int,
    output_ids: list[int],
    generated_unit_ids: list[int],
) -> None:
    """Finish a unit; the native session's request, which prepends the retained turns, is stood in for by its prompt length."""
    data.req = SimpleNamespace(origin_input_ids=[PLAIN_TOKEN_ID] * prompt_length)
    data.output_ids = output_ids
    data.generated_unit_ids = generated_unit_ids
    adapter.result(SESSION, data)


def spoken_output(token_id: int, ends_turn: bool) -> tuple[list[int], list[int]]:
    """The output ids and the generated ids of a unit that speaks token_id, closing the turn when ends_turn."""
    generated_unit_ids = [token_id, *([SPECIAL_TOKENS.turn_eos] if ends_turn else [])]
    return [
        SPECIAL_TOKENS.tts_bos,
        *generated_unit_ids,
        SPECIAL_TOKENS.chunk_eos,
    ], generated_unit_ids


def open_duplex_adapter() -> tuple[ThinkerAdapter, OmniRequest]:
    tokenizer = duplex_tokenizer()
    tokenizer.decode.return_value = ""
    adapter = make_thinker_adapter(tokenizer)
    request = OmniRequest(
        None, params=MiniCPMODuplexSampling(force_listen_count=0).model_dump()
    )
    adapter.open(SESSION, request)
    return adapter, request


def listen_after_a_stop(
    adapter: ThinkerAdapter,
    request: OmniRequest,
    payloads: list[bytes | dict[str, object]],
) -> list[tuple[DuplexUnitRequestData, StagePayload]]:
    """Stop the reply, then build and finish one unit per input payload, each listening, and return each unit with its result."""
    adapter.control(SESSION, interrupt_after(0))
    finished_units: list[tuple[DuplexUnitRequestData, StagePayload]] = []
    for input_seq, payload in enumerate(payloads):
        unit_request = adapter.build(
            SESSION, audio_unit(input_seq, payload), step_plan_payload(request)
        )
        unit_request.output_ids = [SPECIAL_TOKENS.listen]
        finished_units.append((unit_request, adapter.result(SESSION, unit_request)))
    return finished_units


@pytest.mark.parametrize(
    ("is_turn_ended", "interrupted_input_ids"),
    [
        (False, [*UNIT_INPUT_IDS, SPECIAL_TOKENS.turn_eos]),
        (True, UNIT_INPUT_IDS),
    ],
    ids=["mid_reply", "after_the_model_ended_its_turn"],
)
def test_a_stop_holds_the_listen_through_silence_until_the_user_speaks(
    is_turn_ended: bool, interrupted_input_ids: list[int]
) -> None:
    """A client stop: the next unit closes an open turn with turn_eos, listens and reports the end of the turn. The silent units after it still listen, with no second turn_eos and no second end of turn, so the model cannot resume the stopped reply in the silence; the first unit with the user's voice is free again, and so is every unit after it, and freeing adds nothing to their prompts."""
    adapter, request = open_duplex_adapter()
    state = adapter.states[SESSION]
    state.is_turn_ended = is_turn_ended

    finished_units = listen_after_a_stop(
        adapter,
        request,
        [
            SILENT_UNIT_PCM,
            SILENT_UNIT_PCM,
            unit_pcm(speech(UNIT_SAMPLES), VOICE_DBFS),
            SILENT_UNIT_PCM,
        ],
    )

    assert [unit_request.is_listen_forced for unit_request, _ in finished_units] == [
        True,
        True,
        False,
        False,
    ]
    assert [unit_request.input_ids.tolist() for unit_request, _ in finished_units] == [
        interrupted_input_ids,
        *[[SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS]] * 3,
    ]
    assert [result.data for _, result in finished_units] == [
        dict(talker_conditions=[], text="", is_listen=True, end_of_turn=end_of_turn)
        for end_of_turn in (True, False, False, False)
    ]
    assert state.is_turn_ended is True


@pytest.mark.parametrize(
    ("payload", "is_listen_held"),
    [
        (
            unit_pcm(
                np.random.default_rng(0).standard_normal(UNIT_SAMPLES), LOUD_ROOM_DBFS
            ),
            True,
        ),
        (unit_pcm(speech(UNIT_SAMPLES), QUIET_VOICE_DBFS), False),
        (unit_pcm(speech(SHORT_WORD_SAMPLES), QUIET_VOICE_DBFS), False),
        ({"pcm": SILENT_UNIT_PCM, "images": [b"frame"]}, True),
        (
            {"pcm": unit_pcm(speech(UNIT_SAMPLES), VOICE_DBFS), "images": [b"frame"]},
            False,
        ),
    ],
    ids=[
        "a_loud_room",
        "a_quiet_microphone",
        "a_short_quiet_word_at_the_end_of_the_unit",
        "silence_with_a_camera_frame",
        "a_voice_with_a_camera_frame",
    ],
)
def test_after_a_stop_only_the_users_voice_frees_the_model(
    payload: bytes | dict[str, object], is_listen_held: bool
) -> None:
    """After a stop and a silent unit, a loud room's steady noise or a camera frame over silence keeps the model listening, while the user's voice frees it from that unit on at any capture level: a whole unit or a short word from a quiet microphone, or a voice sent with a camera frame."""
    adapter, request = open_duplex_adapter()

    finished_units = listen_after_a_stop(
        adapter, request, [SILENT_UNIT_PCM, SILENT_UNIT_PCM, payload, SILENT_UNIT_PCM]
    )

    assert [
        unit_request.is_listen_forced for unit_request, _ in finished_units[1:]
    ] == [True, is_listen_held, is_listen_held]


def speak_two_units(
    adapter: ThinkerAdapter,
    request: OmniRequest,
    *,
    first_unit_ends_turn: bool,
    second_unit_ends_turn: bool,
) -> None:
    """Two spoken units, PLAIN then OTHER, each closing the turn when asked, with the sampler state as the duplex sampler leaves it."""
    state = adapter.states[SESSION]
    first = adapter.build(SESSION, audio_unit(0), step_plan_payload(request))
    first_output_ids, first_generated_ids = spoken_output(
        PLAIN_TOKEN_ID, first_unit_ends_turn
    )
    state.generated_history = list(first_generated_ids)
    state.is_turn_ended = first_unit_ends_turn
    complete_unit(
        adapter,
        first,
        prompt_length=FIRST_UNIT_PROMPT_LENGTH,
        output_ids=first_output_ids,
        generated_unit_ids=first_generated_ids,
    )
    second = adapter.build(SESSION, audio_unit(1), step_plan_payload(request))
    second_output_ids, second_generated_ids = spoken_output(
        OTHER_TOKEN_ID, second_unit_ends_turn
    )
    state.generated_history = [*first_generated_ids, *second_generated_ids]
    state.is_turn_ended = second_unit_ends_turn
    complete_unit(
        adapter,
        second,
        prompt_length=SECOND_UNIT_PROMPT_LENGTH,
        output_ids=second_output_ids,
        generated_unit_ids=second_generated_ids,
    )


@pytest.mark.parametrize(
    (
        "first_unit_ends_turn",
        "second_unit_ends_turn",
        "listen_unit_follows",
        "last_delivered_unit",
        "kept_token_count",
        "generated_history",
        "interrupted_input_ids",
    ),
    [
        (
            False,
            True,
            False,
            0,
            SECOND_UNIT_PROMPT_LENGTH,
            [PLAIN_TOKEN_ID],
            [
                SPECIAL_TOKENS.turn_eos,
                SPECIAL_TOKENS.listen,
                SPECIAL_TOKENS.unit_end,
                *UNIT_INPUT_IDS,
            ],
        ),
        (
            True,
            False,
            False,
            0,
            SECOND_UNIT_PROMPT_LENGTH,
            [PLAIN_TOKEN_ID, SPECIAL_TOKENS.turn_eos],
            [SPECIAL_TOKENS.listen, SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS],
        ),
        (
            False,
            True,
            False,
            1,
            None,
            [PLAIN_TOKEN_ID, OTHER_TOKEN_ID, SPECIAL_TOKENS.turn_eos],
            [SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS],
        ),
        (
            False,
            True,
            True,
            0,
            SECOND_UNIT_PROMPT_LENGTH,
            [PLAIN_TOKEN_ID],
            [
                SPECIAL_TOKENS.turn_eos,
                SPECIAL_TOKENS.listen,
                SPECIAL_TOKENS.unit_end,
                *UNIT_INPUT_IDS,
                SPECIAL_TOKENS.listen,
                SPECIAL_TOKENS.unit_end,
                *UNIT_INPUT_IDS,
            ],
        ),
    ],
    ids=[
        "the_second_unit_was_not_received_with_the_turn_open_before_it",
        "the_second_unit_was_not_received_with_the_turn_closed_before_it",
        "both_units_were_received",
        "a_listen_unit_followed_the_unreceived_spoken_one",
    ],
)
def test_an_interrupt_drops_the_spoken_unit_the_client_did_not_receive(
    first_unit_ends_turn: bool,
    second_unit_ends_turn: bool,
    listen_unit_follows: bool,
    last_delivered_unit: int,
    kept_token_count: int | None,
    generated_history: list[int],
    interrupted_input_ids: list[int],
) -> None:
    """A client stop that names an earlier unit as the last one received: the next unit keeps the session up to the dropped unit's prompt and refills that unit as the official forced listen, turn_eos then listen when the turn was open before it and listen alone otherwise, the repetition window returns to what it was before it, and the forced listen adds no second turn_eos. A stop after a unit the client received cuts nothing; a listen unit that followed the unreceived spoken unit is re-fed after the refill, its own ids then a listen."""
    adapter, request = open_duplex_adapter()
    state = adapter.states[SESSION]
    speak_two_units(
        adapter,
        request,
        first_unit_ends_turn=first_unit_ends_turn,
        second_unit_ends_turn=second_unit_ends_turn,
    )
    if listen_unit_follows:
        listening = adapter.build(SESSION, audio_unit(2), step_plan_payload(request))
        complete_unit(
            adapter,
            listening,
            prompt_length=THIRD_UNIT_PROMPT_LENGTH,
            output_ids=[SPECIAL_TOKENS.listen],
            generated_unit_ids=[],
        )
        next_input_seq = 3
    else:
        next_input_seq = 2

    adapter.control(SESSION, interrupt_after(last_delivered_unit))
    preparation = adapter.prepare_unit(
        SESSION, audio_unit(next_input_seq), step_plan_payload(request)
    )
    interrupted = adapter.build(
        SESSION, audio_unit(next_input_seq), step_plan_payload(request)
    )
    interrupted.output_ids = [SPECIAL_TOKENS.listen]
    adapter.result(SESSION, interrupted)
    following_preparation = adapter.prepare_unit(
        SESSION, audio_unit(next_input_seq + 1), step_plan_payload(request)
    )
    following = adapter.build(
        SESSION, audio_unit(next_input_seq + 1), step_plan_payload(request)
    )

    assert preparation == ARSessionPreparation(kept_token_count=kept_token_count)
    assert state.generated_history == generated_history
    assert interrupted.input_ids.tolist() == interrupted_input_ids
    assert interrupted.is_listen_forced is True
    assert following_preparation == ARSessionPreparation()
    assert following.input_ids.tolist() == [SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS]


@pytest.mark.parametrize(
    ("payload", "is_listen_forced", "next_input_ids"),
    [
        ({"type": INTERRUPT}, True, [SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS]),
        (
            {"type": INTERRUPT, "keep_through_unit": "0"},
            True,
            [SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS],
        ),
        ({"type": TRUNCATE}, False, [SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS]),
        (
            {"type": TRUNCATE, "keep_through_unit": "0"},
            False,
            [SPECIAL_TOKENS.unit_end, *UNIT_INPUT_IDS],
        ),
    ],
    ids=[
        "an_interrupt_without_a_kept_unit",
        "an_interrupt_with_a_kept_unit_that_is_not_an_int",
        "a_truncate_without_a_kept_unit",
        "a_truncate_with_a_kept_unit_that_is_not_an_int",
    ],
)
def test_a_control_without_a_usable_unit_cuts_nothing(
    payload: dict[str, object], is_listen_forced: bool, next_input_ids: list[int]
) -> None:
    """An interrupt or truncate that does not say which unit to keep, as any session client may send, drops nothing from a reply the model ended: an interrupt still stops the reply with a forced listen, a truncate leaves the next unit alone, and the session keeps the spoken unit."""
    adapter, request = open_duplex_adapter()
    state = adapter.states[SESSION]
    first = adapter.build(SESSION, audio_unit(0), step_plan_payload(request))
    first_output_ids, first_generated_ids = spoken_output(PLAIN_TOKEN_ID, True)
    state.generated_history = list(first_generated_ids)
    state.is_turn_ended = True
    complete_unit(
        adapter,
        first,
        prompt_length=FIRST_UNIT_PROMPT_LENGTH,
        output_ids=first_output_ids,
        generated_unit_ids=first_generated_ids,
    )

    adapter.control(SESSION, session_control(payload))
    preparation = adapter.prepare_unit(
        SESSION, audio_unit(1), step_plan_payload(request)
    )
    following = adapter.build(SESSION, audio_unit(1), step_plan_payload(request))

    assert preparation == ARSessionPreparation()
    assert following.is_listen_forced is is_listen_forced
    assert following.input_ids.tolist() == next_input_ids
    assert state.generated_history == first_generated_ids


def perception_payload(request: OmniRequest, unit_index: int) -> StagePayload:
    """The perception plan of unit unit_index: <unit> and one audio placeholder whose embedding row is filled with the unit index."""
    return StagePayload(
        "unit",
        request,
        {
            "token_ids": [SPECIAL_TOKENS.unit, AUDIO_PLACEHOLDER_ID_BASE + unit_index],
            "input_embeds": torch.full((1, 2), float(unit_index)),
            "embedding_spans": [
                {
                    "modality": "audio",
                    "token_start": 1,
                    "token_end": 2,
                    "embedding_start": 0,
                    "embedding_end": 1,
                }
            ],
        },
    )


def feed_unit(
    adapter: ThinkerAdapter,
    request: OmniRequest,
    unit_index: int,
    *,
    spoken_token_id: int | None,
    ends_turn: bool = False,
) -> DuplexUnitRequestData:
    """Build and finish unit unit_index on its perception plan: a listen unit, or one that speaks spoken_token_id and closes the turn when ends_turn, with the sampler state as the duplex sampler leaves it."""
    state = adapter.states[SESSION]
    unit_request = adapter.build(
        SESSION, audio_unit(unit_index), perception_payload(request, unit_index)
    )
    if spoken_token_id is None:
        output_ids: list[int] = [SPECIAL_TOKENS.listen]
        generated_unit_ids: list[int] = []
    else:
        output_ids, generated_unit_ids = spoken_output(spoken_token_id, ends_turn)
        state.generated_history = [*state.generated_history, *generated_unit_ids]
        state.is_turn_ended = ends_turn
    complete_unit(
        adapter,
        unit_request,
        prompt_length=UNIT_PROMPT_LENGTHS[unit_index],
        output_ids=output_ids,
        generated_unit_ids=generated_unit_ids,
    )
    return unit_request


def spans_of(unit_request: DuplexUnitRequestData) -> list[tuple[int, int, float]]:
    """Each unit span of the request as (start, end, the unit index its embedding row carries)."""
    return [
        (span.start, span.end, span.input_embeds[0, 0].item())
        for span in unit_request.unit_embedding_spans
    ]


@pytest.mark.parametrize(
    "cut_controls",
    [[interrupt_after(0)], [interrupt_after(2), truncate_through(0)]],
    ids=[
        "a_stop_after_the_client_received_the_first_unit_only",
        "a_truncate_played_through_the_first_unit_after_a_stop_that_cut_nothing",
    ],
)
def test_a_cut_at_an_earlier_unit_rewinds_there_and_re_feeds_the_later_units_as_listen_units(
    cut_controls: list[TimedChunk],
) -> None:
    """A stop, or a truncate after a stop, that keeps a three-unit reply through its first unit: the next unit keeps the session up to the second unit's prompt, refills that unit as the forced listen, turn_eos first since the turn was open before it, re-feeds the third unit's own ids and embedding span followed by a listen, then adds its own ids and span after them, with the repetition window from before the second unit."""
    adapter, request = open_duplex_adapter()
    state = adapter.states[SESSION]
    feed_unit(adapter, request, 0, spoken_token_id=PLAIN_TOKEN_ID)
    feed_unit(adapter, request, 1, spoken_token_id=OTHER_TOKEN_ID)
    feed_unit(adapter, request, 2, spoken_token_id=THIRD_TOKEN_ID)

    for cut_control in cut_controls:
        adapter.control(SESSION, cut_control)
    preparation = adapter.prepare_unit(
        SESSION, audio_unit(3), perception_payload(request, 3)
    )
    cut = adapter.build(SESSION, audio_unit(3), perception_payload(request, 3))

    assert preparation == ARSessionPreparation(
        kept_token_count=SECOND_UNIT_PROMPT_LENGTH
    )
    assert cut.input_ids.tolist() == [
        SPECIAL_TOKENS.turn_eos,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        82,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        83,
    ]
    assert spans_of(cut) == [(4, 5, 2.0), (8, 9, 3.0)]
    assert cut.is_listen_forced is True
    assert state.generated_history == [PLAIN_TOKEN_ID]
    assert state.is_turn_ended is True


def test_cuts_only_move_the_kept_position_back_and_re_feed_every_unit_since_it() -> (
    None
):
    """After a stop that cut nothing, a truncate cuts the reply at the unit the client did not play and re-feeds the forced listen unit; a repeated or later truncate changes nothing; an earlier one rewinds further and re-feeds every unit since that position, the ones a cut already re-fed included, each by its own ids and span. The truncates leave the stop holding the silent units after it to a listen."""
    adapter, request = open_duplex_adapter()
    state = adapter.states[SESSION]
    feed_unit(adapter, request, 0, spoken_token_id=PLAIN_TOKEN_ID)
    feed_unit(adapter, request, 1, spoken_token_id=OTHER_TOKEN_ID)
    feed_unit(adapter, request, 2, spoken_token_id=THIRD_TOKEN_ID)
    adapter.control(SESSION, interrupt_after(2))
    stop_preparation = adapter.prepare_unit(
        SESSION, audio_unit(3), perception_payload(request, 3)
    )
    forced_listen = feed_unit(adapter, request, 3, spoken_token_id=None)

    adapter.control(SESSION, truncate_through(1))
    first_cut_preparation = adapter.prepare_unit(
        SESSION, audio_unit(4), perception_payload(request, 4)
    )
    first_cut = feed_unit(adapter, request, 4, spoken_token_id=None)
    first_cut_history = list(state.generated_history)
    adapter.control(SESSION, truncate_through(1))
    adapter.control(SESSION, truncate_through(2))
    unchanged_preparation = adapter.prepare_unit(
        SESSION, audio_unit(5), perception_payload(request, 5)
    )
    unchanged = feed_unit(adapter, request, 5, spoken_token_id=None)
    adapter.control(SESSION, truncate_through(0))
    second_cut_preparation = adapter.prepare_unit(
        SESSION, audio_unit(6), perception_payload(request, 6)
    )
    second_cut = adapter.build(SESSION, audio_unit(6), perception_payload(request, 6))

    assert stop_preparation == ARSessionPreparation()
    assert forced_listen.input_ids.tolist() == [
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        83,
        SPECIAL_TOKENS.turn_eos,
    ]
    assert first_cut_preparation == ARSessionPreparation(
        kept_token_count=THIRD_UNIT_PROMPT_LENGTH
    )
    assert first_cut.input_ids.tolist() == [
        SPECIAL_TOKENS.turn_eos,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        83,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        84,
    ]
    assert spans_of(first_cut) == [(4, 5, 3.0), (8, 9, 4.0)]
    assert first_cut_history == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID]
    assert unchanged_preparation == ARSessionPreparation()
    assert unchanged.input_ids.tolist() == [
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        85,
    ]
    assert spans_of(unchanged) == [(2, 3, 5.0)]
    assert [
        unit_request.is_listen_forced
        for unit_request in (forced_listen, first_cut, unchanged, second_cut)
    ] == [True, True, True, True]
    assert second_cut_preparation == ARSessionPreparation(
        kept_token_count=SECOND_UNIT_PROMPT_LENGTH
    )
    assert second_cut.input_ids.tolist() == [
        SPECIAL_TOKENS.turn_eos,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        82,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        83,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        84,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        85,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        86,
    ]
    assert spans_of(second_cut) == [
        (4, 5, 2.0),
        (8, 9, 3.0),
        (12, 13, 4.0),
        (16, 17, 5.0),
        (20, 21, 6.0),
    ]
    assert state.generated_history == [PLAIN_TOKEN_ID]


def test_a_truncate_cuts_an_ended_reply_and_not_an_open_one() -> None:
    """A truncate cuts the latest reply once the model ended it itself, with the same rewind and re-feed as after a stop, drops nothing from a reply the model is still speaking, whose units the client goes on to hear, and cuts that reply in turn once it ends; a stop still cuts the open reply."""
    adapter, request = open_duplex_adapter()
    state = adapter.states[SESSION]
    feed_unit(adapter, request, 0, spoken_token_id=PLAIN_TOKEN_ID)
    feed_unit(adapter, request, 1, spoken_token_id=OTHER_TOKEN_ID, ends_turn=True)
    feed_unit(adapter, request, 2, spoken_token_id=None)
    adapter.control(SESSION, truncate_through(0))
    ended_preparation = adapter.prepare_unit(
        SESSION, audio_unit(3), perception_payload(request, 3)
    )
    ended_cut = feed_unit(adapter, request, 3, spoken_token_id=None)
    ended_cut_history = list(state.generated_history)
    feed_unit(adapter, request, 4, spoken_token_id=THIRD_TOKEN_ID)
    adapter.control(SESSION, truncate_through(3))
    open_preparation = adapter.prepare_unit(
        SESSION, audio_unit(5), perception_payload(request, 5)
    )
    open_reply = feed_unit(
        adapter, request, 5, spoken_token_id=PLAIN_TOKEN_ID, ends_turn=True
    )
    adapter.control(SESSION, truncate_through(4))
    second_cut_preparation = adapter.prepare_unit(
        SESSION, audio_unit(6), perception_payload(request, 6)
    )
    second_cut = feed_unit(adapter, request, 6, spoken_token_id=None)
    second_cut_history = list(state.generated_history)
    feed_unit(adapter, request, 7, spoken_token_id=OTHER_TOKEN_ID)
    adapter.control(SESSION, interrupt_after(6))
    stop_preparation = adapter.prepare_unit(
        SESSION, audio_unit(8), perception_payload(request, 8)
    )
    stopped = adapter.build(SESSION, audio_unit(8), perception_payload(request, 8))

    assert ended_preparation == ARSessionPreparation(
        kept_token_count=SECOND_UNIT_PROMPT_LENGTH
    )
    assert ended_cut.input_ids.tolist() == [
        SPECIAL_TOKENS.turn_eos,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        82,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        83,
    ]
    assert ended_cut.is_listen_forced is False
    assert ended_cut_history == [PLAIN_TOKEN_ID]
    assert open_preparation == ARSessionPreparation()
    assert open_reply.input_ids.tolist() == [
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        85,
    ]
    assert second_cut_preparation == ARSessionPreparation(
        kept_token_count=UNIT_PROMPT_LENGTHS[5]
    )
    assert second_cut.input_ids.tolist() == [
        SPECIAL_TOKENS.turn_eos,
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        86,
    ]
    assert second_cut_history == [PLAIN_TOKEN_ID, THIRD_TOKEN_ID]
    assert stop_preparation == ARSessionPreparation(
        kept_token_count=UNIT_PROMPT_LENGTHS[7]
    )
    assert stopped.input_ids.tolist() == [
        SPECIAL_TOKENS.listen,
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        88,
    ]
    assert stopped.is_listen_forced is True


def test_a_stopped_reply_is_no_longer_cut_past_the_truncate_window() -> None:
    """A stopped reply stays cuttable through TRUNCATE_WINDOW_UNITS listen units after its last spoken unit, each of which a cut re-feeds; one listen unit later a truncate drops nothing."""
    adapter, request = open_duplex_adapter()
    feed_unit(adapter, request, 0, spoken_token_id=PLAIN_TOKEN_ID)
    adapter.control(SESSION, interrupt_after(0))
    for unit_index in range(1, TRUNCATE_WINDOW_UNITS + 1):
        feed_unit(adapter, request, unit_index, spoken_token_id=None)
    adapter.control(SESSION, truncate_through(-1))
    within_preparation = adapter.prepare_unit(
        SESSION, audio_unit(11), perception_payload(request, 11)
    )
    within = feed_unit(adapter, request, 11, spoken_token_id=None)
    adapter.control(SESSION, truncate_through(-1))
    past_preparation = adapter.prepare_unit(
        SESSION, audio_unit(12), perception_payload(request, 12)
    )
    past = adapter.build(SESSION, audio_unit(12), perception_payload(request, 12))

    assert within_preparation == ARSessionPreparation(
        kept_token_count=FIRST_UNIT_PROMPT_LENGTH
    )
    assert within.input_ids.tolist() == [
        SPECIAL_TOKENS.listen,
        *[
            token_id
            for unit_index in range(1, TRUNCATE_WINDOW_UNITS + 1)
            for token_id in (
                SPECIAL_TOKENS.unit_end,
                SPECIAL_TOKENS.unit,
                AUDIO_PLACEHOLDER_ID_BASE + unit_index,
                SPECIAL_TOKENS.listen,
            )
        ],
        SPECIAL_TOKENS.unit_end,
        SPECIAL_TOKENS.unit,
        91,
    ]
    assert past_preparation == ARSessionPreparation()
    assert past.input_ids.tolist() == [SPECIAL_TOKENS.unit_end, SPECIAL_TOKENS.unit, 92]


def test_chunk_eos_from_the_first_sample_skips_the_second_stage() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state()
    state.is_turn_ended = False
    logits = unit_logits({SPECIAL_TOKENS.chunk_eos: 2.0, PLAIN_TOKEN_ID: 1.0})

    assert sample(runner, logits, [unit_request(state)]) == [SPECIAL_TOKENS.chunk_eos]
    assert state.generated_history == []
    assert state.is_turn_ended is False


def test_second_stage_masks_forbidden_tokens_and_penalizes_each_row_by_its_history() -> (
    None
):
    runner = duplex_runner(SPECIAL_TOKENS)
    masked = session_state()
    penalized = session_state(repetition_penalty=2.0)
    penalized.generated_history = [PLAIN_TOKEN_ID]
    unpenalized = session_state()
    unpenalized.generated_history = [PLAIN_TOKEN_ID]
    logits = unit_logits(
        {BAD_TOKEN_IDS[0]: 9.0, SPECIAL_TOKENS.tts_pad: 8.0, PLAIN_TOKEN_ID: 1.0},
        {PLAIN_TOKEN_ID: 6.0, OTHER_TOKEN_ID: 4.0},
        {PLAIN_TOKEN_ID: 6.0, OTHER_TOKEN_ID: 4.0},
    )

    token_ids = sample(
        runner,
        logits,
        [unit_request(masked), unit_request(penalized), unit_request(unpenalized)],
    )

    assert token_ids == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID, PLAIN_TOKEN_ID]
    assert penalized.generated_history == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID]
    assert unpenalized.generated_history == [PLAIN_TOKEN_ID, PLAIN_TOKEN_ID]


def test_repeated_history_tokens_are_penalized_once() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(repetition_penalty=2.0)
    state.generated_history = [PLAIN_TOKEN_ID, PLAIN_TOKEN_ID]
    logits = unit_logits({PLAIN_TOKEN_ID: 9.0, OTHER_TOKEN_ID: 4.0})

    assert sample(runner, logits, [unit_request(state)]) == [PLAIN_TOKEN_ID]


def test_history_keeps_only_the_repetition_window() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(repetition_window_size=2)
    state.generated_history = [1, 2]

    sample(runner, unit_logits({PLAIN_TOKEN_ID: 1.0}), [unit_request(state)])

    assert state.generated_history == [2, PLAIN_TOKEN_ID]


@pytest.mark.parametrize(
    ("is_turn_ended", "expected_token_id"),
    [(True, SPECIAL_TOKENS.listen), (False, SPECIAL_TOKENS.tts_bos)],
)
def test_listen_becomes_tts_bos_mid_turn(
    is_turn_ended: bool, expected_token_id: int
) -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state()
    state.is_turn_ended = is_turn_ended

    token_ids = sample(
        runner, unit_logits({SPECIAL_TOKENS.listen: 1.0}), [unit_request(state)]
    )

    assert token_ids == [expected_token_id]
    assert state.generated_history == [SPECIAL_TOKENS.listen]
    assert state.is_turn_ended is is_turn_ended


def test_turn_eos_ends_the_turn_and_a_plain_token_reopens_it() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state()
    state.is_turn_ended = False

    assert sample(
        runner, unit_logits({SPECIAL_TOKENS.turn_eos: 1.0}), [unit_request(state)]
    ) == [SPECIAL_TOKENS.turn_eos]
    assert state.is_turn_ended is True
    assert sample(
        runner, unit_logits({PLAIN_TOKEN_ID: 1.0}), [unit_request(state)]
    ) == [PLAIN_TOKEN_ID]
    assert state.is_turn_ended is False


@pytest.mark.parametrize(
    ("listen_prob_scale", "expected_token_id"),
    [(1.0, SPECIAL_TOKENS.listen), (0.5, PLAIN_TOKEN_ID)],
)
def test_listen_prob_scale_rescales_the_listen_logit(
    listen_prob_scale: float, expected_token_id: int
) -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(listen_prob_scale=listen_prob_scale)
    logits = unit_logits({SPECIAL_TOKENS.listen: 10.0, PLAIN_TOKEN_ID: 8.0})

    assert sample(runner, logits, [unit_request(state)]) == [expected_token_id]


def test_sampled_and_greedy_rows_share_one_step() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    greedy = session_state()
    sampled = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    closing = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    logits = unit_logits(
        {PLAIN_TOKEN_ID: 1.0, OTHER_TOKEN_ID: 0.0},
        {OTHER_TOKEN_ID: 1.0},
        {SPECIAL_TOKENS.chunk_eos: 1.0},
    )

    token_ids = sample(
        runner,
        logits,
        [unit_request(greedy), unit_request(sampled), unit_request(closing)],
    )

    assert token_ids == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID, SPECIAL_TOKENS.chunk_eos]
    assert (
        greedy.generated_history,
        sampled.generated_history,
        closing.generated_history,
    ) == ([PLAIN_TOKEN_ID], [OTHER_TOKEN_ID], [])


def test_sampled_rows_draw_from_their_filtered_support() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    sampled = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    closing = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    zero_temperature = session_state(greedy=False, temperature=0.0)
    logits = unit_logits(
        {OTHER_TOKEN_ID: 1.0},
        {SPECIAL_TOKENS.chunk_eos: 1.0},
        {PLAIN_TOKEN_ID: 2.0, OTHER_TOKEN_ID: 1.0},
    )

    token_ids = sample(
        runner,
        logits,
        [unit_request(sampled), unit_request(closing), unit_request(zero_temperature)],
    )

    assert token_ids == [OTHER_TOKEN_ID, SPECIAL_TOKENS.chunk_eos, PLAIN_TOKEN_ID]
    assert (
        sampled.generated_history,
        closing.generated_history,
        zero_temperature.generated_history,
    ) == ([OTHER_TOKEN_ID], [], [PLAIN_TOKEN_ID])


DRAW_COUNT = 6000
FREQUENCY_TOLERANCE = 0.03


@pytest.mark.parametrize(
    ("top_p", "expected_frequencies"),
    [
        (1.0, {PLAIN_TOKEN_ID: 0.7, OTHER_TOKEN_ID: 0.2, THIRD_TOKEN_ID: 0.1}),
        (
            0.75,
            {PLAIN_TOKEN_ID: 0.7 / 0.9, OTHER_TOKEN_ID: 0.2 / 0.9, THIRD_TOKEN_ID: 0.0},
        ),
    ],
    ids=["softmax", "nucleus-drops-the-tail"],
)
def test_sampled_tokens_follow_the_filtered_softmax(
    top_p: float, expected_frequencies: dict[int, float]
) -> None:
    torch.manual_seed(0)
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(greedy=False, temperature=1.0, top_k=-1, top_p=top_p)
    logits = unit_logits(
        {
            PLAIN_TOKEN_ID: math.log(0.7),
            OTHER_TOKEN_ID: math.log(0.2),
            THIRD_TOKEN_ID: math.log(0.1),
        }
    )
    counts = {token_id: 0 for token_id in expected_frequencies}

    for _ in range(DRAW_COUNT):
        (token_id,) = sample(runner, logits, [unit_request(state)])
        counts[token_id] += 1

    for token_id, expected in expected_frequencies.items():
        assert abs(counts[token_id] / DRAW_COUNT - expected) <= FREQUENCY_TOLERANCE


def test_top_k_and_top_p_filter_each_row_of_a_batch() -> None:
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]]))

    top_k_filtered = filter_top_k_top_p(logits, top_k=2, top_p=1.0)
    top_p_filtered = filter_top_k_top_p(logits, top_k=-1, top_p=0.5)

    torch.testing.assert_close(
        top_k_filtered,
        torch.tensor(
            [
                [logits[0, 0], logits[0, 1], -torch.inf],
                [logits[1, 0], logits[1, 1], -torch.inf],
            ]
        ),
    )
    torch.testing.assert_close(
        top_p_filtered,
        torch.tensor(
            [
                [logits[0, 0], -torch.inf, -torch.inf],
                [logits[1, 0], logits[1, 1], -torch.inf],
            ]
        ),
    )


@pytest.mark.parametrize("greedy", [True, False])
def test_one_step_moves_token_ids_to_the_host_exactly_twice(
    monkeypatch: pytest.MonkeyPatch, greedy: bool
) -> None:
    """A step costs two Python-level transfers, not two per row, and never a multinomial sync."""
    transfers: list[str] = []
    multinomial_calls: list[int] = []
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist
    original_multinomial = torch.multinomial

    def counted_item(tensor: torch.Tensor) -> int | float:
        transfers.append("item")
        return original_item(tensor)

    def counted_tolist(tensor: torch.Tensor) -> list[int]:
        transfers.append("tolist")
        return original_tolist(tensor)

    def counted_multinomial(*arguments: object, **keywords: object) -> torch.Tensor:
        multinomial_calls.append(1)
        return original_multinomial(*arguments, **keywords)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    monkeypatch.setattr(torch, "multinomial", counted_multinomial)
    runner = duplex_runner(SPECIAL_TOKENS)
    states = [session_state(greedy=greedy, temperature=0.7) for _ in range(6)]
    logits = unit_logits(*([{PLAIN_TOKEN_ID: 1.0}] * len(states)))

    token_ids = runner.sample_next_token_ids(
        SimpleNamespace(next_token_logits=logits),
        None,
        None,
        [unit_request(state) for state in states],
    )

    assert transfers == ["tolist", "tolist"]
    assert multinomial_calls == []
    assert token_ids.tolist() == [PLAIN_TOKEN_ID] * len(states)
    assert [state.generated_history for state in states] == [[PLAIN_TOKEN_ID]] * len(
        states
    )


def test_mixed_decode_tail_prefill_rows_follow_the_batch_span() -> None:
    """A decode tail mixed into a prefill batch embeds its one-token batch span, not the
    window from its last prefill boundary that the request still records."""
    runner = object.__new__(DuplexThinkerModelRunner)
    runner.embed_tokens = torch.nn.Embedding(VOCAB_SIZE, 4)
    span_rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    fresh = SimpleNamespace(
        data=SGLangARRequestData(
            req=SimpleNamespace(
                prefix_len=0, extend_end=3, get_fill_ids=lambda: [5, 6, 9]
            ),
        )
    )
    tail_request = SimpleNamespace(
        prefix_len=100,
        extend_end=104,
        get_fill_ids=lambda: list(range(104)),
    )
    tail = SimpleNamespace(
        data=SGLangARRequestData(
            req=tail_request,
            session_embedding_spans=[
                EmbeddingSpan(start=101, end=104, input_embeds=span_rows)
            ],
        )
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        replace_embeds=None,
        extend_prefix_lens_cpu=[0, 103],
        extend_seq_lens_cpu=[3, 1],
    )

    runner.custom_prefill_forward(forward_batch, SimpleNamespace(), [fresh, tail])

    rows = get_omni_prefill_inputs(forward_batch).input_embeds
    assert rows.shape[0] == 4
    assert torch.equal(rows[:3], runner.embed_tokens(torch.tensor([5, 6, 9])))
    assert torch.equal(rows[3], span_rows[2])
    assert (tail_request.prefix_len, tail_request.extend_end) == (100, 104)
