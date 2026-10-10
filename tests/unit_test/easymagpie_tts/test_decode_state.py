# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.easymagpie_tts.decode_state import (
    EMIT_COLUMN,
    STOP_COLUMN,
    EasyMagpieDecodeState,
)
from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState

TEXT = [11, 12, 13, 14, 15, 63]


def tts_state(**overrides) -> EasyMagpieTTSState:
    fields = dict(
        text_token_ids=TEXT, phoneme_delay=3, speech_delay=5, text_prefill_num=4
    )
    fields.update(overrides)
    return EasyMagpieTTSState(**fields)


@pytest.fixture
def state(tiny_tts_config) -> EasyMagpieDecodeState:
    return EasyMagpieDecodeState.allocate(
        tiny_tts_config,
        num_slots=4,
        text_capacity=8,
        max_batch=4,
        max_top_k=16,
        device=torch.device("cpu"),
    )


def seed(state, slots, states, prefill_phonemes, seeds=None) -> None:
    state.seed(
        slots=slots,
        states=states,
        seeds=seeds or [7] * len(slots),
        prefill_phonemes=torch.tensor(prefill_phonemes),
    )


def step(state, slots, *, codes=3, phoneme=9, eos=False):
    inputs = state.read(torch.tensor(slots))
    rows = len(slots)
    state.commit(
        inputs,
        codes=torch.full((rows, 4), codes),
        phonemes=torch.full((rows, 1), phoneme),
        eos=torch.full((rows,), eos),
    )
    return inputs


def test_allocation_reserves_the_padding_slot(state) -> None:
    assert state.num_slots == 4
    assert state.offsets.shape == (5,)
    assert state.max_batch == 4
    assert state.step_output.shape == (4, 6)
    with pytest.raises(ValueError, match="positive"):
        EasyMagpieDecodeState.allocate(
            state.config,
            num_slots=0,
            text_capacity=8,
            max_batch=4,
            max_top_k=16,
            device=torch.device("cpu"),
        )


def test_decode_resumes_after_the_prefill_lead_in(state) -> None:
    seed(state, [2], [tts_state()], [[9]])
    inputs = state.read(torch.tensor([2]))
    assert inputs.steps.tolist() == [4]
    assert inputs.text_tokens.tolist() == [15]
    assert inputs.phoneme_tokens.tolist() == [[9]]
    assert inputs.phoneme_valid.tolist() == [True]
    assert inputs.audio_valid.tolist() == [False]
    assert inputs.positions.tolist() == [5 * 4]


def test_decode_applies_phoneme_and_speech_delays(state) -> None:
    seed(state, [1], [tts_state(text_prefill_num=2)], [[9]])

    inputs = step(state, [1])
    assert inputs.phoneme_valid.tolist() == [False]
    assert inputs.audio_valid.tolist() == [False]

    inputs = step(state, [1])
    assert inputs.phoneme_tokens.tolist() == [[17]]
    assert inputs.audio_valid.tolist() == [False]

    step(state, [1])
    inputs = step(state, [1])
    assert inputs.audio_codes.tolist() == [[16] * 4]
    assert inputs.audio_valid.tolist() == [True]
    assert inputs.text_tokens.tolist() == [63]

    inputs = step(state, [1])
    assert inputs.audio_codes.tolist() == [[3] * 4]
    assert inputs.text_valid.tolist() == [False]


def test_a_phoneme_eos_is_fed_once_then_the_channel_closes(state) -> None:
    seed(state, [1], [tts_state()], [[18]])

    inputs = step(state, [1], phoneme=5)
    assert inputs.phoneme_tokens.tolist() == [[18]]
    assert inputs.phoneme_ended.tolist() == [True]

    assert step(state, [1]).phoneme_valid.tolist() == [False]


def test_sampling_controls_follow_the_slot_not_the_row(state) -> None:
    states = [tts_state(temperature=0.5, top_k=3), tts_state(temperature=0.9, top_k=6)]
    seed(state, [1, 3], states, [[9], [9]], seeds=[7, 22])
    step(state, [3])

    inputs = state.read(torch.tensor([3, 1]))
    assert inputs.temperatures.tolist() == pytest.approx([0.9, 0.5])
    assert inputs.top_ks.tolist() == [6, 3]
    assert inputs.seeds.tolist() == [22, 7]
    assert inputs.positions.tolist() == [6 * 4, 5 * 4]


def test_padding_rows_only_touch_the_padding_slot(state) -> None:
    seed(state, [1], [tts_state()], [[9]])
    step(state, [1, 0, 0], codes=4)

    inputs = state.read(torch.tensor([1]))
    assert inputs.steps.tolist() == [5]
    assert inputs.audio_codes.tolist() == [[16] * 4]
    assert state.offsets[2:].tolist() == [0, 0, 0]


def test_seeding_resets_a_reused_slot(state) -> None:
    seed(state, [1], [tts_state()], [[18]])
    step(state, [1])
    seed(state, [1], [tts_state(speech_delay=4)], [[9]])

    inputs = state.read(torch.tensor([1]))
    assert inputs.steps.tolist() == [4]
    assert inputs.phoneme_valid.tolist() == [True]
    assert inputs.audio_codes.tolist() == [[16] * 4]


def test_commit_publishes_codes_audio_rows_and_stop_tokens(state) -> None:
    seed(state, [1, 2], [tts_state(speech_delay=4), tts_state()], [[9], [9]])
    inputs = state.read(torch.tensor([1, 2]))
    state.commit(
        inputs,
        codes=torch.tensor([[0, 1, 2, 3], [5, 5, 5, 5]]),
        phonemes=torch.tensor([[9], [9]]),
        eos=torch.tensor([False, False]),
    )
    output = state.step_output[:2]
    assert output[:, :4].tolist() == [[0, 1, 2, 3], [5, 5, 5, 5]]
    assert output[:, EMIT_COLUMN].tolist() == [1, 0]
    assert output[:, STOP_COLUMN].tolist() == [0, 0]

    state.commit(
        state.read(torch.tensor([1])),
        codes=torch.zeros((1, 4), dtype=torch.long),
        phonemes=torch.tensor([[9]]),
        eos=torch.tensor([True]),
    )
    assert state.step_output[0, EMIT_COLUMN].item() == 0
    assert state.step_output[0, STOP_COLUMN].item() == 1
