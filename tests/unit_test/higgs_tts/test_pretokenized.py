# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Higgs TTS pre-tokenized rollout input."""

from __future__ import annotations

import pytest

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.pretokenized import (
    build_pretokenized_state,
    is_pretokenized_prompt,
)


def test_is_pretokenized_prompt_true_for_nonempty_int_list() -> None:
    assert is_pretokenized_prompt([1, 2, 3]) is True


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"text": "hello"},
        "hello",
        [],
        [1, "x", 3],
        None,
        (1, 2, 3),
    ],
)
def test_is_pretokenized_prompt_false_for_other_shapes(inputs) -> None:
    assert is_pretokenized_prompt(inputs) is False


def test_build_pretokenized_state_uses_ids_verbatim() -> None:
    state = build_pretokenized_state(
        [5, 6, 7],
        {
            "max_new_tokens": 64,
            "temperature": 1.0,
            "seed": 1,
            "return_logprob": True,
            "return_omni_rollout": True,
        },
    )

    assert isinstance(state, HiggsTtsState)
    assert state.prompt_token_ids == [5, 6, 7]
    assert state.reference_codes_delayed is None
    assert state.reference_waveform is None
    assert state.target_text is None
    assert state.reference_text is None
    assert state.max_new_tokens == 64
    assert state.temperature == 1.0
    assert state.top_p is None
    assert state.top_k is None
    assert state.seed == 1
    assert state.return_logprob is True
    assert state.return_omni_rollout is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_p", 0.1),
        ("repetition_penalty", 1.1),
    ],
)
def test_build_pretokenized_state_rejects_unsupported_action_logprobs(
    field, value
) -> None:
    with pytest.raises(ValueError, match="support this sampling transform"):
        build_pretokenized_state(
            [5, 6, 7],
            {
                "return_logprob": True,
                "return_omni_rollout": True,
                field: value,
            },
        )


def test_build_pretokenized_state_accepts_exact_filtered_action_logprobs() -> None:
    state = build_pretokenized_state(
        [5, 6, 7],
        {
            "return_logprob": True,
            "return_omni_rollout": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
        },
    )

    assert state.temperature == 0.7
    assert state.top_p == 0.9
    assert state.top_k == 20


def test_build_pretokenized_state_defaults_and_roundtrip() -> None:
    state = build_pretokenized_state([10, 11], None)

    assert state.max_new_tokens == 2048
    assert state.temperature == 1.0
    assert state.top_p is None
    assert state.top_k is None
    assert state.seed is None
    assert state.num_codebooks == 8
    assert state.codebook_size == 1026

    data = state.to_dict()
    assert data["prompt_token_ids"] == [10, 11]
    assert "reference_codes_delayed" not in data
    assert "reference_waveform" not in data
    assert HiggsTtsState.from_dict(data).prompt_token_ids == [10, 11]


def test_build_pretokenized_state_custom_codebooks() -> None:
    state = build_pretokenized_state([9], {}, num_codebooks=4, codebook_size=512)

    assert state.num_codebooks == 4
    assert state.codebook_size == 512
    assert state.prompt_token_ids == [9]
