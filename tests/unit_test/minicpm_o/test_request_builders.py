# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker request construction."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_sglang_thinker_request

VOCAB_SIZE = 256


def build(params: dict) -> None:
    build_sglang_thinker_request(
        MiniCPMOPipelineState(prompt={"input_ids": torch.tensor([1, 2, 3])}),
        params=params,
        tokenizer=None,
        vocab_size=32000,
    )


def thinker_params(length_penalty: object) -> dict:
    return {"stage_params": {"thinker": {"length_penalty": length_penalty}}}


@pytest.mark.parametrize(
    "params",
    [{}, {"stage_params": {"thinker": None}}, thinker_params(1.0), thinker_params(1.3)],
)
def test_missing_or_positive_length_penalty_builds(params: dict) -> None:
    build(params)


@pytest.mark.parametrize(
    ("length_penalty", "error"),
    [
        (None, TypeError),
        ("1.3", TypeError),
        (0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_invalid_length_penalty_fails_at_build(
    length_penalty: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        build(thinker_params(length_penalty))


def test_prompt_ids_are_checked_before_media_placeholders_are_remapped() -> None:
    media_state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([10, 255, 255, 11])},
        mm_inputs={
            "image": {"bounds": torch.tensor([[1, 3]]), "cache_key": "image:cache"}
        },
        thinker_inputs={"model_inputs": {"pixel_values": torch.ones(1)}},
    )
    text_state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([VOCAB_SIZE - 1, VOCAB_SIZE])}
    )

    media_request = build_sglang_thinker_request(
        media_state,
        params={"max_new_tokens": 3},
        tokenizer=None,
        vocab_size=VOCAB_SIZE,
        request_id="minicpm-o-media",
    )
    with pytest.raises(ValueError, match=f"token id {VOCAB_SIZE} at position 1"):
        build_sglang_thinker_request(
            text_state,
            params={"max_new_tokens": 3},
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
            request_id="minicpm-o-text",
        )

    assert media_request.req.origin_input_ids[1] >= VOCAB_SIZE
