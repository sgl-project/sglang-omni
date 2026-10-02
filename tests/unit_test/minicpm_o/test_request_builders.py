# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker request construction."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_sglang_thinker_request

VOCAB_SIZE = 256


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
