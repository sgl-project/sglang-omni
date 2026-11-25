# SPDX-License-Identifier: Apache-2.0
"""LFM3 request boundary: checkpoint sampling defaults, seeds, empty prompts and the result payload."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.lfm3_audio.hf_config import (
    Lfm3AudioTokens,
    SpeechSamplingDefaults,
)
from sglang_omni.models.lfm3_audio.payload_types import Lfm3AudioState
from sglang_omni.models.lfm3_audio.request_builders import apply_result, build_request
from sglang_omni.proto import OmniRequest, StagePayload

TOKENS = Lfm3AudioTokens()
DEFAULTS = SpeechSamplingDefaults(
    temperature=0.4, top_k=None, repetition_penalty=1.2, repetition_window=4
)


def payload(**fields: object) -> StagePayload:
    state = Lfm3AudioState(input_ids=[1, 2, TOKENS.audio_start_token_id], **fields)
    return StagePayload(
        request_id="request", request=OmniRequest(inputs=""), data=state.to_dict()
    )


def build(**fields: object):
    return build_request(
        payload(**fields),
        vocab_size=200_000,
        tokens=TOKENS,
        speech_defaults=DEFAULTS,
        device="cpu",
    )


def test_unset_speech_fields_take_the_checkpoint_defaults() -> None:
    sampling = build().speech.sampling
    assert (sampling.temperature, sampling.top_k, sampling.repetition_penalty) == (
        0.4,
        None,
        1.2,
    )
    assert sampling.repetition_window == DEFAULTS.repetition_window


def test_set_speech_fields_override_the_defaults() -> None:
    sampling = build(
        audio_temperature=0.0, audio_top_k=8, audio_repetition_penalty=1.0
    ).speech.sampling
    assert (sampling.temperature, sampling.top_k, sampling.repetition_penalty) == (
        0.0,
        8,
        1.0,
    )


def test_a_seed_fixes_the_speech_generator_and_the_text_sampling_seed() -> None:
    first, second = build(seed=7), build(seed=7)
    assert torch.equal(
        torch.rand(4, generator=first.speech.sampling.generator),
        torch.rand(4, generator=second.speech.sampling.generator),
    )
    assert first.req.sampling_params.sampling_seed == 7
    assert first.speech.sampling.seed == second.speech.sampling.seed


def test_a_prompt_ending_in_audio_start_speaks_and_stops_on_im_end() -> None:
    request_data = build(text_only=False)
    assert request_data.speech.needs_frame()
    assert request_data.req.sampling_params.stop_token_ids == {TOKENS.eos_token_ids[0]}


def test_an_empty_prompt_is_rejected() -> None:
    empty = StagePayload(
        request_id="request",
        request=OmniRequest(inputs=""),
        data=Lfm3AudioState(input_ids=[]).to_dict(),
    )
    with pytest.raises(ValueError, match="empty prompt"):
        build_request(
            empty,
            vocab_size=200_000,
            tokens=TOKENS,
            speech_defaults=DEFAULTS,
            device="cpu",
        )


def test_the_result_carries_ids_frames_and_finish_reason() -> None:
    request_data = build(input_features=torch.zeros(1, 128, 4))
    speech = request_data.speech
    speech.apply_frame(torch.full((TOKENS.num_codebooks,), 3))
    speech.apply_frame(torch.full((TOKENS.num_codebooks,), 4))
    request_data.finish_reason = "length"

    state = Lfm3AudioState.from_dict(
        apply_result(request_data.stage_payload, request_data).data
    )

    assert state.generated_ids == [TOKENS.audio_out_token_id] * 2
    assert state.audio_codes.shape == (2, TOKENS.num_codebooks)
    assert state.audio_codes.device.type == "cpu"
    assert state.input_features is None
    assert state.finish_reason == "length"
    assert (state.prompt_tokens, state.completion_tokens) == (3, 2)
