# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.easymagpie_tts.payload_types import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EasyMagpieTTSState,
)
from sglang_omni.models.easymagpie_tts.request_builders import (
    STOP_TOKEN_ID,
    apply_easymagpie_result,
    build_easymagpie_state,
    build_sglang_easymagpie_request,
    easymagpie_stream_output_builder,
    max_decode_tokens,
    prompt_cache_key,
)
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(inputs, params=None, tts_params=None, data=None) -> StagePayload:
    return StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data=data or {},
    )


def test_state_normalizes_openai_input_and_tts_params() -> None:
    state = build_easymagpie_state(
        make_payload(
            {"input": "  hello  "},
            tts_params={"voice": "spk", "top_k": 32, "seed": 7, "context_text": "[DE]"},
        )
    )
    assert (state.text, state.voice, state.context_text) == ("hello", "spk", "[DE]")
    assert (state.top_k, state.seed, state.temperature) == (32, 7, DEFAULT_TEMPERATURE)


def test_endpoint_defaults_do_not_override_model_defaults() -> None:
    params = {"temperature": 1.0, "top_k": 5, "seed": 3, "max_new_tokens": 9}
    implicit = build_easymagpie_state(make_payload("hi", params=params))
    assert (implicit.temperature, implicit.top_k, implicit.seed) == (
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_K,
        None,
    )
    explicit = build_easymagpie_state(
        make_payload(
            "hi",
            params=params,
            tts_params={"explicit_generation_params": list(params)},
        )
    )
    assert (explicit.temperature, explicit.top_k, explicit.seed) == (1.0, 5, 3)
    assert explicit.max_new_frames == 9


@pytest.mark.parametrize("inputs", [None, "", "   ", {"text": "  "}])
def test_state_rejects_empty_text(inputs) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_easymagpie_state(make_payload(inputs))


@pytest.mark.parametrize(
    "tts_params",
    [{"seed": True}, {"temperature": 0}, {"top_k": -1}, {"max_new_frames": -2}],
)
def test_state_rejects_invalid_generation_params(tts_params) -> None:
    with pytest.raises(ValueError):
        build_easymagpie_state(make_payload("hi", tts_params=tts_params))


def preprocessed_state(**overrides) -> EasyMagpieTTSState:
    values = {
        "text": "hello",
        "text_token_ids": [11, 12, 13, 14, 15, 63],
        "context_token_ids": [7, 8],
        "speaker_embedding": torch.ones((3, 8), dtype=torch.float16),
        "phoneme_delay": 3,
        "speech_delay": 5,
        "text_prefill_num": 4,
        "max_new_frames": 10,
        "seed": 5,
    }
    values.update(overrides)
    return EasyMagpieTTSState(**values)


def test_sglang_request_covers_speaker_context_and_text_lead_in() -> None:
    state = preprocessed_state()
    data = build_sglang_easymagpie_request(make_payload("hello", data=state.to_dict()))
    assert data.input_ids.numel() == 3 + 2 + 4
    assert data.decode_offset == state.text_prefill_num
    assert data.req.sampling_params.max_new_tokens == max_decode_tokens(state)
    assert max_decode_tokens(state) == 10 + 5 - 4 + 1
    assert data.req.eos_token_ids == {STOP_TOKEN_ID}
    assert data.sampling_seed == 5
    assert data.output_ids is data.req.output_ids


def test_cache_key_separates_prompts_with_identical_placeholder_ids() -> None:
    base = preprocessed_state()
    assert prompt_cache_key(base) == prompt_cache_key(preprocessed_state())
    assert prompt_cache_key(base) != prompt_cache_key(preprocessed_state(text="bye"))
    assert prompt_cache_key(base) != prompt_cache_key(preprocessed_state(voice="x"))


def test_result_carries_codes_and_usage_but_not_the_speaker_rows() -> None:
    state = preprocessed_state()
    data = build_sglang_easymagpie_request(make_payload("hello", data=state.to_dict()))
    data.output_codes = [torch.arange(4), torch.arange(4) + 1]
    result = EasyMagpieTTSState.from_dict(apply_easymagpie_result(data).data)
    assert result.audio_codes.tolist() == [[0, 1, 2, 3], [1, 2, 3, 4]]
    assert result.speaker_embedding is None
    assert (result.prompt_tokens, result.completion_tokens) == (9, 2)


def test_stream_builder_sends_each_new_frame_once() -> None:
    state = preprocessed_state()
    payload = make_payload("hello", params={"stream": True}, data=state.to_dict())
    data = build_sglang_easymagpie_request(payload)
    assert easymagpie_stream_output_builder("req-0", data, None) == []

    data.output_codes = [torch.arange(4), torch.arange(4) + 1]
    (message,) = easymagpie_stream_output_builder("req-0", data, None)
    assert (message.type, message.target) == ("stream", "vocoder")
    assert message.metadata == {"modality": "audio_codes", "stream": True}
    assert message.data.tolist() == [[0, 1, 2, 3], [1, 2, 3, 4]]

    data.output_codes.append(torch.arange(4) + 2)
    (message,) = easymagpie_stream_output_builder("req-0", data, None)
    assert message.data.tolist() == [[2, 3, 4, 5]]

    result = EasyMagpieTTSState.from_dict(apply_easymagpie_result(data).data)
    assert result.audio_codes is None
    assert result.completion_tokens == 3


def test_stream_builder_ignores_offline_requests() -> None:
    state = preprocessed_state()
    data = build_sglang_easymagpie_request(make_payload("hello", data=state.to_dict()))
    data.output_codes = [torch.arange(4)]
    assert easymagpie_stream_output_builder("req-0", data, None) == []
