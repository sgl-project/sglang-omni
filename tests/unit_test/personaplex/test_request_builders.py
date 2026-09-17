# SPDX-License-Identifier: Apache-2.0
"""One decode step per caller frame; sampling knobs and seeds resolve as documented."""

import pytest
import torch

from sglang_omni.models.personaplex.architecture import (
    DEFAULT_AUDIO_TEMPERATURE,
    DEFAULT_AUDIO_TOP_K,
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_K,
    TEXT_PAD_ID,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
    lm_stream_output_builder,
    resolve_sampling,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_errors import is_bad_request_error


def _payload(
    num_frames: int, params=None, metadata=None, num_samples: int = 0
) -> StagePayload:
    state = PersonaPlexState(
        text_prompt_ids=[11, 12, 13],
        user_codes=torch.zeros(num_frames, 8, dtype=torch.long),
        num_samples=num_samples,
    )
    request = OmniRequest(inputs={}, params=params or {}, metadata=metadata or {})
    return StagePayload("r", request=request, data=state.to_dict())


def test_decode_budget_is_the_frame_count():
    data = build_lm_request(_payload(9), vocab_size=32000)
    assert data.max_new_tokens == 9
    assert data.req.sampling_params.max_new_tokens == 9
    assert data.req.sampling_params.ignore_eos
    timeline = data.talker_model_inputs["timeline"]
    assert len(data.input_ids) == timeline.num_prompt_positions
    assert data.input_ids[0].item() == TEXT_PAD_ID
    assert timeline.num_prompt_positions == 0 + 6 + 3 + 6


def test_sampling_defaults_and_overrides():
    sampling = resolve_sampling({})
    assert sampling.text_temperature == DEFAULT_TEXT_TEMPERATURE
    assert sampling.text_top_k == DEFAULT_TEXT_TOP_K
    assert sampling.audio.temperature == DEFAULT_AUDIO_TEMPERATURE
    assert sampling.seed is None and sampling.text_seed is None

    sampling = resolve_sampling(
        {"temperature": 0.0, "audio_temperature": 0, "seed": 42}
    )
    assert sampling.text_temperature == 0.0 and sampling.audio.greedy
    assert sampling.text_seed != sampling.audio_seed
    assert resolve_sampling({"seed": 42}).text_seed == sampling.text_seed

    data = build_lm_request(
        _payload(2, {"temperature": 0.0, "seed": 42}), vocab_size=32000
    )
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.sampling_seed == sampling.text_seed


def test_result_carries_text_ids_and_frames_and_drops_inputs():
    data = build_lm_request(_payload(2), vocab_size=32000)
    data.output_ids = [3, 17]
    data.talker_model_inputs["frames"] = [torch.arange(8), torch.arange(8) + 8]
    state = PersonaPlexState.from_dict(apply_lm_result(data).data)
    assert state.text_ids == [3, 17]
    assert state.codes.tolist() == [list(range(8)), list(range(8, 16))]
    assert state.user_codes is None and state.waveform is None


def test_stream_builder_ships_pending_frames_to_the_codec():
    data = build_lm_request(_payload(2, num_samples=3000), vocab_size=32000)
    assert lm_stream_output_builder("r", data, None) == []
    data.talker_model_inputs["pending_frames"].append(torch.arange(8))
    messages = lm_stream_output_builder("r", data, None)
    assert len(messages) == 1
    assert messages[0].target == "code2wav"
    assert messages[0].data.shape == (1, 8)
    assert messages[0].metadata["num_samples"] == 3000
    assert data.talker_model_inputs["pending_frames"] == []


def test_request_boundary_rejects_unusable_inputs():
    with pytest.raises(ValueError, match="80 ms frame"):
        build_lm_request(_payload(0), vocab_size=32000)
    no_audio = StagePayload(
        "r",
        request=OmniRequest(inputs={}, params={}),
        data=PersonaPlexState(text_prompt_ids=[11]).to_dict(),
    )
    with pytest.raises(ValueError, match="no encoded caller audio"):
        build_lm_request(no_audio, vocab_size=32000)
    with pytest.raises(ValueError, match="seed must be an integer"):
        resolve_sampling({"seed": True})


def test_client_filler_sampling_values_keep_the_reference_defaults():
    filler = {"temperature": 1.0, "top_k": -1, "seed": None}
    sampling = resolve_sampling(filler)
    assert sampling.text_temperature == DEFAULT_TEXT_TEMPERATURE
    assert sampling.text_top_k == DEFAULT_TEXT_TOP_K

    chosen = resolve_sampling(filler, explicit_fields=["temperature", "top_k"])
    assert chosen.text_temperature == 1.0 and chosen.text_top_k == -1

    staged = resolve_sampling(
        {**filler, "stage_sampling": {"lm": {"temperature": 0.3, "top_k": -1}}}
    )
    assert staged.text_temperature == 0.3 and staged.text_top_k == DEFAULT_TEXT_TOP_K

    stage_params = resolve_sampling(
        {**filler, "stage_params": {"lm": {"temperature": 1.0, "top_k": -1}}}
    )
    assert stage_params.text_temperature == 1.0 and stage_params.text_top_k == -1

    data = build_lm_request(
        _payload(2, filler, {EXPLICIT_GENERATION_PARAMS_KEY: ["temperature"]}),
        vocab_size=32000,
    )
    assert data.req.sampling_params.temperature == 1.0
    assert data.req.sampling_params.top_k == DEFAULT_TEXT_TOP_K


def test_lm_stage_params_set_audio_sampling_and_seed():
    sampling = resolve_sampling(
        {
            "audio_temperature": 0.5,
            "stage_params": {
                "lm": {"audio_temperature": 0.2, "audio_top_k": 7, "seed": 3}
            },
        }
    )
    assert sampling.audio.temperature == 0.2 and sampling.audio.top_k == 7
    assert sampling.seed == 3
    assert resolve_sampling({}).audio.top_k == DEFAULT_AUDIO_TOP_K


def test_request_longer_than_the_context_is_rejected_with_the_limit():
    data = build_lm_request(_payload(4), vocab_size=32000, context_length=4096)
    prompt = data.talker_model_inputs["timeline"].num_prompt_positions
    fits = 4096 - 1 - prompt
    build_lm_request(_payload(fits), vocab_size=32000, context_length=4096)
    with pytest.raises(ValueError, match=r"needs 4096 positions .* holds 4095"):
        build_lm_request(_payload(fits + 1), vocab_size=32000, context_length=4096)


def test_request_errors_are_reported_as_bad_requests():
    raised = []
    for call in (
        lambda: build_lm_request(_payload(0), vocab_size=32000),
        lambda: build_lm_request(_payload(9000), vocab_size=32000, context_length=8192),
        lambda: resolve_sampling({"seed": True}),
    ):
        with pytest.raises(ValueError) as error:
            call()
        raised.append(error.value)
    assert all(is_bad_request_error(error) for error in raised)
