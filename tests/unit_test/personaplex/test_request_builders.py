# SPDX-License-Identifier: Apache-2.0
"""One decode step per caller frame; sampling knobs and seeds resolve as documented."""

from typing import Literal

import pytest
import torch

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
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
from sglang_omni.proto.request import EXPLICIT_STAGE_SAMPLING_PARAMS_KEY, OmniRequest
from sglang_omni.serve.openai_api import (
    build_chat_generate_request,
    build_rollout_generate_request,
)
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import ChatCompletionRequest, RolloutGenerateRequest


def make_payload(
    num_frames: int, params=None, metadata=None, num_samples: int = 0
) -> StagePayload:
    state = PersonaPlexState(
        text_prompt_ids=[11, 12, 13],
        user_codes=torch.zeros(num_frames, 8, dtype=torch.long),
        num_samples=num_samples,
    )
    params = params or {}
    metadata = dict(metadata or {})
    if (
        "stage_sampling" in params
        and EXPLICIT_STAGE_SAMPLING_PARAMS_KEY not in metadata
    ):
        metadata[EXPLICIT_STAGE_SAMPLING_PARAMS_KEY] = {
            stage: list(sampling)
            for stage, sampling in params["stage_sampling"].items()
        }
    else:
        pass
    request = OmniRequest(inputs={}, params=params, metadata=metadata)
    return StagePayload("r", request=request, data=state.to_dict())


def test_decode_budget_is_the_frame_count():
    data = build_lm_request(make_payload(9), vocab_size=32000)
    assert data.max_new_tokens == 9
    assert data.req.sampling_params.max_new_tokens == 9
    assert data.req.sampling_params.ignore_eos
    timeline = data.talker_model_inputs["timeline"]
    assert len(data.input_ids) == timeline.num_prompt_positions
    assert data.input_ids[0].item() == TEXT_PAD_ID
    assert timeline.num_prompt_positions == 0 + 6 + 3 + 6


def test_sampling_defaults_and_overrides():
    sampling = resolve_sampling({}, stage_sampling={})
    assert sampling.text_temperature == DEFAULT_TEXT_TEMPERATURE
    assert sampling.text_top_k == DEFAULT_TEXT_TOP_K
    assert sampling.audio.temperature == DEFAULT_AUDIO_TEMPERATURE
    assert sampling.seed is None and sampling.text_seed is None

    sampling = resolve_sampling(
        {"temperature": 0.0, "audio_temperature": 0, "seed": 42},
        stage_sampling={},
    )
    assert sampling.text_temperature == 0.0 and sampling.audio.greedy
    assert sampling.text_seed != sampling.audio_seed
    assert (
        resolve_sampling({"seed": 42}, stage_sampling={}).text_seed
        == sampling.text_seed
    )

    data = build_lm_request(
        make_payload(2, {"temperature": 0.0, "seed": 42}), vocab_size=32000
    )
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.sampling_seed == sampling.text_seed


def test_stream_builder_ships_pending_frames_to_the_codec():
    data = build_lm_request(make_payload(2, num_samples=3000), vocab_size=32000)
    assert lm_stream_output_builder("r", data, None) == []
    data.talker_model_inputs["pending_frames"].append(torch.arange(8))
    messages = lm_stream_output_builder("r", data, None)
    assert len(messages) == 1
    assert messages[0].target == "code2wav"
    assert messages[0].data.shape == (1, 8)
    assert messages[0].metadata["num_samples"] == 3000
    assert data.talker_model_inputs["pending_frames"] == []


def test_request_boundary_rejects_unusable_inputs():
    with pytest.raises(ValueError, match="80 ms frame") as error:
        build_lm_request(make_payload(0), vocab_size=32000)
    assert is_bad_request_error(error.value)
    no_audio = StagePayload(
        "r",
        request=OmniRequest(inputs={}, params={}),
        data=PersonaPlexState(text_prompt_ids=[11]).to_dict(),
    )
    with pytest.raises(ValueError, match="no encoded caller audio"):
        build_lm_request(no_audio, vocab_size=32000)


def test_client_filler_sampling_values_keep_the_reference_defaults():
    filler = {"temperature": 1.0, "top_k": -1, "seed": None}
    sampling = resolve_sampling(filler, stage_sampling={})
    assert sampling.text_temperature == DEFAULT_TEXT_TEMPERATURE
    assert sampling.text_top_k == DEFAULT_TEXT_TOP_K

    chosen = resolve_sampling(
        filler, explicit_fields=["temperature", "top_k"], stage_sampling={}
    )
    assert chosen.text_temperature == 1.0 and chosen.text_top_k == -1

    data = build_lm_request(
        make_payload(2, filler, {EXPLICIT_GENERATION_PARAMS_KEY: ["temperature"]}),
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
        },
        stage_sampling={},
    )
    assert sampling.audio.temperature == 0.2 and sampling.audio.top_k == 7
    assert sampling.seed == 3
    assert resolve_sampling({}, stage_sampling={}).audio.top_k == DEFAULT_AUDIO_TOP_K


def test_request_longer_than_the_context_is_rejected_with_the_limit():
    data = build_lm_request(make_payload(4), vocab_size=32000, context_length=4096)
    prompt = data.talker_model_inputs["timeline"].num_prompt_positions
    fits = 4096 - 1 - prompt
    build_lm_request(make_payload(fits), vocab_size=32000, context_length=4096)
    with pytest.raises(
        ValueError, match=r"needs 4096 positions .* holds 4095"
    ) as error:
        build_lm_request(make_payload(fits + 1), vocab_size=32000, context_length=4096)
    assert is_bad_request_error(error.value)


@pytest.mark.parametrize("stage_seed,expected", [(None, 9), (0, 0), (42, 42)])
def test_stage_seed_precedence(stage_seed: int | None, expected: int) -> None:
    sampling = resolve_sampling(
        {"seed": 7, "stage_params": {"lm": {"seed": 9}}},
        stage_sampling={"seed": stage_seed},
    )
    assert sampling.seed == expected


@pytest.mark.parametrize("num_frames", [0, 1, 3])
def test_result_preserves_frames_and_history_and_drops_inputs(num_frames: int) -> None:
    data = build_lm_request(make_payload(max(1, num_frames)), vocab_size=32000)
    tokens = list(range(101, 101 + num_frames))
    data.output_ids = tokens.copy()
    frames = torch.arange(num_frames * 8).view(num_frames, 8)
    data.talker_model_inputs["frames"] = list(frames)
    state = PersonaPlexState.from_dict(apply_lm_result(data).data)
    assert state.text_ids == ([TEXT_PAD_ID] + tokens)[:num_frames]
    assert len(state.text_ids) == state.codes.shape[0]
    assert data.output_ids == tokens
    torch.testing.assert_close(state.codes, frames)
    assert state.user_codes is None and state.waveform is None


def test_stage_sampling_seed_and_text_controls_reach_backend() -> None:
    data = build_lm_request(
        make_payload(
            2,
            {
                "seed": 1,
                "stage_params": {"lm": {"seed": 2}},
                "stage_sampling": {
                    "lm": {
                        "temperature": 0.3,
                        "top_k": 10,
                        "seed": 3,
                        "top_p": 0.8,
                        "min_p": 0.1,
                        "repetition_penalty": 1.2,
                    }
                },
            },
        ),
        vocab_size=32000,
    )
    sampling = data.talker_model_inputs["sampling"]
    assert sampling.seed == 3
    backend = data.req.sampling_params
    assert backend.temperature == 0.3
    assert backend.top_k == 10
    assert backend.top_p == 0.8 and backend.min_p == 0.1
    assert backend.repetition_penalty == 1.2
    assert backend.sampling_seed == sampling.text_seed


@pytest.mark.parametrize("source", ["client-stage", "client-top", "chat", "rollout"])
def test_sampling_presence_contract_across_entry_points(
    source: Literal["client-stage", "client-top", "chat", "rollout"]
) -> None:
    seed = 0
    if source == "client-stage":
        request = GenerateRequest(
            prompt="hello", stage_sampling={"lm": SamplingParams(seed=seed)}
        )
    elif source == "client-top":
        request = GenerateRequest(prompt="hello", sampling=SamplingParams(seed=seed))
    elif source == "chat":
        request = build_chat_generate_request(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                stage_sampling={"lm": {"seed": seed}},
            )
        )
    else:
        request = build_rollout_generate_request(
            RolloutGenerateRequest(
                prompt="hello", stage_sampling={"lm": {"seed": seed}}
            )
        )
    lowered = Client.build_omni_request(request)
    data = build_lm_request(
        make_payload(2, lowered.params, lowered.metadata), vocab_size=32000
    )
    sampling = data.talker_model_inputs["sampling"]
    expected = (
        (1.0, -1)
        if source == "client-stage"
        else (DEFAULT_TEXT_TEMPERATURE, DEFAULT_TEXT_TOP_K)
    )
    assert (sampling.text_temperature, sampling.text_top_k) == expected
    assert sampling.seed == seed
    assert (sampling.top_p, sampling.min_p, sampling.repetition_penalty) == (
        1.0,
        0.0,
        1.0,
    )


def test_internal_stage_sampling_requires_presence_metadata() -> None:
    payload = make_payload(2, {"stage_sampling": {"lm": {"seed": 7}}})
    payload.request.metadata.clear()
    with pytest.raises(KeyError, match=EXPLICIT_STAGE_SAMPLING_PARAMS_KEY):
        build_lm_request(payload, vocab_size=32000)


@pytest.mark.parametrize("source", ["chat", "rollout"])
def test_explicit_neutral_stage_controls_override_stage_params(
    source: Literal["chat", "rollout"]
) -> None:
    stage = {
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    overrides = {"lm": {"top_p": 0.8, "min_p": 0.1, "repetition_penalty": 1.2}}
    if source == "chat":
        request = build_chat_generate_request(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                stage_sampling={"lm": stage},
                stage_params=overrides,
            )
        )
    else:
        request = build_rollout_generate_request(
            RolloutGenerateRequest(
                prompt="hello", stage_sampling={"lm": stage}, stage_params=overrides
            )
        )
    lowered = Client.build_omni_request(request)
    data = build_lm_request(
        make_payload(2, lowered.params, lowered.metadata), vocab_size=32000
    )
    sampling = data.talker_model_inputs["sampling"]
    assert (sampling.text_temperature, sampling.text_top_k) == (1.0, -1)
    assert (sampling.top_p, sampling.min_p, sampling.repetition_penalty) == (
        1.0,
        0.0,
        1.0,
    )


def test_rollout_stage_token_limit_alias_preserves_frame_budget() -> None:
    request = build_rollout_generate_request(
        RolloutGenerateRequest(prompt="hello", stage_sampling={"lm": {"max_tokens": 3}})
    )
    lowered = Client.build_omni_request(request)
    data = build_lm_request(
        make_payload(2, lowered.params, lowered.metadata), vocab_size=32000
    )
    assert data.max_new_tokens == 2


@pytest.mark.parametrize(
    "key,value,scope",
    [
        ("audio_temperature", -1, "top_level"),
        ("audio_temperature", float("inf"), "stage_params"),
        ("audio_temperature", float("nan"), "top_level"),
        ("audio_temperature", "0.5", "stage_params"),
        ("audio_temperature", True, "top_level"),
        ("audio_top_k", -5, "stage_params"),
        ("audio_top_k", 2.5, "top_level"),
        ("audio_top_k", True, "stage_params"),
        ("seed", 1.9, "top_level"),
        ("seed", 1.9, "stage_params"),
        ("seed", 1.9, "stage_sampling"),
        ("seed", "abc", "top_level"),
        ("seed", True, "stage_sampling"),
        ("stop", ["hello"], "stage_params"),
        ("stop_token_ids", [3], "stage_sampling"),
    ],
)
def test_invalid_audio_seed_or_stop_is_a_bad_request(
    key: str,
    value: float | int | str | list[str] | list[int],
    scope: Literal["top_level", "stage_params", "stage_sampling"],
) -> None:
    with pytest.raises(ValueError) as error:
        resolve_sampling(
            {key: value} if scope == "top_level" else {scope: {"lm": {key: value}}},
            stage_sampling={key: value} if scope == "stage_sampling" else {},
        )
    assert is_bad_request_error(error.value)
    assert str(error.value).startswith(f"PersonaPlex {key} must be")


def test_valid_sampling_boundaries() -> None:
    sampling = resolve_sampling(
        {
            "stage_params": {
                "lm": {
                    "temperature": 0.0,
                    "audio_temperature": 0.0,
                    "top_k": -1,
                    "audio_top_k": 0,
                    "seed": 0,
                }
            }
        },
        stage_sampling={},
    )
    assert sampling.text_top_k == -1 and sampling.audio.top_k == 0
    assert sampling.audio.greedy and sampling.seed == 0


def test_invalid_text_sampling_is_rejected_by_native_verification() -> None:
    params = {"stage_sampling": {"lm": {"temperature": -0.1}}}
    with pytest.raises(
        ValueError, match="PersonaPlex sampling parameters must be valid:"
    ) as error:
        build_lm_request(make_payload(2, params), vocab_size=32000)
    assert is_bad_request_error(error.value)


def test_overridden_text_value_is_not_validated() -> None:
    data = build_lm_request(
        make_payload(
            2,
            {"temperature": -1, "stage_params": {"lm": {"temperature": 0.7}}},
        ),
        vocab_size=32000,
    )
    assert data.req.sampling_params.temperature == 0.7
