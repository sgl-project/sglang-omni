# SPDX-License-Identifier: Apache-2.0
"""One decode step per caller frame; sampling knobs and seeds resolve as documented."""

from pathlib import Path

import pytest
import torch

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.models.personaplex import request_builders
from sglang_omni.models.personaplex.architecture import (
    DEFAULT_AUDIO_TEMPERATURE,
    DEFAULT_AUDIO_TOP_K,
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_K,
    TEXT_PAD_ID,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.prompts import VoicePrompt
from sglang_omni.models.personaplex.request_builders import (
    RequestSampling,
    apply_lm_result,
    build_lm_request,
    lm_stream_output_builder,
    resolve_sampling,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import ChatCompletionRequest


def make_payload(
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
    data = build_lm_request(make_payload(9), vocab_size=32000, voice_cache={})
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
        {"temperature": 0.0, "audio_temperature": 0, "seed": 42},
        explicit_fields=["temperature"],
    )
    assert sampling.text_temperature == 0.0 and sampling.audio.greedy
    assert sampling.text_seed != sampling.audio_seed
    assert resolve_sampling({"seed": 42}).text_seed == sampling.text_seed

    data = build_lm_request(
        make_payload(
            2,
            {"temperature": 0.0, "seed": 42},
            {EXPLICIT_GENERATION_PARAMS_KEY: ["temperature"]},
        ),
        vocab_size=32000,
        voice_cache={},
    )
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.sampling_seed == sampling.text_seed


def test_result_carries_text_ids_and_frames_and_drops_inputs():
    data = build_lm_request(make_payload(2), vocab_size=32000, voice_cache={})
    data.output_ids = [3, 17]
    data.talker_model_inputs["frames"] = [torch.arange(8), torch.arange(8) + 8]
    state = PersonaPlexState.from_dict(apply_lm_result(data).data)
    assert state.text_ids == [3, 17]
    assert state.codes.tolist() == [list(range(8)), list(range(8, 16))]
    assert state.user_codes is None and state.waveform is None


def test_stream_builder_ships_pending_frames_to_the_codec():
    data = build_lm_request(
        make_payload(2, num_samples=3000), vocab_size=32000, voice_cache={}
    )
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
        build_lm_request(make_payload(0), vocab_size=32000, voice_cache={})
    no_audio = StagePayload(
        "r",
        request=OmniRequest(inputs={}, params={}),
        data=PersonaPlexState(text_prompt_ids=[11]).to_dict(),
    )
    with pytest.raises(ValueError, match="no encoded caller audio"):
        build_lm_request(no_audio, vocab_size=32000, voice_cache={})
    with pytest.raises(ValueError, match="seed must be an integer"):
        resolve_sampling({"seed": True})


def client_lm_sampling(request: GenerateRequest) -> RequestSampling:
    omni_request = Client.build_omni_request(request)
    return resolve_sampling(
        omni_request.params, omni_request.metadata[EXPLICIT_GENERATION_PARAMS_KEY]
    )


def text_sampling(sampling: RequestSampling) -> tuple[float, int]:
    return sampling.text_temperature, sampling.text_top_k


def test_in_process_client_applies_only_the_sampling_its_caller_set():
    defaults = (DEFAULT_TEXT_TEMPERATURE, DEFAULT_TEXT_TOP_K)
    assert text_sampling(client_lm_sampling(GenerateRequest(prompt={}))) == defaults
    assert text_sampling(
        client_lm_sampling(
            GenerateRequest(prompt={}, sampling=SamplingParams(top_k=-1))
        )
    ) == (DEFAULT_TEXT_TEMPERATURE, -1)
    staged = GenerateRequest(
        prompt={}, stage_sampling={"lm": SamplingParams(temperature=1.0)}
    )
    assert text_sampling(client_lm_sampling(staged)) == (1.0, DEFAULT_TEXT_TOP_K)


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, (DEFAULT_TEXT_TEMPERATURE, DEFAULT_TEXT_TOP_K)),
        ({"temperature": 1.0, "top_k": -1}, (1.0, -1)),
        ({"stage_sampling": {"lm": {"temperature": 1.0}}}, (1.0, DEFAULT_TEXT_TOP_K)),
    ],
)
def test_chat_completions_sampling_reaches_the_lm_as_sent(fields, expected):
    chat = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], audios=["a.wav"], **fields
    )
    assert (
        text_sampling(client_lm_sampling(build_chat_generate_request(chat))) == expected
    )


def test_lm_stage_params_choose_text_sampling_without_explicit_fields():
    sampling = resolve_sampling(
        {
            "temperature": 1.0,
            "top_k": -1,
            "stage_params": {"lm": {"temperature": 1.0, "top_k": -1}},
        }
    )
    assert text_sampling(sampling) == (1.0, -1)
    assert text_sampling(resolve_sampling({"temperature": 1.0, "top_k": -1})) == (
        DEFAULT_TEXT_TEMPERATURE,
        DEFAULT_TEXT_TOP_K,
    )


def test_packaged_voice_is_loaded_once_per_stage(tmp_path, monkeypatch):
    loads = []

    def load_packaged_voice(path):
        loads.append(path)
        return VoicePrompt(
            frames=3,
            embeddings=torch.randn(2, 16),
            tail_codes=torch.zeros(2, 8, dtype=torch.long),
        )

    monkeypatch.setattr(request_builders, "load_packaged_voice", load_packaged_voice)
    state = PersonaPlexState(
        user_codes=torch.zeros(2, 8, dtype=torch.long), voice_path="/voices/NATF2.pt"
    )
    payload = StagePayload(
        "r", request=OmniRequest(inputs={}, params={}), data=state.to_dict()
    )
    cache = {}
    first = build_lm_request(payload, vocab_size=32000, voice_cache=cache)
    second = build_lm_request(payload, vocab_size=32000, voice_cache=cache)
    assert loads == [Path("/voices/NATF2.pt")]
    for data in (first, second):
        timeline = data.talker_model_inputs["timeline"]
        assert timeline.prefill_embedding_positions == [0, 1]
        torch.testing.assert_close(
            timeline.prefill_embeddings, cache["/voices/NATF2.pt"].embeddings
        )


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
    data = build_lm_request(
        make_payload(4), vocab_size=32000, voice_cache={}, context_length=4096
    )
    prompt = data.talker_model_inputs["timeline"].num_prompt_positions
    fits = 4096 - 1 - prompt
    build_lm_request(
        make_payload(fits), vocab_size=32000, voice_cache={}, context_length=4096
    )
    with pytest.raises(ValueError, match=r"needs 4096 positions .* holds 4095"):
        build_lm_request(
            make_payload(fits + 1),
            vocab_size=32000,
            voice_cache={},
            context_length=4096,
        )


def test_request_errors_are_reported_as_bad_requests():
    raised = []
    for call in (
        lambda: build_lm_request(make_payload(0), vocab_size=32000, voice_cache={}),
        lambda: build_lm_request(
            make_payload(9000), vocab_size=32000, voice_cache={}, context_length=8192
        ),
        lambda: resolve_sampling({"seed": True}),
    ):
        with pytest.raises(ValueError) as error:
            call()
        raised.append(error.value)
    assert all(is_bad_request_error(error) for error in raised)
