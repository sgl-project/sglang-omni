# SPDX-License-Identifier: Apache-2.0
"""Stage routing for speech synthesis and audio understanding."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.client.client import Client
from sglang_omni.models.ming3_audio import stages
from sglang_omni.models.ming3_audio.routing import route_preprocessing, route_thinker
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SpecTokensDetailsDict
from sglang_omni.serve.protocol import build_sglext


def test_preprocessing_sends_synthesis_to_talker() -> None:
    payload = StagePayload(
        request_id="synthesis-request",
        request=OmniRequest(inputs="hello", metadata={"task": "tts"}),
        data={},
    )
    assert route_preprocessing("synthesis-request", payload) == "talker"


def test_preprocessing_sends_understanding_to_audio_encoder() -> None:
    payload = StagePayload(
        request_id="understanding-request",
        request=OmniRequest(inputs="hello", metadata={}),
        data={},
    )
    assert route_preprocessing("understanding-request", payload) == "audio_encoder"


def test_thinker_sends_audio_output_to_talker() -> None:
    payload = StagePayload(
        request_id="spoken-answer",
        request=OmniRequest(
            inputs="hello", metadata={"output_modalities": ["text", "audio"]}
        ),
        data={},
    )
    assert route_thinker("spoken-answer", payload) == "talker"


def test_thinker_sends_text_output_to_decode() -> None:
    payload = StagePayload(
        request_id="text-answer",
        request=OmniRequest(inputs="hello", metadata={"output_modalities": ["text"]}),
        data={},
    )
    assert route_thinker("text-answer", payload) == "decode"


@pytest.mark.parametrize("speech_output", [False, True])
@pytest.mark.parametrize(
    "speculative_details",
    [
        None,
        {
            "spec_accept_rate": 0.5,
            "spec_accept_length": 2.0,
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "spec_correct_drafts_histogram": [0, 1, 0],
        },
    ],
)
def test_thinker_statistics_reach_the_client_with_text_and_speech(
    monkeypatch: pytest.MonkeyPatch,
    speech_output: bool,
    speculative_details: SpecTokensDetailsDict | None,
) -> None:
    """Thinker acceptance statistics must survive text and speech result conversion."""
    monkeypatch.setattr(stages, "build_sglang_server_args", Mock())
    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure",
        Mock(return_value=(None, None, None, None, None)),
    )
    monkeypatch.setattr(stages, "Ming3AudioModelRunner", Mock())
    monkeypatch.setattr(
        stages, "get_tokenizer", Mock(return_value=Mock(decode=Mock(return_value="hi")))
    )
    monkeypatch.setattr(
        stages.ModelRegistry, "models", dict(stages.ModelRegistry.models)
    )
    scheduler_factory = Mock()
    monkeypatch.setattr(stages, "OmniScheduler", scheduler_factory)
    stages.create_thinker_executor(
        "unused", max_seq_len=8192, enable_async_decode=False, device="cpu"
    )
    payload = StagePayload(
        request_id="reply",
        request=OmniRequest(inputs="hello"),
        data={},
    )
    request = SGLangARRequestData(
        input_ids=torch.tensor([1, 2, 3]),
        output_ids=[7, 8],
        spec_tokens_details=speculative_details,
        stage_payload=payload,
    )
    result_adapter = scheduler_factory.call_args.kwargs["result_adapter"]
    payload = result_adapter(request)
    if speech_output:
        monkeypatch.setattr(
            stages,
            "Ming3SpeechGenerator",
            Mock(
                return_value=Mock(
                    sample_rate=44100, synthesize=Mock(return_value=torch.zeros(16))
                )
            ),
        )
        talker = stages.create_talker_executor(
            "unused",
            max_decode_steps=512,
            seed=123,
            enable_cuda_graph=False,
            device="cpu",
        )
        payload = talker.fn(payload)
    else:
        pass
    payload = stages.create_decode_executor("unused").fn(payload)
    chunk = Client.default_result_builder(payload.request_id, payload.data)
    assert chunk.text == "hi"
    assert chunk.spec_tokens_details == speculative_details
    assert (chunk.audio_data is not None) is speech_output
    extension = build_sglext(True, chunk.spec_tokens_details)
    if speculative_details is None:
        assert extension is None
    else:
        assert extension.spec_tokens_details.model_dump() == speculative_details
