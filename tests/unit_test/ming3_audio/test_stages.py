# SPDX-License-Identifier: Apache-2.0
"""Stage routing for speech synthesis and audio understanding."""

from sglang_omni.models.ming3_audio.routing import route_preprocessing, route_thinker
from sglang_omni.proto.request import OmniRequest, StagePayload


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
