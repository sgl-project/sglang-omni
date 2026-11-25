# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 request contract: defaults, the checks every endpoint gets, and the speech validator's 400s."""

import pytest

from sglang_omni.models.stable_audio3.config import StableAudio3PipelineConfig
from sglang_omni.models.stable_audio3.request_builders import (
    MAX_PROMPT_CHARACTERS,
    build_state,
)
from sglang_omni.models.stable_audio3.speech_validation import (
    StableAudio3SpeechRequestValidator,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.speech_errors import SpeechAPIError


def payload_with(
    params: dict[str, object],
    metadata: dict[str, object] | None = None,
    inputs: object = "ambient pads",
) -> StagePayload:
    return StagePayload(
        request_id="request",
        request=OmniRequest(
            inputs=inputs,
            params=params,
            metadata=(
                {"task": "tts", "tts_params": {}} if metadata is None else metadata
            ),
        ),
        data={},
    )


def test_request_defaults_to_the_official_duration_and_a_random_seed() -> None:
    first, second = (build_state(payload_with({})) for _ in range(2))
    assert first.duration_seconds == 120.0
    assert first.seed != second.seed
    state = build_state(
        payload_with(
            {"seed": 7, "stage_params": {"generation": {"duration_seconds": 30}}}
        )
    )
    assert (state.prompt, state.duration_seconds, state.seed) == ("ambient pads", 30, 7)


def test_requests_that_skip_the_speech_validator_are_refused() -> None:
    """A /generate or chat request skips the speech validator, and /generate can carry the speech metadata itself: chat messages became the prompt, and a negative seed reached the batch."""
    with pytest.raises(ValueError, match="/v1/audio/speech") as error:
        build_state(payload_with({}, metadata={}))
    assert is_bad_request_error(error.value)
    for params, message in (
        ({"seed": -5}, "seed"),
        ({"stage_params": {"generation": {"duration_seconds": 1000}}}, "duration"),
        ({"stage_params": {"generation": {"steps": 4}}}, "steps"),
        ({"stage_params": "fast"}, "stage_params"),
    ):
        with pytest.raises(ValueError, match=message) as error:
            build_state(payload_with(params))
        assert is_bad_request_error(error.value)
    for inputs in (
        "  ",
        [{"role": "user", "content": "ambient pads"}],
        "a" * (MAX_PROMPT_CHARACTERS + 1),
    ):
        with pytest.raises(ValueError, match="prompt") as error:
            build_state(payload_with({}, inputs=inputs))
        assert is_bad_request_error(error.value)


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"voice": "alloy"}, "voice"),
        ({"stream": True, "response_format": "wav"}, "stream"),
        ({"seed": 2**63}, "seed"),
        ({"seed": -1}, "seed"),
        ({"temperature": 0.7}, "temperature"),
        ({"instructions": "dreamy"}, "instructions"),
        ({"stage_params": {"vocoder": {}}}, "stage_params"),
        (
            {"stage_params": {"generation": {"duration_seconds": 381}}},
            "stage_params.generation.duration_seconds",
        ),
        (
            {"stage_params": {"generation": {"duration_seconds": 0.05}}},
            "stage_params.generation.duration_seconds",
        ),
        (
            {"stage_params": {"generation": {"steps": 4}}},
            "stage_params.generation.steps",
        ),
    ],
)
def test_unsupported_requests_are_client_errors(
    fields: dict[str, object], param: str
) -> None:
    validator = StableAudio3SpeechRequestValidator(
        default_model="stable-audio-3",
        supported_speech_fields=StableAudio3PipelineConfig.supported_speech_fields,
    )
    with pytest.raises(SpeechAPIError) as error:
        validator.parse_generation_request({"input": "ambient pads", **fields})
    assert (error.value.status_code, error.value.param) == (400, param)
