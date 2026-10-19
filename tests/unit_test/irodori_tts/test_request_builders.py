# SPDX-License-Identifier: Apache-2.0
"""Request parameter parsing for Irodori-TTS."""

import pytest

from sglang_omni.models.irodori_tts.request_builders import build_irodori_request
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(*, inputs, params=None, metadata=None):
    return StagePayload(
        request_id="test-request",
        request=OmniRequest(
            inputs=inputs, params=params or {}, metadata=metadata or {}
        ),
        data=None,
    )


def test_synthesis_stage_params_override_metadata_parameters() -> None:
    payload = make_payload(
        inputs={"text": "こんにちは"},
        params={
            "stage_params": {
                "synthesis": {
                    "num_steps": 6,
                    "seconds": 2.5,
                    "cfg_scale_text": 4.0,
                    "speed": 1.5,
                }
            }
        },
        metadata={
            "tts_params": {
                "num_steps": 12,
                "duration_scale": 1.2,
                "speed": 1.0,
            }
        },
    )

    request = build_irodori_request(payload)

    assert request.num_steps == 6
    assert request.seconds == 2.5
    assert request.cfg_scale_text == 4.0
    assert request.duration_scale == 1.2
    assert request.speed == 1.5


def test_speed_is_read_from_tts_parameters() -> None:
    request = build_irodori_request(
        make_payload(
            inputs={"text": "こんにちは"}, metadata={"tts_params": {"speed": 2.0}}
        )
    )

    assert request.speed == 2.0


@pytest.mark.parametrize("speed", [0.0, 0.24, 4.01, float("inf"), float("nan")])
def test_invalid_speed_is_rejected(speed: float) -> None:
    payload = make_payload(
        inputs={"text": "こんにちは"},
        metadata={"tts_params": {"speed": speed}},
    )

    with pytest.raises(ValueError, match="speed must be between 0.25 and 4.0"):
        build_irodori_request(payload)


def test_required_text_input_is_validated() -> None:
    with pytest.raises(ValueError, match="requires a text input"):
        build_irodori_request(make_payload(inputs={"caption": "voice style"}))


def test_empty_text_input_is_validated() -> None:
    with pytest.raises(ValueError, match="non-empty Japanese text"):
        build_irodori_request(make_payload(inputs={"text": "  "}))
