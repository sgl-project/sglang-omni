# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.minimax_music3.request_builders import build_ttm_state
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(
    *,
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="req-minimax",
        request=OmniRequest(
            inputs="[Verse]\nNowhere man",
            params=params or {},
            metadata={
                "tts_params": {
                    "instructions": "British art rock with warm vocals",
                    **(tts_params or {}),
                }
            },
        ),
        data={},
    )


def test_build_ttm_state_accepts_the_supported_contract() -> None:
    state = build_ttm_state(
        _payload(params={"max_new_tokens": 250}, tts_params={"seed": 7})
    )

    assert state.seed == 7
    assert state.max_audio_frames == 250
    assert state.prompt is not None and state.prompt.endswith("<|audio_start|>")


@pytest.mark.parametrize("value", [0, 9001, 1.5, "250", True])
def test_build_ttm_state_rejects_invalid_frame_limits(value: object) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        build_ttm_state(_payload(params={"max_new_tokens": value}))


@pytest.mark.parametrize("value", [-1, 2**64, 1.5, "7", True])
def test_build_ttm_state_rejects_invalid_seeds(value: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        build_ttm_state(_payload(tts_params={"seed": value}))


@pytest.mark.parametrize(
    "tts_params",
    [
        {"speed": 1.25},
        {"speed": "1.0"},
        {"voice": "alloy"},
        {"ref_audio": "/tmp/reference.wav"},
        {"language": "en"},
    ],
)
def test_build_ttm_state_rejects_ignored_speech_controls(
    tts_params: dict,
) -> None:
    with pytest.raises(ValueError, match="does not support|only supports"):
        build_ttm_state(_payload(tts_params=tts_params))


@pytest.mark.parametrize("name", ["temperature", "top_p", "top_k"])
def test_build_ttm_state_rejects_explicit_sampling_controls(name: str) -> None:
    with pytest.raises(ValueError, match="sampling parameters"):
        build_ttm_state(
            _payload(
                tts_params={
                    name: 0.5,
                    "explicit_generation_params": [name],
                }
            )
        )


def test_request_data_asks_the_scheduler_to_enforce_limits() -> None:
    """Without this flag validate_input_length never runs for this model."""
    from sglang_omni.models.minimax_music3.sglang_request_builder import (
        MiniMaxMusic3SGLangRequestData,
    )

    assert MiniMaxMusic3SGLangRequestData().enforce_request_limits is True
