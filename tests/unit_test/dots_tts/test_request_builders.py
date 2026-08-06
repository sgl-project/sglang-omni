# SPDX-License-Identifier: Apache-2.0
"""Request-boundary contract for dots.tts."""

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.dots_tts.payload_types import (
    DOTS_TTS_DEFAULT_GUIDANCE_SCALE,
    DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES,
    DOTS_TTS_DEFAULT_NUM_STEPS,
    DOTS_TTS_DEFAULT_SPEAKER_SCALE,
    DotsTtsState,
)
from sglang_omni.models.dots_tts.request_builders import (
    build_dots_tts_state,
    preprocess_dots_tts_payload,
)
from sglang_omni.proto import OmniRequest, StagePayload

REFERENCE = {"audio_path": "ref.wav", "text": "reference transcript"}


def make_payload(
    *,
    inputs: Any,
    params: dict[str, Any] | None = None,
    tts_params: dict[str, Any] | None = None,
) -> StagePayload:
    metadata: dict[str, Any] = {}
    if tts_params is not None:
        metadata["tts_params"] = tts_params
    return StagePayload(
        request_id="req",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata=metadata,
        ),
        data={},
    )


def test_reference_request_uses_model_defaults() -> None:
    payload = make_payload(inputs={"text": "target", "references": [REFERENCE]})

    state = build_dots_tts_state(payload)

    assert state.text == "target"
    assert state.prompt_text == "reference transcript"
    assert state.ref_audio_path == "ref.wav"
    assert state.num_steps == DOTS_TTS_DEFAULT_NUM_STEPS == 4
    assert state.guidance_scale == DOTS_TTS_DEFAULT_GUIDANCE_SCALE
    assert state.speaker_scale == DOTS_TTS_DEFAULT_SPEAKER_SCALE
    assert state.max_audio_patches == DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES
    assert state.seed is None


def test_preprocess_stores_state_on_payload() -> None:
    payload = make_payload(inputs={"text": "target", "references": [REFERENCE]})

    stored = preprocess_dots_tts_payload(payload)

    assert stored is payload
    assert DotsTtsState.from_dict(payload.data).ref_audio_path == "ref.wav"


def test_streaming_requests_are_rejected() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={"stream": True},
    )

    with pytest.raises(ValueError, match="full waveform only"):
        build_dots_tts_state(payload)


@pytest.mark.parametrize(
    "inputs",
    [
        {"text": "target"},
        {"text": "target", "references": [{"audio_path": "ref.wav"}]},
        {"text": "target", "references": [{"text": "reference transcript"}]},
        {"text": "target", "references": [REFERENCE, REFERENCE]},
    ],
)
def test_reference_audio_and_transcript_are_required(inputs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="exactly one reference"):
        build_dots_tts_state(make_payload(inputs=inputs))


def test_empty_target_text_is_rejected() -> None:
    payload = make_payload(inputs={"text": "  ", "references": [REFERENCE]})

    with pytest.raises(ValueError, match="non-empty input text"):
        build_dots_tts_state(payload)


def test_inline_reference_audio_is_rejected() -> None:
    payload = make_payload(
        inputs={
            "text": "target",
            "references": [{"data": "AAAA", "text": "reference transcript"}],
        },
    )

    with pytest.raises(ValueError, match="local file path"):
        build_dots_tts_state(payload)


def test_flat_ref_audio_and_ref_text_are_accepted() -> None:
    payload = make_payload(
        inputs="target",
        tts_params={"ref_audio": "ref.wav", "ref_text": "reference transcript"},
    )

    state = build_dots_tts_state(payload)

    assert state.ref_audio_path == "ref.wav"
    assert state.prompt_text == "reference transcript"


def test_extra_body_overrides_solver_parameters() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={
            "num_steps": 10,
            "guidance_scale": 2.0,
            "speaker_scale": 1.0,
            "max_audio_patches": 128,
        },
    )

    state = build_dots_tts_state(payload)

    assert state.num_steps == 10
    assert state.guidance_scale == 2.0
    assert state.speaker_scale == 1.0
    assert state.max_audio_patches == 128


def test_stage_params_win_over_generic_params() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={
            "num_steps": 10,
            "stage_params": {"tts_engine": {"num_steps": 6}},
        },
    )

    assert build_dots_tts_state(payload).num_steps == 6


@pytest.mark.parametrize("num_steps", [0, -1, 2.5, True, "many"])
def test_invalid_num_steps_is_rejected(num_steps: Any) -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={"num_steps": num_steps},
    )

    with pytest.raises(ValueError, match="num_steps"):
        build_dots_tts_state(payload)


def test_endpoint_sampling_defaults_are_ignored_unless_explicit() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={"temperature": 0.8, "top_p": 0.9, "top_k": 30},
    )

    assert build_dots_tts_state(payload).num_steps == DOTS_TTS_DEFAULT_NUM_STEPS


def test_explicit_sampling_fields_are_rejected() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={"temperature": 0.8},
        tts_params={"explicit_generation_params": ["temperature"]},
    )

    with pytest.raises(ValueError, match="logits sampling field"):
        build_dots_tts_state(payload)


def test_seed_is_taken_from_tts_params_and_explicit_endpoint_field() -> None:
    from_tts_params = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        tts_params={"seed": 7},
    )
    from_endpoint = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={"seed": 9},
        tts_params={"explicit_generation_params": ["seed"]},
    )
    implicit_endpoint = make_payload(
        inputs={"text": "target", "references": [REFERENCE]},
        params={"seed": 9},
    )

    assert build_dots_tts_state(from_tts_params).seed == 7
    assert build_dots_tts_state(from_endpoint).seed == 9
    assert build_dots_tts_state(implicit_endpoint).seed is None
