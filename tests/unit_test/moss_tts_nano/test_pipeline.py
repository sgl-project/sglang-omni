# SPDX-License-Identifier: Apache-2.0
"""Public pipeline and request contracts for MOSS-TTS-Nano."""

from __future__ import annotations

import base64

import pytest

from sglang_omni.models.moss_tts_nano.config import MossTTSNanoPipelineConfig
from sglang_omni.models.moss_tts_nano.request_builders import (
    build_moss_tts_nano_request,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto.request import OmniRequest, StagePayload


def speech_payload(
    *,
    inputs: str | dict[str, str | list[dict[str, str]]] = "Hello.",
    params: dict[str, int | float | bool | None] | None = None,
    tts_params: dict[str, int | float | str | list[str] | None] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="request",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_pipeline_registers_cpu_first_nano_executor() -> None:
    config = MossTTSNanoPipelineConfig(model_path="OpenMOSS-Team/MOSS-TTS-Nano-100M")

    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("MossTTSNanoForCausalLM")
        is MossTTSNanoPipelineConfig
    )
    assert len(config.stages) == 1
    assert config.stages[0].name == "tts"
    assert config.stages[0].terminal is True
    assert config.stages[0].factory.device == "cpu"
    assert config.stages[0].factory.dtype == "float32"


def test_request_maps_reference_and_sampling_parameters() -> None:
    request = build_moss_tts_nano_request(
        speech_payload(
            inputs={
                "text": "Hello.",
                "references": [{"audio_path": "/tmp/reference.wav"}],
            },
            params={"max_new_tokens": 42, "temperature": 0.7, "seed": 9},
            tts_params={"explicit_generation_params": ["temperature", "seed"]},
        )
    )

    assert request.text == "Hello."
    assert request.ref_audio == "/tmp/reference.wav"
    assert request.ref_text is None
    assert request.generation_kwargs["max_new_frames"] == 42
    assert request.generation_kwargs["text_temperature"] == 0.7
    assert request.generation_kwargs["audio_temperature"] == 0.7
    assert request.generation_kwargs["seed"] == 9


def test_request_uses_checkpoint_cli_sampling_defaults() -> None:
    generation = build_moss_tts_nano_request(speech_payload()).generation_kwargs

    assert generation == {
        "max_new_frames": 375,
        "do_sample": True,
        "text_temperature": 1.0,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.8,
        "audio_top_p": 0.95,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.2,
    }


def test_request_rejects_multiple_references() -> None:
    with pytest.raises(ValueError, match="at most one reference"):
        build_moss_tts_nano_request(
            speech_payload(
                inputs={
                    "text": "Hello.",
                    "references": [
                        {"audio_path": "/tmp/one.wav"},
                        {"audio_path": "/tmp/two.wav"},
                    ],
                }
            )
        )


def test_request_rejects_streaming() -> None:
    with pytest.raises(ValueError, match="does not support streaming"):
        build_moss_tts_nano_request(speech_payload(params={"stream": True}))


def test_request_preserves_inline_reference() -> None:
    contents = b"RIFF test audio"
    encoded = base64.b64encode(contents).decode("ascii")

    request = build_moss_tts_nano_request(
        speech_payload(
            inputs={
                "text": "Hello.",
                "references": [{"data": encoded, "media_type": "audio/wav"}],
            }
        )
    )

    assert request.ref_audio == f"data:audio/wav;base64,{encoded}"


@pytest.mark.parametrize(
    "params",
    [{"max_new_tokens": True}, {"seed": 1.5}, {"audio_temperature": float("nan")}],
)
def test_invalid_sampling_is_rejected(
    params: dict[str, int | float | bool | None],
) -> None:
    with pytest.raises(ValueError):
        build_moss_tts_nano_request(speech_payload(params=params))


def test_greedy_sampling_and_unset_frame_limit() -> None:
    request = build_moss_tts_nano_request(
        speech_payload(
            params={"max_new_tokens": None, "temperature": 0, "top_k": -1},
            tts_params={"explicit_generation_params": ["temperature", "top_k"]},
        )
    )
    assert request.generation_kwargs["do_sample"] is False
    assert request.generation_kwargs["max_new_frames"] == 375
    assert request.generation_kwargs["audio_top_k"] == -1
