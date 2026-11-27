# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the LFM3 preprocessing stage's prompt and sampling helpers."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from sglang_omni.models.lfm3_audio import stages
from sglang_omni.serve.openai_errors import is_bad_request_error


def test_tts_prompt_uses_voice_and_language() -> None:
    assert (
        stages.tts_system_prompt({})
        == "Perform TTS in American English using a female voice."
    )
    assert (
        stages.tts_system_prompt({"voice": "Male", "language": "British English"})
        == "Perform TTS in British English using a male voice."
    )


def test_tts_instructions_replace_the_prompt() -> None:
    assert (
        stages.tts_system_prompt({"instructions": " Speak slowly. "}) == "Speak slowly."
    )


def test_tts_rejects_unknown_voices() -> None:
    with pytest.raises(ValueError, match="voices"):
        stages.tts_system_prompt({"voice": "alloy"})


def test_speech_sampling_fields_of_tts_and_chat() -> None:
    """A TTS request's sampling fields, and only a chat request's talker fields, drive the speech frames."""
    params = {"temperature": 0.7, "top_k": 64, "talker_temperature": 0.0}
    assert stages.speech_sampling_params(
        params, is_tts=True
    ) == stages.SpeechSamplingParams(temperature=0.7, top_k=64, repetition_penalty=None)
    assert stages.speech_sampling_params(
        params, is_tts=False
    ) == stages.SpeechSamplingParams(
        temperature=0.0, top_k=None, repetition_penalty=None
    )


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"temperature": "hot"},
        {"top_k": -1},
        {"top_k": 1.5},
        {"repetition_penalty": 0},
        {"repetition_penalty": float("inf")},
        {"temperature": 1e-45},
        {"repetition_penalty": 1e-40},
        {"top_k": 10**20},
    ],
)
def test_speech_sampling_values_the_sampler_rejects_are_request_errors(
    params: dict[str, object],
) -> None:
    """The frame sampler runs on the whole batch: a repetition penalty of 0 or a negative temperature raised inside it and failed every request of the batch. The request alone is now answered with a 400."""
    with pytest.raises(ValueError) as tts_error:
        stages.speech_sampling_params(params, is_tts=True)
    with pytest.raises(ValueError, match="talker_") as chat_error:
        stages.speech_sampling_params(
            {f"talker_{name}": value for name, value in params.items()}, is_tts=False
        )
    assert is_bad_request_error(tts_error.value)
    assert is_bad_request_error(chat_error.value)


def test_token_id_prompts_are_a_request_error() -> None:
    with pytest.raises(ValueError, match="token ids"):
        stages.as_messages([1, 2, 3])


def test_the_codec_process_imports_no_engine_code() -> None:
    """The codec decode runs in its own process; loading its stage must not pull in the AR engine."""
    script = (
        "import sys, sglang_omni.models.lfm3_audio.config, sglang_omni.models.lfm3_audio.stages; "
        "print(chr(10).join(sorted(sys.modules)))"
    )
    modules = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    engine_modules = [
        module
        for module in modules
        if module.startswith(("sglang.srt.managers", "sglang.srt.model_executor"))
        or module
        in (
            "sglang_omni.models.lfm3_audio.engine_builder",
            "sglang_omni.models.lfm3_audio.model_runner",
            "sglang_omni.models.lfm3_audio.sglang_model",
        )
    ]
    assert engine_modules == []


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": -5},
        {"min_p": 2},
        {"repetition_penalty": 0},
        {"repetition_penalty": 2.5},
        {"top_k": 0, "temperature": 0.7},
        {"top_k": 2**40},
    ],
)
def test_text_sampling_values_the_engine_rejects_are_request_errors(
    params: dict[str, object],
) -> None:
    """The engine checks these ranges only after dispatch, where a bad value was a 500."""
    with pytest.raises(ValueError) as error:
        stages.text_sampling_params(params)
    assert is_bad_request_error(error.value)


def test_text_sampling_keeps_the_values_in_range() -> None:
    params = {"temperature": 0.7, "top_k": -1, "top_p": 1, "seed": 3}
    assert stages.text_sampling_params(params) == {
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 1,
    }


@pytest.mark.parametrize("seed", [-1, 2**63, 2**70, "7", True])
def test_seeds_outside_int64_are_request_errors(seed: object) -> None:
    with pytest.raises(ValueError) as error:
        stages.request_seed({"seed": seed})
    assert is_bad_request_error(error.value)


def test_an_undecodable_image_is_a_request_error() -> None:
    preprocessor = stages.Lfm3AudioPreprocessor.__new__(stages.Lfm3AudioPreprocessor)
    preprocessor.has_vision = True
    part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgpub3QgYSBwbmc="},
    }
    with pytest.raises(ValueError, match="could not read the image") as error:
        asyncio.run(preprocessor.read_part(part, stages.Media()))
    assert is_bad_request_error(error.value)


@pytest.mark.parametrize("value", [0, -1, 2.5, "8", True])
def test_a_max_new_tokens_outside_the_positive_integers_is_a_request_error(
    value: object,
) -> None:
    with pytest.raises(ValueError) as error:
        stages.max_new_tokens({"max_new_tokens": value})
    assert is_bad_request_error(error.value)


def test_an_unset_max_new_tokens_takes_the_default() -> None:
    assert stages.max_new_tokens({}) == stages.DEFAULT_MAX_NEW_TOKENS
