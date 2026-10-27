# SPDX-License-Identifier: Apache-2.0
"""CSM sampling resolution ignores leaked speech / generate sentinels."""

from sglang_omni.models.csm_tts.payload_types import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    resolve_csm_generation_params,
)


def test_empty_params_use_csm_defaults():
    gen = resolve_csm_generation_params({})
    assert gen["temperature"] == DEFAULT_TEMPERATURE
    assert gen["top_k"] == DEFAULT_TOP_K
    assert gen["top_p"] is None
    assert gen["depth_temperature"] == DEFAULT_TEMPERATURE
    assert gen["depth_top_k"] == DEFAULT_TOP_K
    assert gen["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS


def test_speech_sentinels_are_ignored_without_explicit_fields():
    gen = resolve_csm_generation_params(
        {
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
        {"voice": "alloy"},
    )
    assert gen["temperature"] == DEFAULT_TEMPERATURE
    assert gen["top_k"] == DEFAULT_TOP_K
    assert gen["top_p"] is None


def test_generate_client_sentinels_are_ignored():
    gen = resolve_csm_generation_params({"temperature": 1.0, "top_p": 1.0, "top_k": -1})
    assert gen["temperature"] == DEFAULT_TEMPERATURE
    assert gen["top_k"] == DEFAULT_TOP_K
    assert gen["top_p"] is None


def test_explicit_speech_sentinels_are_kept():
    gen = resolve_csm_generation_params(
        {"temperature": 0.8, "top_p": 0.8, "top_k": 30},
        {"explicit_generation_params": ["temperature", "top_p", "top_k"]},
    )
    assert gen["temperature"] == 0.8
    assert gen["top_p"] == 0.8
    assert gen["top_k"] == 30
    assert gen["depth_temperature"] == 0.8
    assert gen["depth_top_k"] == 30


def test_non_sentinel_overrides_do_not_need_explicit_flag():
    gen = resolve_csm_generation_params(
        {"temperature": 0.2, "top_k": 10, "max_new_tokens": 32}
    )
    assert gen["temperature"] == 0.2
    assert gen["top_k"] == 10
    assert gen["max_new_tokens"] == 32
    assert gen["top_p"] is None
