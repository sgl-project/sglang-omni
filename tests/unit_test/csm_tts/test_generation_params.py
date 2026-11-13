# SPDX-License-Identifier: Apache-2.0
"""CSM sampling uses caller-set fields, otherwise model defaults."""

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


def test_caller_set_sampling_fields_are_kept():
    gen = resolve_csm_generation_params(
        {"temperature": 0.8, "top_p": 0.8, "top_k": 30},
    )
    assert gen["temperature"] == 0.8
    assert gen["top_p"] == 0.8
    assert gen["top_k"] == 30
    assert gen["depth_temperature"] == 0.8
    assert gen["depth_top_k"] == 30


def test_non_default_overrides_and_max_new_tokens():
    gen = resolve_csm_generation_params(
        {"temperature": 0.2, "top_k": 10, "max_new_tokens": 32}
    )
    assert gen["temperature"] == 0.2
    assert gen["top_k"] == 10
    assert gen["max_new_tokens"] == 32
    assert gen["top_p"] is None


def test_seed_falls_back_to_tts_params():
    gen = resolve_csm_generation_params({}, {"seed": 7})
    assert gen["seed"] == 7
