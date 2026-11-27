# SPDX-License-Identifier: Apache-2.0
"""CSM sampling: depth decoder inherits caller values, seed falls back to tts_params."""

from sglang_omni.models.csm_tts.payload_types import resolve_csm_generation_params


def test_depth_inherits_caller_sampling_and_seed_falls_back():
    gen = resolve_csm_generation_params(
        {"temperature": 0.2, "top_k": 10},
        {"seed": 7},
    )
    assert gen["depth_temperature"] == 0.2
    assert gen["depth_top_k"] == 10
    assert gen["seed"] == 7
