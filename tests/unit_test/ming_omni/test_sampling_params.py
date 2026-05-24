# SPDX-License-Identifier: Apache-2.0
"""Ming thinker sampling parameter forwarding tests."""

from __future__ import annotations


def test_ming_sampling_kwargs_forward_request_controls() -> None:
    from sglang_omni.models.ming_omni.pipeline.sampling import (
        build_ming_sampling_kwargs,
    )

    kwargs = build_ming_sampling_kwargs(
        {
            "max_new_tokens": 33,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "stop": ["\n\n", "Answer:"],
            "stop_token_ids": [123, 456],
            "seed": 99,
        }
    )

    assert kwargs == {
        "max_new_tokens": 33,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "stop": ["\n\n", "Answer:"],
        "stop_token_ids": [123, 456],
        "sampling_seed": 99,
    }


def test_ming_sampling_kwargs_match_qwen3_defaults_when_absent() -> None:
    from sglang_omni.models.ming_omni.pipeline.sampling import (
        build_ming_sampling_kwargs,
    )

    kwargs = build_ming_sampling_kwargs({})

    assert kwargs == {
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "stop": [],
        "stop_token_ids": [],
        "sampling_seed": None,
    }
