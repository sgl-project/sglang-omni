# SPDX-License-Identifier: Apache-2.0
"""Sampling helpers for Ming-Omni thinker requests."""

from __future__ import annotations

from typing import Any


def build_ming_sampling_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": params.get("max_new_tokens", 2048),
        "temperature": params.get("temperature", 0.0),
        "top_p": params.get("top_p", 1.0),
        "top_k": params.get("top_k", -1),
        "min_p": params.get("min_p", 0.0),
        "repetition_penalty": params.get("repetition_penalty", 1.0),
        "stop": params.get("stop") or [],
        "stop_token_ids": params.get("stop_token_ids") or [],
        "sampling_seed": params.get("seed"),
    }


def build_ming_sampling_params(
    params: dict[str, Any],
    *,
    tokenizer: Any,
    vocab_size: int,
):
    from sglang.srt.sampling.sampling_params import SamplingParams

    kwargs = build_ming_sampling_kwargs(params)
    sampling_params = SamplingParams(**kwargs)
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)
    return sampling_params, kwargs["max_new_tokens"], kwargs["temperature"]
