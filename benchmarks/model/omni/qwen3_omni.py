# SPDX-License-Identifier: Apache-2.0
"""Qwen3 Omni model adapter."""

from __future__ import annotations

from benchmarks.model.base import ModelAdapter

THINKER_MAX_NEW_TOKENS = 256


class Qwen3OmniAdapter(ModelAdapter):
    """Qwen3 Omni: chat completions API at ``/v1/chat/completions``."""

    def __init__(self, speaker: str = "Ethan") -> None:
        super().__init__(
            name="qwen3-omni",
            api_endpoint="/v1/chat/completions",
            default_params={
                "max_tokens": THINKER_MAX_NEW_TOKENS,
                "temperature": 0.7,
                "speaker": speaker,
            },
        )
