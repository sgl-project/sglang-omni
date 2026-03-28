# SPDX-License-Identifier: Apache-2.0
"""Fish Audio S2 Pro model adapter."""

from __future__ import annotations

from benchmarks.model.base import ModelAdapter


class S2ProAdapter(ModelAdapter):
    """S2 Pro: OAI TTS API at ``/v1/audio/speech``."""

    def __init__(self) -> None:
        super().__init__(
            name="fishaudio/s2-pro",
            api_endpoint="/v1/audio/speech",
            default_params={"max_new_tokens": 2048, "temperature": 0.8},
        )
