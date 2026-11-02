# SPDX-License-Identifier: Apache-2.0
"""Re-decode a growing Whisper segment with replaceable text hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.client.types import GenerateRequest
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request


@dataclass(slots=True, kw_only=True)
class WhisperStreamingState:
    model_name: str
    language: str


class WhisperStreamingStrategy:
    def create_state(
        self, *, model_name: str, language: str | None
    ) -> WhisperStreamingState:
        return WhisperStreamingState(model_name=model_name, language=language or "en")

    @staticmethod
    def state(state: object) -> WhisperStreamingState:
        if not isinstance(state, WhisperStreamingState):
            raise TypeError("Whisper received incompatible streaming state")
        else:
            return state

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: object,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest:
        """Finality and request identity do not change a full-segment decode."""
        whisper_state = self.state(state)
        return build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime-segment.wav",
            content_type="audio/wav",
            model=whisper_state.model_name,
            language=whisper_state.language,
            prompt=None,
            temperature=0.0,
            stream=False,
        )

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        language: str | None,
        state: object,
    ) -> str:
        whisper_state = self.state(state)
        if language:
            whisper_state.language = language
        else:
            pass
        return generated_text
