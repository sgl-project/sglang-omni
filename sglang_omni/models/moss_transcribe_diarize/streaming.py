from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.client import GenerateRequest
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request


@dataclass(slots=True)
class MossTranscribeDiarizeStreamingState:
    model_name: str
    language: str | None = None
    chunk_id: int = 0
    transcript: str = ""


class MossTranscribeDiarizeStreamingStrategy:
    """Cumulative re-decode strategy for MOSS-TD realtime transcription."""

    def create_state(
        self, *, model_name: str, language: str | None
    ) -> MossTranscribeDiarizeStreamingState:
        return MossTranscribeDiarizeStreamingState(
            model_name=model_name,
            language=language,
        )

    @staticmethod
    def _state(state: object) -> MossTranscribeDiarizeStreamingState:
        if not isinstance(state, MossTranscribeDiarizeStreamingState):
            raise TypeError(
                "MOSS-Transcribe-Diarize received incompatible streaming state"
            )
        return state

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: object,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest:
        del is_final, request_id
        moss_state = self._state(state)
        return build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime-segment.wav",
            content_type="audio/wav",
            model=moss_state.model_name,
            language=moss_state.language,
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
        moss_state = self._state(state)
        if language:
            moss_state.language = language
        moss_state.transcript = generated_text
        moss_state.chunk_id += 1
        return generated_text


__all__ = [
    "MossTranscribeDiarizeStreamingState",
    "MossTranscribeDiarizeStreamingStrategy",
]
