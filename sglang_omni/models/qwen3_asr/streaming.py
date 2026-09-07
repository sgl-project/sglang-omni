from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.client import GenerateRequest
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request

_ROLLBACK_TOKENS = 5
_UNFIXED_CHUNK_NUM = 2


@dataclass(slots=True)
class Qwen3ASRStreamingState:
    model_name: str
    language: str | None = None
    chunk_id: int = 0
    transcript: str = ""


class Qwen3ASRStreamingStrategy:
    def create_state(
        self, *, model_name: str, language: str | None
    ) -> Qwen3ASRStreamingState:
        return Qwen3ASRStreamingState(model_name=model_name, language=language)

    @staticmethod
    def _state(state: object) -> Qwen3ASRStreamingState:
        if not isinstance(state, Qwen3ASRStreamingState):
            raise TypeError("Qwen3-ASR received incompatible streaming state")
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
        qwen_state = self._state(state)
        use_prefix = qwen_state.chunk_id >= _UNFIXED_CHUNK_NUM and bool(
            qwen_state.transcript
        )
        request = build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime-segment.wav",
            content_type="audio/wav",
            model=qwen_state.model_name,
            language=qwen_state.language,
            prompt=None,
            temperature=0.0,
            stream=False,
        )
        request.extra_params.update(
            {
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": (
                    qwen_state.transcript if use_prefix else None
                ),
                "_asr_streaming_rollback_tokens": (
                    _ROLLBACK_TOKENS if use_prefix else 0
                ),
            }
        )
        return request

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        language: str | None,
        state: object,
    ) -> str:
        qwen_state = self._state(state)
        if language:
            qwen_state.language = language
        qwen_state.transcript = generated_text
        qwen_state.chunk_id += 1
        return generated_text


__all__ = ["Qwen3ASRStreamingState", "Qwen3ASRStreamingStrategy"]
