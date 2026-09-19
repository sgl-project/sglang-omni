# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.client import GenerateRequest
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request

_ROLLBACK_CHARS = 8
_UNFIXED_CHUNK_NUM = 2
_STREAMING_REPETITION_PENALTY = 1.3


@dataclass(slots=True)
class FunASRStreamingState:
    model_name: str
    language: str | None = None
    chunk_id: int = 0
    transcript: str = ""


class FunASRStreamingStrategy:
    def create_state(
        self, *, model_name: str, language: str | None
    ) -> FunASRStreamingState:
        return FunASRStreamingState(model_name=model_name, language=language)

    @staticmethod
    def _state(state: object) -> FunASRStreamingState:
        if not isinstance(state, FunASRStreamingState):
            raise TypeError("Fun-ASR received incompatible streaming state")
        return state

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: object,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest:
        del request_id
        fun_state = self._state(state)
        # note (Xinhao Tan): rollback exists to leave a safety margin for
        # audio that has not arrived yet. On the final decode there is no
        # more audio coming, so rolling back only risks re-generating and
        # possibly corrupting text that may already be correct, for no
        # benefit — skip it and trust the accumulated transcript instead.
        if is_final:
            use_prefix = bool(fun_state.transcript)
            rollback_chars = 0
        else:
            use_prefix = fun_state.chunk_id >= _UNFIXED_CHUNK_NUM and bool(
                fun_state.transcript
            )
            rollback_chars = _ROLLBACK_CHARS if use_prefix else 0
        request = build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime-segment.wav",
            content_type="audio/wav",
            model=fun_state.model_name,
            language=fun_state.language,
            prompt=None,
            temperature=0.0,
            repetition_penalty=(_STREAMING_REPETITION_PENALTY if use_prefix else None),
            stream=False,
        )
        request.extra_params.update(
            {
                "_asr_streaming": True,
                "_asr_streaming_prefix_text": (
                    fun_state.transcript if use_prefix else None
                ),
                "_asr_streaming_rollback_chars": rollback_chars,
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
        fun_state = self._state(state)
        if language:
            fun_state.language = language
        fun_state.transcript = generated_text
        fun_state.chunk_id += 1
        return generated_text


__all__ = ["FunASRStreamingState", "FunASRStreamingStrategy"]
