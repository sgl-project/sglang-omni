# SPDX-License-Identifier: Apache-2.0
"""Retain audio turns and translations in the shared realtime ASR session."""

import io
import logging
import wave
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from sglang_omni.client.types import GenerateRequest
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request

logger = logging.getLogger(__name__)


class TranslationTurn(BaseModel):
    end_sample: int = Field(gt=0)
    text: str


@dataclass(kw_only=True)
class HySimulMTStreamingState:
    model_name: str
    source_language: str | None
    history: list[TranslationTurn] = field(default_factory=list)
    pending_end_sample: int = 0
    pending_final: bool = False

    @property
    def committed_end_sample(self) -> int:
        return next(
            (turn.end_sample for turn in reversed(self.history) if turn.text.strip()),
            0,
        )


class HySimulMTStreamingStrategy:
    def create_state(
        self, *, model_name: str, language: str | None
    ) -> HySimulMTStreamingState:
        return HySimulMTStreamingState(model_name=model_name, source_language=language)

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: HySimulMTStreamingState,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest:
        if not isinstance(state, HySimulMTStreamingState):
            raise TypeError("Hy-SimulMT requires its own streaming state")
        else:
            pass
        with wave.open(io.BytesIO(audio), "rb") as waveform:
            if waveform.getframerate() != 16000:
                raise ValueError("Hy-SimulMT realtime audio must be 16 kHz")
            else:
                state.pending_end_sample = waveform.getnframes()
        state.pending_final = is_final
        request = build_speech_to_text_generate_request(
            audio_bytes=audio,
            filename="realtime.wav",
            content_type="audio/wav",
            model=state.model_name,
            language=state.source_language,
            prompt=None,
            temperature=0.0,
            stream=False,
        )
        request.metadata["request_id"] = request_id
        request.extra_params["hy_simulmt_history"] = [
            turn.model_dump() for turn in state.history
        ]
        if is_final:
            request.extra_params["hy_simulmt_final"] = True
        else:
            pass
        return request

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        language: str | None,
        state: HySimulMTStreamingState,
    ) -> str:
        if not isinstance(state, HySimulMTStreamingState):
            raise TypeError("Hy-SimulMT requires its own streaming state")
        else:
            pass
        if state.pending_final and not generated_text.strip():
            if state.pending_end_sample > state.committed_end_sample:
                logger.warning(
                    f"Hy-SimulMT final decode left audio untranslated: "
                    f"start_sample={state.committed_end_sample} "
                    f"end_sample={state.pending_end_sample}"
                )
            else:
                pass
        else:
            if state.pending_final:
                while state.history and not state.history[-1].text.strip():
                    state.history.pop()
            else:
                pass
            if state.pending_final and state.history:
                state.history[-1].text += generated_text
                state.history[-1].end_sample = state.pending_end_sample
            else:
                state.history.append(
                    TranslationTurn(
                        end_sample=state.pending_end_sample, text=generated_text
                    )
                )
        parts = [turn.text for turn in state.history if turn.text.strip()]
        if language == "en":
            translation = ""
            for part in parts:
                if not translation:
                    translation = part
                elif translation[-1].isspace() and part[0].isspace():
                    translation += part.lstrip()
                elif translation[-1].isspace() or part[0].isspace():
                    translation += part
                else:
                    translation += f" {part}"
        else:
            translation = "".join(parts)
        return translation
