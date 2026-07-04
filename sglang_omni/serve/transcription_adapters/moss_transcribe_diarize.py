# SPDX-License-Identifier: Apache-2.0
"""verbose_json adapter for MOSS-Transcribe-Diarize.

The model emits [start][S01] text [end] segments (speaker labels +
timestamps). This adapter parses that markup into OpenAI verbose_json
segments. The segment regex is ported from the upstream SGLang adapter.
"""

from __future__ import annotations

import re

from sglang_omni.serve.protocol import (
    TranscriptionSegment,
    TranscriptionVerboseResponse,
)
from sglang_omni.serve.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)

_SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\s*\[(?P<speaker>S\d{2,})\]"
    r"(?P<text>.*?)"
    r"\s*\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]\s*\[S\d{2,}\]|$))",
    re.DOTALL,
)


@register_transcription_adapter("MossTranscribeDiarize")
class MossTranscribeDiarizeAdapter(TranscriptionAdapter):
    def postprocess_text(self, text: str) -> str:
        return _SPECIAL_TOKEN_RE.sub("", text).strip()

    def build_verbose_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        segments = self._parse_segments(text)
        if not segments:
            segments = self._build_fallback_segments(text, audio_duration_s)
        duration = (
            round(float(audio_duration_s), 2)
            if audio_duration_s > 0
            else max((seg.end for seg in segments), default=0.0)
        )
        return TranscriptionVerboseResponse(
            language=language,
            duration=round(duration, 2),
            text=text,
            segments=segments,
        )

    @staticmethod
    def _parse_segments(text: str) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        for segment_id, match in enumerate(_SEGMENT_RE.finditer(text)):
            speaker = match.group("speaker")
            body = match.group("text").strip()
            segment_text = f"[{speaker}]{body}" if body else f"[{speaker}]"
            segments.append(
                TranscriptionSegment(
                    id=segment_id,
                    start=round(float(match.group("start")), 2),
                    end=round(float(match.group("end")), 2),
                    text=segment_text,
                )
            )
        return segments

    @staticmethod
    def _build_fallback_segments(
        text: str, audio_duration_s: float
    ) -> list[TranscriptionSegment]:
        text = text.strip()
        if not text:
            return []
        if not re.match(r"^\[S\d{2,}\]", text):
            text = f"[S01]{text}"
        return [
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=round(max(float(audio_duration_s), 0.0), 2),
                text=text,
            )
        ]
