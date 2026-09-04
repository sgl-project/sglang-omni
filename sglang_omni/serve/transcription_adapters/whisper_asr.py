# SPDX-License-Identifier: Apache-2.0
"""Segment-timestamp response adapter for Whisper ASR."""

from __future__ import annotations

import logging
import re

from sglang_omni.serve.protocol import (
    TranscriptionSegment,
    TranscriptionVerboseResponse,
)
from sglang_omni.serve.transcription_adapters.base import register_transcription_adapter
from sglang_omni.serve.transcription_adapters.whisper import WhisperTranscriptionAdapter

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"<\|\d+(?:\.\d+)?\|>")
_SEGMENT_RE = re.compile(
    r"<\|(?P<start>\d+(?:\.\d+)?)\|>" r"(?P<text>.*?)" r"<\|(?P<end>\d+(?:\.\d+)?)\|>",
    re.DOTALL,
)


@register_transcription_adapter("Whisper")
class WhisperASRAdapter(WhisperTranscriptionAdapter):
    @property
    def supports_segment_timestamps(self) -> bool:
        return True

    def postprocess_text(self, text: str) -> str:
        return _TIMESTAMP_RE.sub("", text).strip()

    def build_verbose_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        clean_text = self.postprocess_text(text)
        segments = self._parse_segments(text)
        if not segments:
            segments = self._build_fallback_segments(clean_text, audio_duration_s)
        return self._build_response(clean_text, language, audio_duration_s, segments)

    def build_timestamped_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        segments = self._parse_timestamped_segments(text)
        return self._build_response(
            self.postprocess_text(text), language, audio_duration_s, segments
        )

    def _build_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
        segments: list[TranscriptionSegment],
    ) -> TranscriptionVerboseResponse:
        segments = self._sanitize_segments(segments, audio_duration_s)
        duration = (
            round(float(audio_duration_s), 2)
            if audio_duration_s > 0
            else max((segment.end for segment in segments), default=0.0)
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
        for match in _SEGMENT_RE.finditer(text):
            segment_text = match.group("text").strip()
            if not segment_text:
                continue
            segments.append(
                TranscriptionSegment(
                    id=len(segments),
                    start=round(float(match.group("start")), 2),
                    end=round(float(match.group("end")), 2),
                    text=segment_text,
                )
            )
        return segments

    @staticmethod
    def _parse_timestamped_segments(text: str) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        cursor = 0
        previous_end = 0.0
        for match in _SEGMENT_RE.finditer(text):
            if text[cursor : match.start()].strip():
                raise ValueError("model did not produce segment timestamps")

            start = float(match.group("start"))
            end = float(match.group("end"))
            if start < previous_end or end < start:
                raise ValueError("model did not produce segment timestamps")

            segment_text = match.group("text").strip()
            if segment_text:
                segments.append(
                    TranscriptionSegment(
                        id=len(segments),
                        start=round(start, 2),
                        end=round(end, 2),
                        text=segment_text,
                    )
                )
            previous_end = end
            cursor = match.end()

        if text[cursor:].strip() or not segments:
            raise ValueError("model did not produce segment timestamps")
        return segments

    @staticmethod
    def _sanitize_segments(
        segments: list[TranscriptionSegment], audio_duration_s: float
    ) -> list[TranscriptionSegment]:
        if audio_duration_s <= 0:
            return segments
        limit = round(float(audio_duration_s), 2)
        repaired = 0
        for segment in segments:
            start = min(segment.start, limit)
            end = min(max(segment.end, start), limit)
            if start != segment.start or end != segment.end:
                repaired += 1
                segment.start = start
                segment.end = end
        if repaired:
            logger.warning(
                "Clamped %d Whisper transcription segments with timestamps "
                "outside the audio duration",
                repaired,
            )
        return segments

    @staticmethod
    def _build_fallback_segments(
        text: str, audio_duration_s: float
    ) -> list[TranscriptionSegment]:
        text = text.strip()
        if not text:
            return []
        return [
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=round(max(float(audio_duration_s), 0.0), 2),
                text=text,
            )
        ]
