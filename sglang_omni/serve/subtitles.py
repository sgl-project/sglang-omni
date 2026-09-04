# SPDX-License-Identifier: Apache-2.0
"""Subtitle serialization for timestamped transcription segments.

Timestamp and cue-text semantics match the reference openai/whisper
WriteSRT/WriteVTT writers byte for byte.
"""

from __future__ import annotations

from sglang_omni.serve.protocol import TranscriptionSegment


def _format_timestamp(
    seconds: float, *, always_include_hours: bool, decimal_marker: str
) -> str:
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def _cue_text(text: str) -> str:
    # Note (Akazaakane): A literal "-->" inside cue text breaks cue parsing.
    return text.strip().replace("-->", "->")


def segments_to_srt(segments: list[TranscriptionSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp(
            segment.start, always_include_hours=True, decimal_marker=","
        )
        end = _format_timestamp(
            segment.end, always_include_hours=True, decimal_marker=","
        )
        blocks.append(f"{index}\n{start} --> {end}\n{_cue_text(segment.text)}\n\n")
    return "".join(blocks)


def segments_to_vtt(segments: list[TranscriptionSegment]) -> str:
    cues = []
    for segment in segments:
        start = _format_timestamp(
            segment.start, always_include_hours=False, decimal_marker="."
        )
        end = _format_timestamp(
            segment.end, always_include_hours=False, decimal_marker="."
        )
        cues.append(f"{start} --> {end}\n{_cue_text(segment.text)}\n\n")
    return "WEBVTT\n\n" + "".join(cues)
