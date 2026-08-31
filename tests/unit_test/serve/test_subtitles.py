# SPDX-License-Identifier: Apache-2.0

from sglang_omni.serve.protocol import TranscriptionSegment
from sglang_omni.serve.subtitles import segments_to_srt, segments_to_vtt


def _segment(
    segment_id: int, start: float, end: float, text: str
) -> TranscriptionSegment:
    return TranscriptionSegment(id=segment_id, start=start, end=end, text=text)


def test_subtitle_serializers_render_golden_strings() -> None:
    segments = [
        _segment(7, 0.0, 1.234, " Hello --> world. "),
        _segment(9, 2.5, 3661.002, "Second cue."),
    ]

    assert segments_to_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,234\n"
        "Hello -> world.\n\n"
        "2\n"
        "00:00:02,500 --> 01:01:01,002\n"
        "Second cue.\n\n"
    )
    assert segments_to_vtt(segments) == (
        "WEBVTT\n\n"
        "00:00.000 --> 00:01.234\n"
        "Hello -> world.\n\n"
        "00:02.500 --> 01:01:01.002\n"
        "Second cue.\n\n"
    )


def test_subtitle_serializers_handle_empty_segments() -> None:
    assert segments_to_srt([]) == ""
    assert segments_to_vtt([]) == "WEBVTT\n\n"
