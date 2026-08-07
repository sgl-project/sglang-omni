# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-Transcribe-Diarize verbose_json adapter."""

from __future__ import annotations

from sglang_omni.serve.transcription_adapters import resolve_adapter
from sglang_omni.serve.transcription_adapters.base import DefaultTranscriptionAdapter
from sglang_omni.serve.transcription_adapters.moss_transcribe_diarize import (
    _TIMESTAMP_TOLERANCE_S,
    MossTranscribeDiarizeAdapter,
)


def _moss_adapter() -> MossTranscribeDiarizeAdapter:
    return MossTranscribeDiarizeAdapter()


def test_resolve_adapter_matches_moss_architecture() -> None:
    adapter = resolve_adapter(["MossTranscribeDiarizeForConditionalGeneration"])
    assert isinstance(adapter, MossTranscribeDiarizeAdapter)


def test_resolve_adapter_falls_back_to_default() -> None:
    assert isinstance(resolve_adapter(["SomethingElse"]), DefaultTranscriptionAdapter)
    assert isinstance(resolve_adapter([]), DefaultTranscriptionAdapter)
    assert isinstance(resolve_adapter(None), DefaultTranscriptionAdapter)


def test_postprocess_strips_special_tokens_only() -> None:
    adapter = _moss_adapter()
    raw = "<|im_start|>[0.12][S01] hello[3.27]<|im_end|>"
    assert adapter.postprocess_text(raw) == "[0.12][S01] hello[3.27]"


def test_parse_single_segment() -> None:
    adapter = _moss_adapter()
    text = "[0.12][S01] We asked over twenty different people.[3.27]"
    resp = adapter.build_verbose_response(
        text=text, language="en", audio_duration_s=3.3
    )
    assert len(resp.segments) == 1
    seg = resp.segments[0]
    assert seg.id == 0
    assert seg.start == 0.12
    assert seg.end == 3.27
    assert seg.text == "[S01]We asked over twenty different people."
    assert resp.duration == 3.3
    assert resp.language == "en"


def test_parse_multi_speaker_segments() -> None:
    adapter = _moss_adapter()
    text = "[0.00][S01] Hello there.[1.20][1.30][S02] How are you.[3.00]"
    resp = adapter.build_verbose_response(
        text=text, language=None, audio_duration_s=3.0
    )
    assert [(s.id, s.start, s.end) for s in resp.segments] == [
        (0, 0.0, 1.2),
        (1, 1.3, 3.0),
    ]
    assert resp.segments[0].text == "[S01]Hello there."
    assert resp.segments[1].text == "[S02]How are you."
    assert resp.language is None


def test_fallback_when_no_markup() -> None:
    adapter = _moss_adapter()
    resp = adapter.build_verbose_response(
        text="plain transcript no markers", language="en", audio_duration_s=4.5
    )
    assert len(resp.segments) == 1
    seg = resp.segments[0]
    assert seg.start == 0.0
    assert seg.end == 4.5
    assert seg.text == "[S01]plain transcript no markers"
    assert resp.duration == 4.5


def test_empty_text_yields_no_segments() -> None:
    adapter = _moss_adapter()
    resp = adapter.build_verbose_response(
        text="   ", language="en", audio_duration_s=2.0
    )
    assert resp.segments == []


def test_default_adapter_single_segment() -> None:
    adapter = DefaultTranscriptionAdapter()
    resp = adapter.build_verbose_response(
        text="hello world", language=None, audio_duration_s=1.5
    )
    assert len(resp.segments) == 1
    assert resp.segments[0].text == "hello world"
    assert resp.segments[0].end == 1.5


def test_sanitize_clamps_timestamp_beyond_audio_duration() -> None:
    adapter = _moss_adapter()
    duration_s = 4601.3
    text = "[0.00][S01] Hello.[1.20][3797.61][S02] Trade in phones.[4809.84]"
    resp = adapter.build_verbose_response(
        text=text, language=None, audio_duration_s=duration_s
    )
    assert len(resp.segments) == 2
    assert resp.segments[0].start == 0.0
    assert resp.segments[0].end == 1.2
    assert resp.segments[1].start == 3797.61
    assert resp.segments[1].end == round(duration_s + _TIMESTAMP_TOLERANCE_S, 2)


def test_sanitize_keeps_end_within_tolerance_of_duration() -> None:
    adapter = _moss_adapter()
    duration_s = 10.0
    end = duration_s + _TIMESTAMP_TOLERANCE_S / 2
    text = f"[0.00][S01] Hello.[{end:.2f}]"
    resp = adapter.build_verbose_response(
        text=text, language=None, audio_duration_s=duration_s
    )
    assert resp.segments[0].end == round(end, 2)
