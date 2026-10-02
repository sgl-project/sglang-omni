# SPDX-License-Identifier: Apache-2.0
"""CPU checks for NPU ASR fixture validation and stream assertions."""

import json

import pytest

from tests.test_model.test_npu_asr import _load_cases, _normalize, _stream_text


def test_normalize_bilingual_transcript():
    assert _normalize("Hello, WORLD！ 你好。") == "helloworld你好"


def _event(kind, **fields):
    return "data: " + json.dumps({"type": "transcript.text." + kind, **fields})


def test_complete_stream():
    assert (
        _stream_text(
            [
                _event("delta", delta="hello"),
                _event("done", text="hello"),
                "data: [DONE]",
            ]
        )
        == "hello"
    )


@pytest.mark.parametrize(
    "events",
    [
        [_event("delta", delta="hello")],
        [_event("done", text="hello")],
        ["data: [DONE]"],
        [_event("delta", delta="wrong"), _event("done", text="hello"), "data: [DONE]"],
    ],
)
def test_reject_incomplete_or_inconsistent_stream(events):
    with pytest.raises(AssertionError):
        _stream_text(events)


def test_cases_require_both_languages_and_calibrated_ceiling(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.touch()
    path = tmp_path / "cases.json"
    case = {"audio": "clip.wav", "text": "hello", "max_cer": 0.1}
    path.write_text(json.dumps({"English": case}))
    with pytest.raises(AssertionError, match="both language"):
        _load_cases(path)
    path.write_text(json.dumps({"English": case, "Chinese": case}))
    assert _load_cases(path)["English"]["audio"] == str(audio)
    case["max_cer"] = 1
    path.write_text(json.dumps({"English": case, "Chinese": case}))
    with pytest.raises(AssertionError, match="CER ceiling"):
        _load_cases(path)
