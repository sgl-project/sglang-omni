# SPDX-License-Identifier: Apache-2.0
"""Bounded store of finished raw PCM speech stream outcomes."""

from sglang_omni.client.types import UsageInfo
from sglang_omni.serve.speech_stream_outcomes import SpeechStreamOutcomes


def test_store_evicts_the_oldest_outcome_past_max_entries() -> None:
    outcomes = SpeechStreamOutcomes(max_entries=1)
    outcomes.record("speech-1", "stop", None)
    outcomes.record("speech-2", "length", UsageInfo(completion_tokens=120))

    assert outcomes.get("speech-1") is None
    newest = outcomes.get("speech-2")
    assert newest is not None
    assert newest.finish_reason == "length"
    assert newest.usage is not None and newest.usage.completion_tokens == 120


def test_outcome_serialises_missing_usage_as_null() -> None:
    outcomes = SpeechStreamOutcomes(max_entries=1)
    outcomes.record("speech-1", None, None)
    outcome = outcomes.get("speech-1")
    assert outcome is not None
    assert outcome.to_dict() == {
        "request_id": "speech-1",
        "finish_reason": None,
        "usage": None,
    }
