# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.models.moss_tts_realtime.protocol import (
    MossTTSRealtimeInputText,
    MossTTSRealtimeSpeechSessionConfig,
    MossTTSRealtimeTurnUser,
    moss_tts_realtime_event_fingerprint,
    parse_moss_tts_realtime_client_event,
    speech_websocket_session_config_payload,
)


def test_session_config_payload_accepts_flat_and_nested_forms() -> None:
    flat = {
        "type": "session.config",
        "sample_rate": 24000,
    }
    nested = {
        "type": "session.config",
        "session": {
            "sample_rate": 24000,
        },
    }

    assert speech_websocket_session_config_payload(flat) == {
        "sample_rate": 24000,
    }
    assert speech_websocket_session_config_payload(nested) == nested["session"]


def test_session_config_enforces_realtime_pcm_contract() -> None:
    config = MossTTSRealtimeSpeechSessionConfig()

    assert config.response_format == "pcm"
    assert config.stream_audio is True
    assert config.sample_rate == 24000
    with pytest.raises(ValidationError):
        MossTTSRealtimeSpeechSessionConfig(mode="moss_tts_realtime")
    with pytest.raises(ValidationError):
        MossTTSRealtimeSpeechSessionConfig(
            sample_rate=16000,
        )


def test_turn_user_requires_complete_text_audio_pair() -> None:
    assert MossTTSRealtimeTurnUser(text="hello", audio="data:audio/wav;base64,AA==")
    with pytest.raises(ValidationError, match="both text and audio"):
        MossTTSRealtimeTurnUser(text="hello")


def test_client_event_parser_is_strict_and_returns_none_for_unknown_types() -> None:
    event = parse_moss_tts_realtime_client_event(
        {
            "type": "input.text",
            "turn_id": "turn-1",
            "seq_no": 0,
            "text": "hello",
        }
    )

    assert isinstance(event, MossTTSRealtimeInputText)
    assert parse_moss_tts_realtime_client_event({"type": "unknown"}) is None
    with pytest.raises(ValidationError):
        parse_moss_tts_realtime_client_event(
            {
                "type": "input.text",
                "turn_id": "turn-1",
                "seq_no": 0,
                "text": "hello",
                "unexpected": True,
            }
        )


def test_event_fingerprint_is_stable_and_content_sensitive() -> None:
    first = MossTTSRealtimeInputText(
        type="input.text",
        turn_id="turn-1",
        seq_no=0,
        text="hello",
    )
    same = MossTTSRealtimeInputText.model_validate(first.model_dump())
    changed = MossTTSRealtimeInputText(
        type="input.text",
        turn_id="turn-1",
        seq_no=0,
        text="hello!",
    )

    assert moss_tts_realtime_event_fingerprint(first) == (
        moss_tts_realtime_event_fingerprint(same)
    )
    assert moss_tts_realtime_event_fingerprint(first) != (
        moss_tts_realtime_event_fingerprint(changed)
    )
