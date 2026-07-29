from __future__ import annotations

import base64

import numpy as np
import pytest

from sglang_omni.models.minicpmo_4_5.protocol import (
    DuplexProtocolError,
    extract_open_session,
    normalize_append_data,
)
from sglang_omni.proto import OmniRequest, StagePayload


def test_open_session_merges_user_config_with_coordinator_fence() -> None:
    voice = base64.b64encode(np.zeros(16, dtype="<f4").tobytes()).decode("ascii")
    payload = StagePayload(
        request_id="session-1",
        request=OmniRequest(
            inputs={},
            metadata={
                "_duplex_session": {
                    "system_prompt": "keep this prompt",
                    "config": {"sampling": {"temperature": 0.4}},
                    "voice": {"ref_audio_base64": voice},
                },
                "duplex_session": {
                    "session_id": "session-1",
                    "generation": 3,
                    "input_seq": 0,
                    "response_epoch": 0,
                },
            },
        ),
        data=None,
    )

    opened = extract_open_session(payload)

    assert opened.session_id == "session-1"
    assert opened.generation == 3
    assert opened.response_epoch == 0
    assert opened.next_input_seq == 1
    assert opened.system_prompt == "keep this prompt"
    assert opened.config == {"sampling": {"temperature": 0.4}}
    assert opened.voice == {"ref_audio_base64": voice}


def test_open_session_rejects_filesystem_voice_override() -> None:
    payload = StagePayload(
        request_id="session-1",
        request=OmniRequest(
            inputs={},
            metadata={
                "_duplex_session": {"voice": {"ref_audio_path": "/etc/passwd"}},
                "duplex_session": {
                    "session_id": "session-1",
                    "generation": 1,
                    "input_seq": 0,
                    "response_epoch": 0,
                },
            },
        ),
        data=None,
    )

    with pytest.raises(DuplexProtocolError, match="inline reference audio"):
        extract_open_session(payload)


def test_open_session_limits_combined_inline_voice_duration() -> None:
    sixteen_seconds = base64.b64encode(
        np.zeros(16 * 16_000, dtype="<f4").tobytes()
    ).decode("ascii")
    payload = StagePayload(
        request_id="session-1",
        request=OmniRequest(
            inputs={},
            metadata={
                "_duplex_session": {
                    "voice": {
                        "ref_audio_base64": sixteen_seconds,
                        "tts_ref_audio_base64": sixteen_seconds,
                    }
                },
                "duplex_session": {
                    "session_id": "session-1",
                    "generation": 1,
                    "input_seq": 0,
                    "response_epoch": 0,
                },
            },
        ),
        data=None,
    )

    with pytest.raises(DuplexProtocolError, match="combined"):
        extract_open_session(payload)


def test_append_normalizes_one_second_pcm16() -> None:
    pcm = b"\x00\x00" * 16_000

    normalized = normalize_append_data({"audio_pcm16": pcm, "sample_rate": 16_000})

    assert base64.b64decode(normalized["audio_pcm16_b64"]) == pcm
    assert normalized["sample_rate"] == 16_000


@pytest.mark.parametrize(
    "value",
    [b"\x00\x00" * 15_999, b"\x00" * 32_001, "not base64"],
)
def test_append_rejects_malformed_audio_units(value: object) -> None:
    with pytest.raises(DuplexProtocolError):
        normalize_append_data({"audio_pcm16": value, "sample_rate": 16_000})


def test_append_rejects_too_many_video_frames() -> None:
    frame = base64.b64encode(b"jpeg").decode("ascii")
    with pytest.raises(DuplexProtocolError, match="at most 8"):
        normalize_append_data({"video_frames": [frame] * 9})


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_append_rejects_non_boolean_force_listen(value: object) -> None:
    with pytest.raises(DuplexProtocolError, match="force_listen"):
        normalize_append_data(
            {
                "audio_pcm16": b"\x00\x00" * 16_000,
                "force_listen": value,
            }
        )


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_append_rejects_invalid_timestamp(value: object) -> None:
    with pytest.raises(DuplexProtocolError, match="timestamp_ms"):
        normalize_append_data(
            {
                "audio_pcm16": b"\x00\x00" * 16_000,
                "timestamp_ms": value,
            }
        )
