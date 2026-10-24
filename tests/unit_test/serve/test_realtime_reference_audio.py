# SPDX-License-Identifier: Apache-2.0
"""Reference audio is bounded and immutable after native session admission."""

import base64
import io
import json
import wave

import pytest

from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.reference_audio import MAX_REFERENCE_AUDIO_BYTES
from sglang_omni.serve.realtime.schema import JsonValue
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits
from tests.unit_test.serve.test_realtime_duplex_session import (
    ScriptedAdapter,
    build_test_client,
    send_event,
)


def wav_reference(rate: int = 16000, frames: int = 160) -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(bytes(frames * 2))
    return base64.b64encode(output.getvalue()).decode("ascii")


WAV = base64.b64decode(wav_reference())


def encode(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param("%%%%", None, id="not_base64"),
        pytest.param(
            encode(bytes(MAX_REFERENCE_AUDIO_BYTES + 1)), None, id="over_byte_limit"
        ),
        pytest.param(
            encode(b"RIFF" + (4).to_bytes(4, "little") + b"WAVE"),
            None,
            id="header_only",
        ),
        pytest.param(
            encode(b"RIFF" + (12).to_bytes(4, "little") + b"WAVEJUNK" + b"\xff" * 4),
            None,
            id="chunk_overflow",
        ),
        pytest.param(wav_reference(7999, 160), None, id="rate_below_8k"),
        pytest.param(wav_reference(8000, 240001), None, id="over_30_seconds"),
        pytest.param(wav_reference(16000, 0), None, id="empty"),
        pytest.param(
            wav_reference(8000, 240000),
            base64.b64decode(wav_reference(8000, 240000)),
            id="30_seconds_at_8k",
        ),
        pytest.param(
            encode(WAV[:4] + b"\xff" * 4 + WAV[8:40] + b"\xff" * 4 + WAV[44:]),
            WAV,
            id="placeholder_sizes",
        ),
        pytest.param(
            encode(WAV[:4] + (len(WAV) - 9).to_bytes(4, "little") + WAV[8:-1]),
            base64.b64decode(wav_reference(frames=159)),
            id="odd_tail",
        ),
        pytest.param(
            encode(
                WAV[:4]
                + (len(WAV) + 16).to_bytes(4, "little")
                + WAV[8:]
                + WAV[12:24]
                + (1).to_bytes(4, "little")
                + WAV[28:36]
            ),
            WAV,
            id="later_format_chunk",
        ),
    ],
)
def test_reference_is_validated_and_normalized(
    data: JsonValue, expected: bytes | None
) -> None:
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    patch = {"sglang": {"reference_audio": {"media_type": "audio/wav", "data": data}}}
    if expected is None:
        with pytest.raises(ProtocolError) as error:
            negotiation.negotiate({}, "CREATED", patch)
        assert error.value.code == "invalid_request"
    else:
        config, _ = negotiation.negotiate({}, "CREATED", patch)
        assert config["sglang"]["reference_audio"].data == expected


def test_reference_is_never_echoed() -> None:
    reference = {"media_type": "audio/wav", "data": wav_reference(8000, 240000)}
    with build_test_client(
        ScriptedAdapter(), capabilities=Capabilities(supports_reference_audio=True)
    ).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(
            websocket,
            "session.update",
            session={
                "sglang": {
                    "reference_audio": reference,
                    "tts_reference_audio": reference,
                }
            },
        )
        updated = websocket.receive_json()
    assert updated["type"] == "session.updated"
    assert len(json.dumps(updated)) < 4096
