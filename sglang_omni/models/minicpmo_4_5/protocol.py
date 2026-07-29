# SPDX-License-Identifier: Apache-2.0
"""Model-owned protocol helpers for MiniCPM-o native duplex sessions."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Mapping

from sglang_omni.proto import StagePayload

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
INPUT_SAMPLES_PER_UNIT = 16_000
MAX_VIDEO_FRAMES_PER_UNIT = 8
MAX_INLINE_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_AUDIO_SAMPLES = 30 * INPUT_SAMPLE_RATE


class DuplexProtocolError(ValueError):
    """Raised when a MiniCPM-o duplex command violates session ordering."""


@dataclass(frozen=True)
class OpenSession:
    session_id: str
    generation: int
    response_epoch: int
    next_input_seq: int
    system_prompt: str
    config: dict[str, Any]
    voice: dict[str, Any]


@dataclass(frozen=True)
class SessionCommand:
    session_id: str
    generation: int
    input_seq: int
    response_epoch: int
    command: str
    data: dict[str, Any]


def extract_open_session(payload: StagePayload) -> OpenSession:
    request = payload.request
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    inputs = request.inputs if isinstance(request.inputs, dict) else {}

    raw: dict[str, Any] = {}
    user_session = metadata.get("_duplex_session")
    if isinstance(user_session, Mapping):
        raw.update(user_session)
    for key in ("_duplex_session", "duplex_session"):
        input_session = inputs.get(key)
        if isinstance(input_session, Mapping):
            raw.update(input_session)
    coordinator_session = metadata.get("duplex_session")
    if isinstance(coordinator_session, Mapping):
        raw.update(coordinator_session)

    session_id = _required_string(
        raw.get("session_id", payload.request_id), "session_id"
    )
    if session_id != payload.request_id:
        raise DuplexProtocolError("duplex session_id must match request_id")

    generation = _positive_int(raw.get("generation"), "generation")
    response_epoch = _non_negative_int(raw.get("response_epoch", 0), "response_epoch")
    next_input_seq_value = raw.get("next_input_seq")
    if next_input_seq_value is None:
        input_seq = _non_negative_int(raw.get("input_seq", 0), "input_seq")
        next_input_seq_value = input_seq + 1
    next_input_seq = _positive_int(next_input_seq_value, "next_input_seq")
    if response_epoch != 0 or next_input_seq != 1:
        raise DuplexProtocolError(
            "a new duplex generation must start at response_epoch=0 and input_seq=0"
        )
    config = _mapping(raw.get("config"), "config")
    voice = _normalize_session_voice(raw.get("voice"))
    system_prompt = raw.get("system_prompt", config.get("system_prompt"))
    if system_prompt is None:
        system_prompt = "Streaming Omni Conversation."
    system_prompt = _required_string(system_prompt, "system_prompt")

    return OpenSession(
        session_id=session_id,
        generation=generation,
        response_epoch=response_epoch,
        next_input_seq=next_input_seq,
        system_prompt=system_prompt,
        config=config,
        voice=voice,
    )


def extract_session_command(value: Any) -> SessionCommand:
    if isinstance(value, Mapping):
        get = value.get
    else:

        def get(name: str, default: Any = None) -> Any:
            return getattr(value, name, default)

    command = _required_string(get("command"), "command")
    if command not in {"append", "interrupt", "playback_ack", "close"}:
        raise DuplexProtocolError(f"unsupported duplex command {command!r}")
    return SessionCommand(
        session_id=_required_string(get("session_id"), "session_id"),
        generation=_positive_int(get("generation"), "generation"),
        input_seq=_positive_int(get("input_seq"), "input_seq"),
        response_epoch=_non_negative_int(get("response_epoch"), "response_epoch"),
        command=command,
        data=_mapping(get("data", {}), "data"),
    )


def normalize_append_data(data: Mapping[str, Any]) -> dict[str, Any]:
    sample_rate = data.get("sample_rate", INPUT_SAMPLE_RATE)
    if type(sample_rate) is not int or sample_rate != INPUT_SAMPLE_RATE:
        raise DuplexProtocolError(
            f"MiniCPM-o duplex input sample_rate must be {INPUT_SAMPLE_RATE}"
        )

    audio = None
    for key in ("audio_pcm16", "audio_pcm16_b64", "audio"):
        if data.get(key) is not None:
            audio = data[key]
            break

    audio_b64: str | None = None
    payload_bytes = 0
    if audio is not None:
        if isinstance(audio, str):
            encoded = audio.split(",", 1)[1] if audio.startswith("data:") else audio
            try:
                raw_audio = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise DuplexProtocolError("audio_pcm16 is not valid base64") from exc
            audio_b64 = base64.b64encode(raw_audio).decode("ascii")
        elif isinstance(audio, (bytes, bytearray, memoryview)):
            raw_audio = bytes(audio)
            audio_b64 = base64.b64encode(raw_audio).decode("ascii")
        else:
            raise DuplexProtocolError("audio_pcm16 must be bytes or base64 text")
        expected_bytes = INPUT_SAMPLES_PER_UNIT * 2
        if len(raw_audio) != expected_bytes:
            raise DuplexProtocolError(
                f"each audio unit must contain exactly {INPUT_SAMPLES_PER_UNIT} "
                f"PCM16 samples ({expected_bytes} bytes), got {len(raw_audio)} bytes"
            )
        payload_bytes += len(raw_audio)

    raw_frames = data.get("video_frames", data.get("frames", []))
    if raw_frames is None:
        raw_frames = []
    if not isinstance(raw_frames, (list, tuple)):
        raise DuplexProtocolError("video_frames must be a list")
    if len(raw_frames) > MAX_VIDEO_FRAMES_PER_UNIT:
        raise DuplexProtocolError(
            f"video_frames supports at most {MAX_VIDEO_FRAMES_PER_UNIT} frames per unit"
        )
    frames: list[str] = []
    for item in raw_frames:
        encoded_frame, frame_bytes = _normalize_frame(item)
        frames.append(encoded_frame)
        payload_bytes += frame_bytes
    if payload_bytes > MAX_INLINE_PAYLOAD_BYTES:
        raise DuplexProtocolError(
            f"duplex inline unit exceeds {MAX_INLINE_PAYLOAD_BYTES} bytes"
        )
    if audio_b64 is None and not frames:
        raise DuplexProtocolError("append requires audio_pcm16 or video_frames")

    max_slice_nums = data.get("max_slice_nums", 1)
    if type(max_slice_nums) is not int or not 1 <= max_slice_nums <= 9:
        raise DuplexProtocolError("max_slice_nums must be an int between 1 and 9")

    force_listen = data.get("force_listen", False)
    if type(force_listen) is not bool:
        raise DuplexProtocolError("force_listen must be a bool")

    normalized: dict[str, Any] = {
        "sample_rate": INPUT_SAMPLE_RATE,
        "video_frames": frames,
        "force_listen": force_listen,
        "max_slice_nums": max_slice_nums,
    }
    if audio_b64 is not None:
        normalized["audio_pcm16_b64"] = audio_b64
    timestamp_ms = data.get("timestamp_ms")
    if timestamp_ms is not None:
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, (int, float))
            or not math.isfinite(float(timestamp_ms))
            or timestamp_ms < 0
        ):
            raise DuplexProtocolError(
                "timestamp_ms must be a finite non-negative number"
            )
        normalized["timestamp_ms"] = float(timestamp_ms)
    return normalized


def make_envelope(
    *,
    event_type: str,
    session_id: str,
    generation: int,
    input_seq: int,
    response_epoch: int,
    output_seq: int,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "type": event_type,
        "session_id": session_id,
        "generation": generation,
        "input_seq": input_seq,
        "response_epoch": response_epoch,
        "output_seq": output_seq,
        **fields,
    }


def _normalize_frame(value: Any) -> tuple[str, int]:
    if isinstance(value, str):
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise DuplexProtocolError("video frame is not valid base64") from exc
        return base64.b64encode(raw).decode("ascii"), len(raw)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return base64.b64encode(raw).decode("ascii"), len(raw)
    if isinstance(value, Mapping):
        for key in ("base64", "data", "image"):
            if value.get(key) is not None:
                return _normalize_frame(value[key])
    raise DuplexProtocolError("each video frame must be bytes or base64 text")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DuplexProtocolError(f"{name} must be a mapping")
    return dict(value)


def _normalize_session_voice(value: Any) -> dict[str, Any]:
    voice = _mapping(value, "voice")
    if not voice:
        return {}
    allowed = {"ref_audio_base64", "tts_ref_audio_base64"}
    unknown = set(voice) - allowed
    if unknown:
        raise DuplexProtocolError(
            "session voice only accepts inline reference audio; "
            f"unsupported fields: {sorted(unknown)}"
        )
    normalized: dict[str, Any] = {}
    total_samples = 0
    for key, value in voice.items():
        if not isinstance(value, str):
            raise DuplexProtocolError(f"voice.{key} must be base64 text")
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError) as exc:
            raise DuplexProtocolError(f"voice.{key} is not valid base64") from exc
        if not raw or len(raw) % 4:
            raise DuplexProtocolError(f"voice.{key} must contain raw f32le samples")
        total_samples += len(raw) // 4
        normalized[key] = base64.b64encode(raw).decode("ascii")
    if total_samples > MAX_REFERENCE_AUDIO_SAMPLES:
        raise DuplexProtocolError("combined session reference audio exceeds 30 seconds")
    return normalized


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DuplexProtocolError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise DuplexProtocolError(f"{name} must be a positive int")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise DuplexProtocolError(f"{name} must be a non-negative int")
    return value


__all__ = [
    "DuplexProtocolError",
    "INPUT_SAMPLE_RATE",
    "INPUT_SAMPLES_PER_UNIT",
    "MAX_INLINE_PAYLOAD_BYTES",
    "MAX_REFERENCE_AUDIO_SAMPLES",
    "MAX_VIDEO_FRAMES_PER_UNIT",
    "OUTPUT_SAMPLE_RATE",
    "OpenSession",
    "SessionCommand",
    "extract_open_session",
    "extract_session_command",
    "make_envelope",
    "normalize_append_data",
]
