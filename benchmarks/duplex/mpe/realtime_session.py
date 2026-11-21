# SPDX-License-Identifier: Apache-2.0
"""Reactive /v1/realtime WebSocket session driver for Duplex-MPE evaluation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import math
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Protocol as TypingProtocol
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
import soundfile
import websockets
from numpy.typing import NDArray
from pydantic import JsonValue

from benchmarks.dataset.duplex_mpe import (
    DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    DuplexMpeScenarioRecord,
    compute_file_sha256,
    resolve_dataset_audio_path,
)
from benchmarks.duplex.mpe.protocol import (
    N4_PAUSE_SECONDS,
    ORDINARY_GAP_SECONDS,
    TARGET_MAX_RESPONSE_SECONDS,
    TARGET_ONSET_DEADLINE_SECONDS,
    ScenarioScheduleResult,
    ScenarioTurnInput,
    compute_protocol_fingerprint,
    iter_scenario_schedule,
    load_mono_waveform_at_rate,
    resolve_scenario_seed,
)
from benchmarks.duplex.mpe.vad import SileroActivityConfig, SileroVadModel

logger = logging.getLogger(__name__)

RealtimeProtocolMode = Literal["native", "turn_based", "legacy"]
RealtimeProtocolSelection = Literal["auto", "native", "turn_based", "legacy"]

DEFAULT_INPUT_SAMPLE_RATE_HZ = 16000
DEFAULT_OUTPUT_SAMPLE_RATE_HZ = 24000
PACKET_DURATION_MILLISECONDS = 80.0
MAX_WEBSOCKET_MESSAGE_BYTES = 8 * 1024 * 1024
ADMISSION_DENIED_STATUS = 503
ADMISSION_RETRIES = 3
ADMISSION_BACKOFF_S = 0.25
WEBSOCKET_SUPPORTS_PROXY: bool = (
    "proxy" in inspect.signature(websockets.connect).parameters
)
LEGACY_TAIL_WAIT_SECONDS = 1.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 420.0
DEFAULT_MAX_PACING_LAG_SECONDS = 0.5
HANDSHAKE_TIMEOUT_SECONDS = 30.0
SCENARIO_DRAIN_MARGIN_SECONDS = 60.0
RECEIPT_DURATION_TOLERANCE_MILLISECONDS = 1e-3
RETRYABLE_CAPABILITIES_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})
CAPABILITIES_MAX_ATTEMPTS = ADMISSION_RETRIES + 1
SEED_POLICY_METADATA_ONLY = "reference_seed_metadata_only"
KNOWN_SAFE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "admission_rejected",
        "rate_limit_exceeded",
        "session_expired",
        "invalid_request_error",
        "server_error",
        "backend_failure",
        "item_not_found",
    }
)


class ProtocolSessionError(RuntimeError):
    """Sanitized protocol-level session error safe to surface in diagnostics."""


class MonotonicClock(TypingProtocol):
    def __call__(self) -> float: ...


class SessionSender(TypingProtocol):
    async def __call__(self) -> None: ...


class SessionSettler(TypingProtocol):
    async def __call__(self) -> bool: ...


@dataclass
class AdmissionState:
    denials: int = 0


def reject_json_constant(constant: str) -> float:
    raise ValueError(f"Out of range float values are not JSON compliant: {constant}")


SERVER_EVENT_DECODER = json.JSONDecoder(parse_constant=reject_json_constant)


def is_admission_rejection(event: dict[str, JsonValue]) -> bool:
    """Detect a non-fatal admission_rejected error frame that keeps the session open for retry."""
    sglang_metadata = event.get("sglang")
    error = event.get("error")
    return (
        isinstance(error, dict)
        and error.get("code") == "admission_rejected"
        and isinstance(sglang_metadata, dict)
        and sglang_metadata.get("fatal") is False
    )


def turn_detection_type(
    session: JsonValue, protocol: RealtimeProtocolMode
) -> JsonValue:
    """Extract turn_detection.type from top-level legacy session or turn-based audio.input."""
    fields = session if isinstance(session, dict) else {}
    if protocol == "turn_based":
        audio = fields.get("audio")
        audio_input = audio.get("input") if isinstance(audio, dict) else None
        fields = audio_input if isinstance(audio_input, dict) else {}
    else:
        pass
    turn = fields.get("turn_detection")
    return turn.get("type") if isinstance(turn, dict) else None


async def connect_realtime_websocket(
    url: str,
    *,
    admission_state: AdmissionState,
    max_retries: int = ADMISSION_RETRIES,
    backoff_seconds: float = ADMISSION_BACKOFF_S,
    trust_env: bool | None = None,
) -> websockets.ClientConnection | None:
    """Connect to /v1/realtime while charging HTTP 503 denials against admission_state."""
    while True:
        try:
            if trust_env is not None and WEBSOCKET_SUPPORTS_PROXY:
                return await websockets.connect(
                    url,
                    max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
                    compression=None,
                    ping_interval=None,
                    proxy=True if trust_env else None,
                )
            else:
                return await websockets.connect(
                    url,
                    max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
                    compression=None,
                    ping_interval=None,
                )
        except websockets.InvalidStatus as exc:
            if exc.response.status_code != ADMISSION_DENIED_STATUS:
                raise
            else:
                admission_state.denials += 1
                if admission_state.denials > max_retries:
                    return None
                else:
                    await asyncio.sleep(backoff_seconds)


async def negotiate_session_admission(
    *,
    send_session_update: SessionSender,
    settle_session_updated: SessionSettler,
    rejected: asyncio.Event,
    aborted: asyncio.Event,
    admission_state: AdmissionState,
    max_retries: int = ADMISSION_RETRIES,
    backoff_seconds: float = ADMISSION_BACKOFF_S,
) -> bool:
    """Send session.update and retry non-fatal session admission rejections under one shared budget."""
    await send_session_update()
    admitted = await settle_session_updated()
    while rejected.is_set() and admission_state.denials <= max_retries:
        rejected.clear()
        try:
            await asyncio.wait_for(aborted.wait(), backoff_seconds)
        except asyncio.TimeoutError:
            await send_session_update()
            admitted = await settle_session_updated()
        else:
            break
    return admitted


DEFAULT_TURN_DETECTION: dict[str, JsonValue] = {
    "type": "server_vad",
    "silence_duration_ms": 500,
}


@dataclass(frozen=True, kw_only=True)
class RealtimeEndpointSpec:
    websocket_url: str
    capabilities_url: str
    protocol: RealtimeProtocolMode
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    turn_detection: dict[str, JsonValue] | None
    reported_model: str | None = None
    trust_env: bool = False


def validate_realtime_base_url(base_url: str) -> None:
    try:
        parsed = urlsplit(base_url)
        is_valid = (
            parsed.scheme in ("http", "https", "ws", "wss")
            and bool(parsed.hostname)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        is_valid = False
    if not is_valid:
        raise ValueError(
            "Realtime base_url must be an HTTP(S) or WS(S) URL without userinfo, query, or fragment"
        )
    else:
        pass


def sanitize_realtime_origin(endpoint_url: str) -> str:
    parsed = urlsplit(endpoint_url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    else:
        pass
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    else:
        pass
    return urlunsplit((parsed.scheme, host, "", "", ""))


def summarize_transport_exception(exc: BaseException) -> str:
    if isinstance(exc, ProtocolSessionError):
        return str(exc)
    elif isinstance(exc, httpx.HTTPStatusError):
        return f"HTTPStatusError(status={exc.response.status_code})"
    elif isinstance(exc, websockets.InvalidStatus):
        return f"InvalidStatus(status={exc.response.status_code})"
    else:
        return type(exc).__name__


def format_server_error_event(error_event: dict[str, JsonValue]) -> str:
    error_body = error_event.get("error")
    if isinstance(error_body, dict):
        for field_name in ("code", "type"):
            candidate = error_body.get(field_name)
            if isinstance(candidate, str) and candidate in KNOWN_SAFE_ERROR_CODES:
                return f"Server returned error event (code={candidate})"
            else:
                pass
    else:
        pass
    return "Server returned error event"


def derive_realtime_urls(base_url: str) -> tuple[str, str]:
    validate_realtime_base_url(base_url)
    parsed = urlsplit(base_url.rstrip("/"))
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    else:
        pass
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    else:
        pass

    ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    http_scheme = "https" if parsed.scheme in ("https", "wss") else "http"
    raw_path = parsed.path.rstrip("/")
    if raw_path.endswith("/v1/realtime"):
        prefix_path = raw_path[: -len("/v1/realtime")]
    elif raw_path.endswith("/v1"):
        prefix_path = raw_path[: -len("/v1")]
    else:
        prefix_path = raw_path

    websocket_url = urlunsplit((ws_scheme, host, f"{prefix_path}/v1/realtime", "", ""))
    capabilities_url = urlunsplit(
        (http_scheme, host, f"{prefix_path}/v1/realtime/capabilities", "", "")
    )
    return websocket_url, capabilities_url


def extract_session_created_capabilities(
    session_payload: dict[str, JsonValue],
) -> tuple[RealtimeProtocolMode, int | None, int | None, str | None]:
    raw_model = session_payload.get("model")
    reported_model = raw_model if isinstance(raw_model, str) and raw_model else None

    detected_input_rate: int | None = None
    detected_output_rate: int | None = None
    audio_block = session_payload.get("audio")
    if isinstance(audio_block, dict):
        input_block = audio_block.get("input")
        input_format = (
            input_block.get("format") if isinstance(input_block, dict) else None
        )
        if isinstance(input_format, dict) and type(input_format.get("rate")) is int:
            detected_input_rate = int(input_format["rate"])
        else:
            pass
        output_block = audio_block.get("output")
        output_format = (
            output_block.get("format") if isinstance(output_block, dict) else None
        )
        if isinstance(output_format, dict) and type(output_format.get("rate")) is int:
            detected_output_rate = int(output_format["rate"])
        else:
            pass
    else:
        pass

    sglang_extension = session_payload.get("sglang")
    if isinstance(sglang_extension, dict):
        sglang_capabilities = sglang_extension.get(
            "capabilities"
        ) or sglang_extension.get("granted")
        if isinstance(sglang_capabilities, dict):
            input_audio_format = sglang_capabilities.get("input_audio_format")
            if (
                detected_input_rate is None
                and isinstance(input_audio_format, dict)
                and type(input_audio_format.get("rate")) is int
            ):
                detected_input_rate = int(input_audio_format["rate"])
            else:
                pass
            output_audio_format = sglang_capabilities.get("output_audio_format")
            if (
                detected_output_rate is None
                and isinstance(output_audio_format, dict)
                and type(output_audio_format.get("rate")) is int
            ):
                detected_output_rate = int(output_audio_format["rate"])
            else:
                pass
            if (
                sglang_capabilities.get("native_full_duplex") is True
                or sglang_capabilities.get("interaction") == "native"
            ):
                return (
                    "native",
                    detected_input_rate,
                    detected_output_rate,
                    reported_model,
                )
            else:
                return (
                    "turn_based",
                    detected_input_rate,
                    detected_output_rate,
                    reported_model,
                )
        else:
            pass
    else:
        pass

    legacy_caps = session_payload.get("capabilities")
    if isinstance(legacy_caps, dict) and "turn_detection" in legacy_caps:
        return ("legacy", detected_input_rate, detected_output_rate, reported_model)
    elif isinstance(audio_block, dict):
        return ("native", detected_input_rate, detected_output_rate, reported_model)
    else:
        raise ValueError("session.created lacks recognizable realtime capabilities")


async def open_websocket_with_admission_state(
    websocket_url: str,
    *,
    admission_state: AdmissionState,
    trust_env: bool = False,
) -> websockets.ClientConnection:
    origin = sanitize_realtime_origin(websocket_url)
    try:
        connection = await connect_realtime_websocket(
            websocket_url,
            admission_state=admission_state,
            max_retries=ADMISSION_RETRIES,
            backoff_seconds=ADMISSION_BACKOFF_S,
            trust_env=trust_env,
        )
    except websockets.InvalidStatus as exc:
        raise RuntimeError(
            f"WebSocket handshake at {origin} rejected: "
            f"{summarize_transport_exception(exc)}"
        ) from None
    except (OSError, websockets.WebSocketException) as exc:
        raise RuntimeError(
            f"WebSocket connection to {origin} failed: "
            f"{summarize_transport_exception(exc)}"
        ) from None
    if connection is None:
        raise RuntimeError(
            f"WebSocket handshake at {origin} rejected after {ADMISSION_RETRIES} "
            f"HTTP 503 admission retries"
        )
    else:
        return connection


async def probe_session_created_over_websocket(
    websocket_url: str, *, timeout_seconds: float, trust_env: bool = False
) -> tuple[RealtimeProtocolMode, int | None, int | None, str | None]:
    admission_state = AdmissionState()
    connection = await asyncio.wait_for(
        open_websocket_with_admission_state(
            websocket_url, admission_state=admission_state, trust_env=trust_env
        ),
        timeout=timeout_seconds,
    )
    async with connection as websocket:
        raw_frame = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
        frame_text = (
            raw_frame if isinstance(raw_frame, str) else raw_frame.decode("utf-8")
        )
        try:
            parsed_event = SERVER_EVENT_DECODER.decode(frame_text)
        except ValueError:
            raise ValueError(
                "Malformed JSON frame received during capability probe"
            ) from None
        if (
            not isinstance(parsed_event, dict)
            or parsed_event.get("type") != "session.created"
            or not isinstance(parsed_event.get("session"), dict)
        ):
            raise ValueError("Expected session.created object during capability probe")
        else:
            session_payload = parsed_event["session"]
            assert isinstance(session_payload, dict)
            result = extract_session_created_capabilities(session_payload)
            if result[0] != "legacy":
                await websocket.send(
                    json.dumps({"type": "session.close", "event_id": uuid.uuid4().hex})
                )
            else:
                pass
            return result


async def negotiate_realtime_endpoint(
    base_url: str,
    *,
    protocol_selection: RealtimeProtocolSelection = "auto",
    input_sample_rate_hz: int | None = None,
    output_sample_rate_hz: int | None = None,
    turn_detection: dict[str, JsonValue] | None = None,
    timeout_seconds: float = 30.0,
    trust_env: bool = False,
) -> RealtimeEndpointSpec:
    """Inspect /v1/realtime/capabilities or session.created to resolve the session contract."""
    websocket_url, capabilities_url = derive_realtime_urls(base_url)
    endpoint_origin = sanitize_realtime_origin(capabilities_url)
    capabilities_payload: dict[str, JsonValue] | None = None
    http_status: int | None = None
    last_http_error: str | None = None

    async with httpx.AsyncClient(
        timeout=timeout_seconds, trust_env=trust_env
    ) as http_client:
        for attempt_index in range(CAPABILITIES_MAX_ATTEMPTS):
            try:
                response = await http_client.get(capabilities_url)
                http_status = response.status_code
                if http_status in RETRYABLE_CAPABILITIES_STATUS_CODES:
                    last_http_error = f"HTTP {http_status}"
                    await asyncio.sleep(ADMISSION_BACKOFF_S * (attempt_index + 1))
                    continue
                elif http_status == 200:
                    try:
                        decoded = response.json()
                    except ValueError:
                        raise ValueError(
                            f"Capabilities endpoint at {endpoint_origin} returned malformed JSON"
                        ) from None
                    if not isinstance(decoded, dict):
                        raise ValueError(
                            f"Capabilities endpoint at {endpoint_origin} returned non-object JSON"
                        )
                    else:
                        capabilities_payload = decoded
                        break
                else:
                    last_http_error = f"HTTP {http_status}"
                    break
            except httpx.HTTPError as exc:
                last_http_error = summarize_transport_exception(exc)
                await asyncio.sleep(ADMISSION_BACKOFF_S * (attempt_index + 1))

    detected_protocol: RealtimeProtocolMode | None = None
    detected_input_rate: int | None = None
    detected_output_rate: int | None = None
    reported_model: str | None = None

    if capabilities_payload is not None:
        raw_model = capabilities_payload.get("model")
        if isinstance(raw_model, str) and raw_model:
            reported_model = raw_model
        else:
            pass
        input_format = capabilities_payload.get("input_audio_format")
        if isinstance(input_format, dict) and type(input_format.get("rate")) is int:
            detected_input_rate = int(input_format["rate"])
        else:
            pass
        output_format = capabilities_payload.get("output_audio_format")
        if isinstance(output_format, dict) and type(output_format.get("rate")) is int:
            detected_output_rate = int(output_format["rate"])
        else:
            pass
        interaction = capabilities_payload.get("interaction")
        if (
            capabilities_payload.get("native_full_duplex") is True
            or interaction == "native"
        ):
            detected_protocol = "native"
        elif interaction == "turn_based":
            detected_protocol = (
                "turn_based"
                if "turn_control" in capabilities_payload
                or "native_unit_ms" in capabilities_payload
                else "legacy"
            )
        elif "turn_detection" in capabilities_payload:
            detected_protocol = "legacy"
        elif protocol_selection == "auto":
            raise ValueError(f"Unrecognized capabilities schema from {endpoint_origin}")
        else:
            pass
    elif protocol_selection == "auto":
        if http_status == 404:
            try:
                (
                    detected_protocol,
                    detected_input_rate,
                    detected_output_rate,
                    reported_model,
                ) = await probe_session_created_over_websocket(
                    websocket_url,
                    timeout_seconds=timeout_seconds,
                    trust_env=trust_env,
                )
            except (
                OSError,
                RuntimeError,
                TimeoutError,
                ValueError,
                websockets.WebSocketException,
            ) as exc:
                raise RuntimeError(
                    f"Capabilities endpoint at {endpoint_origin} returned 404 and "
                    f"WebSocket session.created probe failed: {summarize_transport_exception(exc)}"
                ) from None
        else:
            raise RuntimeError(
                f"Unable to query capabilities at {endpoint_origin} for protocol auto-detection "
                f"({last_http_error or 'no response'}); pass explicit --protocol"
            )
    else:
        logger.warning(
            f"Capabilities probe at {endpoint_origin} unavailable ({last_http_error}); "
            f"using explicit protocol={protocol_selection}"
        )

    if protocol_selection != "auto":
        if detected_protocol is not None and protocol_selection != detected_protocol:
            raise ValueError(
                f"Requested --protocol={protocol_selection!r} conflicts with server "
                f"capability protocol {detected_protocol!r} at {endpoint_origin}"
            )
        else:
            resolved_protocol: RealtimeProtocolMode = protocol_selection
    elif detected_protocol is not None:
        resolved_protocol = detected_protocol
    else:
        raise RuntimeError(
            f"Could not determine realtime protocol from {endpoint_origin}"
        )

    resolved_input_rate = (
        input_sample_rate_hz
        if input_sample_rate_hz is not None
        else (detected_input_rate or DEFAULT_INPUT_SAMPLE_RATE_HZ)
    )
    resolved_output_rate = (
        output_sample_rate_hz
        if output_sample_rate_hz is not None
        else (detected_output_rate or DEFAULT_OUTPUT_SAMPLE_RATE_HZ)
    )
    if resolved_input_rate <= 0 or resolved_output_rate <= 0:
        raise ValueError("Resolved input and output sample rates must be positive")
    else:
        pass
    if (
        detected_input_rate is not None
        and input_sample_rate_hz is not None
        and input_sample_rate_hz != detected_input_rate
    ):
        raise ValueError(
            f"Requested input_sample_rate_hz={input_sample_rate_hz} conflicts with "
            f"server capability rate {detected_input_rate}"
        )
    elif (
        detected_output_rate is not None
        and output_sample_rate_hz is not None
        and output_sample_rate_hz != detected_output_rate
    ):
        raise ValueError(
            f"Requested output_sample_rate_hz={output_sample_rate_hz} conflicts with "
            f"server capability rate {detected_output_rate}"
        )
    else:
        pass

    resolved_turn_detection: dict[str, JsonValue] | None
    if resolved_protocol == "native":
        if turn_detection is not None:
            raise ValueError("Native full-duplex endpoints reject turn_detection")
        else:
            resolved_turn_detection = None
    else:
        resolved_turn_detection = (
            dict(turn_detection)
            if turn_detection is not None
            else dict(DEFAULT_TURN_DETECTION)
        )

    return RealtimeEndpointSpec(
        websocket_url=websocket_url,
        capabilities_url=capabilities_url,
        protocol=resolved_protocol,
        input_sample_rate_hz=resolved_input_rate,
        output_sample_rate_hz=resolved_output_rate,
        turn_detection=resolved_turn_detection,
        reported_model=reported_model,
        trust_env=trust_env,
    )


class PlaybackTimelineBuffer:
    """Monotonic sample-accurate playout timeline with barge-in truncation."""

    def __init__(self, sample_rate_hz: int) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        else:
            pass
        self.sample_rate_hz: int = int(sample_rate_hz)
        self.queued_segments: list[tuple[int, NDArray[np.float32]]] = []
        self.render_cursor_samples: int = 0
        self.tail_samples: int = 0
        self.cancelled_unplayed_samples: int = 0

    def append_audio(
        self, arrival_seconds: float, pcm_samples: NDArray[np.float32]
    ) -> None:
        flat_samples = np.asarray(pcm_samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(flat_samples).all():
            raise ValueError("Received non-finite PCM audio samples from server")
        elif len(flat_samples) == 0:
            return
        else:
            arrival_sample_index = int(round(arrival_seconds * self.sample_rate_hz))
            start_sample_index = max(
                self.render_cursor_samples, self.tail_samples, arrival_sample_index
            )
            self.queued_segments.append((start_sample_index, flat_samples))
            self.tail_samples = start_sample_index + len(flat_samples)

    def cancel_unplayed(self, cutoff_seconds: float) -> None:
        cutoff_sample_index = max(
            self.render_cursor_samples,
            int(round(cutoff_seconds * self.sample_rate_hz)),
        )
        retained_segments: list[tuple[int, NDArray[np.float32]]] = []
        for start_sample_index, segment_samples in self.queued_segments:
            keep_count = max(
                0, min(len(segment_samples), cutoff_sample_index - start_sample_index)
            )
            self.cancelled_unplayed_samples += len(segment_samples) - keep_count
            if keep_count > 0:
                retained_segments.append(
                    (start_sample_index, segment_samples[:keep_count])
                )
            else:
                pass
        self.queued_segments = retained_segments
        self.tail_samples = max(
            [self.render_cursor_samples]
            + [start_index + len(samples) for start_index, samples in retained_segments]
        )

    def render_interval(
        self, start_sample_index: int, end_sample_index: int
    ) -> NDArray[np.float32]:
        if (
            start_sample_index != self.render_cursor_samples
            or end_sample_index < start_sample_index
        ):
            raise ValueError("PlaybackTimelineBuffer render must advance monotonically")
        else:
            pass
        rendered = np.zeros(end_sample_index - start_sample_index, dtype=np.float32)
        remaining_segments: list[tuple[int, NDArray[np.float32]]] = []
        for segment_offset, segment_samples in self.queued_segments:
            overlap_start = max(start_sample_index, segment_offset)
            overlap_end = min(end_sample_index, segment_offset + len(segment_samples))
            if overlap_end > overlap_start:
                rendered[
                    overlap_start
                    - start_sample_index : overlap_end
                    - start_sample_index
                ] = segment_samples[
                    overlap_start - segment_offset : overlap_end - segment_offset
                ]
            else:
                pass
            if segment_offset + len(segment_samples) > end_sample_index:
                remaining_segments.append((segment_offset, segment_samples))
            else:
                pass
        self.queued_segments = remaining_segments
        self.render_cursor_samples = end_sample_index
        return rendered


def encode_float32_to_pcm16_base64(samples: NDArray[np.float32]) -> str:
    flat_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(flat_samples).all():
        raise ValueError("Input audio samples must be finite")
    else:
        clipped = np.clip(np.round(flat_samples * 32768.0), -32768.0, 32767.0).astype(
            "<i2"
        )
        return base64.b64encode(clipped.tobytes()).decode("ascii")


def decode_pcm16_base64_to_float32(encoded_audio: str) -> NDArray[np.float32]:
    try:
        pcm_bytes = base64.b64decode(encoded_audio, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Invalid base64 audio delta from server") from None
    if len(pcm_bytes) % 2 != 0:
        raise ValueError("Server PCM16 delta has an odd byte length")
    else:
        return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0


def is_retryable_session_rejection(event: dict[str, JsonValue]) -> bool:
    if is_admission_rejection(event):
        return True
    elif event.get("type") != "error":
        return False
    else:
        error_body = event.get("error")
        sglang_meta = event.get("sglang")
        is_fatal = (
            sglang_meta.get("fatal") is True if isinstance(sglang_meta, dict) else False
        )
        if is_fatal:
            return False
        else:
            code = error_body.get("code") if isinstance(error_body, dict) else None
            error_type = (
                error_body.get("type") if isinstance(error_body, dict) else None
            )
            reason = (
                sglang_meta.get("reason") if isinstance(sglang_meta, dict) else None
            )
            return (
                code == "admission_rejected"
                or error_type == "admission_rejected"
                or reason == "admission_rejected"
            )


def build_session_update_payload(
    endpoint_spec: RealtimeEndpointSpec,
) -> dict[str, JsonValue]:
    if endpoint_spec.protocol == "native":
        return {"output_modalities": ["audio"]}
    elif endpoint_spec.protocol == "legacy":
        assert endpoint_spec.turn_detection is not None
        return {
            "modalities": ["text", "audio"],
            "turn_detection": endpoint_spec.turn_detection,
        }
    else:
        assert endpoint_spec.turn_detection is not None
        return {
            "output_modalities": ["audio"],
            "audio": {"input": {"turn_detection": endpoint_spec.turn_detection}},
        }


def validate_accepted_session(
    updated_event: dict[str, JsonValue], endpoint_spec: RealtimeEndpointSpec
) -> None:
    """Verify session.updated echoed the requested detector and compatible PCM sample rates."""
    session_payload = updated_event.get("session")
    if not isinstance(session_payload, dict):
        raise RuntimeError("session.updated lacks a session object")
    else:
        pass
    if endpoint_spec.protocol in ("turn_based", "legacy"):
        assert endpoint_spec.turn_detection is not None
        requested_detector = endpoint_spec.turn_detection.get("type")
        echoed_detector = turn_detection_type(session_payload, endpoint_spec.protocol)
        if echoed_detector != requested_detector:
            raise RuntimeError("session.updated echoed unexpected turn_detection.type")
        else:
            pass
    else:
        pass

    audio_block = session_payload.get("audio")
    if audio_block is not None and not isinstance(audio_block, dict):
        raise RuntimeError("session.updated audio field must be a JSON object")
    else:
        pass
    sglang_extension = session_payload.get("sglang")
    sglang_granted = (
        (sglang_extension.get("granted") or sglang_extension.get("capabilities"))
        if isinstance(sglang_extension, dict)
        else None
    )
    if endpoint_spec.protocol == "native":
        if turn_detection_type(session_payload, "turn_based") is not None or (
            turn_detection_type(session_payload, "legacy") is not None
        ):
            raise RuntimeError(
                "session.updated enabled turn_detection on a native full-duplex session"
            )
        elif isinstance(sglang_granted, dict) and (
            sglang_granted.get("native_full_duplex") is False
            or sglang_granted.get("interaction") not in (None, "native")
        ):
            raise RuntimeError(
                "session.updated granted non-native interaction for a native session"
            )
        else:
            pass
    elif isinstance(sglang_granted, dict) and (
        sglang_granted.get("native_full_duplex") is True
        or sglang_granted.get("interaction") == "native"
    ):
        raise RuntimeError(
            f"session.updated granted native interaction for a {endpoint_spec.protocol} session"
        )
    else:
        pass
    for direction_key, granted_key, expected_rate_hz in (
        ("input", "input_audio_format", endpoint_spec.input_sample_rate_hz),
        ("output", "output_audio_format", endpoint_spec.output_sample_rate_hz),
    ):
        direction_block = (
            audio_block.get(direction_key) if isinstance(audio_block, dict) else None
        )
        format_block: JsonValue = (
            direction_block.get("format") if isinstance(direction_block, dict) else None
        )
        if format_block is None and isinstance(sglang_granted, dict):
            format_block = sglang_granted.get(granted_key)
        else:
            pass
        if endpoint_spec.protocol in ("native", "turn_based"):
            if not isinstance(format_block, dict):
                raise RuntimeError(
                    f"session.updated lacks {direction_key} audio format declaration"
                )
            else:
                format_type = format_block.get("type")
                format_rate = format_block.get("rate")
                if format_type != "audio/pcm":
                    raise RuntimeError(
                        f"session.updated {direction_key} audio format is not audio/pcm"
                    )
                elif type(format_rate) is not int or format_rate != expected_rate_hz:
                    raise RuntimeError(
                        f"session.updated {direction_key} sample rate differs from "
                        f"configured {expected_rate_hz} Hz"
                    )
                else:
                    pass
        elif isinstance(format_block, dict):
            format_type = format_block.get("type")
            format_rate = format_block.get("rate")
            if format_type is not None and format_type != "audio/pcm":
                raise RuntimeError(
                    f"session.updated {direction_key} audio format is not audio/pcm"
                )
            elif format_rate is not None and (
                type(format_rate) is not int or format_rate != expected_rate_hz
            ):
                raise RuntimeError(
                    f"session.updated {direction_key} sample rate differs from "
                    f"configured {expected_rate_hz} Hz"
                )
            else:
                pass
        else:
            pass


def compute_effective_scenario_timeout_seconds(
    *,
    configured_timeout_seconds: float,
    maximum_schedule_duration_seconds: float,
) -> float:
    """Ensure long Duplex-MPE clips (e.g., duplex_mpe_1808 at 292.4s) never expire mid-exchange."""
    if (
        not math.isfinite(configured_timeout_seconds)
        or configured_timeout_seconds <= 0.0
    ):
        raise ValueError("session_timeout_seconds must be finite and positive")
    else:
        required_budget_seconds = (
            maximum_schedule_duration_seconds + SCENARIO_DRAIN_MARGIN_SECONDS
        )
        return max(configured_timeout_seconds, required_budget_seconds)


def prepare_scenario_schedule_sync(
    scenario: DuplexMpeScenarioRecord,
    *,
    dataset_root: Path,
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
    vad_config: SileroActivityConfig | None,
    vad_model: SileroVadModel | None,
    chunk_milliseconds: float,
) -> tuple[
    Generator[NDArray[np.float32], NDArray[np.float32], ScenarioScheduleResult],
    NDArray[np.float32] | None,
    ScenarioScheduleResult | None,
    float,
]:
    preamble_path = resolve_dataset_audio_path(
        dataset_root, DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH
    )
    preamble_samples = load_mono_waveform_at_rate(
        preamble_path, target_sample_rate_hz=input_sample_rate_hz
    )
    turn_inputs = [
        ScenarioTurnInput(
            turn_index=turn.turn_index,
            event_type=turn.event_type,
            speaker=turn.speaker,
            waveform_samples=load_mono_waveform_at_rate(
                resolve_dataset_audio_path(dataset_root, turn.audio_path),
                target_sample_rate_hz=input_sample_rate_hz,
            ),
        )
        for turn in scenario.turns
    ]
    maximum_schedule_duration_seconds = 0.0
    if len(preamble_samples) > 0:
        maximum_schedule_duration_seconds += (
            len(preamble_samples) / input_sample_rate_hz + ORDINARY_GAP_SECONDS
        )
    else:
        pass
    for turn_input in turn_inputs:
        assert turn_input.waveform_samples is not None
        maximum_schedule_duration_seconds += (
            len(turn_input.waveform_samples) / input_sample_rate_hz
        )
        if turn_input.event_type == "T":
            maximum_schedule_duration_seconds += TARGET_MAX_RESPONSE_SECONDS
        elif turn_input.event_type == "N4_Q":
            maximum_schedule_duration_seconds += N4_PAUSE_SECONDS
        else:
            maximum_schedule_duration_seconds += ORDINARY_GAP_SECONDS

    schedule = iter_scenario_schedule(
        turn_inputs,
        input_sample_rate_hz=input_sample_rate_hz,
        output_sample_rate_hz=output_sample_rate_hz,
        vad_config=vad_config,
        vad_model=vad_model,
        chunk_milliseconds=chunk_milliseconds,
        ordinary_gap_seconds=ORDINARY_GAP_SECONDS,
        n4_pause_seconds=N4_PAUSE_SECONDS,
        onset_deadline_seconds=TARGET_ONSET_DEADLINE_SECONDS,
        maximum_response_seconds=TARGET_MAX_RESPONSE_SECONDS,
        preamble_samples=preamble_samples,
        collect_audio=True,
    )
    try:
        first_chunk = next(schedule)
        return schedule, first_chunk, None, maximum_schedule_duration_seconds
    except StopIteration as finished_empty:
        return schedule, None, finished_empty.value, maximum_schedule_duration_seconds


def advance_schedule_step_sync(
    schedule: Generator[
        NDArray[np.float32], NDArray[np.float32], ScenarioScheduleResult
    ],
    rendered_chunk: NDArray[np.float32],
) -> tuple[NDArray[np.float32] | None, ScenarioScheduleResult | None]:
    try:
        next_chunk = schedule.send(rendered_chunk)
        return next_chunk, None
    except StopIteration as finished_schedule:
        return None, finished_schedule.value


def persist_captured_waveform_sync(
    output_audio_path: Path,
    waveform_samples: NDArray[np.float32],
    sample_rate_hz: int,
) -> str:
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_wav = output_audio_path.with_name(f"{output_audio_path.name}.tmp")
    soundfile.write(
        str(temporary_wav),
        waveform_samples,
        sample_rate_hz,
        format="WAV",
        subtype="FLOAT",
    )
    temporary_wav.replace(output_audio_path)
    return compute_file_sha256(output_audio_path)


def extract_numeric_receipt_field(
    event: dict[str, JsonValue], field_name: str, event_type: str
) -> float:
    raw_value = event.get(field_name)
    if not isinstance(raw_value, (int, float)) or type(raw_value) is bool:
        raise ProtocolSessionError(f"{event_type} missing numeric {field_name} field")
    elif not math.isfinite(float(raw_value)):
        raise ProtocolSessionError(
            f"{event_type} carried non-finite {field_name} field"
        )
    else:
        return float(raw_value)


async def run_realtime_scenario(
    scenario: DuplexMpeScenarioRecord,
    *,
    dataset_root: Path,
    endpoint_spec: RealtimeEndpointSpec,
    output_audio_path: Path,
    vad_config: SileroActivityConfig | None = None,
    vad_model: SileroVadModel | None = None,
    chunk_milliseconds: float = PACKET_DURATION_MILLISECONDS,
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
    seed_mapping: dict[str, int] | None = None,
    realtime_pacing: bool = True,
    max_pacing_lag_seconds: float | None = DEFAULT_MAX_PACING_LAG_SECONDS,
    trust_env: bool | None = None,
    clock_fn: MonotonicClock | None = None,
) -> dict[str, JsonValue]:
    """Stream one Duplex-MPE scenario over /v1/realtime and return its mpeval.unified.v3 report."""
    if max_pacing_lag_seconds is not None and max_pacing_lag_seconds <= 0.0:
        raise ValueError("max_pacing_lag_seconds must be positive when set")
    else:
        pass
    monotonic_clock: MonotonicClock = (
        clock_fn if clock_fn is not None else time.perf_counter
    )
    scenario_seed = resolve_scenario_seed(scenario.scenario_id, seed_mapping)
    # note (machengqian): Offload waveform loading, resampling, and cold Silero VAD initialization off the event loop so concurrent sessions never stall active pacing.
    (
        schedule,
        initial_input_chunk,
        immediate_result,
        maximum_schedule_duration_seconds,
    ) = await asyncio.to_thread(
        prepare_scenario_schedule_sync,
        scenario,
        dataset_root=dataset_root,
        input_sample_rate_hz=endpoint_spec.input_sample_rate_hz,
        output_sample_rate_hz=endpoint_spec.output_sample_rate_hz,
        vad_config=vad_config,
        vad_model=vad_model,
        chunk_milliseconds=chunk_milliseconds,
    )
    effective_timeout_seconds = compute_effective_scenario_timeout_seconds(
        configured_timeout_seconds=session_timeout_seconds,
        maximum_schedule_duration_seconds=maximum_schedule_duration_seconds,
    )

    playback_buffer = PlaybackTimelineBuffer(endpoint_spec.output_sample_rate_hz)
    audio_delta_event_type = (
        "response.audio.delta"
        if endpoint_spec.protocol == "legacy"
        else "response.output_audio.delta"
    )
    interrupt_on_speech_started = (
        endpoint_spec.protocol == "legacy"
        and endpoint_spec.turn_detection is not None
        and endpoint_spec.turn_detection.get("interrupt_response") is not False
    )

    effective_trust_env = endpoint_spec.trust_env if trust_env is None else trust_env

    async def execute_exchange() -> (
        tuple[ScenarioScheduleResult, float, float, NDArray[np.float32]]
    ):
        admission_state = AdmissionState()
        connection = await open_websocket_with_admission_state(
            endpoint_spec.websocket_url,
            admission_state=admission_state,
            trust_env=effective_trust_env,
        )
        async with connection as websocket:
            event_waiters: dict[str, asyncio.Event] = {}
            latest_events: dict[str, dict[str, JsonValue]] = {}
            receiver_failure: list[BaseException] = []
            admission_rejected_event = asyncio.Event()
            aborted_event = asyncio.Event()
            receiver_done_event = asyncio.Event()
            awaiting_legacy_terminal_echo = False
            is_closing_expected = False
            stream_epoch_seconds: float | None = None
            total_input_samples = 0
            expected_accepted_end_milliseconds_by_sequence: dict[int, float] = {}
            next_expected_accepted_sequence = 0
            eos_requested = False
            frozen_final_input_samples: int | None = None
            input_ended_receipt_seen = False
            input_drained_receipt_seen = False

            async def send_client_event(
                event_type: str,
                *,
                session_payload: dict[str, JsonValue] | None = None,
                audio_base64: str | None = None,
                sglang_metadata: dict[str, JsonValue] | None = None,
            ) -> None:
                envelope: dict[str, JsonValue] = {
                    "type": event_type,
                    "event_id": uuid.uuid4().hex,
                }
                if session_payload is not None:
                    envelope["session"] = session_payload
                else:
                    pass
                if audio_base64 is not None:
                    envelope["audio"] = audio_base64
                else:
                    pass
                if sglang_metadata is not None:
                    envelope["sglang"] = sglang_metadata
                else:
                    pass
                try:
                    await websocket.send(json.dumps(envelope, allow_nan=False))
                except websockets.ConnectionClosed:
                    raise RuntimeError(
                        f"WebSocket connection closed while sending {event_type}"
                    ) from None

            async def wait_for_server_event(
                event_type: str, wait_timeout_seconds: float | None = None
            ) -> None:
                if receiver_failure:
                    raise RuntimeError(
                        f"Realtime session failed before awaiting {event_type}: "
                        f"{summarize_transport_exception(receiver_failure[0])}"
                    ) from None
                else:
                    pass
                target_waiter = event_waiters.setdefault(event_type, asyncio.Event())
                if target_waiter.is_set():
                    return
                elif receiver_done_event.is_set():
                    raise RuntimeError(
                        f"WebSocket receiver exited before {event_type} arrived"
                    )
                else:
                    pass
                event_task = asyncio.create_task(target_waiter.wait())
                done_task = asyncio.create_task(receiver_done_event.wait())
                try:
                    completed, _ = await asyncio.wait(
                        {event_task, done_task},
                        timeout=wait_timeout_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    event_task.cancel()
                    done_task.cancel()
                    await asyncio.gather(event_task, done_task, return_exceptions=True)
                if receiver_failure:
                    raise RuntimeError(
                        f"Realtime session failed while awaiting {event_type}: "
                        f"{summarize_transport_exception(receiver_failure[0])}"
                    ) from None
                elif not completed:
                    raise TimeoutError(
                        f"Timed out after {wait_timeout_seconds}s awaiting {event_type}"
                    )
                elif not target_waiter.is_set():
                    raise RuntimeError(
                        f"WebSocket connection closed before {event_type} arrived"
                    )
                else:
                    pass

            async def receive_server_frames() -> None:
                nonlocal is_closing_expected, next_expected_accepted_sequence, input_ended_receipt_seen, input_drained_receipt_seen
                try:
                    async for raw_frame in websocket:
                        received_perf_seconds = monotonic_clock()
                        if not realtime_pacing:
                            arrival_seconds = (
                                total_input_samples / endpoint_spec.input_sample_rate_hz
                            )
                        elif stream_epoch_seconds is None:
                            arrival_seconds = 0.0
                        else:
                            arrival_seconds = max(
                                0.0, received_perf_seconds - stream_epoch_seconds
                            )
                        frame_text = (
                            raw_frame
                            if isinstance(raw_frame, str)
                            else raw_frame.decode("utf-8")
                        )
                        try:
                            parsed_event = SERVER_EVENT_DECODER.decode(frame_text)
                        except ValueError:
                            raise ProtocolSessionError(
                                "Malformed JSON frame on /v1/realtime"
                            ) from None
                        if not isinstance(parsed_event, dict) or not isinstance(
                            parsed_event.get("type"), str
                        ):
                            raise ProtocolSessionError(
                                "Malformed server frame on /v1/realtime"
                            )
                        else:
                            event_type = str(parsed_event["type"])
                        latest_events[event_type] = parsed_event
                        if event_type == "session.closed" or (
                            awaiting_legacy_terminal_echo
                            and event_type == "session.updated"
                        ):
                            is_closing_expected = True
                        else:
                            pass
                        if is_retryable_session_rejection(parsed_event):
                            admission_state.denials += 1
                            admission_rejected_event.set()
                        elif event_type == "error":
                            aborted_event.set()
                            raise ProtocolSessionError(
                                format_server_error_event(parsed_event)
                            )
                        elif event_type == audio_delta_event_type:
                            raw_delta = parsed_event.get("delta")
                            if not isinstance(raw_delta, str):
                                raise ProtocolSessionError(
                                    f"Malformed {event_type} payload: delta must be a string"
                                )
                            elif raw_delta:
                                playback_buffer.append_audio(
                                    arrival_seconds,
                                    decode_pcm16_base64_to_float32(raw_delta),
                                )
                            else:
                                pass
                        elif event_type == "sglang.input_audio.accepted":
                            raw_sequence = parsed_event.get("seq")
                            if input_drained_receipt_seen:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.accepted arrived after sglang.input_audio.drained"
                                )
                            elif type(raw_sequence) is not int:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.accepted missing integer seq"
                                )
                            elif raw_sequence != next_expected_accepted_sequence:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.accepted sequence mismatch"
                                )
                            elif (
                                raw_sequence
                                not in expected_accepted_end_milliseconds_by_sequence
                            ):
                                raise ProtocolSessionError(
                                    "sglang.input_audio.accepted exceeded sent sequence"
                                )
                            else:
                                expected_end_milliseconds = (
                                    expected_accepted_end_milliseconds_by_sequence.pop(
                                        raw_sequence
                                    )
                                )
                                accepted_end_milliseconds = (
                                    extract_numeric_receipt_field(
                                        parsed_event,
                                        "accepted_end_ms",
                                        "sglang.input_audio.accepted",
                                    )
                                )
                                if not math.isclose(
                                    accepted_end_milliseconds,
                                    expected_end_milliseconds,
                                    rel_tol=0.0,
                                    abs_tol=RECEIPT_DURATION_TOLERANCE_MILLISECONDS,
                                ):
                                    raise ProtocolSessionError(
                                        "sglang.input_audio.accepted duration mismatch"
                                    )
                                else:
                                    next_expected_accepted_sequence += 1
                        elif event_type == "sglang.input_audio.ended":
                            if not eos_requested or frozen_final_input_samples is None:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.ended arrived before sglang.input_audio.end"
                                )
                            elif input_ended_receipt_seen:
                                raise ProtocolSessionError(
                                    "Duplicate sglang.input_audio.ended receipt"
                                )
                            else:
                                expected_total_milliseconds = (
                                    frozen_final_input_samples
                                    * 1000.0
                                    / endpoint_spec.input_sample_rate_hz
                                )
                                ended_end_milliseconds = extract_numeric_receipt_field(
                                    parsed_event,
                                    "accepted_end_ms",
                                    "sglang.input_audio.ended",
                                )
                                if not math.isclose(
                                    ended_end_milliseconds,
                                    expected_total_milliseconds,
                                    rel_tol=0.0,
                                    abs_tol=RECEIPT_DURATION_TOLERANCE_MILLISECONDS,
                                ):
                                    raise ProtocolSessionError(
                                        "sglang.input_audio.ended duration mismatch"
                                    )
                                else:
                                    input_ended_receipt_seen = True
                        elif event_type == "sglang.input_audio.drained":
                            if not eos_requested or frozen_final_input_samples is None:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.drained arrived before sglang.input_audio.end"
                                )
                            elif input_drained_receipt_seen:
                                raise ProtocolSessionError(
                                    "Duplicate sglang.input_audio.drained receipt"
                                )
                            elif not input_ended_receipt_seen:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.drained arrived before sglang.input_audio.ended"
                                )
                            elif expected_accepted_end_milliseconds_by_sequence:
                                raise ProtocolSessionError(
                                    "sglang.input_audio.drained arrived with unacknowledged input packets"
                                )
                            else:
                                expected_total_milliseconds = (
                                    frozen_final_input_samples
                                    * 1000.0
                                    / endpoint_spec.input_sample_rate_hz
                                )
                                drained_accepted_milliseconds = (
                                    extract_numeric_receipt_field(
                                        parsed_event,
                                        "accepted_end_ms",
                                        "sglang.input_audio.drained",
                                    )
                                )
                                drained_consumed_milliseconds = (
                                    extract_numeric_receipt_field(
                                        parsed_event,
                                        "consumed_ms",
                                        "sglang.input_audio.drained",
                                    )
                                )
                                drained_discarded_milliseconds = (
                                    extract_numeric_receipt_field(
                                        parsed_event,
                                        "discarded_ms",
                                        "sglang.input_audio.drained",
                                    )
                                )
                                if not math.isclose(
                                    drained_accepted_milliseconds,
                                    expected_total_milliseconds,
                                    rel_tol=0.0,
                                    abs_tol=RECEIPT_DURATION_TOLERANCE_MILLISECONDS,
                                ):
                                    raise ProtocolSessionError(
                                        "sglang.input_audio.drained accepted_end_ms mismatch"
                                    )
                                elif not math.isclose(
                                    drained_consumed_milliseconds,
                                    expected_total_milliseconds,
                                    rel_tol=0.0,
                                    abs_tol=RECEIPT_DURATION_TOLERANCE_MILLISECONDS,
                                ):
                                    raise ProtocolSessionError(
                                        "sglang.input_audio.drained consumed_ms mismatch"
                                    )
                                elif not math.isclose(
                                    drained_discarded_milliseconds,
                                    0.0,
                                    rel_tol=0.0,
                                    abs_tol=RECEIPT_DURATION_TOLERANCE_MILLISECONDS,
                                ):
                                    raise ProtocolSessionError(
                                        "sglang.input_audio.drained reported discarded input audio"
                                    )
                                else:
                                    input_drained_receipt_seen = True
                        elif event_type == "response.done":
                            response_payload = parsed_event.get("response")
                            if not isinstance(response_payload, dict):
                                raise ProtocolSessionError(
                                    "Malformed response.done payload: response must be an object"
                                )
                            elif response_payload.get("status") == "failed":
                                raise ProtocolSessionError(
                                    "Server reported failed response.done"
                                )
                            elif (
                                endpoint_spec.protocol in ("turn_based", "legacy")
                                and response_payload.get("status") == "cancelled"
                            ):
                                playback_buffer.cancel_unplayed(arrival_seconds)
                            else:
                                pass
                        elif endpoint_spec.protocol == "legacy" and (
                            event_type == "output_audio_buffer.cleared"
                            or (
                                interrupt_on_speech_started
                                and event_type == "input_audio_buffer.speech_started"
                            )
                        ):
                            playback_buffer.cancel_unplayed(arrival_seconds)
                        else:
                            pass
                        event_waiters.setdefault(event_type, asyncio.Event()).set()
                    if not is_closing_expected:
                        aborted_event.set()
                        receiver_failure.append(
                            ProtocolSessionError(
                                "WebSocket connection closed before session completion"
                            )
                        )
                    else:
                        pass
                except asyncio.CancelledError:
                    raise
                except websockets.ConnectionClosed as exc:
                    if not is_closing_expected:
                        aborted_event.set()
                        receiver_failure.append(
                            ProtocolSessionError(
                                f"WebSocket connection closed prematurely: "
                                f"{summarize_transport_exception(exc)}"
                            )
                        )
                    else:
                        pass
                except BaseException as exc:
                    aborted_event.set()
                    receiver_failure.append(exc)
                finally:
                    receiver_done_event.set()

            receiver_task = asyncio.create_task(receive_server_frames())
            try:
                await wait_for_server_event(
                    "session.created", HANDSHAKE_TIMEOUT_SECONDS
                )
                configure_payload = build_session_update_payload(endpoint_spec)

                async def send_session_update() -> None:
                    event_waiters.setdefault("session.updated", asyncio.Event()).clear()
                    await send_client_event(
                        "session.update", session_payload=configure_payload
                    )

                async def settle_session_updated() -> bool:
                    updated_waiter = asyncio.create_task(
                        event_waiters.setdefault(
                            "session.updated", asyncio.Event()
                        ).wait()
                    )
                    rejected_waiter = asyncio.create_task(
                        admission_rejected_event.wait()
                    )
                    done_waiter = asyncio.create_task(receiver_done_event.wait())
                    try:
                        completed, _ = await asyncio.wait(
                            {updated_waiter, rejected_waiter, done_waiter},
                            timeout=HANDSHAKE_TIMEOUT_SECONDS,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        updated_waiter.cancel()
                        rejected_waiter.cancel()
                        done_waiter.cancel()
                        await asyncio.gather(
                            updated_waiter,
                            rejected_waiter,
                            done_waiter,
                            return_exceptions=True,
                        )
                    if receiver_failure:
                        raise RuntimeError(
                            f"Realtime session aborted during session.update: "
                            f"{summarize_transport_exception(receiver_failure[0])}"
                        ) from None
                    elif event_waiters["session.updated"].is_set():
                        return True
                    elif admission_rejected_event.is_set():
                        return False
                    elif not completed:
                        raise TimeoutError(
                            f"Timed out after {HANDSHAKE_TIMEOUT_SECONDS}s awaiting session.updated"
                        )
                    else:
                        raise RuntimeError(
                            "WebSocket connection closed before session.updated arrived"
                        )

                admitted = await negotiate_session_admission(
                    send_session_update=send_session_update,
                    settle_session_updated=settle_session_updated,
                    rejected=admission_rejected_event,
                    aborted=aborted_event,
                    admission_state=admission_state,
                    max_retries=ADMISSION_RETRIES,
                    backoff_seconds=ADMISSION_BACKOFF_S,
                )
                if receiver_failure:
                    raise RuntimeError(
                        f"Realtime session aborted during session.update: "
                        f"{summarize_transport_exception(receiver_failure[0])}"
                    ) from None
                elif not admitted:
                    raise RuntimeError(
                        f"Session admission rejected after {admission_state.denials} "
                        f"combined admission denials (budget {ADMISSION_RETRIES} retries)"
                    )
                else:
                    validate_accepted_session(
                        latest_events["session.updated"], endpoint_spec
                    )

                rendered_output_samples = 0
                packet_sequence = 0
                max_observed_pacing_lag_seconds = 0.0

                def record_and_validate_pacing_lag(
                    nominal_deadline_seconds: float,
                ) -> None:
                    nonlocal max_observed_pacing_lag_seconds
                    observed_lag_seconds = max(
                        0.0, monotonic_clock() - nominal_deadline_seconds
                    )
                    max_observed_pacing_lag_seconds = max(
                        max_observed_pacing_lag_seconds, observed_lag_seconds
                    )
                    if (
                        max_pacing_lag_seconds is not None
                        and observed_lag_seconds > max_pacing_lag_seconds
                    ):
                        raise RuntimeError(
                            f"Realtime input pacing lagged by {observed_lag_seconds:.3f}s "
                            f"(limit {max_pacing_lag_seconds:.3f}s)"
                        )
                    else:
                        pass

                if immediate_result is not None:
                    schedule_result = immediate_result
                else:
                    assert initial_input_chunk is not None
                    next_input_chunk: NDArray[np.float32] | None = initial_input_chunk
                    stream_epoch_seconds = monotonic_clock()
                    completed_schedule_result: ScenarioScheduleResult | None = None
                    while next_input_chunk is not None:
                        if receiver_failure:
                            raise RuntimeError(
                                f"Realtime receiver failed mid-scenario: "
                                f"{summarize_transport_exception(receiver_failure[0])}"
                            ) from None
                        else:
                            pass
                        if realtime_pacing and packet_sequence > 0:
                            record_and_validate_pacing_lag(
                                stream_epoch_seconds
                                + total_input_samples
                                / endpoint_spec.input_sample_rate_hz
                            )
                        else:
                            pass
                        chunk_start_milliseconds = (
                            total_input_samples
                            * 1000.0
                            / endpoint_spec.input_sample_rate_hz
                        )
                        next_total_samples = total_input_samples + len(next_input_chunk)
                        expected_accepted_end_milliseconds_by_sequence[
                            packet_sequence
                        ] = (
                            next_total_samples
                            * 1000.0
                            / endpoint_spec.input_sample_rate_hz
                        )
                        total_input_samples = next_total_samples
                        await send_client_event(
                            "input_audio_buffer.append",
                            audio_base64=encode_float32_to_pcm16_base64(
                                next_input_chunk
                            ),
                            sglang_metadata={
                                "seq": packet_sequence,
                                "t_start_ms": chunk_start_milliseconds,
                            },
                        )
                        packet_sequence += 1
                        target_perf_deadline = (
                            stream_epoch_seconds
                            + total_input_samples / endpoint_spec.input_sample_rate_hz
                        )
                        if realtime_pacing:
                            await asyncio.sleep(
                                max(0.0, target_perf_deadline - monotonic_clock())
                            )
                            record_and_validate_pacing_lag(target_perf_deadline)
                        else:
                            await asyncio.sleep(0)

                        if receiver_failure:
                            raise RuntimeError(
                                f"Realtime receiver failed while awaiting output: "
                                f"{summarize_transport_exception(receiver_failure[0])}"
                            ) from None
                        else:
                            pass
                        target_output_samples = int(
                            round(
                                total_input_samples
                                * endpoint_spec.output_sample_rate_hz
                                / endpoint_spec.input_sample_rate_hz
                            )
                        )
                        rendered_chunk = playback_buffer.render_interval(
                            rendered_output_samples, target_output_samples
                        )
                        rendered_output_samples = target_output_samples
                        if realtime_pacing:
                            (
                                next_input_chunk,
                                completed_schedule_result,
                            ) = await asyncio.to_thread(
                                advance_schedule_step_sync, schedule, rendered_chunk
                            )
                        else:
                            next_input_chunk, completed_schedule_result = (
                                advance_schedule_step_sync(schedule, rendered_chunk)
                            )
                    if realtime_pacing:
                        record_and_validate_pacing_lag(
                            stream_epoch_seconds
                            + total_input_samples / endpoint_spec.input_sample_rate_hz
                        )
                    else:
                        pass
                    assert completed_schedule_result is not None
                    schedule_result = completed_schedule_result

                elapsed_wall_seconds = (
                    0.0
                    if stream_epoch_seconds is None
                    else monotonic_clock() - stream_epoch_seconds
                )
                if endpoint_spec.protocol == "legacy":
                    if realtime_pacing:
                        await asyncio.sleep(LEGACY_TAIL_WAIT_SECONDS)
                    else:
                        pass
                    event_waiters["session.updated"].clear()
                    awaiting_legacy_terminal_echo = True
                    await send_client_event("session.update", session_payload={})
                    await wait_for_server_event("session.updated")
                else:
                    event_waiters.setdefault(
                        "sglang.input_audio.drained", asyncio.Event()
                    ).clear()
                    frozen_final_input_samples = total_input_samples
                    eos_requested = True
                    await send_client_event("sglang.input_audio.end")
                    await wait_for_server_event("sglang.input_audio.drained")
                    closed_waiter = event_waiters.setdefault(
                        "session.closed", asyncio.Event()
                    )
                    if not closed_waiter.is_set():
                        await send_client_event("session.close")
                        await wait_for_server_event("session.closed")
                    else:
                        pass

                if playback_buffer.tail_samples > rendered_output_samples:
                    drain_tail_waveform = playback_buffer.render_interval(
                        rendered_output_samples, playback_buffer.tail_samples
                    )
                else:
                    drain_tail_waveform = np.zeros(0, dtype=np.float32)
                return (
                    schedule_result,
                    elapsed_wall_seconds,
                    max_observed_pacing_lag_seconds,
                    drain_tail_waveform,
                )
            finally:
                receiver_task.cancel()
                await asyncio.gather(receiver_task, return_exceptions=True)

    (
        schedule_result,
        elapsed_wall_seconds,
        max_observed_pacing_lag_seconds,
        drain_tail_waveform,
    ) = await asyncio.wait_for(execute_exchange(), timeout=effective_timeout_seconds)
    assert schedule_result.assistant_waveform is not None
    full_captured_waveform = (
        np.concatenate([schedule_result.assistant_waveform, drain_tail_waveform])
        if len(drain_tail_waveform) > 0
        else schedule_result.assistant_waveform
    )

    audio_sha256 = await asyncio.to_thread(
        persist_captured_waveform_sync,
        output_audio_path,
        full_captured_waveform,
        endpoint_spec.output_sample_rate_hz,
    )

    report = schedule_result.to_report_dict()
    report.update(
        {
            "scenario_id": scenario.scenario_id,
            "pair_id": scenario.pair_id,
            "addressing": scenario.addressing,
            "run_seed": scenario_seed,
            "seed_policy": SEED_POLICY_METADATA_ONLY,
            "server_seed_applied": False,
            "code_fingerprint": compute_protocol_fingerprint(),
            "assistant_wav_sr": endpoint_spec.output_sample_rate_hz,
            "evaluation_clock": (
                "realtime_wall_clock"
                if realtime_pacing
                else "audio_samples_excluding_compute_time"
            ),
            "realtime_protocol": endpoint_spec.protocol,
            "elapsed_wall_s": round(elapsed_wall_seconds, 4),
            "max_pacing_lag_ms": round(max_observed_pacing_lag_seconds * 1000.0, 3),
            "effective_session_timeout_s": round(effective_timeout_seconds, 4),
            "cancelled_unplayed_samples": playback_buffer.cancelled_unplayed_samples,
            "drain_tail_samples": int(len(drain_tail_waveform)),
            "assistant_audio_sha256": audio_sha256,
        }
    )
    return report
