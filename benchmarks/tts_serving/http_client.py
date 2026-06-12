# SPDX-License-Identifier: Apache-2.0
"""HTTP and SSE clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import binascii
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from benchmarks.tts_serving.audio_validation import (
    MIN_PCM_AUDIO_BYTES,
    PCM_CONTENT_TYPES,
    validate_audio_response,
    validate_pcm_chunk,
)
from benchmarks.tts_serving.batch_client import handle_batch_success
from benchmarks.tts_serving.http_contracts import (
    MAX_HTTP_RESPONSE_BYTES,
    ResponseBodyTooLarge,
    _classify_http_failure,
    _is_unsupported_http_status,
    _mark_protocol_error,
    _mark_success,
    _mark_unexpected_success,
    _mark_unsupported_contract,
    _result_has_terminal_error,
    read_response_body,
)
from benchmarks.tts_serving.metrics import (
    PCM_SAMPLE_RATE,
    SSE_DONE_MARKER,
    ScenarioResult,
    classify_http_status,
    finish_timing,
    parse_sse_audio_event,
)
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec
from benchmarks.tts_serving.urls import api_url
from benchmarks.tts_serving.voice_client import (
    handle_voice_success,
    request_body,
    request_size,
    run_voice_cache_pressure_sequence,
    run_voice_lifecycle,
    run_voice_overwrite,
    run_voice_speaker_cap_sequence,
    run_voice_upload,
    run_voice_upload_delete_race,
    run_voice_upload_metadata_sequence,
)


@dataclass
class SseAudioState:
    total_bytes: int = 0
    duration_s: float = 0.0
    has_signal: bool = False


async def run_http_scenario(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
) -> ScenarioResult:
    result = ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        capability_key=scenario.capability_key,
        expected_success=scenario.expect_success,
        response_format=_scenario_response_format(scenario),
        batch_size=scenario.planned_metadata.get("batch_size"),
    )
    url = api_url(spec.base_url, scenario.path)
    start = time.perf_counter()
    try:
        if scenario.method == "VOICE_LIFECYCLE":
            await run_voice_lifecycle(
                session,
                spec,
                scenario,
                result,
            )
        elif scenario.method == "VOICE_OVERWRITE":
            await run_voice_overwrite(session, spec, scenario, result)
        elif scenario.method == "VOICE_UPLOAD_DELETE_RACE":
            await run_voice_upload_delete_race(session, spec, scenario, result)
        elif scenario.method == "VOICE_SPEAKER_CAP_SEQUENCE":
            await run_voice_speaker_cap_sequence(session, spec, scenario, result)
        elif scenario.method == "VOICE_UPLOAD_METADATA_SEQUENCE":
            await run_voice_upload_metadata_sequence(session, spec, scenario, result)
        elif scenario.method == "VOICE_CACHE_PRESSURE_SEQUENCE":
            await run_voice_cache_pressure_sequence(session, spec, scenario, result)
        elif scenario.capability_key == "voices.upload" and scenario.expect_success:
            await run_voice_upload(session, spec, scenario, result)
        elif scenario.method == "GET":
            async with session.get(url) as response:
                await _handle_probe_response(response, result, scenario)
        elif scenario.method == "DELETE":
            async with session.delete(url) as response:
                await _handle_binary_response(response, result, start, scenario)
        else:
            body = request_body(scenario)
            kwargs = (
                {"data": body}
                if scenario.body_type == "multipart"
                else {"json": scenario.payload}
            )
            result.request_bytes = request_size(scenario)
            async with session.post(url, **kwargs) as response:
                if scenario.endpoint == "speech_sse":
                    await _handle_sse_response(
                        response,
                        result,
                        start,
                        scenario,
                    )
                else:
                    await _handle_binary_response(response, result, start, scenario)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "transport_error"
        result.error = str(exc)
    except Exception as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "client_error"
        result.error = f"HTTP benchmark scenario failed before classification: {exc}"
    finally:
        finish_timing(result, start)
    return result


async def _handle_probe_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    try:
        body, body_text = await _response_body_and_text(response)
    except ResponseBodyTooLarge as exc:
        _mark_response_body_too_large(result, exc)
        return
    result.response_bytes = len(body)
    if _is_unsupported_http_status(response.status, scenario):
        _mark_unsupported_contract(
            result,
            scenario,
            body=body_text,
        )
        return
    if 200 <= response.status < 300:
        if scenario.endpoint == "voices":
            handle_voice_success(body, result, scenario)
            return
        _mark_success(result, capability="pass")
        return
    _classify_http_failure(response.status, body_text, result, scenario)


async def _handle_binary_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    start: float,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    try:
        body = await read_response_body(response)
    except ResponseBodyTooLarge as exc:
        finish_timing(result, start)
        _mark_response_body_too_large(result, exc)
        return
    finish_timing(result, start)
    result.response_bytes = len(body)
    if _is_unsupported_http_status(response.status, scenario):
        _mark_unsupported_contract(
            result,
            scenario,
            body=body.decode("utf-8", errors="replace"),
        )
        return
    if 200 <= response.status < 300:
        if not scenario.expect_success:
            _mark_unexpected_success(result, scenario)
            return
        if scenario.endpoint == "batch":
            await asyncio.to_thread(handle_batch_success, body, result, scenario)
            return
        if scenario.endpoint == "voices":
            handle_voice_success(body, result, scenario)
            return
        response_format = str(scenario.payload.get("response_format", ""))
        validation = await asyncio.to_thread(
            validate_audio_response,
            body,
            response_format=response_format,
            content_type=response.headers.get("Content-Type"),
        )
        if not validation.ok:
            _mark_protocol_error(
                result,
                status="invalid_audio_response",
                error=(
                    "speech endpoint returned 2xx without the requested audio "
                    f"contract (format={response_format!r}, "
                    f"content-type={response.headers.get('Content-Type')!r}, "
                    f"bytes={len(body)}, validation_error={validation.error})"
                ),
            )
            return
        result.audio_bytes = len(body)
        result.audio_duration_s = validation.duration_s
        _mark_success(result)
        return
    _classify_http_failure(
        response.status, body.decode("utf-8", errors="replace"), result, scenario
    )


async def _handle_sse_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    start: float,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    if response.status != 200:
        try:
            body, body_text = await _response_body_and_text(response)
        except ResponseBodyTooLarge as exc:
            _mark_response_body_too_large(result, exc)
            return
        result.response_bytes = len(body)
        if _is_unsupported_http_status(response.status, scenario):
            _mark_unsupported_contract(result, scenario, body=body_text)
            return
        _classify_http_failure(response.status, body_text, result, scenario)
        return
    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "text/event-stream" not in content_type:
        try:
            body = await read_response_body(response)
        except ResponseBodyTooLarge as exc:
            _mark_response_body_too_large(result, exc)
            return
        result.response_bytes = len(body)
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE speech endpoint returned 2xx without text/event-stream "
                f"content-type: {response.headers.get('Content-Type')!r}"
            ),
        )
        return

    buffer = bytearray()
    audio_state = SseAudioState()
    chunk_times: list[float] = []
    streamed_response_bytes = 0
    saw_done = False
    async for chunk in response.content.iter_any():
        streamed_response_bytes += len(chunk)
        if streamed_response_bytes > MAX_HTTP_RESPONSE_BYTES:
            _mark_response_body_too_large(
                result,
                ResponseBodyTooLarge(
                    bytes_read=streamed_response_bytes,
                    max_bytes=MAX_HTTP_RESPONSE_BYTES,
                ),
            )
            return
        buffer.extend(chunk)
        while b"\n" in buffer:
            raw_line, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            saw_done = (
                _merge_sse_line(
                    raw_line.decode("utf-8", errors="replace").strip(),
                    result,
                    start,
                    chunk_times,
                    scenario,
                    audio_state,
                )
                or saw_done
            )
            if _result_has_terminal_error(result):
                break
        if _result_has_terminal_error(result):
            break
    if not _result_has_terminal_error(result) and buffer.strip():
        saw_done = (
            _merge_sse_line(
                bytes(buffer).decode("utf-8", errors="replace").strip(),
                result,
                start,
                chunk_times,
                scenario,
                audio_state,
            )
            or saw_done
        )
    if _result_has_terminal_error(result):
        result.success = False
        return
    if not scenario.expect_success:
        _mark_unexpected_success(result, scenario)
        return
    if result.audio_bytes <= 0:
        _mark_protocol_error(
            result,
            status="empty_stream",
            error="SSE speech endpoint completed without audio bytes",
        )
        result.response_bytes = result.audio_bytes
        return
    if not saw_done:
        _mark_protocol_error(
            result,
            status="incomplete_sse_stream",
            error="SSE speech endpoint completed without terminal data: [DONE]",
        )
        result.response_bytes = result.audio_bytes
        return
    if audio_state.total_bytes < MIN_PCM_AUDIO_BYTES:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE stream completed before the minimum generated-audio duration "
                f"(bytes={audio_state.total_bytes}, minimum={MIN_PCM_AUDIO_BYTES})"
            ),
        )
        result.response_bytes = result.audio_bytes
        return
    if not audio_state.has_signal:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error="SSE stream completed with only zero-amplitude PCM chunks",
        )
        result.response_bytes = result.audio_bytes
        return
    result.audio_duration_s = audio_state.duration_s
    _mark_success(result)
    result.response_bytes = result.audio_bytes


async def _response_body_and_text(
    response: aiohttp.ClientResponse,
) -> tuple[bytes, str]:
    body = await read_response_body(response)
    return body, body.decode("utf-8", errors="replace")


def _mark_response_body_too_large(
    result: ScenarioResult,
    exc: ResponseBodyTooLarge,
) -> None:
    result.response_bytes = exc.bytes_read
    _mark_protocol_error(
        result,
        status="response_too_large",
        error=(
            "HTTP response exceeded benchmark read cap "
            f"(bytes_read={exc.bytes_read}, max_bytes={exc.max_bytes})"
        ),
    )


def _merge_sse_line(
    line: str,
    result: ScenarioResult,
    start: float,
    chunk_times: list[float],
    scenario: Scenario,
    audio_state: SseAudioState,
) -> bool:
    if not line or line.startswith(":"):
        return False
    if line == SSE_DONE_MARKER:
        return True
    try:
        audio_bytes, event = parse_sse_audio_event(line)
    except (TypeError, ValueError, binascii.Error) as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "protocol_error"
        result.error = f"malformed SSE audio event: {exc}"
        return False
    if event is None:
        return False
    if audio_bytes is None:
        if _is_terminal_sse_json_event(event):
            return False
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE event did not include base64 audio payload: {event}",
        )
        return False
    audio = event.get("audio") if isinstance(event, dict) else None
    if not isinstance(audio, dict):
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE audio event has invalid audio metadata: {event}",
        )
        return False
    audio_format = audio.get("format")
    expected_format = str(scenario.payload.get("response_format", ""))
    if not isinstance(audio_format, str) or audio_format != expected_format:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE audio.format must match requested response_format "
                f"(expected={expected_format!r}, observed={audio_format!r})"
            ),
        )
        return False
    mime_type = audio.get("mime_type")
    if mime_type is not None and not isinstance(mime_type, str):
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE audio.mime_type must be a string when present: {event}",
        )
        return False
    normalized_mime_type = (
        (mime_type or "application/octet-stream").lower().split(";", 1)[0]
    )
    if audio_format != "pcm" or normalized_mime_type not in PCM_CONTENT_TYPES:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE stream=true audio chunks must be PCM "
                f"(format={audio_format!r}, mime_type={mime_type!r})"
            ),
        )
        return False
    sample_rate = audio.get("sample_rate", PCM_SAMPLE_RATE)
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE audio.sample_rate must be a positive integer: {event}",
        )
        return False
    chunk_validation = validate_pcm_chunk(
        audio_bytes,
        sample_rate=sample_rate,
    )
    if not chunk_validation.ok:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE audio.data must decode to valid 16-bit PCM chunk "
                f"(decoded_bytes={len(audio_bytes)}, "
                f"validation_error={chunk_validation.error})"
            ),
        )
        return False
    now = time.perf_counter()
    if result.ttfa_s is None:
        result.ttfa_s = now - start
    elif chunk_times:
        result.inter_chunk_s.append(now - chunk_times[-1])
    chunk_times.append(now)
    audio_state.total_bytes += len(audio_bytes)
    audio_state.duration_s += chunk_validation.duration_s
    audio_state.has_signal = audio_state.has_signal or any(audio_bytes)
    result.audio_bytes += len(audio_bytes)
    result.response_bytes += len(audio_bytes)
    return False


def _is_terminal_sse_json_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    if event.get("audio") is not None:
        return False
    finish_reason = event.get("finish_reason")
    return isinstance(finish_reason, str) and bool(finish_reason)


def _scenario_response_format(scenario: Scenario) -> str | None:
    response_format = scenario.planned_metadata.get("response_format")
    if response_format is None:
        response_format = scenario.payload.get("response_format")
    return str(response_format) if response_format is not None else None
