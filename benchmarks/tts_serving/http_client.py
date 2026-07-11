# SPDX-License-Identifier: Apache-2.0
"""HTTP clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import aiohttp

from benchmarks.tts_serving.audio_validation import validate_audio_response
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
    read_response_body,
)
from benchmarks.tts_serving.metrics import (
    PCM_SAMPLE_RATE,
    ScenarioResult,
    classify_http_status,
    finish_timing,
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
                if scenario.endpoint == "speech_stream":
                    await _handle_streaming_audio_response(
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
            sample_rate=_response_sample_rate(response),
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


async def _handle_streaming_audio_response(
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

    if not scenario.expect_success:
        try:
            body = await read_response_body(response)
        except ResponseBodyTooLarge as exc:
            _mark_response_body_too_large(result, exc)
            return
        result.response_bytes = len(body)
        _mark_unexpected_success(result, scenario)
        return

    body = bytearray()
    chunk_times: list[float] = []
    async for chunk, chunk_time in _iter_response_http_chunks(response):
        if not chunk_times:
            result.ttfa_s = chunk_time - start
            result.first_audio_payload_bytes = len(chunk)
        chunk_times.append(chunk_time)
        body.extend(chunk)
        if len(body) > MAX_HTTP_RESPONSE_BYTES:
            _mark_response_body_too_large(
                result,
                ResponseBodyTooLarge(
                    bytes_read=len(body),
                    max_bytes=MAX_HTTP_RESPONSE_BYTES,
                ),
            )
            return

    if chunk_times:
        result.inter_chunk_s = [
            now - prev for prev, now in zip(chunk_times, chunk_times[1:])
        ]
    result.audio_chunk_count = len(chunk_times)
    finish_timing(result, start)
    result.response_bytes = len(body)
    response_format = str(scenario.payload.get("response_format", ""))
    validation = await asyncio.to_thread(
        validate_audio_response,
        bytes(body),
        response_format=response_format,
        content_type=response.headers.get("Content-Type"),
        sample_rate=_response_sample_rate(response),
    )
    if not validation.ok:
        _mark_protocol_error(
            result,
            status="invalid_streaming_audio_response",
            error=(
                "speech streaming endpoint returned 2xx without the requested "
                f"audio contract (format={response_format!r}, "
                f"content-type={response.headers.get('Content-Type')!r}, "
                f"bytes={len(body)}, validation_error={validation.error})"
            ),
        )
        return
    result.audio_bytes = len(body)
    result.audio_duration_s = validation.duration_s
    _mark_success(result)


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


def _response_sample_rate(response: aiohttp.ClientResponse) -> int:
    value = response.headers.get("X-Sample-Rate")
    if value is None:
        return PCM_SAMPLE_RATE
    try:
        sample_rate = int(value)
    except ValueError:
        return PCM_SAMPLE_RATE
    return sample_rate if sample_rate > 0 else PCM_SAMPLE_RATE


async def _iter_response_http_chunks(
    response: aiohttp.ClientResponse,
) -> AsyncIterator[tuple[bytes, float]]:
    pending = bytearray()
    pending_start_s: float | None = None
    async for data, end_of_http_chunk in response.content.iter_chunks():
        now = time.perf_counter()
        if data:
            if not pending:
                pending_start_s = now
            pending.extend(data)
        if end_of_http_chunk and pending:
            yield bytes(pending), pending_start_s or now
            pending.clear()
            pending_start_s = None

    if pending:
        yield bytes(pending), pending_start_s or time.perf_counter()


def _scenario_response_format(scenario: Scenario) -> str | None:
    response_format = scenario.planned_metadata.get("response_format")
    if response_format is None:
        response_format = scenario.payload.get("response_format")
    return str(response_format) if response_format is not None else None
