# SPDX-License-Identifier: Apache-2.0
"""HTTP clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator

import aiohttp

from benchmarks.tts_serving.audio_validation import (
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
    run_voice_named_batch_sequence,
    run_voice_named_speech_sequence,
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
        workload=scenario.workload,
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
        elif scenario.method == "VOICE_NAMED_SPEECH_SEQUENCE":
            await run_voice_named_speech_sequence(session, spec, scenario, result)
        elif scenario.method == "VOICE_NAMED_BATCH_SEQUENCE":
            await run_voice_named_batch_sequence(session, spec, scenario, result)
        elif scenario.capability_key == "voices.upload" and scenario.expect_success:
            await run_voice_upload(session, spec, scenario, result)
        elif scenario.method == "GET":
            async with session.get(url) as response:
                await _handle_probe_response(response, result, scenario)
        elif scenario.method == "DELETE":
            async with session.delete(url) as response:
                await _handle_binary_response(response, result, start, scenario)
        elif scenario.method == "HTTP_DISCONNECT":
            await _run_disconnect_scenario(session, spec, scenario, result)
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


async def _run_disconnect_scenario(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    result.request_bytes = request_size(scenario)
    if scenario.endpoint == "speech_stream":
        disconnected = await _disconnect_streaming_response(
            session,
            spec,
            scenario,
            result,
        )
    else:
        disconnected = await _disconnect_before_response(
            session,
            spec,
            scenario,
            result,
        )
    if not disconnected:
        return

    result.was_cancelled = True
    await _run_speech_liveness_probe(session, spec, result)


async def _disconnect_before_response(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> bool:
    request_body_sent = asyncio.Event()

    async def mark_request_body_sent(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: object,
        _params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        request_body_sent.set()

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_chunk_sent.append(mark_request_body_sent)
    connector = aiohttp.TCPConnector(limit=1, force_close=True)
    async with aiohttp.ClientSession(
        timeout=session.timeout,
        headers=session.headers,
        connector=connector,
        trace_configs=[trace_config],
    ) as disconnect_session:
        request_task = asyncio.create_task(
            disconnect_session.post(
                api_url(spec.base_url, scenario.path),
                json=scenario.payload,
            )
        )
        body_sent_task = asyncio.create_task(request_body_sent.wait())
        try:
            done, _ = await asyncio.wait(
                {request_task, body_sent_task},
                timeout=spec.params.timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                _mark_protocol_error(
                    result,
                    status="disconnect_not_reached",
                    error="request body was not sent before the configured request timeout",
                )
                return False
            if request_task in done:
                response = request_task.result()
                result.http_status = response.status
                result.http_status_class = classify_http_status(response.status)
                response.close()
                _mark_protocol_error(
                    result,
                    status="disconnect_completed_too_early",
                    error="request completed before the benchmark client disconnected",
                )
                return False

            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            return True
        finally:
            body_sent_task.cancel()
            await asyncio.gather(body_sent_task, return_exceptions=True)
            if not request_task.done():
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)


async def _disconnect_streaming_response(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> bool:
    response = await session.post(
        api_url(spec.base_url, scenario.path),
        json=scenario.payload,
    )
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    try:
        if response.status != 200:
            body, body_text = await _response_body_and_text(response)
            result.response_bytes = len(body)
            _classify_http_failure(response.status, body_text, result, scenario)
            return False

        chunk_iterator = _iter_response_http_chunks(response)
        try:
            chunk, _ = await anext(chunk_iterator)
        except StopAsyncIteration:
            _mark_protocol_error(
                result,
                status="disconnect_missing_audio",
                error="streaming response ended before producing an audio chunk",
            )
            return False

        validation = validate_pcm_chunk(
            chunk,
            sample_rate=_response_sample_rate(response),
        )
        if not validation.ok:
            _mark_protocol_error(
                result,
                status="disconnect_invalid_audio",
                error=f"first streaming audio chunk is invalid: {validation.error}",
            )
            return False
        if response.content.at_eof():
            _mark_protocol_error(
                result,
                status="disconnect_completed_too_early",
                error="streaming response completed before the client disconnected",
            )
            return False

        result.first_audio_payload_bytes = len(chunk)
        result.audio_chunk_count = 1
        return True
    finally:
        response.close()


async def _run_speech_liveness_probe(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    result: ScenarioResult,
) -> None:
    payload = {
        "model": spec.model_name,
        "input": "Post-disconnect liveness probe.",
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0,
    }
    async with session.post(
        api_url(spec.base_url, "/v1/audio/speech"),
        json=payload,
    ) as response:
        try:
            body = await read_response_body(response)
        except ResponseBodyTooLarge as exc:
            _mark_protocol_error(
                result,
                status="disconnect_liveness_response_too_large",
                error=str(exc),
            )
            return
        if response.status < 200 or response.status >= 300:
            _mark_protocol_error(
                result,
                status="disconnect_liveness_failed",
                error=(
                    "post-disconnect speech probe failed "
                    f"(status={response.status}, "
                    f"body={body.decode('utf-8', errors='replace')})"
                ),
            )
            return

        validation = await asyncio.to_thread(
            validate_audio_response,
            body,
            response_format="wav",
            content_type=response.headers.get("Content-Type"),
            sample_rate=_response_sample_rate(response),
        )
        if not validation.ok:
            _mark_protocol_error(
                result,
                status="disconnect_liveness_invalid_audio",
                error=f"post-disconnect speech probe returned invalid audio: {validation.error}",
            )
            return
        _mark_success(result)


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
        if not _parse_sglang_usage_headers(response, result):
            return
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


def _parse_sglang_usage_headers(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
) -> bool:
    """Capture optional SGLang usage headers without requiring provider extensions."""
    header_specs = (
        ("X-Prompt-Tokens", "prompt_tokens", int),
        ("X-Completion-Tokens", "completion_tokens", int),
        ("X-Engine-Time", "engine_time_s", float),
    )
    for header_name, attribute, parser in header_specs:
        raw_value = response.headers.get(header_name)
        if raw_value is None:
            continue
        try:
            value = parser(raw_value)
        except (TypeError, ValueError):
            _mark_protocol_error(
                result,
                status="invalid_usage_headers",
                error=f"{header_name} must be numeric, observed={raw_value!r}",
            )
            return False
        if isinstance(value, float) and not math.isfinite(value):
            _mark_protocol_error(
                result,
                status="invalid_usage_headers",
                error=f"{header_name} must be finite, observed={raw_value!r}",
            )
            return False
        if value < 0:
            _mark_protocol_error(
                result,
                status="invalid_usage_headers",
                error=f"{header_name} must be non-negative, observed={raw_value!r}",
            )
            return False
        setattr(result, attribute, value)

    return True


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
