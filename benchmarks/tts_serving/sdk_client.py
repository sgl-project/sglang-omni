# SPDX-License-Identifier: Apache-2.0
"""OpenAI Python SDK client for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from benchmarks.tts_serving.audio_validation import validate_audio_response
from benchmarks.tts_serving.error_contract import is_openai_error_response
from benchmarks.tts_serving.http_contracts import (
    MAX_HTTP_RESPONSE_BYTES,
    UNSUPPORTED_HTTP_STATUSES,
)
from benchmarks.tts_serving.metrics import ScenarioResult, finish_timing
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec


async def run_sdk_scenario(spec: BenchmarkSpec, scenario: Scenario) -> ScenarioResult:
    start = time.perf_counter()
    result = ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        capability_key=scenario.capability_key,
        workload=scenario.workload,
        expected_success=scenario.expect_success,
        response_format=_scenario_response_format(scenario),
    )
    try:
        await asyncio.to_thread(
            _run_openai_speech_create, spec, scenario, result, start
        )
    except Exception as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "client_error"
        result.error = (
            f"OpenAI SDK benchmark scenario failed before classification: {exc}"
        )
    finally:
        finish_timing(result, start)
    return result


def _run_openai_speech_create(
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    start: float,
) -> None:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
    except ImportError as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "dependency_error"
        result.error = "openai package is required for SDK benchmark scenarios"
        return

    client = OpenAI(
        base_url=_sdk_base_url(spec.base_url),
        api_key=_sdk_api_key(spec),
        timeout=spec.params.timeout_s,
    )
    request = _sdk_request(scenario.payload, spec.model_name)
    result.request_bytes = _json_size(request)
    try:
        with tempfile.TemporaryDirectory(prefix="tts-serving-sdk-") as tmp_dir:
            response_format = _scenario_response_format(scenario)
            output_path = Path(tmp_dir) / f"speech.{response_format}"
            response = client.audio.speech.create(**request)
            response.stream_to_file(str(output_path))
            output_size = output_path.stat().st_size
            if output_size > MAX_HTTP_RESPONSE_BYTES:
                result.response_bytes = output_size
                result.status = "response_too_large"
                result.capability = "fail"
                result.error_class = "protocol_error"
                result.error = (
                    "OpenAI SDK speech.create output exceeded benchmark read cap "
                    f"(bytes_read={output_size}, max_bytes={MAX_HTTP_RESPONSE_BYTES})"
                )
                return
            body = output_path.read_bytes()
            finish_timing(result, start)
    except APIStatusError as exc:
        _classify_sdk_status_error(exc, result, scenario)
        return
    except (APIConnectionError, APITimeoutError) as exc:
        result.status = "transport_error"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "transport_error"
        result.error = str(exc)
        return
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            with suppress(Exception):
                close()

    result.response_bytes = len(body)
    response_format = _scenario_response_format(scenario)
    validation = validate_audio_response(
        body,
        response_format=response_format,
        require_content_type=False,
    )
    if not validation.ok:
        result.status = "invalid_audio_response"
        result.capability = "fail"
        result.error_class = "protocol_error"
        result.error = (
            "OpenAI SDK speech.create stream_to_file did not produce requested "
            f"audio bytes (format={response_format!r}, bytes={len(body)}, "
            f"validation_error={validation.error})"
        )
        return
    result.audio_bytes = len(body)
    result.audio_duration_s = validation.duration_s
    result.status = "ok"
    result.success = True
    result.capability = "pass"


def _sdk_request(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    return {
        "model": str(payload.get("model") or model_name),
        "input": str(payload.get("input") or ""),
        "voice": str(payload.get("voice") or "default"),
        "response_format": str(payload.get("response_format") or "wav"),
        "speed": float(payload.get("speed") or 1.0),
    }


def _scenario_response_format(scenario: Scenario) -> str:
    return str(
        scenario.planned_metadata.get("sdk_response_format")
        or scenario.planned_metadata.get("response_format")
        or scenario.payload.get("response_format")
        or "wav"
    )


def _sdk_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _sdk_api_key(spec: BenchmarkSpec) -> str:
    if spec.auth.api_key_env:
        token = os.environ.get(spec.auth.api_key_env)
        if not token:
            raise RuntimeError(
                f"auth environment variable is not set: {spec.auth.api_key_env}"
            )
        return token
    return "benchmark"


def _json_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _classify_sdk_status_error(
    exc: Any,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    status_code = int(getattr(exc, "status_code", 0) or 0)
    result.http_status = status_code or None
    result.http_status_class = f"{status_code // 100}xx" if status_code else None
    result.success = False
    result.error_type = exc.__class__.__name__
    body = _sdk_error_body(exc)
    result.error = body or str(exc)
    if status_code in UNSUPPORTED_HTTP_STATUSES and scenario.expect_success:
        result.status = "unsupported_contract"
        result.capability = "fail"
        result.error_class = "unsupported_endpoint"
        return
    if status_code >= 500:
        result.status = "failed"
        result.capability = "fail"
        result.error_class = "server_error"
        return
    if (
        400 <= status_code < 500
        and not scenario.expect_success
        and scenario.expected_status_class == "client_error"
    ):
        expected_status = scenario.expected_http_status or 400
        if status_code != expected_status:
            result.status = "invalid_error_response"
            result.capability = "fail"
            result.error_class = "protocol_error"
            result.error = (
                "OpenAI SDK expected-error scenario returned wrong HTTP status "
                f"(expected={expected_status}, observed={status_code}): {result.error}"
            )
            return
        if not _is_openai_error_body(
            body,
            expected_status=expected_status,
        ):
            result.status = "invalid_error_response"
            result.capability = "fail"
            result.error_class = "protocol_error"
            result.error = (
                "OpenAI SDK expected-error scenario did not expose "
                f"OpenAI-compatible error JSON: {result.error}"
            )
            return
        result.status = "expected_error"
        result.capability = "pass"
        result.error_class = "expected_client_error"
        return
    result.status = "failed"
    result.capability = "fail"
    result.error_class = "http_error"


def _sdk_error_body(exc: Any) -> str:
    response = getattr(exc, "response", None)
    text = getattr(response, "text", "") if response is not None else ""
    if isinstance(text, str) and text:
        return text
    parsed_body = getattr(exc, "body", None)
    if isinstance(parsed_body, dict):
        return json.dumps(parsed_body, ensure_ascii=False)
    return ""


def _is_openai_error_body(
    body: str,
    *,
    expected_status: int,
) -> bool:
    return is_openai_error_response(body, expected_status=expected_status)
