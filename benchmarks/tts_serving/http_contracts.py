# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP result contracts for the TTS serving benchmark."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from benchmarks.tts_serving.error_contract import is_openai_error_response
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.scenarios import Scenario

UNSUPPORTED_HTTP_STATUSES = {404, 405, 501}
MAX_HTTP_RESPONSE_BYTES = 64 * 1024 * 1024
HTTP_RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class ResponseBodyTooLarge(RuntimeError):
    def __init__(self, *, bytes_read: int, max_bytes: int) -> None:
        super().__init__(
            f"HTTP response exceeded benchmark read cap "
            f"(bytes_read={bytes_read}, max_bytes={max_bytes})"
        )
        self.bytes_read = bytes_read
        self.max_bytes = max_bytes


async def read_response_body(response: aiohttp.ClientResponse) -> bytes:
    body = bytearray()
    async for chunk in response.content.iter_chunked(HTTP_RESPONSE_READ_CHUNK_BYTES):
        next_size = len(body) + len(chunk)
        if next_size > MAX_HTTP_RESPONSE_BYTES:
            raise ResponseBodyTooLarge(
                bytes_read=next_size,
                max_bytes=MAX_HTTP_RESPONSE_BYTES,
            )
        body.extend(chunk)
    return bytes(body)


def _mark_unexpected_success(result: ScenarioResult, scenario: Scenario) -> None:
    result.success = False
    result.status = "unexpected_success"
    result.capability = "fail"
    result.error_class = "unexpected_success"
    result.error = (
        f"scenario {scenario.id} expected an error but received HTTP "
        f"{result.http_status}"
    )


def _result_has_terminal_error(result: ScenarioResult) -> bool:
    return result.error_class is not None or (
        result.status not in {"error", "ok"} and not result.success
    )


def _mark_success(result: ScenarioResult, *, capability: str | None = "pass") -> None:
    result.success = True
    result.status = "ok"
    if capability is not None:
        result.capability = capability


def _mark_unsupported_contract(
    result: ScenarioResult,
    scenario: Scenario,
    *,
    body: str,
    path: str | None = None,
) -> None:
    result.success = False
    result.status = "unsupported_contract"
    result.capability = "fail"
    result.error_class = "unsupported_endpoint"
    result.error = (
        "enabled benchmark contract is unsupported: "
        f"endpoint={scenario.endpoint}, operation={scenario.capability_key}, "
        f"path={path or scenario.path}, http_status={result.http_status}, body={body}"
    )


def _classify_http_failure(
    status: int,
    body: str,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.success = False
    result.error = body
    if scenario.capability_key == "voices.delete" and not scenario.expect_success:
        if _is_valid_missing_voice_delete_response(status, body):
            result.status = "expected_error"
            result.error_class = "expected_client_error"
            result.capability = "pass"
            return
        if _is_unsupported_http_status(status, scenario):
            _mark_unsupported_contract(result, scenario, body=body)
            return
        result.status = "invalid_voice_response"
        result.error_class = "protocol_error"
        result.capability = "fail"
        result.error = (
            "missing voice delete must return HTTP 404 JSON with "
            f"success=false and error details; received status={status}, body={body}"
        )
        return
    if 400 <= status < 500 and _is_expected_client_error_scenario(scenario):
        expected_status = _expected_client_error_status(scenario)
        if status != expected_status:
            _mark_protocol_error(
                result,
                status="invalid_error_response",
                error=(
                    "expected client-error scenario returned wrong HTTP status "
                    f"(expected={expected_status}, observed={status}): {body}"
                ),
            )
            return
        if not _is_valid_error_response(
            status,
            body,
            expected_status=expected_status,
        ):
            _mark_protocol_error(
                result,
                status="invalid_error_response",
                error=(
                    "expected client-error scenario returned HTTP "
                    f"{status} without OpenAI-compatible error JSON: {body}"
                ),
            )
            return
        result.status = "expected_error"
        result.error_class = "expected_client_error"
        result.capability = "pass"
        return
    if _is_unsupported_http_status(status, scenario):
        _mark_unsupported_contract(result, scenario, body=body)
        return
    if 500 <= status:
        result.status = "failed"
        result.error_class = "server_error"
        result.capability = "fail"
        return
    result.status = "failed"
    result.error_class = "http_error"
    if scenario.expect_success:
        result.capability = "fail"


def _is_unsupported_http_status(status: int, scenario: Scenario) -> bool:
    if status not in UNSUPPORTED_HTTP_STATUSES:
        return False
    if scenario.capability_key == "voices.delete" and not scenario.expect_success:
        return status != 404
    if _is_expected_client_error_scenario(scenario):
        return False
    return True


def _is_expected_client_error_scenario(scenario: Scenario) -> bool:
    return (
        not scenario.expect_success and scenario.expected_status_class == "client_error"
    )


def _expected_client_error_status(scenario: Scenario) -> int:
    return scenario.expected_http_status or 400


def _json_object_from_bytes(
    body: bytes,
    result: ScenarioResult,
    *,
    status: str,
    error_prefix: str,
) -> dict[str, Any] | None:
    payload = _json_from_bytes(
        body,
        result,
        status=status,
        error_prefix=error_prefix,
        default_empty={},
    )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        _mark_protocol_error(
            result,
            status=status,
            error=f"{error_prefix}: response must be a JSON object",
        )
        return None
    return payload


def _json_from_bytes(
    body: bytes,
    result: ScenarioResult,
    *,
    status: str,
    error_prefix: str,
    default_empty: Any = None,
) -> Any | None:
    try:
        return (
            json.loads(body.decode("utf-8", errors="replace"))
            if body
            else default_empty
        )
    except json.JSONDecodeError as exc:
        _mark_protocol_error(
            result,
            status=status,
            error=f"{error_prefix}: {exc}",
        )
        result.error_type = exc.__class__.__name__
        return None


def _is_valid_missing_voice_delete_response(status: int, body: str) -> bool:
    if status != 404:
        return False
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("success") is not False:
        return False
    error = payload.get("error")
    return isinstance(error, (dict, str)) and bool(error)


def _is_valid_error_response(
    status: int,
    body: str,
    *,
    expected_status: int,
) -> bool:
    return status == expected_status and is_openai_error_response(
        body,
        expected_status=expected_status,
    )


def _mark_protocol_error(result: ScenarioResult, *, status: str, error: str) -> None:
    result.status = status
    result.success = False
    result.capability = "fail"
    result.error_class = "protocol_error"
    result.error = error
