# SPDX-License-Identifier: Apache-2.0
"""Batch response validation for the TTS serving benchmark."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

from benchmarks.tts_serving.audio_validation import (
    EXPECTED_AUDIO_CONTENT_TYPES,
    validate_audio_response,
)
from benchmarks.tts_serving.http_contracts import (
    _json_from_bytes,
    _mark_protocol_error,
    _mark_success,
)
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.scenarios import Scenario

AUDIO_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "flac", "aac", "opus"}
MIN_SPEECH_SPEED = 0.25
MAX_SPEECH_SPEED = 4.0
SUCCESS_BATCH_STATUSES = {"ok", "success", "succeeded"}
FAILED_BATCH_STATUSES = {"error", "failed"}
VALID_BATCH_TASK_TYPES = {"Base", "CustomVoice", "VoiceDesign"}
EXPECTED_BATCH_MEDIA_TYPES = EXPECTED_AUDIO_CONTENT_TYPES


@dataclass(frozen=True)
class BatchItemValidation:
    error: str | None = None
    audio_bytes: int = 0
    audio_duration_s: float = 0.0


def handle_batch_success(
    body: bytes, result: ScenarioResult, scenario: Scenario
) -> None:
    payload = _json_from_bytes(
        body,
        result,
        status="invalid_batch_response",
        error_prefix="batch endpoint returned invalid JSON",
        default_empty={},
    )
    if payload is None:
        return
    required_keys = {"id", "results", "total", "succeeded", "failed"}
    if not isinstance(payload, dict) or not required_keys <= set(payload):
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint returned JSON without id/results/total/succeeded/failed",
        )
        return
    batch_size = int(scenario.planned_metadata.get("batch_size") or 0)
    results = payload.get("results")
    total = payload.get("total")
    succeeded = payload.get("succeeded")
    failed = payload.get("failed")
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint id must be a non-empty string",
        )
        return
    if (
        not isinstance(results, list)
        or not isinstance(total, int)
        or not isinstance(succeeded, int)
        or not isinstance(failed, int)
    ):
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint returned non-integer counts or non-list results",
        )
        return
    if total != batch_size or len(results) != batch_size:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint result count mismatch "
                f"(expected={batch_size}, total={total}, results={len(results)})"
            ),
        )
        return
    if succeeded + failed != total:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint succeeded + failed does not equal total",
        )
        return
    expected_item_failures = _expected_batch_item_failures(scenario)
    expected_failed = len(expected_item_failures)
    if failed != expected_failed or succeeded != total - expected_failed:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint did not report the exact expected item-level "
                f"outcome counts (expected_succeeded={total - expected_failed}, "
                f"succeeded={succeeded}, expected_failed={expected_failed}, "
                f"failed={failed})"
            ),
        )
        return
    observed_success = 0
    observed_failed = 0
    successful_audio_bytes = 0
    successful_audio_duration_s = 0.0
    for index, item in enumerate(results):
        expect_item_failure = index in expected_item_failures
        expected_format = _expected_batch_response_format(scenario, index)
        validation_error = _validate_batch_item(
            item,
            expected_index=index,
            expected_format=expected_format,
            expect_failure=expect_item_failure,
        )
        if validation_error.error is not None:
            _mark_protocol_error(
                result,
                status="invalid_batch_response",
                error=f"batch endpoint result item {index}: {validation_error.error}",
            )
            return
        if expect_item_failure:
            observed_failed += 1
        else:
            observed_success += 1
            successful_audio_bytes += validation_error.audio_bytes
            successful_audio_duration_s += validation_error.audio_duration_s
    if observed_success != succeeded or observed_failed != failed:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint item statuses do not match top-level counts "
                f"(item_success={observed_success}, succeeded={succeeded}, "
                f"item_failed={observed_failed}, failed={failed})"
            ),
        )
        return
    result.audio_bytes += successful_audio_bytes
    result.audio_duration_s += successful_audio_duration_s
    _mark_success(result, capability="pass")


def _expected_batch_item_failures(scenario: Scenario) -> set[int]:
    items = scenario.payload.get("items", [])
    if not isinstance(items, list):
        return set()
    return {
        index
        for index, item in enumerate(items)
        if _is_expected_batch_item_failure(item)
    }


def _is_expected_batch_item_failure(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    input_text = item.get("input")
    if not isinstance(input_text, str) or not input_text.strip():
        return True
    response_format = item.get("response_format")
    if response_format is not None and response_format not in AUDIO_RESPONSE_FORMATS:
        return True
    task_type = item.get("task_type")
    if task_type is not None and task_type not in VALID_BATCH_TASK_TYPES:
        return True
    speed = item.get("speed")
    return speed is not None and (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or speed < MIN_SPEECH_SPEED
        or speed > MAX_SPEECH_SPEED
    )


def _expected_batch_response_format(scenario: Scenario, index: int) -> str:
    items = scenario.payload.get("items", [])
    item_format = None
    if (
        isinstance(items, list)
        and index < len(items)
        and isinstance(items[index], dict)
    ):
        item_format = items[index].get("response_format")
    return str(item_format or scenario.payload.get("response_format") or "wav")


def _validate_batch_item(
    item: Any,
    *,
    expected_index: int,
    expected_format: str,
    expect_failure: bool,
) -> BatchItemValidation:
    if not isinstance(item, dict):
        return BatchItemValidation(error="result item must be a JSON object")
    if item.get("index") != expected_index:
        return BatchItemValidation(
            error=(
                "index mismatch "
                f"(expected={expected_index}, observed={item.get('index')})"
            )
        )
    status = item.get("status")
    if expect_failure:
        if status not in FAILED_BATCH_STATUSES:
            return BatchItemValidation(
                error=f"expected failed status, observed={status!r}"
            )
        error = item.get("error")
        if not isinstance(error, (dict, str)) or not error:
            return BatchItemValidation(
                error="failed item must include non-empty error details"
            )
        return BatchItemValidation()
    if status not in SUCCESS_BATCH_STATUSES:
        return BatchItemValidation(
            error=f"expected success status, observed={status!r}"
        )
    audio_data = item.get("audio_data")
    media_type = item.get("media_type")
    if not isinstance(audio_data, str) or not audio_data:
        return BatchItemValidation(
            error="successful item must include non-empty base64 audio_data"
        )
    if not isinstance(media_type, str) or not _is_valid_batch_media_type(
        media_type, expected_format=expected_format
    ):
        return BatchItemValidation(
            error=(
                "successful item media_type does not match requested format "
                f"(format={expected_format!r}, media_type={media_type!r})"
            )
        )
    try:
        decoded = base64.b64decode(audio_data, validate=True)
    except binascii.Error as exc:
        return BatchItemValidation(
            error=f"successful item audio_data is not valid base64: {exc}"
        )
    if not decoded:
        return BatchItemValidation(
            error="successful item audio_data decoded to empty bytes"
        )
    validation = validate_audio_response(
        decoded,
        response_format=expected_format,
        content_type=media_type,
    )
    if not validation.ok:
        return BatchItemValidation(
            error=(
                "successful item audio_data does not match requested audio "
                f"contract (format={expected_format!r}, media_type={media_type!r}, "
                f"decoded_bytes={len(decoded)}, validation_error={validation.error})"
            )
        )
    return BatchItemValidation(
        audio_bytes=len(decoded),
        audio_duration_s=validation.duration_s,
    )


def _is_valid_batch_media_type(media_type: str, *, expected_format: str) -> bool:
    normalized = media_type.lower().split(";", 1)[0]
    return normalized in EXPECTED_BATCH_MEDIA_TYPES.get(expected_format.lower(), set())
