# SPDX-License-Identifier: Apache-2.0
"""Voice API client flows for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import aiohttp

from benchmarks.tts_serving import voice_contracts
from benchmarks.tts_serving.audio_validation import validate_audio_response
from benchmarks.tts_serving.http_contracts import (
    UNSUPPORTED_HTTP_STATUSES,
    ResponseBodyTooLarge,
    _classify_http_failure,
    _is_valid_error_response,
    _json_from_bytes,
    _json_object_from_bytes,
    _mark_protocol_error,
    _mark_success,
    _mark_unsupported_contract,
    read_response_body,
)
from benchmarks.tts_serving.metrics import ScenarioResult, classify_http_status
from benchmarks.tts_serving.scenarios import (
    VOICE_SMALL_UPLOAD_BYTES,
    VOICE_UPLOAD_SUCCESS_FORMATS,
    Scenario,
)
from benchmarks.tts_serving.spec import BenchmarkSpec
from benchmarks.tts_serving.urls import api_url
from benchmarks.tts_serving.voice_upload_fixtures import (
    get_near_limit_voice_upload_fixture,
    get_voice_upload_fixture,
    get_wav_upload_fixture,
)

RawVoiceResponse = tuple[int, bytes, dict[str, str]]
MAX_CLEANUP_FAILURE_DETAILS = 20


def request_body(
    scenario: Scenario, *, form_fields: Mapping[str, str] | None = None
) -> aiohttp.FormData:
    return _voice_upload_body(
        scenario,
        form_fields=form_fields or scenario.form_fields,
        upload_format=str(scenario.planned_metadata.get("upload_format", "wav")),
        content_type=scenario.upload_content_type or "audio/wav",
        upload_size=scenario.upload_size_bytes,
        upload_case=str(scenario.planned_metadata.get("upload_case", "format")),
        filename=scenario.upload_filename or "audio.wav",
    )


def _voice_upload_body(
    scenario: Scenario,
    *,
    form_fields: Mapping[str, str],
    upload_format: str,
    content_type: str,
    upload_size: int,
    upload_case: str,
    filename: str,
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for key, value in form_fields.items():
        form.add_field(key, value)
    if scenario.upload_field:
        form.add_field(
            scenario.upload_field,
            _synthetic_upload_bytes_for(
                upload_case=upload_case,
                upload_format=upload_format,
                upload_size=upload_size,
            ),
            filename=filename,
            content_type=content_type,
        )
    return form


def _synthetic_upload_bytes_for(
    *,
    upload_case: str,
    upload_format: str,
    upload_size: int,
) -> bytes:
    if upload_case == "corrupt_audio":
        return _pad_bytes(b"not-a-valid-audio-upload", upload_size)
    if upload_case in {"near_limit", "cache_eviction"}:
        return get_near_limit_voice_upload_fixture(upload_format, upload_size)
    if upload_case == "format":
        return get_voice_upload_fixture(upload_format)
    return _synthetic_audio_bytes(upload_size, upload_format)


def _synthetic_audio_bytes(size: int, upload_format: str = "wav") -> bytes:
    if size <= 0:
        return b""
    if upload_format == "mp3":
        return _pad_bytes(b"ID3", size)
    if upload_format == "flac":
        return _pad_bytes(b"fLaC", size)
    if upload_format == "ogg":
        return _pad_bytes(b"OggS", size)
    if upload_format == "aac":
        return _pad_bytes(b"\xff\xf1", size)
    if upload_format == "webm":
        return _pad_bytes(b"\x1a\x45\xdf\xa3", size)
    if upload_format == "mp4":
        return _pad_bytes(b"\x00\x00\x00\x18ftypmp42", size)
    return get_wav_upload_fixture(size)


def _pad_bytes(prefix: bytes, size: int) -> bytes:
    if size <= len(prefix):
        return prefix[:size]
    return prefix + (b"\0" * (size - len(prefix)))


def request_size(scenario: Scenario) -> int:
    if scenario.body_type == "multipart":
        return _voice_request_size(scenario, form_fields=scenario.form_fields)
    try:
        return len(json.dumps(scenario.payload, ensure_ascii=False).encode("utf-8"))
    except TypeError:
        return 0


def _voice_request_size(
    scenario: Scenario,
    *,
    form_fields: Mapping[str, str],
) -> int:
    return _voice_request_size_for(
        form_fields=form_fields,
        upload_size=scenario.upload_size_bytes,
    )


def _voice_request_size_for(
    *,
    form_fields: Mapping[str, str],
    upload_size: int,
) -> int:
    return (
        sum(len(key) + len(value) for key, value in form_fields.items()) + upload_size
    )


def _voice_sequence_upload_size(upload_format: str) -> int:
    if upload_format == "wav":
        return VOICE_SMALL_UPLOAD_BYTES
    return len(get_voice_upload_fixture(upload_format))


def _voice_sequence_form_fields(
    scenario: Scenario,
    *,
    voice_name: str,
    ref_text: str,
    speaker_description: str,
) -> dict[str, str]:
    form_fields = dict(scenario.form_fields)
    form_fields["name"] = voice_name
    form_fields["ref_text"] = ref_text
    form_fields["speaker_description"] = speaker_description
    return form_fields


def _cache_revisit_voice_names(voice_names: list[str]) -> list[str]:
    if len(voice_names) <= 2:
        return voice_names
    midpoint = len(voice_names) // 2
    return [voice_names[0], voice_names[midpoint], voice_names[-1], voice_names[0]]


def _speaker_cap_form_fields(
    scenario: Scenario,
    voice_name: str,
) -> dict[str, str]:
    form_fields = dict(scenario.form_fields)
    form_fields["name"] = voice_name
    return form_fields


def _metadata_positive_int(scenario: Scenario, key: str) -> int | None:
    value = scenario.planned_metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


async def run_voice_lifecycle(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    created_voice_names: list[str] = []
    try:
        result.request_bytes = request_size(scenario)
        payload = await _post_voice_upload(
            session,
            upload_url,
            scenario,
            result,
            form_fields=scenario.form_fields,
        )
        if payload is None:
            return
        created_voice_names.append(voice_name)
        if not _require_voice_upload_identifier(
            payload,
            result,
            error="voice lifecycle upload response must include an identifier",
        ):
            return
        before_delete = await _require_uploaded_voice_present(
            session,
            spec,
            scenario,
            result,
            voice_name,
            operation="voice lifecycle upload",
        )
        if before_delete is None:
            return
        if not await _delete_voice_by_name(session, spec, scenario, result, voice_name):
            return
        after_delete = await _get_voice_list(session, spec, scenario, result)
        if after_delete is None:
            return
        if not _require_voice_absent_in_list(after_delete, result, voice_name):
            return
        if not await _expect_deleted_voice_speech_404(
            session, spec, scenario, result, voice_name
        ):
            return
        if not _validate_delete_invalidation_counter(
            before_delete,
            after_delete,
            result,
        ):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def run_voice_upload(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    created_voice_names: list[str] = []
    try:
        result.request_bytes = request_size(scenario)
        payload = await _post_voice_upload(
            session,
            upload_url,
            scenario,
            result,
            form_fields=scenario.form_fields,
        )
        if payload is None:
            return
        created_voice_names.append(voice_name)
        if not _require_voice_upload_identifier(
            payload,
            result,
            error="voice upload response must include id, voice_id, or name",
        ):
            return
        if (
            await _require_uploaded_voice_present(
                session,
                spec,
                scenario,
                result,
                voice_name,
                operation="voice upload",
            )
            is None
        ):
            return
        if not await _delete_voice_by_name(session, spec, scenario, result, voice_name):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def run_voice_overwrite(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    result.request_bytes = request_size(scenario) * 2
    created_voice_names: list[str] = []

    before_description = "Synthetic benchmark voice before overwrite."
    after_description = "Synthetic benchmark voice after overwrite."
    first_fields = dict(scenario.form_fields)
    second_fields = dict(scenario.form_fields)
    first_fields["speaker_description"] = before_description
    second_fields["speaker_description"] = after_description

    try:
        first_payload = await _post_voice_upload(
            session,
            upload_url,
            scenario,
            result,
            form_fields=first_fields,
        )
        if first_payload is None:
            return
        created_voice_names.append(voice_name)
        if not _require_voice_upload_identifier(
            first_payload,
            result,
            error="first same-name voice upload response must include an identifier",
        ):
            return

        second_payload = await _post_voice_upload(
            session,
            upload_url,
            scenario,
            result,
            form_fields=second_fields,
        )
        if second_payload is None:
            return
        if not _require_voice_upload_identifier(
            second_payload,
            result,
            error="second same-name voice upload response must include an identifier",
        ):
            return
        if not voice_contracts.is_voice_overwrite_ack(second_payload):
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error=(
                    "second same-name upload must include an overwrite warning or "
                    f"replacement indicator: {second_payload}"
                ),
            )
            return

        voice_list = await _get_voice_list(session, spec, scenario, result)
        if voice_list is None:
            return
        entries = voice_contracts.uploaded_voice_entries(voice_list, voice_name)
        if not _validate_overwritten_voice_entry(
            entries,
            result,
            voice_name=voice_name,
            expected_speaker_description=after_description,
        ):
            return
        if not await _delete_voice_by_name(session, spec, scenario, result, voice_name):
            return
        created_voice_names.clear()
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def run_voice_upload_delete_race(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    result.request_bytes = request_size(scenario) * 2
    created_voice_names: list[str] = []

    try:
        initial_payload = await _post_voice_upload(
            session,
            upload_url,
            scenario,
            result,
            form_fields=scenario.form_fields,
        )
        if initial_payload is None:
            return
        created_voice_names.append(voice_name)
        if not _require_voice_upload_identifier(
            initial_payload,
            result,
            error="initial race voice upload response must include an identifier",
        ):
            return

        race_upload_fields = dict(scenario.form_fields)
        race_upload_fields["speaker_description"] = (
            "Synthetic benchmark voice uploaded concurrently with delete."
        )
        upload_body = request_body(scenario, form_fields=race_upload_fields)
        delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
        upload_response, delete_response = await asyncio.gather(
            _raw_post(session, upload_url, upload_body),
            _raw_delete(session, delete_url),
        )
        _merge_raw_voice_response(upload_response, result)
        _merge_raw_voice_response(delete_response, result)
        if not _classify_voice_race_response(
            upload_response,
            result,
            scenario,
            operation="concurrent voice upload",
            requires_voice_identifier=True,
        ):
            return
        if not _classify_voice_race_response(
            delete_response,
            result,
            scenario,
            operation="concurrent voice delete",
            requires_delete_success=True,
        ):
            return

        voice_list = await _get_voice_list(session, spec, scenario, result)
        if voice_list is None:
            return
        entries = voice_contracts.uploaded_voice_entries(voice_list, voice_name)
        if len(entries) > 1:
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error=(
                    "same-name upload/delete race must not leave duplicate uploaded "
                    f"voices named {voice_name!r}; observed={len(entries)}"
                ),
            )
            return
        if entries and not await _delete_voice_by_name(
            session, spec, scenario, result, voice_name
        ):
            return
        created_voice_names.clear()
        _mark_success(result, capability="pass")
    except ResponseBodyTooLarge as exc:
        _mark_voice_response_too_large(result, exc)
        return
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def run_voice_speaker_cap_sequence(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    sequence_config = _speaker_cap_sequence_config(scenario, result)
    if sequence_config is None:
        return
    attempt_count, voice_name_prefix, speaker_max_uploaded = sequence_config

    upload_url = api_url(spec.base_url, scenario.path)
    created_voice_names: list[str] = []
    try:
        uploaded_voices = await _speaker_cap_uploaded_voices_after_stale_cleanup(
            session,
            spec,
            scenario,
            result,
            voice_name_prefix=voice_name_prefix,
        )
        if uploaded_voices is None:
            return

        remaining_before_cap = max(speaker_max_uploaded - len(uploaded_voices), 0)
        if not _speaker_cap_attempts_cross_cap(
            result,
            uploaded_count=len(uploaded_voices),
            attempt_count=attempt_count,
            remaining_before_cap=remaining_before_cap,
            speaker_max_uploaded=speaker_max_uploaded,
        ):
            return

        if not await _fill_speaker_cap(
            session,
            upload_url,
            scenario,
            result,
            voice_name_prefix=voice_name_prefix,
            upload_count=remaining_before_cap,
            created_voice_names=created_voice_names,
        ):
            return
        overflow_name = f"{voice_name_prefix}_overflow"
        created_voice_names.append(overflow_name)
        if not await _expect_speaker_cap_rejection(
            session,
            upload_url,
            scenario,
            result,
            voice_name=overflow_name,
            speaker_max_uploaded=speaker_max_uploaded,
        ):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(
            session,
            spec,
            created_voice_names,
        )
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def run_voice_upload_metadata_sequence(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    voice_name_prefix = str(scenario.planned_metadata.get("voice_name_prefix", ""))
    if not voice_name_prefix:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="voice metadata sequence requires voice_name_prefix",
        )
        return

    upload_url = api_url(spec.base_url, scenario.path)
    created_voice_names: list[str] = []
    expected_entries: dict[str, dict[str, str]] = {}
    try:
        for upload_format, content_type in VOICE_UPLOAD_SUCCESS_FORMATS:
            voice_name = f"{voice_name_prefix}_{upload_format}"
            fields = _voice_sequence_form_fields(
                scenario,
                voice_name=voice_name,
                ref_text=f"Voice metadata reference text for {upload_format}.",
                speaker_description=(
                    f"Synthetic metadata sequence voice in {upload_format} format."
                ),
            )
            created_voice_names.append(voice_name)
            expected_entries[voice_name] = {
                "ref_text": fields["ref_text"],
                "speaker_description": fields["speaker_description"],
            }
            if not await _post_expected_voice_upload(
                session,
                upload_url,
                scenario,
                result,
                form_fields=fields,
                upload_format=upload_format,
                content_type=content_type,
                upload_size=_voice_sequence_upload_size(upload_format),
                upload_case="format",
                operation=f"metadata upload {upload_format}",
            ):
                return

        voice_list = await _get_voice_list(session, spec, scenario, result)
        if voice_list is None:
            return
        if not _validate_uploaded_voice_metadata_sequence(
            voice_list,
            expected_entries,
            result,
        ):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def run_voice_cache_pressure_sequence(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    voice_count = _metadata_positive_int(scenario, "voice_count")
    voice_name_prefix = str(scenario.planned_metadata.get("voice_name_prefix", ""))
    if voice_count is None or not voice_name_prefix:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="voice cache pressure sequence requires voice_count and voice_name_prefix",
        )
        return

    created_voice_names: list[str] = []
    try:
        if not await _run_voice_cache_pressure_primary_path(
            session,
            spec,
            scenario,
            result,
            voice_name_prefix=voice_name_prefix,
            voice_count=voice_count,
            created_voice_names=created_voice_names,
        ):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def _run_voice_cache_pressure_primary_path(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name_prefix: str,
    voice_count: int,
    created_voice_names: list[str],
) -> bool:
    before_stats = await _voice_cache_stats_snapshot(
        session,
        spec,
        scenario,
        result,
        operation="before cache pressure",
    )
    if before_stats is None:
        return False

    after_unique_stats = await _cache_pressure_unique_phase(
        session,
        spec,
        scenario,
        result,
        before_stats=before_stats,
        voice_name_prefix=voice_name_prefix,
        voice_count=voice_count,
        created_voice_names=created_voice_names,
    )
    if after_unique_stats is None:
        return False

    after_revisit_stats = await _cache_pressure_revisit_phase(
        session,
        spec,
        scenario,
        result,
        after_unique_stats=after_unique_stats,
        created_voice_names=created_voice_names,
    )
    if after_revisit_stats is None:
        return False

    return await _cache_pressure_cleanup_phase(
        session,
        spec,
        scenario,
        result,
        after_revisit_stats=after_revisit_stats,
        created_voice_names=created_voice_names,
    )


async def _cache_pressure_unique_phase(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    before_stats: dict[str, int],
    voice_name_prefix: str,
    voice_count: int,
    created_voice_names: list[str],
) -> dict[str, int] | None:
    if not await _upload_and_synthesize_cache_pressure_voices(
        session,
        spec,
        scenario,
        result,
        voice_name_prefix=voice_name_prefix,
        voice_count=voice_count,
        created_voice_names=created_voice_names,
    ):
        return None

    after_unique_stats = await _voice_cache_stats_snapshot(
        session,
        spec,
        scenario,
        result,
        operation="after unique cache pressure requests",
    )
    if after_unique_stats is None:
        return None
    if not voice_contracts.validate_cache_pressure_unique_stats(
        before_stats,
        after_unique_stats,
        voice_count=voice_count,
        result=result,
    ):
        return None
    return after_unique_stats


async def _cache_pressure_revisit_phase(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    after_unique_stats: dict[str, int],
    created_voice_names: list[str],
) -> dict[str, int] | None:
    revisit_voice_names = await _revisit_cache_pressure_voices(
        session,
        spec,
        scenario,
        result,
        created_voice_names,
    )
    if revisit_voice_names is None:
        return None

    after_revisit_stats = await _voice_cache_stats_snapshot(
        session,
        spec,
        scenario,
        result,
        operation="after cache revisit requests",
    )
    if after_revisit_stats is None:
        return None
    if not voice_contracts.validate_cache_pressure_revisit_stats(
        after_unique_stats,
        after_revisit_stats,
        revisit_count=len(revisit_voice_names),
        result=result,
    ):
        return None
    return after_revisit_stats


async def _cache_pressure_cleanup_phase(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    after_revisit_stats: dict[str, int],
    created_voice_names: list[str],
) -> bool:
    cleanup = await _cleanup_cache_pressure_voices(
        session,
        spec,
        scenario,
        result,
        created_voice_names,
    )
    if cleanup is None:
        return False
    after_cleanup_stats, cleanup_count = cleanup
    return voice_contracts.validate_cache_pressure_cleanup_stats(
        after_revisit_stats,
        after_cleanup_stats,
        deleted_count=cleanup_count,
        result=result,
    )


async def _voice_cache_stats_snapshot(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    operation: str,
) -> dict[str, int] | None:
    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return None
    return voice_contracts.require_voice_cache_stats(
        voice_list, result, operation=operation
    )


async def _upload_and_synthesize_cache_pressure_voices(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name_prefix: str,
    voice_count: int,
    created_voice_names: list[str],
) -> bool:
    upload_url = api_url(spec.base_url, scenario.path)
    for voice_index in range(voice_count):
        voice_name = f"{voice_name_prefix}_{voice_index:04d}"
        fields = _voice_sequence_form_fields(
            scenario,
            voice_name=voice_name,
            ref_text=f"Voice cache pressure reference text {voice_index}.",
            speaker_description=f"Synthetic cache pressure voice number {voice_index}.",
        )
        created_voice_names.append(voice_name)
        if not await _post_expected_voice_upload(
            session,
            upload_url,
            scenario,
            result,
            form_fields=fields,
            upload_format="wav",
            content_type="audio/wav",
            upload_size=VOICE_SMALL_UPLOAD_BYTES,
            upload_case="format",
            operation=f"cache pressure upload {voice_index}",
        ):
            return False
        if not await _post_speech_with_uploaded_voice(
            session,
            spec,
            scenario,
            result,
            voice_name=voice_name,
            prompt=f"Cache pressure synthesis request {voice_index}.",
        ):
            return False
    return True


async def _revisit_cache_pressure_voices(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    voice_names: list[str],
) -> list[str] | None:
    revisit_voice_names = _cache_revisit_voice_names(voice_names)
    for voice_name in revisit_voice_names:
        if not await _post_speech_with_uploaded_voice(
            session,
            spec,
            scenario,
            result,
            voice_name=voice_name,
            prompt=f"Cache revisit synthesis request for {voice_name}.",
        ):
            return None
    return revisit_voice_names


async def _cleanup_cache_pressure_voices(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    created_voice_names: list[str],
) -> tuple[dict[str, int], int] | None:
    cleanup_count = len(created_voice_names)
    cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
    if cleanup_error is not None:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=cleanup_error,
        )
        return None
    created_voice_names.clear()
    after_cleanup_stats = await _voice_cache_stats_snapshot(
        session,
        spec,
        scenario,
        result,
        operation="after cache pressure cleanup",
    )
    if after_cleanup_stats is None:
        return None
    return after_cleanup_stats, cleanup_count


def _speaker_cap_sequence_config(
    scenario: Scenario,
    result: ScenarioResult,
) -> tuple[int, str, int] | None:
    attempt_count = _metadata_positive_int(scenario, "attempt_count")
    if attempt_count is None:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="speaker cap sequence requires positive integer attempt_count",
        )
        return None
    voice_name_prefix = str(scenario.planned_metadata.get("voice_name_prefix", ""))
    if not voice_name_prefix:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="speaker cap sequence requires voice_name_prefix",
        )
        return None
    speaker_max_uploaded = _metadata_positive_int(scenario, "speaker_max_uploaded")
    if speaker_max_uploaded is None:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="speaker cap sequence requires positive integer speaker_max_uploaded",
        )
        return None
    return attempt_count, voice_name_prefix, speaker_max_uploaded


async def _speaker_cap_uploaded_voices_after_stale_cleanup(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name_prefix: str,
) -> list[dict[str, Any]] | None:
    uploaded_voices = await _get_uploaded_voices(session, spec, scenario, result)
    if uploaded_voices is None:
        return None

    stale_voice_names = voice_contracts.uploaded_voice_names_with_prefix(
        uploaded_voices,
        voice_name_prefix,
    )
    cleanup_error = await _cleanup_voice_names(session, spec, stale_voice_names)
    if cleanup_error is not None:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=cleanup_error,
        )
        return None
    if not stale_voice_names:
        return uploaded_voices
    return await _get_uploaded_voices(session, spec, scenario, result)


def _mark_cleanup_error_if_primary_path_passed(
    result: ScenarioResult,
    cleanup_error: str,
) -> None:
    if result.error_class is None:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=cleanup_error,
        )


def _speaker_cap_attempts_cross_cap(
    result: ScenarioResult,
    *,
    uploaded_count: int,
    attempt_count: int,
    remaining_before_cap: int,
    speaker_max_uploaded: int,
) -> bool:
    if attempt_count > remaining_before_cap:
        return True
    _mark_protocol_error(
        result,
        status="invalid_benchmark_scenario",
        error=(
            "speaker cap sequence did not include enough upload attempts to "
            f"cross the cap (uploaded={uploaded_count}, cap={speaker_max_uploaded}, "
            f"attempts={attempt_count})"
        ),
    )
    return False


async def _fill_speaker_cap(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name_prefix: str,
    upload_count: int,
    created_voice_names: list[str],
) -> bool:
    for cap_index in range(upload_count):
        voice_name = f"{voice_name_prefix}_{cap_index:04d}"
        created_voice_names.append(voice_name)
        if not await _upload_expected_speaker_cap_voice(
            session,
            upload_url,
            scenario,
            result,
            voice_name=voice_name,
        ):
            return False
    return True


async def _upload_expected_speaker_cap_voice(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name: str,
) -> bool:
    form_fields = _speaker_cap_form_fields(scenario, voice_name)
    result.request_bytes += _voice_request_size(scenario, form_fields=form_fields)
    payload = await _post_voice_upload(
        session,
        upload_url,
        scenario,
        result,
        form_fields=form_fields,
    )
    if payload is None:
        return False
    return _require_voice_upload_identifier(
        payload,
        result,
        error=f"speaker cap upload response must include an identifier for {voice_name!r}",
    )


async def _expect_speaker_cap_rejection(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name: str,
    speaker_max_uploaded: int,
) -> bool:
    form_fields = _speaker_cap_form_fields(scenario, voice_name)
    body = request_body(scenario, form_fields=form_fields)
    result.request_bytes += _voice_request_size(scenario, form_fields=form_fields)
    try:
        status, response_body, headers = await _raw_post(session, upload_url, body)
    except ResponseBodyTooLarge as exc:
        _mark_voice_response_too_large(result, exc)
        return False
    result.http_status = status
    result.http_status_class = classify_http_status(status)
    result.response_headers = headers
    result.response_bytes += len(response_body)
    body_text = response_body.decode("utf-8", errors="replace")

    if status in UNSUPPORTED_HTTP_STATUSES:
        _mark_unsupported_contract(result, scenario, body=body_text)
        return False
    if 200 <= status < 300:
        _mark_protocol_error(
            result,
            status="unexpected_success",
            error=(
                "speaker cap overflow upload unexpectedly succeeded after "
                f"{speaker_max_uploaded} uploaded voices"
            ),
        )
        result.error_class = "unexpected_success"
        return False
    if status == 400:
        if not _is_valid_error_response(status, body_text, expected_status=400):
            _mark_protocol_error(
                result,
                status="invalid_error_response",
                error=(
                    "speaker cap overflow returned HTTP "
                    f"{status} without structured error JSON: {body_text}"
                ),
            )
            return False
        return True
    if 400 <= status < 500:
        _mark_protocol_error(
            result,
            status="invalid_error_response",
            error=(
                "speaker cap overflow must return HTTP 400 with structured error "
                f"JSON, got HTTP {status}: {body_text}"
            ),
        )
        return False

    result.status = "failed"
    result.success = False
    result.error_class = "server_error" if status >= 500 else "http_error"
    result.capability = "fail"
    result.error = body_text
    return False


async def _post_voice_upload(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    form_fields: Mapping[str, str],
) -> dict[str, Any] | None:
    return await _post_voice_upload_audio(
        session,
        upload_url,
        scenario,
        result,
        form_fields=form_fields,
        upload_format=str(scenario.planned_metadata.get("upload_format", "wav")),
        content_type=scenario.upload_content_type or "audio/wav",
        upload_size=scenario.upload_size_bytes,
        upload_case=str(scenario.planned_metadata.get("upload_case", "format")),
        filename=scenario.upload_filename or "audio.wav",
    )


async def _post_expected_voice_upload(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    form_fields: Mapping[str, str],
    upload_format: str,
    content_type: str,
    upload_size: int,
    upload_case: str,
    operation: str,
) -> bool:
    result.request_bytes += _voice_request_size_for(
        form_fields=form_fields,
        upload_size=upload_size,
    )
    payload = await _post_voice_upload_audio(
        session,
        upload_url,
        scenario,
        result,
        form_fields=form_fields,
        upload_format=upload_format,
        content_type=content_type,
        upload_size=upload_size,
        upload_case=upload_case,
        filename=f"{form_fields.get('name', 'voice')}.{upload_format}",
    )
    if payload is None:
        return False
    return _require_voice_upload_identifier(
        payload,
        result,
        error=f"{operation} response must include an identifier",
    )


async def _post_voice_upload_audio(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    form_fields: Mapping[str, str],
    upload_format: str,
    content_type: str,
    upload_size: int,
    upload_case: str,
    filename: str,
) -> dict[str, Any] | None:
    body = _voice_upload_body(
        scenario,
        form_fields=form_fields,
        upload_format=upload_format,
        content_type=content_type,
        upload_size=upload_size,
        upload_case=upload_case,
        filename=filename,
    )
    async with session.post(upload_url, data=body) as response:
        result.http_status = response.status
        result.http_status_class = classify_http_status(response.status)
        result.response_headers = dict(response.headers)
        response_body = await _read_voice_response_body(response, result)
        if response_body is None:
            return None
        if response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=response_body.decode("utf-8", errors="replace"),
            )
            return None
        if not 200 <= response.status < 300:
            _classify_http_failure(
                response.status,
                response_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return None
    return _json_object_from_bytes(
        response_body,
        result,
        status="invalid_voice_response",
        error_prefix="voice upload response returned invalid JSON",
    )


async def _post_speech_with_uploaded_voice(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name: str,
    prompt: str,
) -> bool:
    payload = {
        "model": spec.model_name,
        "input": prompt,
        "voice": voice_name,
        "response_format": "pcm",
        "speed": 1.0,
    }
    result.request_bytes += len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    speech_url = api_url(spec.base_url, "/v1/audio/speech")
    async with session.post(speech_url, json=payload) as response:
        result.http_status = response.status
        result.http_status_class = classify_http_status(response.status)
        result.response_headers = dict(response.headers)
        body = await _read_voice_response_body(response, result)
        if body is None:
            return False
        body_text = body.decode("utf-8", errors="replace")
        if response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=body_text,
                path="/v1/audio/speech",
            )
            return False
        if not 200 <= response.status < 300:
            _classify_http_failure(response.status, body_text, result, scenario)
            return False
        validation = validate_audio_response(
            body,
            response_format="pcm",
            content_type=response.headers.get("Content-Type"),
        )
        if not validation.ok:
            _mark_protocol_error(
                result,
                status="invalid_audio_response",
                error=(
                    "speech endpoint returned 2xx without PCM audio while using "
                    f"uploaded voice {voice_name!r}: {validation.error}"
                ),
            )
            return False
        result.audio_bytes += len(body)
        result.audio_duration_s += validation.duration_s
        return True


async def _expect_deleted_voice_speech_404(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    voice_name: str,
) -> bool:
    payload = {
        "model": spec.model_name,
        "input": "This request must fail because the uploaded voice was deleted.",
        "voice": voice_name,
        "response_format": "pcm",
        "speed": 1.0,
    }
    result.request_bytes += len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    speech_url = api_url(spec.base_url, "/v1/audio/speech")
    async with session.post(speech_url, json=payload) as response:
        result.http_status = response.status
        result.http_status_class = classify_http_status(response.status)
        result.response_headers = dict(response.headers)
        body = await _read_voice_response_body(response, result)
        if body is None:
            return False
        status = response.status
    body_text = body.decode("utf-8", errors="replace")
    if status != 404:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "speech with a deleted uploaded voice must return HTTP 404 "
                f"with structured error JSON; got HTTP {status}: {body_text}"
            ),
        )
        return False
    if not _is_valid_error_response(status, body_text, expected_status=404):
        _mark_protocol_error(
            result,
            status="invalid_error_response",
            error=(
                "speech with a deleted uploaded voice returned HTTP 404 without "
                f"OpenAI-compatible error JSON: {body_text}"
            ),
        )
        return False
    return True


async def _raw_post(
    session: aiohttp.ClientSession,
    url: str,
    body: aiohttp.FormData,
) -> RawVoiceResponse:
    async with session.post(url, data=body) as response:
        body = await read_response_body(response)
        return response.status, body, dict(response.headers)


async def _raw_delete(
    session: aiohttp.ClientSession,
    url: str,
) -> RawVoiceResponse:
    async with session.delete(url) as response:
        body = await read_response_body(response)
        return response.status, body, dict(response.headers)


async def _raw_get(
    session: aiohttp.ClientSession,
    url: str,
) -> RawVoiceResponse:
    async with session.get(url) as response:
        body = await read_response_body(response)
        return response.status, body, dict(response.headers)


async def _read_voice_response_body(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
) -> bytes | None:
    try:
        body = await read_response_body(response)
    except ResponseBodyTooLarge as exc:
        _mark_voice_response_too_large(result, exc)
        return None
    result.response_bytes += len(body)
    return body


def _mark_voice_response_too_large(
    result: ScenarioResult,
    exc: ResponseBodyTooLarge,
) -> None:
    result.response_bytes += exc.bytes_read
    _mark_protocol_error(
        result,
        status="response_too_large",
        error=(
            "HTTP response exceeded benchmark read cap "
            f"(bytes_read={exc.bytes_read}, max_bytes={exc.max_bytes})"
        ),
    )


def _merge_raw_voice_response(
    response: RawVoiceResponse,
    result: ScenarioResult,
) -> None:
    status, body, headers = response
    result.http_status = status
    result.http_status_class = classify_http_status(status)
    result.response_headers = headers
    result.response_bytes += len(body)


def _classify_voice_race_response(
    response: RawVoiceResponse,
    result: ScenarioResult,
    scenario: Scenario,
    *,
    operation: str,
    requires_voice_identifier: bool = False,
    requires_delete_success: bool = False,
) -> bool:
    status, body, _ = response
    body_text = body.decode("utf-8", errors="replace")
    if status in UNSUPPORTED_HTTP_STATUSES:
        _mark_unsupported_contract(result, scenario, body=body_text)
        return False
    if not 200 <= status < 300:
        _classify_http_failure(status, body_text, result, scenario)
        if result.error_class == "http_error":
            result.error = f"{operation} failed: {body_text}"
        return False
    if requires_voice_identifier:
        payload = _json_object_from_bytes(
            body,
            result,
            status="invalid_voice_response",
            error_prefix=f"{operation} response returned invalid JSON",
        )
        if payload is None:
            return False
        if not voice_contracts.voice_upload_response_identifier(payload):
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error=f"{operation} response must include an identifier",
            )
            return False
    if requires_delete_success and not voice_contracts.is_valid_voice_delete_success(
        body
    ):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=f"{operation} response must be success JSON",
        )
        return False
    return True


def _require_voice_upload_identifier(
    payload: dict[str, Any],
    result: ScenarioResult,
    *,
    error: str,
) -> bool:
    if voice_contracts.voice_upload_response_identifier(payload):
        return True
    _mark_protocol_error(
        result,
        status="invalid_voice_response",
        error=error,
    )
    return False


def _validate_overwritten_voice_entry(
    entries: list[dict],
    result: ScenarioResult,
    *,
    voice_name: str,
    expected_speaker_description: str,
) -> bool:
    if len(entries) != 1:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "same-name voice overwrite must leave exactly one uploaded "
                f"voice named {voice_name!r}; observed={len(entries)}"
            ),
        )
        return False
    if entries[0].get("speaker_description") != expected_speaker_description:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "same-name voice overwrite must expose the second upload metadata "
                f"for {voice_name!r}"
            ),
        )
        return False
    return True


def _validate_uploaded_voice_metadata_sequence(
    payload: dict[str, Any],
    expected_entries: Mapping[str, Mapping[str, str]],
    result: ScenarioResult,
) -> bool:
    if not voice_contracts.is_valid_voice_list_response(payload):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice metadata sequence requires voice list response with "
                "valid preset and uploaded voice metadata"
            ),
        )
        return False
    for voice_name, expected_fields in expected_entries.items():
        entries = voice_contracts.uploaded_voice_entries(payload, voice_name)
        if len(entries) != 1:
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error=(
                    "voice metadata sequence must expose exactly one uploaded "
                    f"voice named {voice_name!r}; observed={len(entries)}"
                ),
            )
            return False
        entry = entries[0]
        for key, expected_value in expected_fields.items():
            if entry.get(key) != expected_value:
                _mark_protocol_error(
                    result,
                    status="invalid_voice_response",
                    error=(
                        "voice metadata sequence did not preserve "
                        f"{key} for {voice_name!r}"
                    ),
                )
                return False
    return True


def _validate_delete_invalidation_counter(
    before_delete: dict[str, Any],
    after_delete: dict[str, Any],
    result: ScenarioResult,
) -> bool:
    before_counter = _recursive_counter_value(
        before_delete, "delete_invalidation_counter"
    )
    after_counter = _recursive_counter_value(
        after_delete, "delete_invalidation_counter"
    )
    if before_counter is None or after_counter is None:
        return True
    if after_counter > before_counter:
        return True
    _mark_protocol_error(
        result,
        status="invalid_voice_response",
        error=(
            "voice delete did not advance delete_invalidation_counter "
            f"(before={before_counter}, after={after_counter})"
        ),
    )
    return False


def _recursive_counter_value(payload: Any, key: str) -> int | None:
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        for child in payload.values():
            found = _recursive_counter_value(child, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for child in payload:
            found = _recursive_counter_value(child, key)
            if found is not None:
                return found
    return None


async def _get_voice_list(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> dict[str, Any] | None:
    list_url = api_url(spec.base_url, "/v1/audio/voices")
    async with session.get(list_url) as list_response:
        result.http_status = list_response.status
        result.http_status_class = classify_http_status(list_response.status)
        result.response_headers = dict(list_response.headers)
        list_body = await _read_voice_response_body(list_response, result)
        if list_body is None:
            return None
        if list_response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=list_body.decode("utf-8", errors="replace"),
            )
            return None
        if not 200 <= list_response.status < 300:
            _classify_http_failure(
                list_response.status,
                list_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return None
    return _json_object_from_bytes(
        list_body,
        result,
        status="invalid_voice_response",
        error_prefix="voice list response returned invalid JSON",
    )


async def _get_uploaded_voices(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> list[dict[str, Any]] | None:
    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return None
    if not voice_contracts.is_valid_voice_list_response(voice_list):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice list response must be an object with voices and "
                "uploaded_voices before speaker cap validation"
            ),
        )
        return None
    uploaded_voices = voice_list["uploaded_voices"]
    return [voice for voice in uploaded_voices if isinstance(voice, dict)]


async def _require_uploaded_voice_present(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    voice_name: str,
    *,
    operation: str,
) -> dict[str, Any] | None:
    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return None
    if not voice_contracts.is_valid_voice_list_response(voice_list):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                f"{operation} requires voice list response with valid preset "
                "and uploaded voice metadata"
            ),
        )
        return None
    entries = voice_contracts.uploaded_voice_entries(voice_list, voice_name)
    if len(entries) != 1:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                f"{operation} must expose exactly one uploaded voice named "
                f"{voice_name!r}; observed={len(entries)}"
            ),
        )
        return None
    return voice_list


def _require_voice_absent_in_list(
    voice_list: dict[str, Any],
    result: ScenarioResult,
    voice_name: str,
) -> bool:
    if not voice_contracts.is_valid_voice_list_response(voice_list):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "post-delete voice list response must include valid voices and "
                "uploaded_voices metadata"
            ),
        )
        return False
    entries = voice_contracts.uploaded_voice_entries(voice_list, voice_name)
    if entries:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice delete reported success but uploaded_voices still contains "
                f"{voice_name!r}"
            ),
        )
        return False
    return True


async def _delete_voice_by_name(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    voice_name: str,
) -> bool:
    delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
    async with session.delete(delete_url) as delete_response:
        result.http_status = delete_response.status
        result.http_status_class = classify_http_status(delete_response.status)
        result.response_headers = dict(delete_response.headers)
        delete_body = await _read_voice_response_body(delete_response, result)
        if delete_body is None:
            return False
        if delete_response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=delete_body.decode("utf-8", errors="replace"),
            )
            return False
        if not 200 <= delete_response.status < 300:
            _classify_http_failure(
                delete_response.status,
                delete_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return False
        if not voice_contracts.is_valid_voice_delete_success(delete_body):
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error="voice cleanup delete response must be success JSON",
            )
            return False
    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return False
    if not _require_voice_absent_in_list(voice_list, result, voice_name):
        return False
    return True


async def _cleanup_voice_names(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    voice_names: list[str],
) -> str | None:
    failures: list[str] = []
    for voice_name in reversed(voice_names):
        delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
        try:
            status, body, _ = await _raw_delete(session, delete_url)
        except (aiohttp.ClientError, asyncio.TimeoutError, ResponseBodyTooLarge) as exc:
            failures.append(f"voice cleanup failed for {voice_name!r}: {exc}")
            continue
        body_text = body.decode("utf-8", errors="replace")
        if status == 404:
            absent_error = await _cleanup_voice_absence_error(session, spec, voice_name)
            if absent_error is not None:
                failures.append(absent_error)
            continue
        if (
            not 200 <= status < 300
            or not voice_contracts.is_valid_voice_delete_success(body)
        ):
            failures.append(
                f"voice cleanup failed for {voice_name!r}: "
                f"status={status}, body={body_text}"
            )
            continue
        absent_error = await _cleanup_voice_absence_error(session, spec, voice_name)
        if absent_error is not None:
            failures.append(absent_error)
    return _voice_cleanup_error_message(failures)


def _voice_cleanup_error_message(failures: list[str]) -> str | None:
    if not failures:
        return None
    visible_failures = failures[:MAX_CLEANUP_FAILURE_DETAILS]
    hidden_count = len(failures) - len(visible_failures)
    suffix = f"; ... {hidden_count} additional cleanup failures" if hidden_count else ""
    return "; ".join(visible_failures) + suffix


async def _cleanup_voice_absence_error(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    voice_name: str,
) -> str | None:
    list_url = api_url(spec.base_url, "/v1/audio/voices")
    try:
        status, body, _ = await _raw_get(session, list_url)
    except (aiohttp.ClientError, asyncio.TimeoutError, ResponseBodyTooLarge) as exc:
        return f"voice cleanup list verification failed for {voice_name!r}: {exc}"
    body_text = body.decode("utf-8", errors="replace")
    if status in UNSUPPORTED_HTTP_STATUSES:
        return (
            f"voice cleanup list verification failed for {voice_name!r}: "
            f"voice list endpoint unsupported after delete, status={status}, body={body_text}"
        )
    if not 200 <= status < 300:
        return (
            f"voice cleanup list verification failed for {voice_name!r}: "
            f"status={status}, body={body_text}"
        )
    try:
        payload = json.loads(body_text) if body else {}
    except json.JSONDecodeError as exc:
        return (
            f"voice cleanup list verification failed for {voice_name!r}: "
            f"invalid JSON: {exc}"
        )
    if not isinstance(payload, dict):
        return (
            f"voice cleanup list verification failed for {voice_name!r}: "
            "voice list response must be a JSON object"
        )
    if not voice_contracts.is_valid_voice_list_response(payload):
        return (
            f"voice cleanup list verification failed for {voice_name!r}: "
            "voice list response must include valid voices and uploaded_voices metadata"
        )
    if voice_contracts.uploaded_voice_entries(payload, voice_name):
        return (
            f"voice cleanup failed for {voice_name!r}: delete returned success "
            "but uploaded_voices still contains the voice"
        )
    return None


def handle_voice_success(
    body: bytes, result: ScenarioResult, scenario: Scenario
) -> None:
    payload = _json_from_bytes(
        body,
        result,
        status="invalid_voice_response",
        error_prefix="voice endpoint returned invalid JSON",
        default_empty={},
    )
    if payload is None:
        return
    if scenario.capability_key == "voices.list":
        if voice_contracts.is_valid_voice_list_response(payload):
            _mark_success(result, capability="pass")
            return
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice list response must be an object with voices and "
                "uploaded_voices; uploaded entries require name, consent, "
                "created_at, file_size, and mime_type"
            ),
        )
        return
    if scenario.capability_key in {"voices.upload", "voices.lifecycle"}:
        if voice_contracts.voice_upload_response_identifier(payload):
            _mark_success(result, capability="pass")
            return
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error="voice upload response must include id, voice_id, or name",
        )
        return
    _mark_success(result, capability="pass")
