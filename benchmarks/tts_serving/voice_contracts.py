# SPDX-License-Identifier: Apache-2.0
"""Voice response contracts for the TTS serving benchmark."""

from __future__ import annotations

import json
from typing import Any

from benchmarks.tts_serving.http_contracts import _mark_protocol_error
from benchmarks.tts_serving.metrics import ScenarioResult

VOICE_CACHE_STATS_KEY = "cache_stats"
VOICE_CACHE_OBSERVABILITY_COUNTERS = (
    "entries",
    "memory_bytes",
    "max_bytes",
    "eviction_count",
    "hit_count",
    "miss_count",
    "delete_invalidation_counter",
)


def is_valid_voice_list_response(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    voices = payload.get("voices")
    uploaded_voices = payload.get("uploaded_voices")
    if not isinstance(voices, list) or not isinstance(uploaded_voices, list):
        return False
    return all(_is_valid_preset_voice(item) for item in voices) and all(
        _is_valid_uploaded_voice_metadata(item) for item in uploaded_voices
    )


def voice_upload_response_identifier(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("success") is False:
        return None
    for key in ("id", "voice_id", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    voice = payload.get("voice")
    if isinstance(voice, str) and voice:
        return voice
    if isinstance(voice, dict):
        for key in ("id", "voice_id", "name"):
            value = voice.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def is_voice_overwrite_ack(payload: dict[str, Any]) -> bool:
    if payload.get("overwritten") is True or payload.get("replaced") is True:
        return True
    for key in ("warning", "message"):
        value = payload.get(key)
        if isinstance(value, str) and "overwrit" in value.lower():
            return True
    return False


def uploaded_voice_entries(payload: dict[str, Any], voice_name: str) -> list[dict]:
    uploaded_voices = payload.get("uploaded_voices")
    if not isinstance(uploaded_voices, list):
        return []
    return [
        item
        for item in uploaded_voices
        if isinstance(item, dict) and item.get("name") == voice_name
    ]


def uploaded_voice_names_with_prefix(
    uploaded_voices: list[dict[str, Any]],
    voice_name_prefix: str,
) -> list[str]:
    names: list[str] = []
    for voice in uploaded_voices:
        name = voice.get("name")
        if isinstance(name, str) and name.startswith(voice_name_prefix):
            names.append(name)
    return names


def is_valid_voice_delete_success(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace")) if body else {}
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return False
    if payload.get("deleted") is True:
        return True
    return payload.get("success") is True or any(
        isinstance(payload.get(key), str) and payload[key]
        for key in ("id", "voice_id", "name")
    )


def require_voice_cache_stats(
    payload: dict[str, Any],
    result: ScenarioResult,
    *,
    operation: str,
) -> dict[str, int] | None:
    if not is_valid_voice_list_response(payload):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                f"{operation} requires voice list response with valid preset "
                "and uploaded voice metadata"
            ),
        )
        return None
    cache_stats = payload.get(VOICE_CACHE_STATS_KEY)
    if not isinstance(cache_stats, dict):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=f"{operation} requires voice list response with cache_stats object",
        )
        return None

    parsed: dict[str, int] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for key in VOICE_CACHE_OBSERVABILITY_COUNTERS:
        value = cache_stats.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            invalid.append(key)
            continue
        parsed[key] = value
    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if invalid:
            details.append(f"invalid={invalid}")
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=f"{operation} returned invalid cache_stats: {', '.join(details)}",
        )
        return None
    if parsed["max_bytes"] <= 0:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=f"{operation} cache_stats.max_bytes must be greater than 0",
        )
        return None
    if parsed["memory_bytes"] > parsed["max_bytes"]:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                f"{operation} cache_stats.memory_bytes exceeds max_bytes "
                f"(memory_bytes={parsed['memory_bytes']}, "
                f"max_bytes={parsed['max_bytes']})"
            ),
        )
        return None
    return parsed


def validate_cache_pressure_unique_stats(
    before: dict[str, int],
    after: dict[str, int],
    *,
    voice_count: int,
    result: ScenarioResult,
) -> bool:
    miss_delta = after["miss_count"] - before["miss_count"]
    if miss_delta < voice_count:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice cache pressure must record cache misses for unique voices "
                f"(miss_delta={miss_delta}, voice_count={voice_count})"
            ),
        )
        return False
    eviction_delta = after["eviction_count"] - before["eviction_count"]
    if eviction_delta <= 0 and _voice_cache_at_capacity(after):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice cache pressure reached cache capacity without advancing "
                "eviction_count "
                f"(before={before['eviction_count']}, "
                f"after={after['eviction_count']}, "
                f"memory_bytes={after['memory_bytes']}, "
                f"max_bytes={after['max_bytes']})"
            ),
        )
        return False
    if after["memory_bytes"] <= 0 or after["entries"] <= 0:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice cache pressure must populate cache_stats entries and "
                f"memory_bytes (entries={after['entries']}, "
                f"memory_bytes={after['memory_bytes']})"
            ),
        )
        return False
    return True


def validate_cache_pressure_revisit_stats(
    before: dict[str, int],
    after: dict[str, int],
    *,
    revisit_count: int,
    result: ScenarioResult,
) -> bool:
    lookup_delta = _cache_lookup_count(after) - _cache_lookup_count(before)
    if lookup_delta < revisit_count:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice cache revisit requests must advance hit/miss counters "
                f"(lookup_delta={lookup_delta}, revisit_count={revisit_count})"
            ),
        )
        return False
    if after["hit_count"] <= before["hit_count"]:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice cache revisit requests must record at least one hit "
                f"(before={before['hit_count']}, after={after['hit_count']})"
            ),
        )
        return False
    return True


def validate_cache_pressure_cleanup_stats(
    before: dict[str, int],
    after: dict[str, int],
    *,
    deleted_count: int,
    result: ScenarioResult,
) -> bool:
    invalidation_delta = (
        after["delete_invalidation_counter"] - before["delete_invalidation_counter"]
    )
    if invalidation_delta <= 0:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice cache cleanup must invalidate deleted voice artifacts "
                f"(invalidation_delta={invalidation_delta}, "
                f"deleted_count={deleted_count})"
            ),
        )
        return False
    return True


def _is_valid_preset_voice(item: Any) -> bool:
    if isinstance(item, str):
        return bool(item)
    return isinstance(item, dict) and any(
        isinstance(item.get(key), str) and item[key] for key in ("name", "voice", "id")
    )


def _is_valid_uploaded_voice_metadata(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required = ("name", "consent", "created_at", "file_size", "mime_type")
    return (
        all(key in item for key in required)
        and isinstance(item["name"], str)
        and bool(item["name"])
        and isinstance(item["consent"], (bool, str))
        and bool(str(item["consent"]))
        and isinstance(item["created_at"], (str, int, float))
        and bool(str(item["created_at"]))
        and _is_nonnegative_file_size(item["file_size"])
        and isinstance(item["mime_type"], str)
        and bool(item["mime_type"])
    )


def _is_nonnegative_file_size(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value >= 0
    if isinstance(value, str) and value.strip().isdigit():
        return int(value) >= 0
    return False


def _cache_lookup_count(stats: dict[str, int]) -> int:
    return stats["hit_count"] + stats["miss_count"]


def _voice_cache_at_capacity(stats: dict[str, int]) -> bool:
    return stats["memory_bytes"] >= stats["max_bytes"]
