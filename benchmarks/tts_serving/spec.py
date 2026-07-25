# SPDX-License-Identifier: Apache-2.0
"""Spec parsing for the TTS serving benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VALID_TEST_TYPES = {"engine", "e2e", "external"}
DEFAULT_PROFILE = "stress"
VALID_PROFILES = {DEFAULT_PROFILE}
VALID_LOAD_MODES = {"closed_loop", "open_loop", "ramp", "burst", "soak"}
VALID_ARRIVAL_DISTRIBUTIONS = {"deterministic", "poisson"}
DEFAULT_ENDPOINTS = ("speech", "speech_stream", "voices", "batch", "websocket")
DEFAULT_SPEAKER_MAX_UPLOADED = 1000
DEFAULT_SEED = 0
SAFE_STAGE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SENSITIVE_METADATA_KEY_TERMS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "headers",
    "password",
    "secret",
    "test_env",
    "token",
)
REDACTED_METADATA_VALUE = "[REDACTED]"
TOP_LEVEL_KEYS = {
    "base_url",
    "model_name",
    "test_type",
    "run_id",
    "seed",
    "auth",
    "params",
}
AUTH_KEYS = {"api_key_env"}
PARAM_KEYS = {
    "profile",
    "total_requests",
    "max_concurrency",
    "concurrency_levels",
    "load_stages",
    "request_rate",
    "timeout_s",
    "enabled_endpoints",
    "seedtts_ref_audio",
    "seedtts_ref_text",
    "file_ref_audio",
    "file_ref_text",
    "voice_cache_pressure_voice_count",
    "voice_speaker_cap_count",
    "speaker_max_uploaded",
    "provider_label",
    "implementation_label",
}
LOAD_STAGE_KEYS = {
    "id",
    "mode",
    "request_count",
    "total_requests",
    "max_concurrency",
    "concurrency",
    "request_rate",
    "start_request_rate",
    "duration_s",
    "arrival_distribution",
    "enabled_endpoints",
    "voice_cache_pressure_voice_count",
    "voice_speaker_cap_count",
    "speaker_max_uploaded",
}


class SpecError(ValueError):
    """Raised when /etc/benchmark/spec.json is invalid."""


@dataclass(frozen=True)
class AuthSpec:
    api_key_env: str | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> AuthSpec:
        if obj is None:
            return cls()
        if not isinstance(obj, dict):
            raise SpecError("auth must be an object when provided")
        _reject_unknown_keys(obj, AUTH_KEYS, "auth")
        api_key_env = obj.get("api_key_env")
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not api_key_env.strip():
                raise SpecError("auth.api_key_env must be a non-empty string")
            api_key_env = api_key_env.strip()
        return cls(api_key_env=api_key_env)


@dataclass(frozen=True)
class LoadStage:
    id: str
    mode: str
    request_count: int
    max_concurrency: int
    request_rate: float = float("inf")
    start_request_rate: float | None = None
    duration_s: float | None = None
    arrival_distribution: str = "deterministic"
    enabled_endpoints: tuple[str, ...] | None = None
    voice_cache_pressure_voice_count: int = 0
    voice_speaker_cap_count: int = 0
    speaker_max_uploaded: int | None = None

    @classmethod
    def from_obj(cls, obj: Any, *, index: int) -> LoadStage:
        if not isinstance(obj, dict):
            raise SpecError("params.load_stages entries must be objects")
        _reject_unknown_keys(obj, LOAD_STAGE_KEYS, "params.load_stages[]")
        stage_id = _safe_stage_id(_str_value(obj, "id", f"stage-{index + 1}"))
        mode = _str_value(obj, "mode", "closed_loop")
        if mode not in VALID_LOAD_MODES:
            raise SpecError(
                f"params.load_stages[].mode must be one of {sorted(VALID_LOAD_MODES)}"
            )
        if mode == "soak" and "request_rate" in obj:
            raise SpecError(
                "params.load_stages[].request_rate is derived from request_count "
                "and duration_s for soak stages"
            )

        request_count = _positive_int(
            obj,
            "request_count",
            _positive_int(obj, "total_requests", 100, path="params.load_stages[]"),
            path="params.load_stages[]",
        )
        max_concurrency = _positive_int(
            obj,
            "max_concurrency",
            _positive_int(obj, "concurrency", 8, path="params.load_stages[]"),
            path="params.load_stages[]",
        )
        request_rate = _request_rate(
            obj.get("request_rate", float("inf")),
            path="params.load_stages[].request_rate",
        )
        start_request_rate = _optional_request_rate(
            obj.get("start_request_rate"),
            path="params.load_stages[].start_request_rate",
        )
        duration_s = _optional_positive_float(obj.get("duration_s"), "duration_s")
        arrival_distribution = _str_value(obj, "arrival_distribution", "deterministic")
        if arrival_distribution not in VALID_ARRIVAL_DISTRIBUTIONS:
            raise SpecError(
                "params.load_stages[].arrival_distribution must be one of "
                f"{sorted(VALID_ARRIVAL_DISTRIBUTIONS)}"
            )
        if mode in {"open_loop", "ramp"} and request_rate == float("inf"):
            raise SpecError(
                "params.load_stages[].request_rate must be finite for " f"{mode} stages"
            )
        if mode == "ramp" and start_request_rate is None:
            raise SpecError(
                "params.load_stages[].start_request_rate is required for ramp stages"
            )
        if mode == "soak" and duration_s is None:
            raise SpecError(
                "params.load_stages[].duration_s is required for soak stages"
            )
        if mode == "soak":
            assert duration_s is not None
            request_rate = request_count / duration_s
        enabled_endpoints = _optional_enabled_endpoints(
            obj.get("enabled_endpoints"),
            "params.load_stages[].enabled_endpoints",
        )
        voice_cache_pressure_voice_count = _nonnegative_int_value(
            obj.get("voice_cache_pressure_voice_count", 0),
            "params.load_stages[].voice_cache_pressure_voice_count",
        )
        voice_speaker_cap_count = _nonnegative_int_value(
            obj.get("voice_speaker_cap_count", 0),
            "params.load_stages[].voice_speaker_cap_count",
        )
        speaker_max_uploaded = _optional_positive_int_value(
            obj.get("speaker_max_uploaded"),
            "params.load_stages[].speaker_max_uploaded",
        )
        return cls(
            id=stage_id,
            mode=mode,
            request_count=request_count,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
            start_request_rate=start_request_rate,
            duration_s=duration_s,
            arrival_distribution=arrival_distribution,
            enabled_endpoints=enabled_endpoints,
            voice_cache_pressure_voice_count=voice_cache_pressure_voice_count,
            voice_speaker_cap_count=voice_speaker_cap_count,
            speaker_max_uploaded=speaker_max_uploaded,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "request_count": self.request_count,
            "max_concurrency": self.max_concurrency,
            "request_rate": (
                "inf" if self.request_rate == float("inf") else self.request_rate
            ),
            "start_request_rate": self.start_request_rate,
            "duration_s": self.duration_s,
            "arrival_distribution": self.arrival_distribution,
            "enabled_endpoints": (
                list(self.enabled_endpoints)
                if self.enabled_endpoints is not None
                else None
            ),
            "voice_cache_pressure_voice_count": self.voice_cache_pressure_voice_count,
            "voice_speaker_cap_count": self.voice_speaker_cap_count,
            "speaker_max_uploaded": self.speaker_max_uploaded,
        }


@dataclass(frozen=True)
class BenchmarkParams:
    profile: str = DEFAULT_PROFILE
    total_requests: int = 100
    max_concurrency: int = 8
    concurrency_levels: tuple[int, ...] | None = None
    load_stages: tuple[LoadStage, ...] = field(default_factory=tuple)
    request_rate: float = float("inf")
    timeout_s: int = 120
    enabled_endpoints: tuple[str, ...] = DEFAULT_ENDPOINTS
    seedtts_ref_audio: str | None = None
    seedtts_ref_text: str | None = None
    file_ref_audio: str | None = None
    file_ref_text: str | None = None
    voice_cache_pressure_voice_count: int = 0
    voice_speaker_cap_count: int = 0
    speaker_max_uploaded: int = DEFAULT_SPEAKER_MAX_UPLOADED
    provider_label: str | None = None
    implementation_label: str | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> BenchmarkParams:
        if obj is None:
            obj = {}
        if not isinstance(obj, dict):
            raise SpecError("params must be an object when provided")
        _reject_unknown_keys(obj, PARAM_KEYS, "params")

        profile = _str_value(obj, "profile", cls.profile)
        if profile not in VALID_PROFILES:
            raise SpecError(f"params.profile must be one of {sorted(VALID_PROFILES)}")

        total_requests = _positive_int(obj, "total_requests", cls.total_requests)
        max_concurrency = _positive_int(obj, "max_concurrency", cls.max_concurrency)
        concurrency_levels = _concurrency_levels(obj.get("concurrency_levels"))
        timeout_s = _positive_int(obj, "timeout_s", cls.timeout_s)
        request_rate = _request_rate(obj.get("request_rate", cls.request_rate))
        enabled = _enabled_endpoints(
            obj.get("enabled_endpoints", list(DEFAULT_ENDPOINTS)),
            "params.enabled_endpoints",
        )
        load_stages = _load_stages(
            obj.get("load_stages"),
            total_requests=total_requests,
            max_concurrency=max_concurrency,
            concurrency_levels=concurrency_levels,
            request_rate=request_rate,
        )
        voice_cache_pressure_voice_count = _nonnegative_int(
            obj,
            "voice_cache_pressure_voice_count",
            cls.voice_cache_pressure_voice_count,
        )
        voice_speaker_cap_count = _nonnegative_int(
            obj,
            "voice_speaker_cap_count",
            cls.voice_speaker_cap_count,
        )
        speaker_max_uploaded = _positive_int(
            obj,
            "speaker_max_uploaded",
            cls.speaker_max_uploaded,
        )
        _validate_voice_speaker_cap_stages(
            load_stages,
            default_enabled_endpoints=enabled,
            default_voice_speaker_cap_count=voice_speaker_cap_count,
        )

        return cls(
            profile=profile,
            total_requests=total_requests,
            max_concurrency=(
                max(stage.max_concurrency for stage in load_stages)
                if load_stages
                else max_concurrency
            ),
            concurrency_levels=concurrency_levels,
            load_stages=load_stages,
            request_rate=request_rate,
            timeout_s=timeout_s,
            enabled_endpoints=enabled,
            seedtts_ref_audio=_optional_str(obj, "seedtts_ref_audio"),
            seedtts_ref_text=_optional_str(obj, "seedtts_ref_text"),
            file_ref_audio=_optional_file_uri(obj, "file_ref_audio"),
            file_ref_text=_optional_str(obj, "file_ref_text"),
            voice_cache_pressure_voice_count=voice_cache_pressure_voice_count,
            voice_speaker_cap_count=voice_speaker_cap_count,
            speaker_max_uploaded=speaker_max_uploaded,
            provider_label=_optional_str(obj, "provider_label"),
            implementation_label=_optional_str(obj, "implementation_label"),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    base_url: str
    model_name: str
    test_type: str = "engine"
    run_id: str | None = None
    seed: int = DEFAULT_SEED
    auth: AuthSpec = field(default_factory=AuthSpec)
    params: BenchmarkParams = field(default_factory=BenchmarkParams)
    platform_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> BenchmarkSpec:
        if not isinstance(obj, dict):
            raise SpecError("spec must be a JSON object")
        platform_metadata = {
            key: value for key, value in obj.items() if key not in TOP_LEVEL_KEYS
        }
        base_url = _validate_base_url(_required_str(obj, "base_url"))
        model_name = _required_str(obj, "model_name")
        test_type = _str_value(obj, "test_type", "engine")
        if test_type not in VALID_TEST_TYPES:
            raise SpecError(f"test_type must be one of {sorted(VALID_TEST_TYPES)}")
        seed = obj.get("seed", DEFAULT_SEED)
        if not isinstance(seed, int):
            raise SpecError("seed must be an integer")
        run_id = _optional_str(obj, "run_id")
        return cls(
            base_url=base_url,
            model_name=model_name,
            test_type=test_type,
            run_id=run_id,
            seed=seed,
            auth=AuthSpec.from_obj(obj.get("auth")),
            params=BenchmarkParams.from_obj(obj.get("params")),
            platform_metadata=platform_metadata,
        )


def load_spec(path: str | Path) -> BenchmarkSpec:
    spec_path = Path(path)
    if not spec_path.is_file():
        raise SpecError(f"spec file not found: {spec_path}")
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON in spec file: {exc}") from exc
    return BenchmarkSpec.from_obj(raw)


def workload_spec_hash(spec: BenchmarkSpec) -> str:
    payload = _canonical_json_value(asdict(spec))
    payload.pop("base_url", None)
    payload.pop("run_id", None)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def redact_sensitive_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                REDACTED_METADATA_VALUE
                if _is_sensitive_metadata_key(str(key))
                else redact_sensitive_metadata(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_metadata(item) for item in value]
    return value


def _is_sensitive_metadata_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(term in normalized for term in SENSITIVE_METADATA_KEY_TERMS)


def _required_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{key} must be a non-empty string")
    return value


def _reject_unknown_keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise SpecError(f"unknown {path} keys: {unknown}")


def _optional_str(obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{key} must be a non-empty string when provided")
    return value


def _optional_file_uri(obj: dict[str, Any], key: str) -> str | None:
    value = _optional_str(obj, key)
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file" or not parsed.path:
        raise SpecError(f"params.{key} must be a file:// URI when provided")
    return value


def _str_value(obj: dict[str, Any], key: str, default: str) -> str:
    value = obj.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{key} must be a non-empty string")
    return value


def _validate_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SpecError("base_url must be an absolute http(s) URL")
    return base_url


def _safe_stage_id(value: str) -> str:
    if not SAFE_STAGE_ID_RE.fullmatch(value):
        raise SpecError(
            "params.load_stages[].id must contain only ASCII letters, digits, "
            "'.', '_', or '-', and must be at most 64 characters"
        )
    return value


def _positive_int(
    obj: dict[str, Any],
    key: str,
    default: int,
    *,
    path: str = "params",
) -> int:
    value = obj.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecError(f"{path}.{key} must be a positive integer")
    return value


def _nonnegative_int(obj: dict[str, Any], key: str, default: int) -> int:
    return _nonnegative_int_value(obj.get(key, default), f"params.{key}")


def _nonnegative_int_value(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecError(f"{key} must be a non-negative integer")
    return value


def _optional_positive_int_value(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecError(f"{key} must be a positive integer")
    return value


def _request_rate(value: Any, *, path: str = "params.request_rate") -> float:
    if value == "inf":
        return float("inf")
    if not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0:
        return float(value)
    raise SpecError(f"{path} must be a positive number or 'inf'")


def _optional_request_rate(
    value: Any,
    *,
    path: str = "params.load_stages[].start_request_rate",
) -> float | None:
    if value is None:
        return None
    rate = _request_rate(value, path=path)
    if rate == float("inf"):
        raise SpecError(f"{path} must be finite")
    return rate


def _optional_positive_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SpecError(f"params.load_stages[].{key} must be a positive number")
    return float(value)


def _concurrency_levels(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise SpecError("params.concurrency_levels must be a non-empty list")
    levels: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise SpecError("params.concurrency_levels must contain positive integers")
        if item not in levels:
            levels.append(item)
    return tuple(levels)


def _enabled_endpoints(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecError(f"{key} must be a list of strings")
    if not value:
        raise SpecError(f"{key} must not be empty")
    unknown = sorted(set(value) - set(DEFAULT_ENDPOINTS))
    if unknown:
        raise SpecError(f"unknown {key}: {unknown}")
    deduped = list(dict.fromkeys(value))
    return tuple(deduped)


def _optional_enabled_endpoints(value: Any, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _enabled_endpoints(value, key)


def _load_stages(
    value: Any,
    *,
    total_requests: int,
    max_concurrency: int,
    concurrency_levels: tuple[int, ...] | None,
    request_rate: float,
) -> tuple[LoadStage, ...]:
    if value is not None:
        if not isinstance(value, list) or not value:
            raise SpecError("params.load_stages must be a non-empty list")
        stages = tuple(
            LoadStage.from_obj(item, index=index) for index, item in enumerate(value)
        )
        _validate_unique_stage_ids(stages)
        return stages

    levels = concurrency_levels or (max_concurrency,)
    return tuple(
        LoadStage(
            id=f"c{level}",
            mode="closed_loop",
            request_count=total_requests,
            max_concurrency=level,
            request_rate=request_rate,
            enabled_endpoints=None,
        )
        for level in levels
    )


def _validate_unique_stage_ids(stages: tuple[LoadStage, ...]) -> None:
    seen: set[str] = set()
    for stage in stages:
        if stage.id in seen:
            raise SpecError(f"duplicate params.load_stages[].id: {stage.id}")
        seen.add(stage.id)


def _validate_voice_speaker_cap_stages(
    stages: tuple[LoadStage, ...],
    *,
    default_enabled_endpoints: tuple[str, ...],
    default_voice_speaker_cap_count: int,
) -> None:
    default_count_stage_ids: list[str] = []
    for stage in stages:
        voice_speaker_cap_count = _effective_voice_speaker_cap_count(
            stage,
            default_enabled_endpoints=default_enabled_endpoints,
            default_voice_speaker_cap_count=default_voice_speaker_cap_count,
        )
        if not voice_speaker_cap_count:
            continue
        if not stage.voice_speaker_cap_count:
            default_count_stage_ids.append(stage.id)
        endpoints = stage.enabled_endpoints or default_enabled_endpoints
        if endpoints != ("voices",):
            raise SpecError(
                "voice_speaker_cap_count stages must enable only ['voices']"
            )
        if stage.mode != "closed_loop":
            raise SpecError("voice_speaker_cap_count stages must use closed_loop mode")
        if stage.max_concurrency != 1:
            raise SpecError("voice_speaker_cap_count stages must use max_concurrency=1")
        if stage.request_count != 2:
            raise SpecError("voice_speaker_cap_count stages must use request_count=2")
    if default_voice_speaker_cap_count and len(default_count_stage_ids) != 1:
        raise SpecError(
            "params.voice_speaker_cap_count requires exactly one dedicated load "
            "stage with enabled_endpoints ['voices']; prefer stage-level "
            "voice_speaker_cap_count when multiple voices stages exist"
        )


def _effective_voice_speaker_cap_count(
    stage: LoadStage,
    *,
    default_enabled_endpoints: tuple[str, ...],
    default_voice_speaker_cap_count: int,
) -> int:
    if stage.voice_speaker_cap_count:
        return stage.voice_speaker_cap_count
    endpoints = stage.enabled_endpoints or default_enabled_endpoints
    if endpoints == ("voices",):
        return default_voice_speaker_cap_count
    return 0
