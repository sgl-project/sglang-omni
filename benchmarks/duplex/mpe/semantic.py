# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/Duplex-MPE (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex/mpe/LICENSE).
"""ASR window transcription and OpenAI-compatible semantic judging for Duplex-MPE."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from benchmarks.dataset.duplex_mpe import (
    DuplexMpeScenarioRecord,
    compute_file_sha256,
    read_json_file,
    resolve_retained_scenario_audio_path,
    validate_safe_identifier,
    write_json_atomic,
)
from benchmarks.duplex.mpe.metrics import (
    collect_non_target_episodes,
    collect_target_answer_cases,
)
from benchmarks.duplex.mpe.prompts import (
    SemanticTaskKind,
    compute_prompt_sha256,
    render_judge_prompt,
)
from benchmarks.duplex.mpe.protocol import resample_linear_mono

__all__ = [
    "DEFAULT_ASR_MODEL_ID",
    "JudgeEndpointConfig",
    "JudgeRunResult",
    "JudgeTaskDecodingConfig",
    "build_semantic_tasks",
    "judge_tasks_via_http",
    "load_judge_endpoint_config",
    "load_validated_existing_verdicts",
    "make_deterministic_empty_asr_verdict",
    "public_judge_metadata",
    "resolve_chat_completions_url",
    "resolve_transcriptions_url",
    "sanitize_endpoint_origin",
    "semantic_task_key",
    "transcribe_tasks_via_http",
    "validate_http_endpoint_url",
    "validate_transcribed_against_tasks",
]

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
SHORT_CLIP_EMPTY_THRESHOLD_SECONDS = 0.15
ASR_TARGET_SAMPLE_RATE_HZ = 16000
TIMELINE_ROUNDING_TOLERANCE_SECONDS = 0.04
JUDGE_MAX_OUTPUT_TOKENS = 4096
JUDGE_MAX_ATTEMPTS = 3
RETRYABLE_CLIENT_STATUS_CODES: frozenset[int] = frozenset({408, 429})
UNFINISHED_STOP_REASONS: frozenset[str] = frozenset(
    {"max_tokens", "length", "refusal", "content_filter"}
)


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in RETRYABLE_CLIENT_STATUS_CODES or 500 <= status_code < 600


def validate_http_endpoint_url(base_url: str) -> None:
    try:
        parsed = urlsplit(base_url)
        is_valid = (
            parsed.scheme in ("http", "https")
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
            "Endpoint base_url must be an HTTP(S) URL without userinfo, query, or fragment"
        )
    else:
        pass


class JudgeTaskDecodingConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, hide_input_in_errors=True
    )

    temperature: float | int | None = Field(default=None, ge=0.0)
    top_p: float | int | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int = Field(default=JUDGE_MAX_OUTPUT_TOKENS, ge=1)

    @model_validator(mode="after")
    def validate_decoding_fields(self) -> "JudgeTaskDecodingConfig":
        if self.temperature is not None and not math.isfinite(float(self.temperature)):
            raise ValueError(
                "Judge decoding temperature must be a finite number or null"
            )
        elif self.top_p is not None and not math.isfinite(float(self.top_p)):
            raise ValueError("Judge decoding top_p must be a finite number or null")
        elif self.top_k is not None and (self.top_k == 0 or self.top_k < -1):
            raise ValueError("Judge decoding top_k must be -1 or >= 1")
        else:
            return self

    def to_metadata_dict(self) -> dict[str, JsonValue]:
        return {
            "temperature": (
                float(self.temperature) if self.temperature is not None else None
            ),
            "top_p": float(self.top_p) if self.top_p is not None else None,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }


DEFAULT_ANSWER_DECODING = JudgeTaskDecodingConfig(
    temperature=None, max_tokens=JUDGE_MAX_OUTPUT_TOKENS
)
DEFAULT_SILENCE_DECODING = JudgeTaskDecodingConfig(
    temperature=0.0, max_tokens=JUDGE_MAX_OUTPUT_TOKENS
)


class JudgeEndpointConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, hide_input_in_errors=True
    )

    model: str = Field(min_length=1)
    model_revision: str | None = None
    base_url: str = Field(min_length=1)
    api_key_env: str | None = None
    max_concurrency: int = Field(default=8, ge=1)
    enable_thinking: bool | None = None
    answer_decoding: JudgeTaskDecodingConfig = DEFAULT_ANSWER_DECODING
    silence_decoding: JudgeTaskDecodingConfig = DEFAULT_SILENCE_DECODING

    @model_validator(mode="after")
    def validate_endpoint_fields(self) -> "JudgeEndpointConfig":
        if not self.model.strip():
            raise ValueError("Judge config model must be non-empty")
        elif self.model_revision is not None and (
            not self.model_revision.strip()
            or self.model_revision != self.model_revision.strip()
        ):
            raise ValueError(
                "Judge config model_revision must be null or a trimmed identifier"
            )
        elif not self.base_url.strip():
            raise ValueError("Judge config base_url must be non-empty")
        else:
            validate_http_endpoint_url(self.base_url.strip())
        if self.api_key_env is not None and (
            not self.api_key_env or self.api_key_env != self.api_key_env.strip()
        ):
            raise ValueError(
                "Judge config api_key_env must be null or a trimmed variable name"
            )
        else:
            return self

    def decoding_for_kind(self, kind: SemanticTaskKind) -> JudgeTaskDecodingConfig:
        if kind == "answer":
            return self.answer_decoding
        else:
            return self.silence_decoding


@dataclass(frozen=True, kw_only=True)
class JudgeRunResult:
    verdicts: list[dict[str, JsonValue]]
    failed_task_keys: tuple[str, ...]
    failure_reasons: dict[str, str]

    def __iter__(self) -> Iterator[dict[str, JsonValue]]:
        return iter(self.verdicts)

    def __len__(self) -> int:
        return len(self.verdicts)

    def __getitem__(self, index: int) -> dict[str, JsonValue]:
        return self.verdicts[index]


def sanitize_endpoint_origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
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


def validate_bearer_token(raw_token: str) -> str:
    if not raw_token or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in raw_token):
        raise ValueError(
            "API key environment variable contains invalid or whitespace characters"
        )
    else:
        return raw_token


def summarize_http_exception(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTPStatusError(status={exc.response.status_code})"
    else:
        return type(exc).__name__


def public_judge_metadata(judge_config: JudgeEndpointConfig) -> dict[str, JsonValue]:
    return {
        "model": judge_config.model,
        "model_revision": judge_config.model_revision,
        "base_url": sanitize_endpoint_origin(judge_config.base_url),
        "api_key_env": judge_config.api_key_env,
        "max_concurrency": judge_config.max_concurrency,
        "enable_thinking": judge_config.enable_thinking,
        "answer_decoding": judge_config.answer_decoding.to_metadata_dict(),
        "silence_decoding": judge_config.silence_decoding.to_metadata_dict(),
    }


def load_judge_endpoint_config(config_path: Path) -> JudgeEndpointConfig:
    return JudgeEndpointConfig.model_validate(read_json_file(config_path))


def resolve_transcriptions_url(base_url: str) -> str:
    validate_http_endpoint_url(base_url)
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/audio/transcriptions"):
        return trimmed
    elif trimmed.endswith("/v1"):
        return f"{trimmed}/audio/transcriptions"
    else:
        return f"{trimmed}/v1/audio/transcriptions"


def resolve_chat_completions_url(base_url: str) -> str:
    validate_http_endpoint_url(base_url)
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    elif trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    else:
        return f"{trimmed}/v1/chat/completions"


def build_semantic_tasks(
    reports_by_id: dict[str, dict[str, JsonValue]],
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord],
) -> list[dict[str, JsonValue]]:
    """Construct deterministic silence-classification and answer-grading tasks."""
    tasks: list[dict[str, JsonValue]] = []
    for episode in collect_non_target_episodes(reports_by_id, scenarios_by_id):
        tasks.append(
            {
                "scenario_id": episode.scenario_id,
                "turn_index": episode.turn_index,
                "event_type": episode.event_type,
                "win_start": episode.window_start_seconds,
                "win_end": episode.window_end_seconds,
                "segments": [
                    [segment_start, segment_end]
                    for segment_start, segment_end in episode.speech_segments
                ],
                "duration_s": episode.duration_seconds,
                "overlap_s": episode.overlap_seconds,
                "onset_in_window": episode.has_onset_in_window,
                "context": episode.conversation_context,
                "kind": "silence",
            }
        )
    for answer_case in collect_target_answer_cases(reports_by_id):
        scenario = scenarios_by_id[answer_case.scenario_id]
        context_lines = "\n".join(
            f"{turn.speaker}: {turn.text}"
            for turn in scenario.turns[: answer_case.turn_index]
            if turn.text
        )
        tasks.append(
            {
                "scenario_id": answer_case.scenario_id,
                "turn_index": answer_case.turn_index,
                "win_start": answer_case.window_start_seconds,
                "win_end": answer_case.window_end_seconds,
                "segments": [
                    [segment_start, segment_end]
                    for segment_start, segment_end in answer_case.speech_segments
                ],
                "t_outcome": answer_case.target_outcome,
                "kind": "answer",
                "context": context_lines if context_lines else "(none)",
                "request": scenario.turns[answer_case.turn_index].text,
                "gold": scenario.gold_answer,
            }
        )
    for task in tasks:
        scenario_id = str(task["scenario_id"])
        audio_sha256 = reports_by_id[scenario_id].get("assistant_audio_sha256")
        if isinstance(audio_sha256, str) and audio_sha256:
            task["expected_audio_sha256"] = audio_sha256
        else:
            pass

    def task_sort_key(row: dict[str, JsonValue]) -> tuple[str, int, str]:
        raw_turn_index = row.get("turn_index")
        if type(raw_turn_index) is not int:
            raise ValueError("Task turn_index must be int")
        else:
            return (str(row["scenario_id"]), raw_turn_index, str(row["kind"]))

    return sorted(tasks, key=task_sort_key)


def canonical_semantic_sort_key(row: dict[str, JsonValue]) -> tuple[str, int, str]:
    raw_turn_index = row.get("turn_index")
    if type(raw_turn_index) is not int:
        raise ValueError("Task turn_index must be int")
    else:
        return (str(row["scenario_id"]), raw_turn_index, str(row["kind"]))


def semantic_task_key(task: dict[str, JsonValue]) -> str:
    raw_scenario_id = task.get("scenario_id")
    if not isinstance(raw_scenario_id, str):
        raise ValueError("Task scenario_id must be a string")
    else:
        scenario_id = validate_safe_identifier(raw_scenario_id, "scenario_id")
    kind = task.get("kind")
    if kind not in ("answer", "silence"):
        raise ValueError(f"Unknown semantic task kind: {kind!r}")
    else:
        return f"{scenario_id}:{task['turn_index']}:{kind}"


def effective_task_audio_sha256(task: dict[str, JsonValue]) -> str | None:
    actual = task.get("assistant_audio_sha256")
    expected = task.get("expected_audio_sha256")
    actual_str = actual if isinstance(actual, str) and actual else None
    expected_str = expected if isinstance(expected, str) and expected else None
    if (
        actual_str is not None
        and expected_str is not None
        and actual_str != expected_str
    ):
        raise ValueError(f"Conflicting audio SHA-256 on {semantic_task_key(task)}")
    elif actual_str is not None:
        return actual_str
    else:
        return expected_str


def bind_verdict_audio_sha256(
    verdict: dict[str, JsonValue], task: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    expected_sha256 = effective_task_audio_sha256(task)
    stored_sha256 = verdict.get("assistant_audio_sha256")
    if expected_sha256 is not None:
        if (
            isinstance(stored_sha256, str)
            and stored_sha256
            and stored_sha256 != expected_sha256
        ):
            raise ValueError(
                f"Verdict audio SHA-256 differs from task on {semantic_task_key(task)}"
            )
        else:
            return {**verdict, "assistant_audio_sha256": expected_sha256}
    else:
        return verdict


def index_semantic_tasks(
    tasks: list[dict[str, JsonValue]], *, require_hypothesis: bool = True
) -> dict[str, dict[str, JsonValue]]:
    indexed: dict[str, dict[str, JsonValue]] = {}
    for task in tasks:
        task_key = semantic_task_key(task)
        if task_key in indexed:
            raise ValueError(f"Duplicate semantic task {task_key}")
        elif require_hypothesis and not isinstance(task.get("hyp"), str):
            raise ValueError(f"Missing ASR transcript hyp on {task_key}")
        else:
            effective_task_audio_sha256(task)
            indexed[task_key] = task
    return indexed


def validate_transcribed_against_tasks(
    expected_tasks: list[dict[str, JsonValue]],
    transcribed_tasks: list[dict[str, JsonValue]],
    *,
    expected_asr_model: str | None = None,
    expected_asr_endpoint_sha256: str | None = None,
    allowed_unselected_task_keys: frozenset[str] | None = None,
) -> dict[str, dict[str, JsonValue]]:
    """Ensure cached ASR records match current scenario contexts, gold answers, and segments."""
    indexed_expected = index_semantic_tasks(expected_tasks, require_hypothesis=False)
    indexed_transcribed = index_semantic_tasks(
        transcribed_tasks, require_hypothesis=True
    )
    missing_keys = set(indexed_expected) - set(indexed_transcribed)
    extra_keys = set(indexed_transcribed) - set(indexed_expected)
    if missing_keys:
        raise ValueError(
            "Cached transcribed.json task keys do not match current scenario tasks"
        )
    elif extra_keys:
        for extra_key in extra_keys:
            if (
                allowed_unselected_task_keys is None
                or extra_key not in allowed_unselected_task_keys
            ):
                raise ValueError(
                    "Cached transcribed.json task keys do not match current scenario tasks"
                )
            else:
                pass
    else:
        pass
    selected_transcribed: dict[str, dict[str, JsonValue]] = {}
    for task_key, expected_task in indexed_expected.items():
        cached_task = indexed_transcribed[task_key]
        normalized_expected = json.loads(json.dumps(expected_task))
        for field_name, field_value in normalized_expected.items():
            if cached_task.get(field_name) != field_value:
                raise ValueError(
                    f"Cached transcribed task {task_key} differs on field {field_name!r}"
                )
            else:
                pass
        if (
            expected_asr_model is not None
            and cached_task.get("asr_model") != expected_asr_model
        ):
            raise ValueError(
                f"Cached transcribed task {task_key} used asr_model="
                f"{cached_task.get('asr_model')!r}, not {expected_asr_model!r}"
            )
        elif (
            expected_asr_endpoint_sha256 is not None
            and cached_task.get("asr_endpoint_sha256") != expected_asr_endpoint_sha256
        ):
            raise ValueError(
                f"Cached transcribed task {task_key} used a different ASR endpoint"
            )
        else:
            selected_transcribed[task_key] = cached_task
    return selected_transcribed


def parse_judge_verdict_json(
    response_text: str, kind: SemanticTaskKind
) -> dict[str, JsonValue] | None:
    json_match = re.search(r"\{.*\}", response_text, re.S)
    if json_match is None:
        return None
    else:
        pass
    try:
        parsed = json.loads(json_match.group())
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    elif kind == "answer":
        return parsed if type(parsed.get("correct")) is bool else None
    else:
        return (
            parsed
            if parsed.get("label") in ("BACKCHANNEL", "SUBSTANTIVE_INTRUSION")
            else None
        )


def make_deterministic_empty_asr_verdict(
    task: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    decision_fields: dict[str, JsonValue] = (
        {"correct": False} if task["kind"] == "answer" else {"label": "JUDGE_ERROR"}
    )
    return bind_verdict_audio_sha256(
        {
            "scenario_id": task["scenario_id"],
            "turn_index": task["turn_index"],
            "kind": task["kind"],
            **decision_fields,
            "error_kind": "asr_empty",
            "judge_model": "deterministic_empty_asr_policy",
        },
        task,
    )


def validate_task_segments(
    task: dict[str, JsonValue], waveform_duration_seconds: float
) -> list[tuple[float, float]]:
    raw_segments = task.get("segments")
    task_key = semantic_task_key(task)
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"ASR task requires non-empty segments: {task_key}")
    else:
        pass
    validated: list[tuple[float, float]] = []
    previous_end_seconds = 0.0
    upper_limit = waveform_duration_seconds + TIMELINE_ROUNDING_TOLERANCE_SECONDS
    for pair in raw_segments:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], (int, float))
            or not isinstance(pair[1], (int, float))
        ):
            raise ValueError(f"Invalid segment bounds on {task_key}")
        else:
            start_seconds = float(pair[0])
            end_seconds = float(pair[1])
            if (
                not (math.isfinite(start_seconds) and math.isfinite(end_seconds))
                or not 0.0 <= start_seconds < end_seconds <= upper_limit
                or start_seconds < previous_end_seconds
            ):
                raise ValueError(f"Out-of-bounds or overlapping segments on {task_key}")
            else:
                validated.append((start_seconds, end_seconds))
                previous_end_seconds = end_seconds
    return validated


def encode_wav_bytes_pcm16(
    samples_16khz: NDArray[np.float32], sample_rate_hz: int = ASR_TARGET_SAMPLE_RATE_HZ
) -> bytes:
    memory_buffer = io.BytesIO()
    soundfile.write(
        memory_buffer,
        samples_16khz,
        sample_rate_hz,
        format="WAV",
        subtype="PCM_16",
    )
    return memory_buffer.getvalue()


async def transcribe_tasks_via_http(
    tasks: list[dict[str, JsonValue]],
    *,
    audio_directory: Path,
    output_path: Path,
    asr_base_url: str,
    asr_model_id: str = DEFAULT_ASR_MODEL_ID,
    api_key_env: str | None = None,
    max_concurrency: int = 8,
    timeout_seconds: float = 120.0,
    trust_env: bool = False,
    allowed_unselected_task_keys: frozenset[str] | None = None,
) -> list[dict[str, JsonValue]]:
    """Slice voiced window segments and transcribe them via /v1/audio/transcriptions."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    else:
        pass
    endpoint_url = resolve_transcriptions_url(asr_base_url)
    endpoint_sha256 = hashlib.sha256(endpoint_url.encode("utf-8")).hexdigest()
    request_headers: dict[str, str] = {}
    if api_key_env:
        token = os.environ.get(api_key_env)
        if not token:
            raise RuntimeError(
                f"ASR API key environment variable is not set: {api_key_env}"
            )
        else:
            validated_token = validate_bearer_token(token)
            request_headers["Authorization"] = f"Bearer {validated_token}"
    else:
        pass

    indexed_tasks = index_semantic_tasks(tasks, require_hypothesis=False)
    cached_by_key: dict[str, dict[str, JsonValue]] = {}
    if output_path.is_file():
        raw_cached = read_json_file(output_path)
        if isinstance(raw_cached, list):
            cached_by_key = index_semantic_tasks(
                [row for row in raw_cached if isinstance(row, dict)],
                require_hypothesis=True,
            )
        else:
            pass
    else:
        pass

    preserved_unselected: list[dict[str, JsonValue]] = []
    if allowed_unselected_task_keys is not None:
        for cached_key, cached_row in cached_by_key.items():
            if cached_key in indexed_tasks:
                continue
            elif cached_key not in allowed_unselected_task_keys:
                raise ValueError(
                    f"Cached transcribed.json contains unknown task {cached_key}"
                )
            elif (
                cached_row.get("asr_model") is not None
                and cached_row.get("asr_model") != asr_model_id
            ) or (
                cached_row.get("asr_endpoint_sha256") is not None
                and cached_row.get("asr_endpoint_sha256") != endpoint_sha256
            ):
                raise ValueError(
                    f"Subset ASR configuration conflicts with unselected cached task {cached_key}; "
                    "use a new output path"
                )
            else:
                preserved_unselected.append(cached_row)
    else:
        pass

    tasks_by_scenario: dict[str, list[dict[str, JsonValue]]] = {}
    for task in indexed_tasks.values():
        tasks_by_scenario.setdefault(str(task["scenario_id"]), []).append(task)

    pending_jobs: list[
        tuple[str, dict[str, JsonValue], Path, list[tuple[float, float]]]
    ] = []
    completed_by_key: dict[str, dict[str, JsonValue]] = {}

    for scenario_id, group in tasks_by_scenario.items():
        wav_path = resolve_retained_scenario_audio_path(
            audio_directory, scenario_id, must_exist=True
        )
        audio_sha256 = compute_file_sha256(wav_path)
        audio_header = soundfile.info(str(wav_path))
        if audio_header.channels != 1:
            raise ValueError(f"Assistant audio must be mono: {wav_path}")
        else:
            waveform_duration_seconds = audio_header.frames / audio_header.samplerate

        for task in group:
            task_key = semantic_task_key(task)
            expected_sha256 = task.get("expected_audio_sha256")
            if (
                isinstance(expected_sha256, str)
                and expected_sha256
                and expected_sha256 != audio_sha256
            ):
                raise ValueError(
                    f"Assistant audio on disk differs from timing report: {scenario_id}"
                )
            else:
                pass
            segments = validate_task_segments(task, waveform_duration_seconds)
            base_record: dict[str, JsonValue] = {
                **task,
                "asr_model": asr_model_id,
                "asr_language": "en",
                "asr_transport": "openai_audio_transcriptions",
                "asr_endpoint_sha256": endpoint_sha256,
                "assistant_audio_sha256": audio_sha256,
            }
            normalized_record = json.loads(json.dumps(base_record))
            previous = cached_by_key.get(task_key)
            if previous is not None and all(
                previous.get(field) == field_value
                for field, field_value in normalized_record.items()
            ):
                completed_by_key[task_key] = previous
                continue
            else:
                pass
            total_clip_frames = sum(
                max(
                    0,
                    min(
                        audio_header.frames,
                        int(segment_end_seconds * audio_header.samplerate),
                    )
                    - max(
                        0,
                        min(
                            audio_header.frames,
                            int(segment_start_seconds * audio_header.samplerate),
                        ),
                    ),
                )
                for segment_start_seconds, segment_end_seconds in segments
            )
            if (
                total_clip_frames
                < audio_header.samplerate * SHORT_CLIP_EMPTY_THRESHOLD_SECONDS
            ):
                completed_by_key[task_key] = {**base_record, "hyp": ""}
            else:
                pending_jobs.append((task_key, base_record, wav_path, segments))

    checkpoint_lock = asyncio.Lock()

    async def persist_checkpoint() -> None:
        async with checkpoint_lock:
            partial_ordered = sorted(
                [
                    completed_by_key[task_key]
                    for task_key in indexed_tasks
                    if task_key in completed_by_key
                ]
                + preserved_unselected,
                key=canonical_semantic_sort_key,
            )
            write_json_atomic(output_path, partial_ordered)

    if completed_by_key and not pending_jobs:
        await persist_checkpoint()
    elif pending_jobs:
        if completed_by_key:
            await persist_checkpoint()
        else:
            pass
        semaphore = asyncio.Semaphore(max_concurrency)
        async with httpx.AsyncClient(
            timeout=timeout_seconds, trust_env=trust_env
        ) as http_client:

            async def transcribe_one(
                job_key: str,
                record_template: dict[str, JsonValue],
                source_wav_path: Path,
                segments: list[tuple[float, float]],
            ) -> None:
                async with semaphore:
                    waveform, sample_rate_hz = soundfile.read(
                        str(source_wav_path), dtype="float32"
                    )
                    concatenated_clip = np.concatenate(
                        [
                            waveform[
                                int(segment_start_seconds * sample_rate_hz) : int(
                                    segment_end_seconds * sample_rate_hz
                                )
                            ]
                            for segment_start_seconds, segment_end_seconds in segments
                        ]
                    ).astype(np.float32)
                    if (
                        len(concatenated_clip)
                        < sample_rate_hz * SHORT_CLIP_EMPTY_THRESHOLD_SECONDS
                    ):
                        completed_by_key[job_key] = {
                            **record_template,
                            "hyp": "",
                        }
                        await persist_checkpoint()
                        return
                    else:
                        pass
                    clip_16khz = resample_linear_mono(
                        concatenated_clip,
                        source_sample_rate_hz=int(sample_rate_hz),
                        target_sample_rate_hz=ASR_TARGET_SAMPLE_RATE_HZ,
                    )
                    wav_payload = encode_wav_bytes_pcm16(clip_16khz)
                    last_error = "no attempt executed"
                    for attempt_index in range(JUDGE_MAX_ATTEMPTS):
                        try:
                            response = await http_client.post(
                                endpoint_url,
                                headers=request_headers,
                                data={"model": asr_model_id, "language": "en"},
                                files={
                                    "file": ("window.wav", wav_payload, "audio/wav")
                                },
                            )
                            if is_retryable_http_status(response.status_code):
                                last_error = f"HTTP {response.status_code}"
                                await asyncio.sleep(0.5 * (attempt_index + 1))
                                continue
                            elif response.status_code >= 400:
                                raise RuntimeError(
                                    f"ASR transcription rejected for {job_key}: "
                                    f"HTTPStatusError(status={response.status_code})"
                                )
                            else:
                                payload = response.json()
                                if not isinstance(payload, dict) or not isinstance(
                                    payload.get("text"), str
                                ):
                                    raise ValueError(
                                        "ASR response JSON lacks string field text"
                                    )
                                else:
                                    completed_by_key[job_key] = {
                                        **record_template,
                                        "hyp": payload["text"].strip(),
                                    }
                                    await persist_checkpoint()
                                    return
                        except (httpx.HTTPError, ValueError) as exc:
                            last_error = summarize_http_exception(exc)
                            await asyncio.sleep(0.5 * (attempt_index + 1))
                    raise RuntimeError(
                        f"ASR transcription failed for {job_key}: {last_error}"
                    )

            outcomes = await asyncio.gather(
                *(
                    transcribe_one(job_key, record_template, source_wav_path, segments)
                    for job_key, record_template, source_wav_path, segments in pending_jobs
                ),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
                else:
                    pass
    else:
        write_json_atomic(
            output_path,
            sorted(preserved_unselected, key=canonical_semantic_sort_key),
        )

    ordered_output = [completed_by_key[task_key] for task_key in indexed_tasks]
    write_json_atomic(
        output_path,
        sorted(ordered_output + preserved_unselected, key=canonical_semantic_sort_key),
    )
    return ordered_output


def validate_verdict_against_judge_config(
    verdict: dict[str, JsonValue],
    *,
    task_key: str,
    kind: SemanticTaskKind,
    judge_config: JudgeEndpointConfig | None,
    expected_judge_model: str | None,
    expected_model_revision: str | None,
    expected_enable_thinking: bool | None,
    check_enable_thinking: bool,
    expected_endpoint_sha256: str | None,
) -> None:
    effective_model = (
        judge_config.model if judge_config is not None else expected_judge_model
    )
    if effective_model is not None and verdict.get("judge_model") != effective_model:
        raise ValueError(
            f"Existing verdict on {task_key} was produced by judge "
            f"{verdict.get('judge_model')!r}, not {effective_model!r}; "
            "use a new output path"
        )
    else:
        pass
    effective_revision = (
        judge_config.model_revision
        if judge_config is not None
        else expected_model_revision
    )
    has_revision_check = judge_config is not None or expected_model_revision is not None
    if has_revision_check and "judge_model_revision" in verdict:
        if verdict.get("judge_model_revision") != effective_revision:
            raise ValueError(
                f"Existing verdict on {task_key} used judge_model_revision="
                f"{verdict.get('judge_model_revision')!r}, not {effective_revision!r}; "
                "use a new output path"
            )
        else:
            pass
    elif effective_revision is not None and "judge_model_revision" not in verdict:
        raise ValueError(
            f"Existing verdict on {task_key} lacks declared judge_model_revision; "
            "use a new output path"
        )
    else:
        pass
    effective_thinking = (
        judge_config.enable_thinking
        if judge_config is not None
        else expected_enable_thinking
    )
    should_check_thinking = judge_config is not None or check_enable_thinking
    if should_check_thinking and "judge_enable_thinking" in verdict:
        if verdict.get("judge_enable_thinking") != effective_thinking:
            raise ValueError(
                f"Existing verdict on {task_key} used enable_thinking="
                f"{verdict.get('judge_enable_thinking')!r}, not "
                f"{effective_thinking!r}; use a new output path"
            )
        else:
            pass
    elif effective_thinking is not None and "judge_enable_thinking" not in verdict:
        raise ValueError(
            f"Existing verdict on {task_key} lacks judge_enable_thinking; "
            "use a new output path"
        )
    else:
        pass
    if (
        expected_endpoint_sha256 is not None
        and verdict.get("judge_endpoint_sha256") != expected_endpoint_sha256
    ):
        raise ValueError(
            f"Existing verdict on {task_key} used a different judge endpoint; "
            "use a new output path"
        )
    else:
        pass
    if judge_config is not None:
        expected_decoding = judge_config.decoding_for_kind(kind).to_metadata_dict()
        default_decoding = (
            DEFAULT_ANSWER_DECODING.to_metadata_dict()
            if kind == "answer"
            else DEFAULT_SILENCE_DECODING.to_metadata_dict()
        )
        if "judge_decoding" in verdict:
            if verdict.get("judge_decoding") != expected_decoding:
                raise ValueError(
                    f"Existing verdict on {task_key} used judge_decoding="
                    f"{verdict.get('judge_decoding')!r}, not {expected_decoding!r}; "
                    "use a new output path"
                )
            else:
                pass
        elif expected_decoding != default_decoding:
            raise ValueError(
                f"Existing verdict on {task_key} lacks judge_decoding for custom decoding; "
                "use a new output path"
            )
        else:
            pass
    else:
        pass


def load_validated_existing_verdicts(
    indexed_tasks: dict[str, dict[str, JsonValue]],
    existing_verdicts: list[dict[str, JsonValue]] | None,
    *,
    judge_config: JudgeEndpointConfig | None = None,
    expected_judge_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_enable_thinking: bool | None = None,
    expected_endpoint_sha256: str | None = None,
    allowed_unselected_task_keys: frozenset[str] | None = None,
) -> dict[str, dict[str, JsonValue]]:
    validated: dict[str, dict[str, JsonValue]] = {}
    seen_keys: set[str] = set()
    inferred_identity: tuple[JsonValue, JsonValue, JsonValue, JsonValue] | None = None
    inferred_decoding_by_kind: dict[SemanticTaskKind, JsonValue] = {}
    for raw_verdict in existing_verdicts or []:
        task_key = semantic_task_key(raw_verdict)
        if task_key in seen_keys:
            raise ValueError(f"Duplicate existing semantic verdict {task_key}")
        else:
            seen_keys.add(task_key)
        if task_key not in indexed_tasks:
            if (
                allowed_unselected_task_keys is not None
                and task_key in allowed_unselected_task_keys
            ):
                continue
            else:
                raise ValueError(
                    f"Existing semantic verdict references unknown task {task_key}"
                )
        else:
            pass
        task = indexed_tasks[task_key]
        bound_verdict = bind_verdict_audio_sha256(raw_verdict, task)
        hypothesis = str(task.get("hyp", "")).strip()
        if not hypothesis:
            expected_empty = make_deterministic_empty_asr_verdict(task)
            decision_field = "correct" if task["kind"] == "answer" else "label"
            if (
                bound_verdict.get("error_kind") != expected_empty["error_kind"]
                or bound_verdict.get(decision_field) != expected_empty[decision_field]
            ):
                raise ValueError(
                    f"Existing verdict conflicts with deterministic empty-ASR rule on {task_key}"
                )
            else:
                validated[task_key] = expected_empty
                continue
        else:
            pass
        kind: SemanticTaskKind = "answer" if task["kind"] == "answer" else "silence"
        if judge_config is None:
            row_identity = (
                bound_verdict.get("judge_model"),
                bound_verdict.get("judge_model_revision"),
                bound_verdict.get("judge_enable_thinking"),
                bound_verdict.get("judge_endpoint_sha256"),
            )
            if inferred_identity is None:
                inferred_identity = row_identity
            elif row_identity != inferred_identity:
                raise ValueError(
                    f"Mixed judge provenance across cached verdicts on {task_key}; "
                    "use a new output path"
                )
            else:
                pass
            row_decoding = bound_verdict.get("judge_decoding")
            if kind not in inferred_decoding_by_kind:
                inferred_decoding_by_kind[kind] = row_decoding
            elif inferred_decoding_by_kind[kind] != row_decoding:
                raise ValueError(
                    f"Mixed judge_decoding across cached {kind} verdicts on {task_key}; "
                    "use a new output path"
                )
            else:
                pass
        else:
            pass
        validate_verdict_against_judge_config(
            bound_verdict,
            task_key=task_key,
            kind=kind,
            judge_config=judge_config,
            expected_judge_model=expected_judge_model,
            expected_model_revision=expected_model_revision,
            expected_enable_thinking=expected_enable_thinking,
            check_enable_thinking=(expected_enable_thinking is not None),
            expected_endpoint_sha256=expected_endpoint_sha256,
        )
        if bound_verdict.get("prompt_sha256") != compute_prompt_sha256(task):
            raise ValueError(f"Prompt SHA-256 mismatch on existing verdict {task_key}")
        elif any(
            bound_verdict.get(reason_key) in UNFINISHED_STOP_REASONS
            for reason_key in ("finish_reason", "stop_reason")
        ):
            continue
        elif bound_verdict.get("error_kind") == "asr_empty":
            raise ValueError(
                f"Empty-ASR verdict cannot apply to non-empty transcript on {task_key}"
            )
        elif task["kind"] == "answer" and type(bound_verdict.get("correct")) is bool:
            validated[task_key] = bound_verdict
        elif task["kind"] == "silence" and bound_verdict.get("label") in (
            "BACKCHANNEL",
            "SUBSTANTIVE_INTRUSION",
        ):
            validated[task_key] = bound_verdict
        else:
            pass
    return validated


async def judge_tasks_via_http(
    transcribed_tasks: list[dict[str, JsonValue]],
    *,
    judge_config: JudgeEndpointConfig,
    output_path: Path,
    timeout_seconds: float = 180.0,
    trust_env: bool = False,
    allowed_unselected_task_keys: frozenset[str] | None = None,
) -> JudgeRunResult:
    """Grade transcribed answer and silence tasks via an OpenAI-compatible judge."""
    endpoint_url = resolve_chat_completions_url(judge_config.base_url)
    endpoint_sha256 = hashlib.sha256(endpoint_url.encode("utf-8")).hexdigest()
    request_headers: dict[str, str] = {"Content-Type": "application/json"}
    if judge_config.api_key_env:
        token = os.environ.get(judge_config.api_key_env)
        if not token:
            raise RuntimeError(
                f"Judge API key environment variable is not set: {judge_config.api_key_env}"
            )
        else:
            validated_token = validate_bearer_token(token)
            request_headers["Authorization"] = f"Bearer {validated_token}"
    else:
        pass

    indexed_tasks = index_semantic_tasks(transcribed_tasks, require_hypothesis=True)
    existing_verdicts: list[dict[str, JsonValue]] | None = None
    if output_path.is_file():
        raw_existing = read_json_file(output_path)
        if isinstance(raw_existing, list):
            existing_verdicts = [row for row in raw_existing if isinstance(row, dict)]
        else:
            pass
    else:
        pass

    verdicts_by_key = load_validated_existing_verdicts(
        indexed_tasks,
        existing_verdicts,
        judge_config=judge_config,
        expected_endpoint_sha256=endpoint_sha256,
        allowed_unselected_task_keys=allowed_unselected_task_keys,
    )
    preserved_unselected_verdicts: list[dict[str, JsonValue]] = []
    if allowed_unselected_task_keys is not None:
        for raw_verdict in existing_verdicts or []:
            verdict_key = semantic_task_key(raw_verdict)
            if verdict_key in indexed_tasks:
                continue
            elif raw_verdict.get("error_kind") != "asr_empty":
                unselected_kind: SemanticTaskKind = (
                    "answer" if raw_verdict.get("kind") == "answer" else "silence"
                )
                try:
                    validate_verdict_against_judge_config(
                        raw_verdict,
                        task_key=verdict_key,
                        kind=unselected_kind,
                        judge_config=judge_config,
                        expected_judge_model=judge_config.model,
                        expected_model_revision=judge_config.model_revision,
                        expected_enable_thinking=judge_config.enable_thinking,
                        check_enable_thinking=True,
                        expected_endpoint_sha256=(
                            endpoint_sha256
                            if raw_verdict.get("judge_endpoint_sha256") is not None
                            else None
                        ),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Subset judge configuration conflicts with unselected verdict "
                        f"{verdict_key}; use a new output path"
                    ) from exc
                preserved_unselected_verdicts.append(raw_verdict)
            else:
                preserved_unselected_verdicts.append(raw_verdict)
    else:
        pass
    pending_tasks: list[tuple[str, dict[str, JsonValue]]] = []
    for task_key, task in indexed_tasks.items():
        if not str(task.get("hyp", "")).strip():
            verdicts_by_key[task_key] = make_deterministic_empty_asr_verdict(task)
        elif task_key not in verdicts_by_key:
            pending_tasks.append((task_key, task))
        else:
            pass

    failed_reasons: dict[str, str] = {}
    if pending_tasks:
        semaphore = asyncio.Semaphore(judge_config.max_concurrency)
        checkpoint_lock = asyncio.Lock()
        async with httpx.AsyncClient(
            timeout=timeout_seconds, trust_env=trust_env
        ) as http_client:

            async def judge_one(
                task_key: str, task: dict[str, JsonValue]
            ) -> tuple[str, dict[str, JsonValue] | None]:
                kind: SemanticTaskKind = (
                    "answer" if task["kind"] == "answer" else "silence"
                )
                prompt_text = render_judge_prompt(task)
                prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                decoding = judge_config.decoding_for_kind(kind)
                request_body: dict[str, JsonValue] = {
                    "model": judge_config.model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": decoding.max_tokens,
                }
                if decoding.temperature is not None:
                    request_body["temperature"] = decoding.temperature
                else:
                    pass
                if decoding.top_p is not None:
                    request_body["top_p"] = decoding.top_p
                else:
                    pass
                if decoding.top_k is not None:
                    request_body["top_k"] = decoding.top_k
                else:
                    pass
                if decoding.seed is not None:
                    request_body["seed"] = decoding.seed
                else:
                    pass
                if judge_config.enable_thinking is not None:
                    request_body["chat_template_kwargs"] = {
                        "enable_thinking": judge_config.enable_thinking
                    }
                else:
                    pass

                last_failure_reason = "no_valid_verdict"
                async with semaphore:
                    for attempt_index in range(JUDGE_MAX_ATTEMPTS):
                        try:
                            response = await http_client.post(
                                endpoint_url,
                                headers=request_headers,
                                json=request_body,
                            )
                            if is_retryable_http_status(response.status_code):
                                last_failure_reason = (
                                    f"HTTPStatusError(status={response.status_code})"
                                )
                                await asyncio.sleep(0.5 * (attempt_index + 1))
                                continue
                            elif response.status_code >= 400:
                                last_failure_reason = (
                                    f"HTTPStatusError(status={response.status_code})"
                                )
                                logger.error(
                                    f"Judge non-retryable HTTP failure on {task_key}: "
                                    f"{last_failure_reason}"
                                )
                                break
                            else:
                                payload = response.json()
                            choices = (
                                payload.get("choices")
                                if isinstance(payload, dict)
                                else None
                            )
                            if not isinstance(choices, list) or not choices:
                                last_failure_reason = "missing_choices"
                                continue
                            else:
                                first_choice = choices[0]
                            if not isinstance(first_choice, dict):
                                last_failure_reason = "malformed_choice"
                                continue
                            else:
                                finish_reason = first_choice.get("finish_reason")
                                choice_stop_reason = first_choice.get("stop_reason")
                                top_stop_reason = (
                                    payload.get("stop_reason")
                                    if isinstance(payload, dict)
                                    else None
                                )
                            if any(
                                marker in UNFINISHED_STOP_REASONS
                                for marker in (
                                    finish_reason,
                                    choice_stop_reason,
                                    top_stop_reason,
                                )
                            ):
                                last_failure_reason = "unfinished_stop_reason"
                                continue
                            else:
                                message = first_choice.get("message")
                            content = (
                                message.get("content")
                                if isinstance(message, dict)
                                else None
                            )
                            if not isinstance(content, str):
                                last_failure_reason = "missing_message_content"
                                continue
                            else:
                                parsed_verdict = parse_judge_verdict_json(content, kind)
                            if parsed_verdict is None:
                                last_failure_reason = "invalid_verdict_json"
                                continue
                            else:
                                decision: dict[str, JsonValue] = (
                                    {"correct": bool(parsed_verdict["correct"])}
                                    if kind == "answer"
                                    else {"label": str(parsed_verdict["label"])}
                                )
                                record: dict[str, JsonValue] = {
                                    "scenario_id": task["scenario_id"],
                                    "turn_index": task["turn_index"],
                                    "kind": kind,
                                    **decision,
                                    "judge_model": judge_config.model,
                                    "judge_model_revision": judge_config.model_revision,
                                    "judge_enable_thinking": (
                                        judge_config.enable_thinking
                                    ),
                                    "judge_decoding": decoding.to_metadata_dict(),
                                    "judge_endpoint_sha256": endpoint_sha256,
                                    "prompt_sha256": prompt_sha256,
                                }
                                if isinstance(finish_reason, str):
                                    record["finish_reason"] = finish_reason
                                else:
                                    pass
                                reason_value = parsed_verdict.get("reason")
                                if isinstance(reason_value, str):
                                    record["reason"] = reason_value
                                else:
                                    pass
                                bound_record = bind_verdict_audio_sha256(record, task)
                                async with checkpoint_lock:
                                    verdicts_by_key[task_key] = bound_record
                                    write_json_atomic(
                                        output_path,
                                        sorted(
                                            [
                                                verdicts_by_key[key]
                                                for key in indexed_tasks
                                                if key in verdicts_by_key
                                            ]
                                            + preserved_unselected_verdicts,
                                            key=canonical_semantic_sort_key,
                                        ),
                                    )
                                return (task_key, bound_record)
                        except (httpx.HTTPError, ValueError) as exc:
                            last_failure_reason = summarize_http_exception(exc)
                            logger.warning(
                                f"Judge attempt {attempt_index + 1} failed on {task_key}: "
                                f"{last_failure_reason}"
                            )
                            await asyncio.sleep(0.5 * (attempt_index + 1))
                async with checkpoint_lock:
                    failed_reasons[task_key] = last_failure_reason
                return (task_key, None)

            await asyncio.gather(
                *(judge_one(task_key, task) for task_key, task in pending_tasks)
            )
    else:
        pass

    ordered_verdicts = [
        verdicts_by_key[task_key]
        for task_key in indexed_tasks
        if task_key in verdicts_by_key
    ]
    write_json_atomic(
        output_path,
        sorted(
            ordered_verdicts + preserved_unselected_verdicts,
            key=canonical_semantic_sort_key,
        ),
    )
    ordered_failed_keys = tuple(
        task_key for task_key in indexed_tasks if task_key in failed_reasons
    )
    return JudgeRunResult(
        verdicts=ordered_verdicts,
        failed_task_keys=ordered_failed_keys,
        failure_reasons={key: failed_reasons[key] for key in ordered_failed_keys},
    )
