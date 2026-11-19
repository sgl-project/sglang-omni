# SPDX-License-Identifier: Apache-2.0
"""Parse CLI options into the single immutable stress-test configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = "stress_results.jsonl"
SAMPLE_ID_FIELDS: tuple[str, ...] = ("id", "key", "sample_id", "sample_key", "uid")


@dataclass(frozen=True, kw_only=True)
class CorpusEntry:
    """One prompt parsed from the input jsonl, handed to the sender and builder."""

    sample_id: str
    text: str
    source_line: int


@dataclass(frozen=True, kw_only=True)
class StressConfig:
    """Immutable configuration shared by the sender, responder, and builder."""

    base_url: str
    endpoint: str
    url: str
    api_key: str | None
    headers: dict[str, str]

    payload_template: dict[str, Any]
    corpus: tuple[CorpusEntry, ...]
    corpus_path: Path
    text_field: str
    text_id_field: str | None
    text_repeat: str

    qpm: float | None
    request_rate: float
    total_requests: int
    duration_s: float | None
    warmup_requests: int
    max_concurrency: int
    arrival_mode: str
    arrival_seed: int
    ramp_up_s: float
    ramp_start_fraction: float
    arrival_offsets: tuple[float, ...]
    lag_threshold_s: float

    timeout_s: float
    connect_timeout_s: float
    chunk_timeout_s: float
    outcome_timeout_s: float
    collect_outcomes: bool

    response_format: str
    stream_format: str
    sample_rate: int | None
    fallback_sample_rate: int
    channels: int
    bit_depth: int

    underrun_thresholds_ms: tuple[int, ...]
    broken_underrun_ms: float
    stall_broken_s: float
    min_audio_duration_s: float

    output: Path
    summary_json: Path | None
    omit_chunks: bool
    run_id: str
    labels: dict[str, str]
    quiet: bool
    log_every: int
    dry_run: bool
    strict: bool

    @property
    def block_align(self) -> int:
        return self.channels * self.bit_depth // 8

    def entry_for(self, index: int) -> CorpusEntry:
        return self.corpus[index % len(self.corpus)]

    def payload_for(self, index: int, text: str | None = None) -> dict[str, Any]:
        payload = dict(self.payload_template)
        payload["input"] = text if text is not None else self.entry_for(index).text
        return payload

    def snapshot(self) -> dict[str, Any]:
        """Return the reproducible configuration recorded as the first jsonl row."""
        return {
            "run_id": self.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "labels": self.labels,
            "target": {
                "base_url": self.base_url,
                "endpoint": self.endpoint,
                "url": self.url,
            },
            "request": {
                "payload_template": redact_payload(self.payload_template),
                "stream_format": self.stream_format,
                "response_format": self.response_format,
            },
            "corpus": {
                "text_jsonl": str(self.corpus_path),
                "text_field": self.text_field,
                "text_id_field": self.text_id_field,
                "corpus_size": len(self.corpus),
                "first_sample_ids": [entry.sample_id for entry in self.corpus[:5]],
            },
            "load": {
                "qpm": self.qpm,
                "request_rate": round(self.request_rate, 6),
                "total_requests": self.total_requests,
                "duration_s": self.duration_s,
                "warmup_requests": self.warmup_requests,
                "max_concurrency": self.max_concurrency,
                "arrival_mode": self.arrival_mode,
                "arrival_seed": self.arrival_seed,
                "ramp_up_s": self.ramp_up_s,
                "ramp_start_fraction": self.ramp_start_fraction,
                "planned_wall_clock_s": (
                    round(self.arrival_offsets[-1], 3) if self.arrival_offsets else 0.0
                ),
                "lag_threshold_s": self.lag_threshold_s,
            },
            "timeouts": {
                "total_s": self.timeout_s,
                "connect_s": self.connect_timeout_s,
                "chunk_s": self.chunk_timeout_s,
                "outcome_s": self.outcome_timeout_s,
            },
            "audio": {
                "sample_rate": self.sample_rate,
                "fallback_sample_rate": self.fallback_sample_rate,
                "channels": self.channels,
                "bit_depth": self.bit_depth,
            },
            "stream_health": {
                "underrun_thresholds_ms": list(self.underrun_thresholds_ms),
                "broken_underrun_ms": self.broken_underrun_ms,
                "stall_broken_s": self.stall_broken_s,
                "min_audio_duration_s": self.min_audio_duration_s,
                "collect_outcomes": self.collect_outcomes,
            },
            "output": {
                "jsonl": str(self.output),
                "omit_chunks": self.omit_chunks,
                "summary_json": (str(self.summary_json) if self.summary_json else None),
            },
        }


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Hide long inline audio references before writing the configuration row."""
    cleaned = dict(payload)
    for field_name in ("ref_audio", "ref_text"):
        if isinstance(cleaned.get(field_name), str) and len(cleaned[field_name]) > 128:
            cleaned[field_name] = "<redacted>"
    return cleaned


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-stream-stress",
        description=(
            "Stream TTS load test: drive /v1/audio/speech at a target QPM and "
            "write per-chunk TTFA, RTF, and stream-health metrics as jsonl."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    target = parser.add_argument_group("target service")
    target.add_argument(
        "--base-url",
        default=None,
        help="full base URL, for example http://127.0.0.1:30000; overrides --host/--port",
    )
    target.add_argument("--host", default="127.0.0.1", help="service host")
    target.add_argument("--port", type=int, default=8000, help="service port")
    target.add_argument("--scheme", choices=("http", "https"), default="http")
    target.add_argument("--endpoint", default="/v1/audio/speech", help="TTS path")
    target.add_argument("--api-key", default=None, help="Bearer token passed literally")
    target.add_argument(
        "--api-key-env",
        default=None,
        help="environment variable holding the Bearer token",
    )
    target.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="K:V",
        help="extra request header, repeatable",
    )
    target.add_argument(
        "--timeout-s", type=float, default=300.0, help="total timeout for one request"
    )
    target.add_argument(
        "--connect-timeout-s", type=float, default=10.0, help="connection timeout"
    )
    target.add_argument(
        "--chunk-timeout-s",
        type=float,
        default=30.0,
        help="maximum wait between chunks; exceeding it marks the stream broken",
    )
    target.add_argument(
        "--outcome-timeout-s",
        type=float,
        default=5.0,
        help="timeout for the stream outcome lookup",
    )

    request = parser.add_argument_group("request body")
    request.add_argument(
        "--model", default=None, help="model name; omit to use the server default"
    )
    request.add_argument("--voice", default="default", help="voice name")
    request.add_argument(
        "--response-format",
        choices=("pcm", "wav"),
        default="pcm",
        help="streaming only supports pcm or wav",
    )
    request.add_argument(
        "--stream-format",
        choices=("audio", "sse"),
        default="audio",
        help="raw binary stream or server-sent events",
    )
    request.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="speech speed, server range 0.25 to 4.0",
    )
    request.add_argument("--language", default=None)
    request.add_argument("--task-type", default=None)
    request.add_argument("--instructions", default=None)
    request.add_argument(
        "--ref-audio", default=None, help="reference audio, server path or data URI"
    )
    request.add_argument("--ref-text", default=None)
    request.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="JSON",
        help="one references entry, repeatable",
    )
    request.add_argument("--initial-codec-chunk-frames", type=int, default=None)
    request.add_argument("--max-new-tokens", type=int, default=None)
    request.add_argument("--temperature", type=float, default=None)
    request.add_argument("--top-p", type=float, default=None)
    request.add_argument("--top-k", type=int, default=None)
    request.add_argument("--repetition-penalty", type=float, default=None)
    request.add_argument(
        "--seed", type=int, default=None, help="generation sampling seed"
    )
    request.add_argument(
        "--extra-body",
        default=None,
        metavar="JSON",
        help="extra JSON body fields, may override the options above",
    )
    request.add_argument(
        "--request-file",
        default=None,
        metavar="PATH",
        help="JSON request template, lower priority than CLI options",
    )

    corpus = parser.add_argument_group("text corpus")
    corpus.add_argument(
        "--text-jsonl",
        required=True,
        metavar="PATH",
        help="jsonl corpus; every non-empty line must be an object with the text field",
    )
    corpus.add_argument(
        "--text-field", default="text", help="prompt field name in each jsonl object"
    )
    corpus.add_argument(
        "--text-id-field",
        default=None,
        metavar="FIELD",
        help="sample id field; defaults to the first of id/key/sample_id/sample_key/uid, else the line number",
    )
    corpus.add_argument(
        "--max-texts",
        type=int,
        default=0,
        help="maximum number of prompts, 0 means all",
    )
    corpus.add_argument(
        "--text-repeat",
        choices=("cycle", "random"),
        default="cycle",
        help="cycle or randomly sample prompts when the corpus is reused",
    )

    load = parser.add_argument_group("load and QPM")
    rate = load.add_mutually_exclusive_group()
    rate.add_argument(
        "--qpm", type=float, default=None, help="requests per minute, defaults to 60"
    )
    rate.add_argument(
        "--request-rate",
        type=float,
        default=None,
        metavar="RPS",
        help="requests per second, mutually exclusive with --qpm",
    )
    load.add_argument(
        "--total-requests", type=int, default=100, help="number of measured requests"
    )
    load.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="run for this many seconds, overrides --total-requests",
    )
    load.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="warmup requests excluded from the metrics",
    )
    load.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="maximum in-flight requests, 0 means unlimited",
    )
    load.add_argument(
        "--arrival-mode",
        choices=("uniform", "poisson", "burst"),
        default="uniform",
        help="uniform keeps QPM steady, poisson samples arrivals, burst sends everything at once",
    )
    load.add_argument(
        "--arrival-seed",
        type=int,
        default=20261118,
        help="seed for the arrival sequence",
    )
    load.add_argument(
        "--ramp-up-s",
        type=float,
        default=0.0,
        help="ramp duration from --ramp-start-fraction to the target QPM",
    )
    load.add_argument("--ramp-start-fraction", type=float, default=0.25)
    load.add_argument(
        "--lag-threshold-s",
        type=float,
        default=0.1,
        help="scheduler lag that marks load_generator_lagged",
    )

    audio = parser.add_argument_group("audio and metrics")
    audio.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="force a sample rate; otherwise read X-Sample-Rate",
    )
    audio.add_argument(
        "--fallback-sample-rate",
        type=int,
        default=24000,
        help="sample rate when the response header is absent",
    )
    audio.add_argument(
        "--channels",
        type=int,
        default=1,
        help="channel count when the response header is absent",
    )
    audio.add_argument(
        "--bit-depth",
        type=int,
        default=16,
        help="bit depth when the response header is absent",
    )
    audio.add_argument(
        "--underrun-thresholds-ms",
        default="50,100,200",
        help="continuity pass-rate thresholds in milliseconds",
    )
    audio.add_argument(
        "--broken-underrun-ms",
        type=float,
        default=200.0,
        help="max playback underrun that marks a broken stream",
    )
    audio.add_argument(
        "--stall-broken-s",
        type=float,
        default=2.0,
        help="chunk gap that marks a broken stream",
    )
    audio.add_argument(
        "--min-audio-duration-s",
        type=float,
        default=0.05,
        help="shorter audio is treated as invalid",
    )
    audio.add_argument(
        "--collect-outcomes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after a raw stream, look up finish_reason at /v1/audio/speech/{speech_id}",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output", default=DEFAULT_OUTPUT, help="result jsonl path")
    output.add_argument(
        "--summary-json", default=None, help="also write a standalone summary JSON"
    )
    output.add_argument(
        "--omit-chunks",
        action="store_true",
        help="omit per-chunk details from the jsonl",
    )
    output.add_argument(
        "--run-id", default=None, help="run identifier, defaults to a timestamp"
    )
    output.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="K=V",
        help="experiment label written to every row, repeatable",
    )
    output.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the console summary and progress logs",
    )
    output.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="log progress every N completed requests, 0 disables it",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan without sending requests",
    )
    output.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 1 when any request fails or any stream breaks",
    )

    return parser


def parse_config(argv: Sequence[str] | None = None) -> StressConfig:
    """Validate CLI options and derive the structured configuration."""
    options = build_arg_parser().parse_args(argv)
    return resolve_options(options)


def resolve_options(options: argparse.Namespace) -> StressConfig:
    errors: list[str] = []

    base_url = (
        options.base_url or f"{options.scheme}://{options.host}:{options.port}"
    ).rstrip("/")
    endpoint = (
        options.endpoint if options.endpoint.startswith("/") else f"/{options.endpoint}"
    )
    url = f"{base_url}{endpoint}"

    api_key = options.api_key
    if options.api_key_env:
        api_key = os.environ.get(options.api_key_env) or api_key
        if not api_key:
            errors.append(
                f"environment variable {options.api_key_env} is unset or empty"
            )

    headers = {"Content-Type": "application/json", "Accept": "*/*"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for item in options.header:
        field_name, separator, field_value = item.partition(":")
        if not separator or not field_name.strip():
            errors.append(f"--header must be Key:Value, got {item!r}")
        else:
            headers[field_name.strip()] = field_value.strip()

    qpm: float | None
    if options.request_rate is not None:
        if options.request_rate <= 0:
            errors.append("--request-rate must be > 0")
        request_rate = options.request_rate
        qpm = request_rate * 60.0
    else:
        qpm = 60.0 if options.qpm is None else options.qpm
        if qpm <= 0:
            errors.append("--qpm must be > 0")
        request_rate = qpm / 60.0

    if options.duration_s is not None and options.duration_s <= 0:
        errors.append("--duration-s must be > 0")
    if options.duration_s is None and options.total_requests <= 0:
        errors.append("--total-requests must be > 0")
    if options.warmup < 0:
        errors.append("--warmup cannot be negative")
    if options.max_concurrency < 0:
        errors.append("--max-concurrency cannot be negative")
    if not 0 < options.ramp_start_fraction <= 1:
        errors.append("--ramp-start-fraction must be in (0, 1]")
    if options.ramp_up_s < 0:
        errors.append("--ramp-up-s cannot be negative")
    for field_name in (
        "timeout_s",
        "connect_timeout_s",
        "chunk_timeout_s",
        "outcome_timeout_s",
    ):
        if getattr(options, field_name) <= 0:
            errors.append(f"--{field_name.replace('_', '-')} must be > 0")
    if not 0.25 <= options.speed <= 4.0:
        errors.append("--speed must be within the server range 0.25 to 4.0")
    if options.bit_depth % 8 or options.bit_depth <= 0 or options.channels <= 0:
        errors.append("--channels / --bit-depth is invalid")
    if options.fallback_sample_rate <= 0:
        errors.append("--fallback-sample-rate must be > 0")
    if (
        options.broken_underrun_ms < 0
        or options.stall_broken_s <= 0
        or options.min_audio_duration_s < 0
    ):
        errors.append(
            "stream-health thresholds cannot be negative and --stall-broken-s must be > 0"
        )

    try:
        thresholds = tuple(
            int(value.strip())
            for value in options.underrun_thresholds_ms.split(",")
            if value.strip()
        )
    except ValueError:
        errors.append("--underrun-thresholds-ms must be comma-separated integers")
        thresholds = ()
    if not thresholds or any(value <= 0 for value in thresholds):
        errors.append("--underrun-thresholds-ms needs at least one positive integer")

    labels: dict[str, str] = {}
    for item in options.tag:
        field_name, separator, field_value = item.partition("=")
        if not separator or not field_name.strip():
            errors.append(f"--tag must be K=V, got {item!r}")
        else:
            labels[field_name.strip()] = field_value.strip()

    extra_body = load_json_object(options.extra_body, "--extra-body", errors)
    file_template: dict[str, Any] = {}
    if options.request_file:
        request_path = Path(options.request_file).expanduser()
        if not request_path.is_file():
            errors.append(f"--request-file does not exist: {request_path}")
        else:
            try:
                file_template = load_json_object(
                    request_path.read_text(encoding="utf-8"),
                    "--request-file",
                    errors,
                )
            except OSError as exc:
                errors.append(f"cannot read --request-file: {exc}")

    references: list[Any] = []
    for raw_reference in options.reference:
        try:
            references.append(json.loads(raw_reference))
        except json.JSONDecodeError as exc:
            errors.append(f"--reference is not valid JSON: {exc}")

    corpus = load_corpus(options, errors)
    payload_template = build_payload(
        options, file_template, extra_body, references, errors
    )
    raise_if(errors)

    offsets, total_requests = arrival_offsets(
        count=options.total_requests,
        duration_s=options.duration_s,
        request_rate=request_rate,
        mode=options.arrival_mode,
        seed=options.arrival_seed,
        ramp_up_s=options.ramp_up_s,
        ramp_start_fraction=options.ramp_start_fraction,
    )
    if total_requests <= 0:
        raise_if(
            [
                "the current QPM and --duration-s produce no requests; "
                "raise QPM or lengthen the run"
            ]
        )
    raise_if(errors)

    output = Path(options.output).expanduser().resolve()
    summary_json = (
        Path(options.summary_json).expanduser().resolve()
        if options.summary_json
        else None
    )
    run_id = options.run_id or f"tts-stream-{time.strftime('%Y%m%d-%H%M%S')}"

    return StressConfig(
        base_url=base_url,
        endpoint=endpoint,
        url=url,
        api_key=api_key,
        headers=headers,
        payload_template=payload_template,
        corpus=corpus,
        corpus_path=Path(options.text_jsonl).expanduser().resolve(),
        text_field=options.text_field,
        text_id_field=options.text_id_field,
        text_repeat=options.text_repeat,
        qpm=qpm,
        request_rate=request_rate,
        total_requests=total_requests,
        duration_s=options.duration_s,
        warmup_requests=options.warmup,
        max_concurrency=options.max_concurrency,
        arrival_mode=options.arrival_mode,
        arrival_seed=options.arrival_seed,
        ramp_up_s=options.ramp_up_s,
        ramp_start_fraction=options.ramp_start_fraction,
        arrival_offsets=offsets,
        lag_threshold_s=options.lag_threshold_s,
        timeout_s=options.timeout_s,
        connect_timeout_s=options.connect_timeout_s,
        chunk_timeout_s=options.chunk_timeout_s,
        outcome_timeout_s=options.outcome_timeout_s,
        collect_outcomes=options.collect_outcomes,
        response_format=options.response_format,
        stream_format=options.stream_format,
        sample_rate=options.sample_rate,
        fallback_sample_rate=options.fallback_sample_rate,
        channels=options.channels,
        bit_depth=options.bit_depth,
        underrun_thresholds_ms=thresholds,
        broken_underrun_ms=options.broken_underrun_ms,
        stall_broken_s=options.stall_broken_s,
        min_audio_duration_s=options.min_audio_duration_s,
        output=output,
        summary_json=summary_json,
        omit_chunks=options.omit_chunks,
        run_id=run_id,
        labels=labels,
        quiet=options.quiet,
        log_every=options.log_every,
        dry_run=options.dry_run,
        strict=options.strict,
    )


def raise_if(errors: list[str]) -> None:
    if errors:
        raise ValueError("invalid arguments:\n  - " + "\n  - ".join(errors))
    else:
        pass


def load_json_object(raw: str | None, label: str, errors: list[str]) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"{label} is not valid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    return value


def load_corpus(
    options: argparse.Namespace, errors: list[str]
) -> tuple[CorpusEntry, ...]:
    """Parse the requested jsonl into structured prompt entries."""
    path = Path(options.text_jsonl).expanduser().resolve()
    if not path.is_file():
        errors.append(f"--text-jsonl does not exist: {path}")
        return ()

    id_fields = (options.text_id_field,) if options.text_id_field else SAMPLE_ID_FIELDS
    entries: list[CorpusEntry] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number} cannot parse JSON: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path}:{line_number} must be a JSON object")
                    continue
                value = record.get(options.text_field)
                if not isinstance(value, str) or not value.strip():
                    available = ", ".join(sorted(record.keys())) or "none"
                    errors.append(
                        f"{path}:{line_number} lacks a non-empty string field "
                        f"{options.text_field!r}; available fields: {available}"
                    )
                    continue
                entries.append(
                    CorpusEntry(
                        sample_id=resolve_sample_id(record, id_fields, line_number),
                        text=value.strip(),
                        source_line=line_number,
                    )
                )
                if options.max_texts > 0 and len(entries) >= options.max_texts:
                    break
    except OSError as exc:
        errors.append(f"cannot read --text-jsonl: {exc}")
    if not entries:
        errors.append(f"--text-jsonl produced no usable prompts: {path}")
    return tuple(entries)


def resolve_sample_id(
    record: dict[str, Any], id_fields: tuple[str, ...], line_number: int
) -> str:
    for field_name in id_fields:
        value = record.get(field_name)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"line-{line_number:06d}"


def build_payload(
    options: argparse.Namespace,
    file_template: dict[str, Any],
    extra_body: dict[str, Any],
    references: list[Any],
    errors: list[str],
) -> dict[str, Any]:
    optional = {
        "model": options.model,
        "language": options.language,
        "task_type": options.task_type,
        "instructions": options.instructions,
        "ref_audio": options.ref_audio,
        "ref_text": options.ref_text,
        "initial_codec_chunk_frames": options.initial_codec_chunk_frames,
        "max_new_tokens": options.max_new_tokens,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "top_k": options.top_k,
        "repetition_penalty": options.repetition_penalty,
        "seed": options.seed,
    }
    payload: dict[str, Any] = {
        "model": options.model,
        "voice": options.voice,
        "input": "",
        "response_format": options.response_format,
        "speed": options.speed,
        "stream": True,
        "stream_format": options.stream_format,
    }
    payload.update(
        {
            field_name: value
            for field_name, value in file_template.items()
            if value is not None
        }
    )
    payload.update(
        {
            field_name: value
            for field_name, value in optional.items()
            if value is not None
        }
    )
    if references:
        payload["references"] = references
    payload.update(extra_body)

    payload["stream"] = bool(payload.get("stream", True))
    payload["stream_format"] = options.stream_format
    payload["response_format"] = options.response_format
    if not payload.get("input"):
        payload["input"] = ""
    if payload.get("response_format") not in ("pcm", "wav"):
        errors.append("streaming response_format must be pcm or wav")
    if not payload["stream"]:
        errors.append("this tool only runs streaming workloads; do not disable stream")
    if "input" in extra_body and extra_body["input"]:
        errors.append("--extra-body must not set input; use the text options")
    if file_template.get("input"):
        errors.append("--request-file must not set input; use the text options")
    return {
        field_name: value for field_name, value in payload.items() if value is not None
    }


def arrival_offsets(
    *,
    count: int,
    duration_s: float | None,
    request_rate: float,
    mode: str,
    seed: int,
    ramp_up_s: float,
    ramp_start_fraction: float,
) -> tuple[tuple[float, ...], int]:
    """Build open-loop arrival times; duration_s takes precedence over count."""
    if request_rate <= 0:
        return (), 0
    if mode == "burst":
        return (0.0,) * count, count

    random_generator = random.Random(seed)
    limit = duration_s if duration_s is not None else math.inf
    max_offsets = count if duration_s is None else 1_000_000
    offsets: list[float] = []
    cumulative = 0.0
    while len(offsets) < max_offsets:
        if not offsets:
            cumulative = 0.0
        elif mode == "uniform":
            cumulative = float(len(offsets))
        else:
            cumulative += random_generator.expovariate(1.0)
        moment = inverse_arrival_intensity(
            cumulative, request_rate, ramp_up_s, ramp_start_fraction
        )
        if moment >= limit:
            break
        offsets.append(moment)
    return tuple(offsets), len(offsets)


def inverse_arrival_intensity(
    cumulative: float,
    request_rate: float,
    ramp_up_s: float,
    start_fraction: float,
) -> float:
    """Invert the integrated arrival rate so ramp-up stays smooth."""
    if cumulative <= 0:
        return 0.0
    if ramp_up_s <= 0 or start_fraction >= 1.0:
        return cumulative / request_rate
    lambda_at_ramp_end = request_rate * ramp_up_s * (start_fraction + 1.0) / 2.0
    if cumulative >= lambda_at_ramp_end:
        return ramp_up_s + (cumulative - lambda_at_ramp_end) / request_rate
    quadratic = (1.0 - start_fraction) / (2.0 * ramp_up_s)
    if quadratic < 1e-12:
        return cumulative / (request_rate * start_fraction)
    linear = start_fraction
    constant = -cumulative / request_rate
    root = (-linear + math.sqrt(linear * linear - 4.0 * quadratic * constant)) / (
        2.0 * quadratic
    )
    return max(0.0, root)


def describe_plan(config: StressConfig) -> str:
    """Render the resolved load plan for --dry-run."""
    config_snapshot = config.snapshot()
    offsets = config.arrival_offsets
    head = ", ".join(f"{value:.3f}" for value in offsets[:5])
    tail = (
        ", ".join(f"{value:.3f}" for value in offsets[-3:]) if len(offsets) > 5 else ""
    )
    lines = [
        "TTS stream stress plan (dry run, no requests sent)",
        f"run_id           : {config.run_id}",
        f"target           : POST {config.url}",
        f"stream protocol  : stream_format={config.stream_format}, response_format={config.response_format}",
        f"QPM / RPS        : {config.qpm} / {config.request_rate:.6f}",
        f"arrival          : mode={config.arrival_mode}, ramp_up_s={config.ramp_up_s}, seed={config.arrival_seed}",
        f"requests         : {config.total_requests} + warmup {config.warmup_requests}, concurrency={config.max_concurrency or 'unlimited'}",
        (
            f"planned duration : {offsets[-1]:.3f}s"
            if offsets
            else "planned duration : 0s"
        ),
        f"first offsets    : [{head}{' ... ' + tail if tail else ''}]",
        "payload sample   : "
        + json.dumps(config.payload_for(0), ensure_ascii=False)[:600],
        f"text corpus      : {len(config.corpus)} prompts from {config.corpus_path} "
        f"(field={config.text_field}), first sample_id={config.corpus[0].sample_id} "
        f"text={config.corpus[0].text[:60]!r}",
        f"output           : {config.output}",
        "config snapshot  : "
        + json.dumps(config_snapshot, ensure_ascii=False)[:300]
        + " ...",
    ]
    return "\n".join(lines)
