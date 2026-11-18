# SPDX-License-Identifier: Apache-2.0
"""Identify Qwen3-TTS codec EOS errors in CI without accepting other failures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NotRequired, TypedDict

from benchmarks.eval.benchmark_tts_generation_budget import (
    codec_eos_request_ids,
    is_codec_eos_http_error,
)
from tests.utils import MetricCheckCollector

logger = logging.getLogger(__name__)


class TtsCiRequestOutcome(TypedDict):
    id: str
    is_success: bool
    error: NotRequired[str | None]
    speech_outcome_id: NotRequired[str | None]
    speech_stream_interrupted: NotRequired[bool]
    codec_eos_missing: NotRequired[bool]


def record_codec_eos_failures(
    per_request: list[TtsCiRequestOutcome], worker_logs: list[Path]
) -> None:
    failures = [row for row in per_request if row["is_success"] is False]
    if not failures:
        return
    else:
        pass
    request_ids = codec_eos_request_ids(worker_logs)
    for row in failures:
        if is_codec_eos_http_error(row.get("error")) or (
            row.get("speech_stream_interrupted") is True
            and row.get("speech_outcome_id") in request_ids
        ):
            row["codec_eos_missing"] = True
        else:
            pass


def assert_expected_codec_eos_failures(
    per_request: list[TtsCiRequestOutcome],
    *,
    max_codec_eos_failures: int,
    label: str,
    collector: MetricCheckCollector,
) -> None:
    codec_eos_failures: list[str] = []
    unexpected_failures: list[str] = []
    for request in per_request:
        if request.get("is_success") is True:
            pass
        elif request.get("codec_eos_missing") is True:
            codec_eos_failures.append(request["id"])
        else:
            unexpected_failures.append(request["id"])
    collector.check(
        not unexpected_failures,
        f"{label}: unexpected failures {unexpected_failures[:5]}",
    )
    collector.check(
        len(codec_eos_failures) <= max_codec_eos_failures,
        f"{label}: codec EOS failures {codec_eos_failures[:5]} "
        f"({len(codec_eos_failures)} > {max_codec_eos_failures})",
    )
    if codec_eos_failures:
        logger.warning(
            f"{label}: {len(codec_eos_failures)} missing codec EOS failures {codec_eos_failures[:5]}"
        )
    else:
        pass
