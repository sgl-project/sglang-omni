# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import importlib
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.transcribe_diarize_metrics import (
    DiarizationRow,
    compute_diarization_metrics,
)

MOVIES800_REPO_ID: Final[str] = "zhaochenyang20/movies800time"
EXPECTED_SAMPLE_COUNT: Final[int] = 800
TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(r"\[\d+(?:\.\d+)?\]")
SPEAKER_RE: Final[re.Pattern[str]] = re.compile(r"\[\s*s0*(\d+)\s*\]", re.IGNORECASE)
SPECIAL_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"<\|(?:im_start|im_end|endoftext)\|>"
)

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class Summary(TypedDict):
    total_samples: int
    evaluated: int
    skipped: int
    exact_matches: int
    mismatches: int
    exact_match_rate: float


class PerSampleRecord(TypedDict):
    id: str
    audio_path: str
    expected_text: str
    result_text: str
    expected_normalized: str
    result_normalized: str
    is_exact_match: bool
    latency_s: float | None
    audio_duration_s: float | None
    error: str | None
    cer_no_spk: float | None
    cp_cer: float | None
    cer_valid: bool | None
    cp_cer_valid: bool | None
    cp_invalid_reason: str | None
    speaker_timestamp_der_valid: bool | None
    speaker_timestamp_der_invalid_reason: str | None
    speaker_timestamp_ref_segments: int | None
    speaker_timestamp_pred_segments: int | None
    speaker_timestamp_der: float | None
    speaker_timestamp_der_total_seconds: float | None
    speaker_timestamp_der_false_alarm: float | None
    speaker_timestamp_der_missed_detection: float | None
    speaker_timestamp_der_confusion: float | None


class EvaluationConfig(TypedDict):
    repo_id: str
    split: str
    model_path: str
    concurrency: int
    wall_clock_s: float


class EvaluationPayload(TypedDict):
    config: EvaluationConfig
    summary: Summary
    speed: dict[str, JSONScalar]
    diarization_metrics: dict[str, float | int | None]
    diarization_metrics_percent: dict[str, float | int | None]
    per_sample: list[PerSampleRecord]


@dataclass(frozen=True, slots=True)
class Movies800Sample:
    sample_id: str
    audio_path: str
    expected_text: str


def load_movies800_samples(
    repo_id: str,
    split: str,
    audio_column: str,
    expected_column: str,
    max_samples: int | None = None,
    expected_sample_count: int | None = EXPECTED_SAMPLE_COUNT,
) -> list[Movies800Sample]:
    datasets_module = importlib.import_module("datasets")
    audio_type = getattr(datasets_module, "Audio")
    load_dataset = getattr(datasets_module, "load_dataset")
    dataset = load_dataset(repo_id, split=split)
    if expected_column not in dataset.column_names:
        raise ValueError(
            f"Dataset {repo_id}/{split} is missing expected column {expected_column!r}. "
            f"Available columns: {dataset.column_names}"
        )
    if audio_column not in dataset.column_names:
        raise ValueError(
            f"Dataset {repo_id}/{split} is missing audio column {audio_column!r}. "
            f"Available columns: {dataset.column_names}"
        )
    dataset = dataset.cast_column(audio_column, audio_type(decode=False))
    if max_samples is not None:
        dataset = dataset.select(list(range(min(max_samples, len(dataset)))))

    staging_dir = Path(tempfile.mkdtemp(prefix=f"movies800_{split}_"))
    atexit.register(shutil.rmtree, str(staging_dir), True)
    samples = [
        Movies800Sample(
            sample_id=_sample_id_from_row(row, index),
            audio_path=_resolve_audio_path(
                row=row,
                index=index,
                audio_column=audio_column,
                staging_dir=staging_dir,
            ),
            expected_text=_require_string(row, expected_column, allow_empty=True),
        )
        for index, row in enumerate(dataset)
    ]
    if (
        max_samples is None
        and expected_sample_count is not None
        and len(samples) != expected_sample_count
    ):
        raise ValueError(
            f"Expected {expected_sample_count} samples for the full {repo_id} run, got {len(samples)}"
        )
    return samples


CONCAT_SEGMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"\[(\d+(?:\.\d+)?)\]\[S0*(\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL
)


def merge_concat_references(
    clip_references: Sequence[str],
    clip_offsets_s: Sequence[float],
    keep_until_s: Sequence[float | None],
) -> str:
    # note (db-ol): speaker labels are per clip ordinals, so the same tag in
    # two clips names two different people. Remapping each clip to a disjoint
    # range keeps the merged reference readable as one meeting.
    merged: list[str] = []
    speaker_base = 0
    for reference, offset_s, limit_s in zip(
        clip_references, clip_offsets_s, keep_until_s, strict=True
    ):
        max_speaker = 0
        for match in CONCAT_SEGMENT_RE.finditer(reference):
            start = float(match.group(1))
            speaker = int(match.group(2))
            end = float(match.group(4))
            max_speaker = max(max_speaker, speaker)
            if limit_s is not None and end > limit_s:
                continue
            merged.append(
                f"[{start + offset_s:.2f}][S{speaker + speaker_base:02d}]"
                f"{match.group(3)}[{end + offset_s:.2f}]"
            )
        if max_speaker == 0:
            raise ValueError("Clip reference has no parseable timestamped segments")
        speaker_base += max_speaker
    return "".join(merged)


def _filler_cut_boundary_s(reference: str, limit_s: float) -> float:
    ends = [
        end
        for match in CONCAT_SEGMENT_RE.finditer(reference)
        if (end := float(match.group(4))) <= limit_s
    ]
    if not ends:
        raise ValueError(
            f"filler clip has no reference segment ending within {limit_s:.2f}s"
        )
    return max(ends)


def build_long_audio_concat_sample(
    clips: Sequence[Movies800Sample],
    target_duration_s: float,
    gap_s: float = 0.8,
    sample_id: str = "long_audio_concat",
) -> Movies800Sample:
    # note (db-ol): the first clip is the filler, cut at a reference boundary
    # and silence padded so the total lands on target_duration_s. Later clips
    # are appended whole, so their content sits at the deepest positions.
    import numpy as np
    import soundfile as sf

    waveforms = []
    sample_rate = None
    for clip in clips:
        waveform, rate = sf.read(clip.audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(
                f"Clip {clip.sample_id} sample rate {rate} != {sample_rate}"
            )
        waveforms.append(waveform.astype(np.float32))

    tail_s = sum(len(w) for w in waveforms[1:]) / sample_rate
    filler_s = target_duration_s - tail_s - gap_s * (len(waveforms) - 1)
    if filler_s <= 0:
        raise ValueError(
            f"target_duration_s={target_duration_s} leaves no room for the "
            f"filler clip ({tail_s:.1f}s of whole clips already)"
        )
    if filler_s > len(waveforms[0]) / sample_rate:
        raise ValueError(
            f"filler clip is {len(waveforms[0]) / sample_rate:.1f}s but "
            f"{filler_s:.1f}s are needed to reach target_duration_s"
        )
    cut_s = _filler_cut_boundary_s(clips[0].expected_text, filler_s)
    kept = waveforms[0][: int(cut_s * sample_rate)]
    padding = np.zeros(int(filler_s * sample_rate) - len(kept), dtype=np.float32)
    waveforms[0] = np.concatenate([kept, padding])

    gap = np.zeros(int(gap_s * sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    offsets: list[float] = []
    cursor = 0.0
    for position, waveform in enumerate(waveforms):
        if position > 0:
            parts.append(gap)
            cursor += gap_s
        offsets.append(cursor)
        parts.append(waveform)
        cursor += len(waveform) / sample_rate

    merged_reference = merge_concat_references(
        [clip.expected_text for clip in clips],
        offsets,
        [cut_s] + [None] * (len(clips) - 1),
    )

    staging_dir = Path(tempfile.mkdtemp(prefix=f"{sample_id}_"))
    atexit.register(shutil.rmtree, str(staging_dir), True)
    audio_path = staging_dir / f"{sample_id}.wav"
    sf.write(str(audio_path), np.concatenate(parts), sample_rate)
    return Movies800Sample(
        sample_id=sample_id,
        audio_path=str(audio_path),
        expected_text=merged_reference,
    )


def extract_prediction_text(payload: Mapping[str, JSONValue]) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    segments = payload.get("segments")
    if isinstance(segments, list):
        texts = [
            segment_text
            for segment in segments
            if (segment_text := _segment_text(segment))
        ]
        if texts:
            return " ".join(texts)
    if not isinstance(text, str):
        raise ValueError("Transcription response is missing a string 'text' field")
    return text.strip()


def normalize_transcribe_diarize_text(text: str) -> str:
    normalized = SPECIAL_TOKEN_RE.sub(" ", text)
    normalized = TIMESTAMP_RE.sub(" ", normalized)
    normalized = SPEAKER_RE.sub(
        lambda match: f"[S{int(match.group(1)):02d}]", normalized
    )
    normalized = re.sub(r"(\[S\d+\])(?=\S)", r"\1 ", normalized)
    normalized = re.sub(r"\s+([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def build_evaluation_payload(
    samples: list[Movies800Sample],
    outputs: list[RequestResult],
    wall_clock_s: float,
    model_path: str,
    concurrency: int,
    *,
    repo_id: str = MOVIES800_REPO_ID,
    split: str = "validation",
    dataset: str | None = None,
) -> EvaluationPayload:
    result_by_id = {result.request_id: result for result in outputs}
    successful_rows: list[DiarizationRow] = []
    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        if result and result.is_success:
            successful_rows.append(
                DiarizationRow(
                    sample_id=sample.sample_id,
                    audio_path=sample.audio_path,
                    reference_text=sample.expected_text,
                    prediction_text=result.text,
                )
            )
    diarization_metrics = compute_diarization_metrics(successful_rows)
    diarization_by_id = {
        sample_metric.sample_id: sample_metric
        for sample_metric in diarization_metrics.samples
    }
    per_sample: list[PerSampleRecord] = []
    exact_matches = 0
    evaluated = 0
    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        diarization_sample = diarization_by_id.get(sample.sample_id)
        actual_text = result.text if result and result.is_success else ""
        expected_norm = normalize_transcribe_diarize_text(sample.expected_text)
        actual_norm = (
            normalize_transcribe_diarize_text(actual_text) if actual_text else ""
        )
        is_exact_match = bool(
            result and result.is_success and expected_norm == actual_norm
        )
        if result and result.is_success:
            evaluated += 1
        if is_exact_match:
            exact_matches += 1
        per_sample.append(
            {
                "id": sample.sample_id,
                "audio_path": sample.audio_path,
                "expected_text": sample.expected_text,
                "result_text": actual_text,
                "expected_normalized": expected_norm,
                "result_normalized": actual_norm,
                "is_exact_match": is_exact_match,
                "latency_s": result.latency_s if result else None,
                "audio_duration_s": result.audio_duration_s if result else None,
                "error": result.error if result and result.error else None,
                "cer_no_spk": (
                    diarization_sample.cer_no_spk if diarization_sample else None
                ),
                "cp_cer": diarization_sample.cp_cer if diarization_sample else None,
                "cer_valid": (
                    diarization_sample.cer_valid if diarization_sample else None
                ),
                "cp_cer_valid": (
                    diarization_sample.cp_cer_valid if diarization_sample else None
                ),
                "cp_invalid_reason": (
                    diarization_sample.cp_invalid_reason if diarization_sample else None
                ),
                "speaker_timestamp_der_valid": (
                    diarization_sample.speaker_timestamp_der_valid
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_der_invalid_reason": (
                    diarization_sample.speaker_timestamp_der_invalid_reason
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_ref_segments": (
                    diarization_sample.speaker_timestamp_ref_segments
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_pred_segments": (
                    diarization_sample.speaker_timestamp_pred_segments
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_der": (
                    diarization_sample.speaker_timestamp_der
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_der_total_seconds": (
                    diarization_sample.speaker_timestamp_der_total_seconds
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_der_false_alarm": (
                    diarization_sample.speaker_timestamp_der_false_alarm
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_der_missed_detection": (
                    diarization_sample.speaker_timestamp_der_missed_detection
                    if diarization_sample
                    else None
                ),
                "speaker_timestamp_der_confusion": (
                    diarization_sample.speaker_timestamp_der_confusion
                    if diarization_sample
                    else None
                ),
            }
        )
    summary: Summary = {
        "total_samples": len(samples),
        "evaluated": evaluated,
        "skipped": len(samples) - evaluated,
        "exact_matches": exact_matches,
        "mismatches": evaluated - exact_matches,
        "exact_match_rate": exact_matches / evaluated if evaluated else 0.0,
    }
    speed = {
        key: value
        for key, value in compute_speed_metrics(
            outputs, wall_clock_s=wall_clock_s
        ).items()
        if isinstance(value, str | int | float | bool) or value is None
    }
    return {
        "config": {
            "dataset": dataset,
            "repo_id": repo_id,
            "split": split,
            "model_path": model_path,
            "concurrency": concurrency,
            "wall_clock_s": wall_clock_s,
        },
        "summary": summary,
        "speed": speed,
        "diarization_metrics": diarization_metrics.metrics,
        "diarization_metrics_percent": diarization_metrics.metrics_percent,
        "per_sample": per_sample,
    }


def _sample_id_from_row(row: Mapping[str, JSONValue], index: int) -> str:
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name:
        return file_name
    return f"sample-{index:06d}"


def _resolve_audio_path(
    *,
    row: Mapping[str, JSONValue],
    index: int,
    audio_column: str,
    staging_dir: Path,
) -> str:
    audio_value = row.get(audio_column)
    if isinstance(audio_value, Mapping):
        audio_path = audio_value.get("path")
        if isinstance(audio_path, str) and Path(audio_path).is_file():
            return audio_path
        audio_bytes = audio_value.get("bytes")
        if isinstance(audio_bytes, bytes | bytearray):
            suffix = Path(_sample_id_from_row(row, index)).suffix or ".wav"
            staged_path = staging_dir / f"sample_{index:06d}{suffix}"
            staged_path.write_bytes(bytes(audio_bytes))
            return str(staged_path)
    raise ValueError(
        f"Row {index} is missing loadable audio bytes/path in column {audio_column!r}"
    )


def _require_string(
    row: Mapping[str, JSONValue],
    field_name: str,
    allow_empty: bool = False,
) -> str:
    value = row.get(field_name)
    expected_description = "a string" if allow_empty else "a non-empty string"
    if not isinstance(value, str):
        raise ValueError(f"Field {field_name!r} must be {expected_description}")
    if allow_empty:
        return value.strip()
    if not value.strip():
        raise ValueError(f"Field {field_name!r} must be {expected_description}")
    return value.strip()


def _segment_text(segment: JSONValue) -> str | None:
    if isinstance(segment, Mapping):
        text = segment.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None
