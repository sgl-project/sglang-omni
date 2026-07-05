# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH

TIMESTAMP_RE = re.compile(r"\[\d+(?:\.\d+)?\]")
SPEAKER_TAG_RE = re.compile(r"\[S0*(\d+)\]", re.IGNORECASE)
SPEAKER_TAG_CANON_RE = re.compile(r"\[S\d+\]", re.IGNORECASE)
BRACKET_EVENT_RE = re.compile(r"\[(?!S\d+\])[^]]+\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DiarizationRow:
    sample_id: str
    audio_path: str
    reference_text: str
    prediction_text: str


@dataclass(frozen=True, slots=True)
class DiarizationSampleMetric:
    sample_id: str
    cer_valid: bool
    cp_cer_valid: bool
    cer_no_spk: float | None
    cp_cer: float | None
    cp_invalid_reason: str


@dataclass(frozen=True, slots=True)
class _CharStats:
    cer: float | None
    ref_chars: int
    pred_chars: int
    errors: int


@dataclass(frozen=True, slots=True)
class _CpCerStats(_CharStats):
    valid: bool
    invalid_reason: str


@dataclass(frozen=True, slots=True)
class DiarizationMetricsResult:
    metrics: dict[str, float | int | None]
    metrics_percent: dict[str, float | int | None]
    samples: list[DiarizationSampleMetric]


def print_diarization_accuracy_summary(
    *,
    summary: Mapping[str, object],
    diarization_metrics: Mapping[str, object],
    model_name: str,
    concurrency: int,
) -> None:
    line_width = SPEED_LINE_WIDTH
    label_width = SPEED_LABEL_WIDTH
    print(f"\n{'=' * line_width}")
    print(f"{'ASR Accuracy Benchmark Result':^{line_width}}")
    print(f"{'=' * line_width}")
    print(f"  {'ASR model:':<{label_width}} {model_name}")
    print(f"  {'Concurrency:':<{label_width}} {concurrency}")
    print(
        f"  {'Evaluated / Total:':<{label_width}} "
        f"{summary['evaluated']}/{summary['total_samples']}"
    )
    print(f"  {'Skipped:':<{label_width}} {summary['skipped']}")
    print(f"{'-' * line_width}")
    print(
        f"  {'Exact match rate:':<{label_width}} "
        f"{summary['exact_match_rate']:.4f} ({summary['exact_match_rate'] * 100:.2f}%)"
    )
    print(f"  {'CER:':<{label_width}} {_format_ratio(diarization_metrics.get('cer'))}")
    print(
        f"  {'CER no speaker:':<{label_width}} "
        f"{_format_ratio(diarization_metrics.get('cer_no_spk'))}"
    )
    print(
        f"  {'cpCER:':<{label_width}} {_format_ratio(diarization_metrics.get('cp_cer'))}"
    )
    print(
        f"  {'Delta CER:':<{label_width}} "
        f"{_format_ratio(diarization_metrics.get('delta_cer'))}"
    )
    print(
        f"  {'CER-valid samples:':<{label_width}} "
        f"{diarization_metrics['cer_valid_samples']}"
    )
    print(
        f"  {'cpCER-valid samples:':<{label_width}} "
        f"{diarization_metrics['cp_cer_valid_samples']}"
    )
    print(f"{'=' * line_width}")


def print_diarization_speed_summary(
    *,
    speed: Mapping[str, object],
    model_name: str,
    concurrency: int,
) -> None:
    line_width = SPEED_LINE_WIDTH
    label_width = SPEED_LABEL_WIDTH
    print(f"\n{'=' * line_width}")
    print(f"{'ASR Speed Benchmark Result':^{line_width}}")
    print(f"{'=' * line_width}")
    print(f"  {'ASR model:':<{label_width}} {model_name}")
    print(f"  {'Concurrency:':<{label_width}} {concurrency}")
    print(f"  {'Completed requests:':<{label_width}} {speed['completed_requests']}")
    print(f"  {'Failed requests:':<{label_width}} {speed['failed_requests']}")
    print(f"{'-' * line_width}")
    print(
        f"  {'Throughput (req/s):':<{label_width}} "
        f"{_format_decimal(speed.get('throughput_qps'), digits=3)}"
    )
    print(
        f"  {'Latency mean / p95 (s):':<{label_width}} "
        f"{_format_decimal(speed.get('latency_mean_s'), digits=3)} / "
        f"{_format_decimal(speed.get('latency_p95_s'), digits=3)}"
    )
    print(
        f"  {'RTF mean / p95:':<{label_width}} "
        f"{_format_decimal(speed.get('rtf_mean'), digits=4)} / "
        f"{_format_decimal(speed.get('rtf_p95'), digits=4)}"
    )
    print(
        f"  {'Audio throughput (s/s):':<{label_width}} "
        f"{_format_decimal(speed.get('audio_throughput_s_per_s'), digits=3)}"
    )
    print(f"{'=' * line_width}")


def compute_diarization_metrics(
    rows: Sequence[DiarizationRow],
) -> DiarizationMetricsResult:
    sample_metrics: list[DiarizationSampleMetric] = []
    cer_stats: list[_CharStats] = []
    cp_stats: list[_CpCerStats] = []
    cer_stats_on_cp_valid: list[_CharStats] = []
    for row in rows:
        sample_cer_stats = _char_stats(
            clean_no_speaker(row.reference_text),
            clean_no_speaker(row.prediction_text),
        )
        sample_cp_stats = cp_cer_stats(row.reference_text, row.prediction_text)
        cer_valid = sample_cer_stats.ref_chars > 0
        cp_valid = sample_cp_stats.valid
        if cer_valid:
            cer_stats.append(sample_cer_stats)
        if cp_valid:
            cp_stats.append(sample_cp_stats)
        if cer_valid and cp_valid:
            cer_stats_on_cp_valid.append(sample_cer_stats)
        sample_metrics.append(
            DiarizationSampleMetric(
                sample_id=row.sample_id,
                cer_valid=cer_valid,
                cp_cer_valid=cp_valid,
                cer_no_spk=sample_cer_stats.cer,
                cp_cer=sample_cp_stats.cer,
                cp_invalid_reason=sample_cp_stats.invalid_reason,
            )
        )

    cer_summary = _sum_char_stats(cer_stats)
    cp_summary = _sum_char_stats(cp_stats)
    cer_on_cp_summary = _sum_char_stats(cer_stats_on_cp_valid)
    delta_cer = None
    if cp_summary.cer is not None and cer_on_cp_summary.cer is not None:
        delta_cer = cp_summary.cer - cer_on_cp_summary.cer
    metrics = {
        "cer_no_spk": cer_summary.cer,
        "cer": cer_summary.cer,
        "cp_cer": cp_summary.cer,
        "cer_no_spk_cp_valid": cer_on_cp_summary.cer,
        "delta_cer": delta_cer,
        "cer_valid_samples": sum(1 for item in sample_metrics if item.cer_valid),
        "cp_cer_valid_samples": sum(1 for item in sample_metrics if item.cp_cer_valid),
        "count": len(sample_metrics),
    }
    metrics_percent = {
        key: (
            value * 100.0
            if key
            in {"cer_no_spk", "cer", "cp_cer", "cer_no_spk_cp_valid", "delta_cer"}
            and value is not None
            else value
        )
        for key, value in metrics.items()
    }
    return DiarizationMetricsResult(
        metrics=metrics,
        metrics_percent=metrics_percent,
        samples=sample_metrics,
    )


def canonicalize_speaker_tags(text: str) -> str:
    return SPEAKER_TAG_RE.sub(lambda match: f"[S{int(match.group(1))}]", text or "")


def clean_no_speaker(text: str) -> str:
    cleaned = _preclean(text)
    return _remove_punct_and_space(SPEAKER_TAG_CANON_RE.sub(" ", cleaned))


def cp_cer_stats(reference: str, prediction: str) -> _CpCerStats:
    if not has_speaker_tags(reference):
        return _invalid_cp_stats("no_ref_speaker_tags", reference, prediction)
    reference_speakers = split_clean_by_speaker(
        reference, implicit_single_speaker=False
    )
    prediction_speakers = split_clean_by_speaker(
        prediction, implicit_single_speaker=True
    )
    reference_texts = list(reference_speakers.values())
    prediction_texts = list(prediction_speakers.values())
    speaker_count = max(len(reference_texts), len(prediction_texts))
    reference_texts.extend([""] * (speaker_count - len(reference_texts)))
    prediction_texts.extend([""] * (speaker_count - len(prediction_texts)))
    if speaker_count == 0 or sum(len(text) for text in reference_texts) == 0:
        return _invalid_cp_stats("empty_ref_after_clean", reference, prediction)
    cost = np.zeros((speaker_count, speaker_count), dtype=np.int64)
    stats_matrix: list[list[_CharStats]] = []
    for row_index, reference_text in enumerate(reference_texts):
        stats_row: list[_CharStats] = []
        for column_index, prediction_text in enumerate(prediction_texts):
            stats = _char_stats(reference_text, prediction_text)
            cost[row_index, column_index] = stats.errors
            stats_row.append(stats)
        stats_matrix.append(stats_row)
    row_indexes, column_indexes = linear_sum_assignment(cost)
    assigned = [
        stats_matrix[row_index][column_index]
        for row_index, column_index in zip(row_indexes, column_indexes, strict=False)
    ]
    summary = _sum_char_stats(assigned)
    is_valid = summary.ref_chars > 0
    return _CpCerStats(
        cer=summary.cer,
        ref_chars=summary.ref_chars,
        pred_chars=summary.pred_chars,
        errors=summary.errors,
        valid=is_valid,
        invalid_reason="" if is_valid else "empty_ref_after_clean",
    )


def has_speaker_tags(text: str) -> bool:
    return bool(SPEAKER_TAG_RE.search(text or ""))


def split_clean_by_speaker(
    text: str, *, implicit_single_speaker: bool
) -> dict[str, str]:
    cleaned = _preclean(text)
    positions = [
        (match.start(), match.end(), match.group())
        for match in SPEAKER_TAG_CANON_RE.finditer(cleaned)
    ]
    if not positions:
        flattened = _remove_punct_and_space(cleaned)
        if not implicit_single_speaker or not flattened:
            return {}
        return {"[S1]": flattened}
    speaker_text: dict[str, str] = {}
    for index, (_start, end, speaker) in enumerate(positions):
        next_start = (
            positions[index + 1][0] if index + 1 < len(positions) else len(cleaned)
        )
        content = _remove_punct_and_space(cleaned[end:next_start])
        if content:
            speaker_text[speaker] = speaker_text.get(speaker, "") + content
    return speaker_text


def _char_stats(reference: str, prediction: str) -> _CharStats:
    if not reference:
        errors = len(prediction or "")
        return _CharStats(
            cer=None if errors else 0.0,
            ref_chars=0,
            pred_chars=len(prediction or ""),
            errors=errors,
        )
    ref_chars = len(reference)
    errors = _levenshtein_distance(reference, prediction or "")
    return _CharStats(
        cer=(errors / ref_chars) if ref_chars > 0 else None,
        ref_chars=ref_chars,
        pred_chars=len(prediction or ""),
        errors=errors,
    )


def _invalid_cp_stats(reason: str, reference: str, prediction: str) -> _CpCerStats:
    prediction_speakers = split_clean_by_speaker(
        prediction, implicit_single_speaker=True
    )
    return _CpCerStats(
        cer=None,
        ref_chars=0,
        pred_chars=sum(len(text) for text in prediction_speakers.values()),
        errors=0,
        valid=False,
        invalid_reason=reason,
    )


def _preclean(text: str) -> str:
    cleaned = TIMESTAMP_RE.sub(" ", text or "")
    cleaned = canonicalize_speaker_tags(cleaned)
    cleaned = re.sub(r"【[^】]*】", " ", cleaned)
    cleaned = re.sub(r"<[^>]*>", " ", cleaned)
    cleaned = re.sub(r"&[^&]{0,40}&", " ", cleaned)
    return BRACKET_EVENT_RE.sub(" ", cleaned)


def _remove_punct_and_space(text: str) -> str:
    return "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    ).lower()


def _sum_char_stats(items: Sequence[_CharStats]) -> _CharStats:
    ref_chars = sum(item.ref_chars for item in items)
    pred_chars = sum(item.pred_chars for item in items)
    errors = sum(item.errors for item in items)
    return _CharStats(
        cer=(errors / ref_chars) if ref_chars > 0 else None,
        ref_chars=ref_chars,
        pred_chars=pred_chars,
        errors=errors,
    )


def _print_metric_line(
    label_width: int,
    label: str,
    metrics: Mapping[str, object],
    key: str,
) -> None:
    value = metrics.get(key)
    if value is None:
        return
    print(f"  {label:<{label_width}} {value}")


def _format_ratio(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f} ({float(value) * 100:.2f}%)"


def _format_decimal(value: float | int | None, *, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _levenshtein_distance(reference: str, prediction: str) -> int:
    previous_row = list(range(len(prediction) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current_row = [reference_index]
        for prediction_index, prediction_character in enumerate(prediction, start=1):
            substitution_cost = 0 if reference_character == prediction_character else 1
            current_row.append(
                min(
                    previous_row[prediction_index] + 1,
                    current_row[prediction_index - 1] + 1,
                    previous_row[prediction_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]
