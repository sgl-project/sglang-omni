# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH

TIMESTAMP_RE = re.compile(r"\[\d+(?:\.\d+)?\]")
TIMESTAMP_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")
SPEAKER_TOKEN_RE = re.compile(r"^S0*(\d+)$", re.IGNORECASE)
SPEAKER_TAG_RE = re.compile(r"\[S0*(\d+)\]", re.IGNORECASE)
SPEAKER_TAG_CANON_RE = re.compile(r"\[S\d+\]", re.IGNORECASE)
BRACKET_EVENT_RE = re.compile(r"\[(?!S\d+\])[^]]+\]", re.IGNORECASE)
_CER_ABOVE_50_FRACTION = 0.5


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
    speaker_timestamp_der_valid: bool
    speaker_timestamp_der_invalid_reason: str
    speaker_timestamp_ref_segments: int
    speaker_timestamp_pred_segments: int
    speaker_timestamp_der: float | None
    speaker_timestamp_der_total_seconds: float
    speaker_timestamp_der_false_alarm: float
    speaker_timestamp_der_missed_detection: float
    speaker_timestamp_der_confusion: float


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
class _CerPartitionStats:
    below_50_corpus_cer: float | None
    n_above_50: int
    pct_above_50: float


@dataclass(frozen=True, slots=True)
class DiarizationMetricsResult:
    metrics: dict[str, float | int | None]
    metrics_percent: dict[str, float | int | None]
    samples: list[DiarizationSampleMetric]


@dataclass(frozen=True, slots=True)
class _TimestampDerSample:
    valid: bool
    invalid_reason: str
    ref_segments: int
    pred_segments: int
    der: float | None
    total_seconds: float
    false_alarm: float
    missed_detection: float
    confusion: float


@dataclass(frozen=True, slots=True)
class _TimestampDerAggregate:
    overall_der: float | None
    valid_samples: int
    skipped: int
    skipped_parse_error: int
    skipped_no_ref_segments: int
    skipped_no_pred_segments: int
    compute_error: str
    total_seconds: float
    false_alarm: float
    missed_detection: float
    confusion: float
    per_sample: list[_TimestampDerSample]


class _TimestampDerDetail(TypedDict):
    der: float | None
    total: float
    false_alarm: float
    missed_detection: float
    confusion: float


class _TimestampDerSampleDetail(_TimestampDerDetail):
    sample_index: int


class _TimestampDerAggregateDetail(TypedDict):
    overall_der: float | None
    sample_details: list[_TimestampDerSampleDetail]
    total: float
    false_alarm: float
    missed_detection: float
    confusion: float


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
        f"{_as_float(summary, 'exact_match_rate'):.4f} ({_as_float(summary, 'exact_match_rate') * 100:.2f}%)"
    )
    print(
        f"  {'CER:':<{label_width}} {_format_ratio(_as_optional_number(diarization_metrics, 'cer'))}"
    )
    print(
        f"  {'CER no speaker:':<{label_width}} "
        f"{_format_ratio(_as_optional_number(diarization_metrics, 'cer_no_spk'))}"
    )
    print(
        f"  {'CER no-spk corpus (excl >50%):':<{label_width}} "
        f"{_format_ratio(_as_optional_number(diarization_metrics, 'cer_no_spk_below_50_corpus'))}"
    )
    print(
        f"  {'>50% CER samples:':<{label_width}} "
        f"{diarization_metrics.get('n_above_50_pct_cer', 0)} "
        f"({diarization_metrics.get('pct_above_50_pct_cer', 0):.1f}%)"
    )
    print(
        f"  {'cpCER:':<{label_width}} {_format_ratio(_as_optional_number(diarization_metrics, 'cp_cer'))}"
    )
    print(
        f"  {'Delta CER:':<{label_width}} "
        f"{_format_ratio(_as_optional_number(diarization_metrics, 'delta_cer'))}"
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
        f"{_format_decimal(_as_optional_number(speed, 'throughput_qps'), digits=3)}"
    )
    print(
        f"  {'Latency mean / p95 (s):':<{label_width}} "
        f"{_format_decimal(_as_optional_number(speed, 'latency_mean_s'), digits=3)} / "
        f"{_format_decimal(_as_optional_number(speed, 'latency_p95_s'), digits=3)}"
    )
    print(
        f"  {'RTF mean / p95:':<{label_width}} "
        f"{_format_decimal(_as_optional_number(speed, 'rtf_mean'), digits=4)} / "
        f"{_format_decimal(_as_optional_number(speed, 'rtf_p95'), digits=4)}"
    )
    print(
        f"  {'Audio throughput (s/s):':<{label_width}} "
        f"{_format_decimal(_as_optional_number(speed, 'audio_throughput_s_per_s'), digits=3)}"
    )
    print(f"{'=' * line_width}")


def compute_diarization_metrics(
    rows: Sequence[DiarizationRow],
) -> DiarizationMetricsResult:
    timestamp_der = _compute_timestamp_der(
        [(row.reference_text, row.prediction_text) for row in rows],
        collar=0.0,
    )
    sample_metrics: list[DiarizationSampleMetric] = []
    cer_stats: list[_CharStats] = []
    cp_stats: list[_CpCerStats] = []
    cer_stats_on_cp_valid: list[_CharStats] = []
    for row, timestamp_sample in zip(rows, timestamp_der.per_sample, strict=False):
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
                speaker_timestamp_der_valid=timestamp_sample.valid,
                speaker_timestamp_der_invalid_reason=timestamp_sample.invalid_reason,
                speaker_timestamp_ref_segments=timestamp_sample.ref_segments,
                speaker_timestamp_pred_segments=timestamp_sample.pred_segments,
                speaker_timestamp_der=timestamp_sample.der,
                speaker_timestamp_der_total_seconds=timestamp_sample.total_seconds,
                speaker_timestamp_der_false_alarm=timestamp_sample.false_alarm,
                speaker_timestamp_der_missed_detection=timestamp_sample.missed_detection,
                speaker_timestamp_der_confusion=timestamp_sample.confusion,
            )
        )

    cer_summary = _sum_char_stats(cer_stats)
    cp_summary = _sum_char_stats(cp_stats)
    cer_on_cp_summary = _sum_char_stats(cer_stats_on_cp_valid)
    cer_partition = _partition_cer_stats(cer_stats)
    delta_cer = None
    if cp_summary.cer is not None and cer_on_cp_summary.cer is not None:
        delta_cer = cp_summary.cer - cer_on_cp_summary.cer
    metrics = {
        "cer_no_spk": cer_summary.cer,
        "cer": cer_summary.cer,
        "cer_no_spk_below_50_corpus": cer_partition.below_50_corpus_cer,
        "n_above_50_pct_cer": cer_partition.n_above_50,
        "pct_above_50_pct_cer": cer_partition.pct_above_50,
        "cp_cer": cp_summary.cer,
        "cer_no_spk_cp_valid": cer_on_cp_summary.cer,
        "delta_cer": delta_cer,
        "cer_valid_samples": sum(1 for item in sample_metrics if item.cer_valid),
        "cp_cer_valid_samples": sum(1 for item in sample_metrics if item.cp_cer_valid),
        "speaker_timestamp_der": timestamp_der.overall_der,
        "speaker_timestamp_der_collar": 0.0,
        "speaker_timestamp_der_valid_samples": timestamp_der.valid_samples,
        "speaker_timestamp_der_skipped": timestamp_der.skipped,
        "speaker_timestamp_der_skipped_parse_error": timestamp_der.skipped_parse_error,
        "speaker_timestamp_der_skipped_no_ref_segments": timestamp_der.skipped_no_ref_segments,
        "speaker_timestamp_der_skipped_no_pred_segments": timestamp_der.skipped_no_pred_segments,
        "speaker_timestamp_der_compute_error": timestamp_der.compute_error,
        "speaker_timestamp_der_total_seconds": timestamp_der.total_seconds,
        "speaker_timestamp_der_false_alarm": timestamp_der.false_alarm,
        "speaker_timestamp_der_missed_detection": timestamp_der.missed_detection,
        "speaker_timestamp_der_confusion": timestamp_der.confusion,
        "count": len(sample_metrics),
    }
    metrics_percent = {
        key: (
            value * 100.0
            if key
            in {
                "cer_no_spk",
                "cer",
                "cer_no_spk_below_50_corpus",
                "cp_cer",
                "cer_no_spk_cp_valid",
                "delta_cer",
                "speaker_timestamp_der",
            }
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


def _partition_cer_stats(cer_stats: Sequence[_CharStats]) -> _CerPartitionStats:
    """Partition corpus CER like WER: global stats plus an excl->50% subset."""
    if not cer_stats:
        return _CerPartitionStats(
            below_50_corpus_cer=None,
            n_above_50=0,
            pct_above_50=0.0,
        )
    n_above_50 = sum(
        1
        for stats in cer_stats
        if stats.cer is not None and stats.cer > _CER_ABOVE_50_FRACTION
    )
    below_50_stats = [
        stats
        for stats in cer_stats
        if stats.cer is not None and stats.cer <= _CER_ABOVE_50_FRACTION
    ]
    below_50_summary = _sum_char_stats(below_50_stats)
    return _CerPartitionStats(
        below_50_corpus_cer=below_50_summary.cer,
        n_above_50=n_above_50,
        pct_above_50=(n_above_50 / len(cer_stats) * 100.0 if cer_stats else 0.0),
    )


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


def _as_float(metrics: Mapping[str, object], key: str) -> float:
    value = metrics[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Metric {key!r} must be numeric, got {type(value).__name__}")
    return float(value)


def _as_optional_number(metrics: Mapping[str, object], key: str) -> float | int | None:
    value = metrics.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(
            f"Metric {key!r} must be numeric or None, got {type(value).__name__}"
        )
    return value


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


def _compute_timestamp_der(
    rows: Sequence[tuple[str, str]],
    *,
    collar: float = 0.0,
) -> _TimestampDerAggregate:
    per_sample = [
        _timestamp_der_validity(reference, prediction) for reference, prediction in rows
    ]
    valid_items = [
        (index, reference, prediction)
        for index, ((reference, prediction), sample) in enumerate(
            zip(rows, per_sample, strict=False)
        )
        if sample.valid
    ]
    overall_der: float | None = None
    total_seconds = 0.0
    false_alarm = 0.0
    missed_detection = 0.0
    confusion = 0.0
    compute_error = ""
    if valid_items:
        try:
            details = _timestamp_der_detail(
                predictions=[prediction for _, _, prediction in valid_items],
                references=[reference for _, reference, _ in valid_items],
                collar=collar,
            )
            overall_der = details["overall_der"]
            total_seconds = details["total"]
            false_alarm = details["false_alarm"]
            missed_detection = details["missed_detection"]
            confusion = details["confusion"]
            for (sample_index, _reference, _prediction), detail in zip(
                valid_items,
                details["sample_details"],
                strict=False,
            ):
                sample = per_sample[sample_index]
                per_sample[sample_index] = _TimestampDerSample(
                    valid=sample.valid,
                    invalid_reason=sample.invalid_reason,
                    ref_segments=sample.ref_segments,
                    pred_segments=sample.pred_segments,
                    der=detail["der"],
                    total_seconds=detail["total"],
                    false_alarm=detail["false_alarm"],
                    missed_detection=detail["missed_detection"],
                    confusion=detail["confusion"],
                )
        except Exception as exc:
            compute_error = str(exc)
            for sample_index, _reference, _prediction in valid_items:
                sample = per_sample[sample_index]
                per_sample[sample_index] = _TimestampDerSample(
                    valid=False,
                    invalid_reason="der_compute_error",
                    ref_segments=sample.ref_segments,
                    pred_segments=sample.pred_segments,
                    der=None,
                    total_seconds=0.0,
                    false_alarm=0.0,
                    missed_detection=0.0,
                    confusion=0.0,
                )

    return _TimestampDerAggregate(
        overall_der=overall_der,
        valid_samples=sum(1 for item in per_sample if item.valid),
        skipped=sum(1 for item in per_sample if not item.valid),
        skipped_parse_error=sum(
            1 for item in per_sample if item.invalid_reason == "timestamp_parse_error"
        ),
        skipped_no_ref_segments=sum(
            1
            for item in per_sample
            if item.invalid_reason == "no_ref_timestamped_speaker_segments"
        ),
        skipped_no_pred_segments=sum(
            1
            for item in per_sample
            if item.invalid_reason == "no_pred_timestamped_speaker_segments"
        ),
        compute_error=compute_error,
        total_seconds=total_seconds,
        false_alarm=false_alarm,
        missed_detection=missed_detection,
        confusion=confusion,
        per_sample=per_sample,
    )


def _timestamp_der_validity(reference: str, prediction: str) -> _TimestampDerSample:
    try:
        ref_segments = len(_parse_timestamped_speaker_segments(reference or ""))
        pred_segments = len(_parse_timestamped_speaker_segments(prediction or ""))
    except Exception:
        return _TimestampDerSample(
            valid=False,
            invalid_reason="timestamp_parse_error",
            ref_segments=0,
            pred_segments=0,
            der=None,
            total_seconds=0.0,
            false_alarm=0.0,
            missed_detection=0.0,
            confusion=0.0,
        )
    if ref_segments <= 0:
        return _TimestampDerSample(
            valid=False,
            invalid_reason="no_ref_timestamped_speaker_segments",
            ref_segments=ref_segments,
            pred_segments=pred_segments,
            der=None,
            total_seconds=0.0,
            false_alarm=0.0,
            missed_detection=0.0,
            confusion=0.0,
        )
    if pred_segments <= 0:
        return _TimestampDerSample(
            valid=False,
            invalid_reason="no_pred_timestamped_speaker_segments",
            ref_segments=ref_segments,
            pred_segments=pred_segments,
            der=None,
            total_seconds=0.0,
            false_alarm=0.0,
            missed_detection=0.0,
            confusion=0.0,
        )
    return _TimestampDerSample(
        valid=True,
        invalid_reason="",
        ref_segments=ref_segments,
        pred_segments=pred_segments,
        der=None,
        total_seconds=0.0,
        false_alarm=0.0,
        missed_detection=0.0,
        confusion=0.0,
    )


def _parse_timestamped_speaker_segments(text: str) -> list[tuple[float, float, str]]:
    tokens: list[tuple[str, float | str]] = []
    for match in re.finditer(r"\[([^\]]+)\]", text or ""):
        raw = match.group(1).strip()
        if TIMESTAMP_TOKEN_RE.fullmatch(raw):
            tokens.append(("time", float(raw)))
            continue
        speaker_match = SPEAKER_TOKEN_RE.fullmatch(raw)
        if speaker_match:
            tokens.append(("spk", f"[S{int(speaker_match.group(1))}]"))

    segments: list[tuple[float, float, str]] = []
    last_speaker: str | None = None
    index = 0
    token_count = len(tokens)
    while index < token_count:
        kind, value = tokens[index]
        if kind != "time":
            index += 1
            continue
        start = float(value)
        next_index = index + 1
        segment_speaker: str | None = None
        while next_index < token_count and tokens[next_index][0] != "time":
            if tokens[next_index][0] == "spk":
                segment_speaker = str(tokens[next_index][1])
            next_index += 1
        if next_index >= token_count:
            break
        end = float(tokens[next_index][1])
        speaker = segment_speaker or last_speaker
        if speaker is not None and end > start:
            segments.append((start, end, speaker))
            last_speaker = speaker
        index = next_index + 1
    return segments


def _timestamp_der_detail(
    predictions: list[str],
    references: list[str],
    *,
    collar: float,
) -> _TimestampDerAggregateDetail:
    sample_details: list[_TimestampDerSampleDetail] = []
    total = 0.0
    false_alarm = 0.0
    missed_detection = 0.0
    confusion = 0.0
    for sample_index, (prediction, reference) in enumerate(
        zip(predictions, references, strict=False)
    ):
        detail = _timestamp_der_one(reference, prediction, collar=collar)
        sample_details.append(
            {
                "sample_index": sample_index,
                "der": detail["der"],
                "total": detail["total"],
                "false_alarm": detail["false_alarm"],
                "missed_detection": detail["missed_detection"],
                "confusion": detail["confusion"],
            }
        )
        total += detail["total"]
        false_alarm += detail["false_alarm"]
        missed_detection += detail["missed_detection"]
        confusion += detail["confusion"]
    errors = false_alarm + missed_detection + confusion
    return {
        "overall_der": errors / total if total > 0 else None,
        "sample_details": sample_details,
        "total": total,
        "false_alarm": false_alarm,
        "missed_detection": missed_detection,
        "confusion": confusion,
    }


def _timestamp_der_one(
    reference: str,
    prediction: str,
    *,
    collar: float,
) -> _TimestampDerDetail:
    reference_segments = _parse_timestamped_speaker_segments(reference)
    prediction_segments = _parse_timestamped_speaker_segments(prediction)
    mapping = _best_hyp_to_ref_mapping(reference_segments, prediction_segments)
    collar_regions = _reference_collar_regions(reference_segments, collar)
    boundaries = sorted(
        {
            point
            for start, end, _speaker in reference_segments + prediction_segments
            for point in (start, end)
        }
        | {point for start, end in collar_regions for point in (start, end)}
    )

    total = 0.0
    false_alarm = 0.0
    missed_detection = 0.0
    confusion = 0.0
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        duration = end - start
        if duration <= 0:
            continue
        midpoint = (start + end) / 2.0
        if _inside_intervals(midpoint, collar_regions):
            continue
        reference_active = _active_speakers(reference_segments, start, end)
        prediction_active_raw = _active_speakers(prediction_segments, start, end)
        if not reference_active and not prediction_active_raw:
            continue
        prediction_active = {
            mapping.get(speaker, f"__hyp_unmapped_{speaker}")
            for speaker in prediction_active_raw
        }
        total += len(reference_active) * duration
        false_alarm += (
            max(0, len(prediction_active_raw) - len(reference_active)) * duration
        )
        missed_detection += (
            max(0, len(reference_active) - len(prediction_active_raw)) * duration
        )
        confusion += (
            max(
                0,
                min(len(reference_active), len(prediction_active_raw))
                - len(reference_active & prediction_active),
            )
            * duration
        )
    errors = false_alarm + missed_detection + confusion
    return {
        "der": errors / total if total > 0 else None,
        "total": total,
        "false_alarm": false_alarm,
        "missed_detection": missed_detection,
        "confusion": confusion,
    }


def _best_hyp_to_ref_mapping(
    reference_segments: list[tuple[float, float, str]],
    prediction_segments: list[tuple[float, float, str]],
) -> dict[str, str]:
    reference_labels = sorted(
        {segment[2] for segment in reference_segments}, key=_speaker_sort_key
    )
    prediction_labels = sorted(
        {segment[2] for segment in prediction_segments}, key=_speaker_sort_key
    )
    if not reference_labels or not prediction_labels:
        return {}
    overlap = np.zeros((len(reference_labels), len(prediction_labels)), dtype=float)
    reference_index = {label: index for index, label in enumerate(reference_labels)}
    prediction_index = {label: index for index, label in enumerate(prediction_labels)}
    for reference_segment in reference_segments:
        for prediction_segment in prediction_segments:
            duration = _segment_overlap(reference_segment, prediction_segment)
            if duration > 0.0:
                overlap[
                    reference_index[reference_segment[2]],
                    prediction_index[prediction_segment[2]],
                ] += duration
    row_indexes, column_indexes = linear_sum_assignment(-overlap)
    return {
        prediction_labels[column_index]: reference_labels[row_index]
        for row_index, column_index in zip(row_indexes, column_indexes, strict=False)
    }


def _segment_overlap(
    left: tuple[float, float, str],
    right: tuple[float, float, str],
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _speaker_sort_key(label: str) -> tuple[int, str]:
    match = SPEAKER_TAG_RE.fullmatch(label)
    return (int(match.group(1)), label) if match else (10**9, label)


def _reference_collar_regions(
    reference_segments: list[tuple[float, float, str]],
    collar: float,
) -> list[tuple[float, float]]:
    if collar <= 0:
        return []
    intervals = []
    for start, end, _speaker in reference_segments:
        intervals.append((max(0.0, start - collar), start + collar))
        intervals.append((max(0.0, end - collar), end + collar))
    return _merge_intervals(intervals)


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((item for item in intervals if item[1] > item[0])):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _inside_intervals(
    time_point: float,
    intervals: list[tuple[float, float]],
) -> bool:
    return any(start <= time_point < end for start, end in intervals)


def _active_speakers(
    segments: list[tuple[float, float, str]],
    start: float,
    end: float,
) -> set[str]:
    return {
        speaker
        for segment_start, segment_end, speaker in segments
        if segment_start < end and segment_end > start
    }
