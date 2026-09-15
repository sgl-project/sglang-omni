# SPDX-License-Identifier: Apache-2.0
"""Audio validity and transcript agreement for Restage TTS trials."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import apply_wer
from sglang_omni.restage.evaluation import QualityReport

# Note (Jiaxin Deng): a runaway generation inserts many words; it fails its own
# request instead of dragging the corpus WER of every other clip.
RUNAWAY_WER = 0.5


async def evaluate_tts_quality(
    results: Sequence[RequestResult],
    *,
    targets: Mapping[str, str],
    transcribe: Callable[[list[SampleInput]], Awaitable[list[RequestResult]]],
    output: Path,
    lang: str,
    max_wer: float,
) -> QualityReport:
    """Check saved audio and transcript agreement; this does not assess voice similarity."""
    if not math.isfinite(max_wer) or max_wer < 0:
        raise ValueError("max_wer must be finite and nonnegative")
    if len({r.request_id for r in results}) != len(results):
        raise ValueError("TTS request IDs must be unique")
    details: dict[str, SampleOutput] = {}
    samples = []
    for result in results:
        detail = SampleOutput(
            sample_id=result.request_id,
            target_text=targets.get(result.request_id, ""),
        )
        details[result.request_id] = detail
        if not result.is_success:
            detail.error = result.error or "TTS request failed"
            continue
        if not detail.target_text.strip():
            detail.error = "Missing target text"
            continue
        try:
            audio, sample_rate = sf.read(result.wav_path, dtype="float32")
        except (OSError, RuntimeError) as exc:
            detail.error = f"Cannot decode audio: {exc}"
            continue
        if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
            detail.error = "Empty, nonfinite, or silent audio"
            continue
        detail.audio_duration_s = len(audio) / sample_rate
        samples.append(
            SampleInput(
                sample_id=result.request_id,
                ref_text=detail.target_text,
                ref_audio=result.wav_path,
                target_text=detail.target_text,
            )
        )

    transcripts = await transcribe(samples)
    if len({r.request_id for r in transcripts}) != len(transcripts):
        raise ValueError("ASR request IDs must be unique")
    by_id = {r.request_id: r for r in transcripts}
    for sample in samples:
        detail = details[sample.sample_id]
        transcript = by_id.get(sample.sample_id)
        if transcript is None or not transcript.is_success:
            detail.error = (
                transcript.error or "ASR request failed"
                if transcript is not None
                else "Missing ASR result"
            )
            continue
        detail.asr_latency_s = transcript.latency_s
        apply_wer(detail, transcript.text, lang)
    return corpus_report(details, output=output, lang=lang, max_wer=max_wer)


def corpus_report(
    details: Mapping[str, SampleOutput], *, output: Path, lang: str, max_wer: float
) -> QualityReport:
    """Per-request transcription success plus the corpus WER gate, saved as JSON.

    A request passes when its transcript was scored and is not a runaway
    (WER above ``RUNAWAY_WER``, the hallucination signature); the remaining
    word errors accumulate into one corpus WER compared against ``max_wer``,
    so a single short clip cannot fail the trial on its own.
    """
    verdicts = {
        key: detail.is_success and detail.wer <= RUNAWAY_WER
        for key, detail in details.items()
    }
    scored = [detail for key, detail in details.items() if verdicts[key]]
    metrics = calculate_wer_metrics(scored, lang)
    corpus_wer = metrics["wer_corpus"] if metrics["evaluated"] else None
    corpus_pass = corpus_wer is not None and corpus_wer <= max_wer
    output.write_text(
        json.dumps(
            {
                "lang": lang,
                "max_wer": max_wer,
                "runaway_wer": RUNAWAY_WER,
                "corpus_wer": corpus_wer,
                "corpus_pass": corpus_pass,
                "evaluated": metrics["evaluated"],
                "runaway": sum(
                    1
                    for key, detail in details.items()
                    if detail.is_success and not verdicts[key]
                ),
                "requests": {
                    key: {**asdict(detail), "quality_pass": verdicts[key]}
                    for key, detail in details.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return QualityReport(
        verdicts=verdicts,
        corpus_wer=corpus_wer,
        corpus_pass=corpus_pass,
        max_wer=max_wer,
    )
