# SPDX-License-Identifier: Apache-2.0
"""ASR trials with transcript agreement against the input audio reference."""

import json
import math
from dataclasses import asdict
from pathlib import Path

from benchmarks.benchmarker.restage_corpus import repeat_corpus
from benchmarks.benchmarker.restage_quality import corpus_report
from benchmarks.benchmarker.restage_trial import execute_trial
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import apply_wer, make_asr_send_fn
from sglang_omni.restage.evaluation import SLO, Evaluation


async def execute_asr_trial(
    *,
    config_path: Path,
    model_path: str,
    samples: list[SampleInput],
    slo: SLO,
    rate: float,
    destination: Path,
    port: int,
    lang: str,
    max_wer: float,
    stream: bool = False,
    warmup: int = 1,
    warmup_sample: SampleInput | None = None,
    startup_timeout_s: int = 1800,
    request_timeout_s: int = 300,
    arrival_seed: int | None = None,
    corpus_repeats: int = 1,
) -> Evaluation:
    """Measure latency/RTF and score the returned transcript without another service.

    Audio-output SLOs do not apply to recognition. Streaming transcript timing
    remains in request evidence; this adapter does not yet select on text TTFT.
    Transcript accuracy is gated at corpus level through ``max_wer``.
    """
    if slo.max_ttfa_s is not None or slo.max_underrun_s is not None:
        raise ValueError("ASR does not support audio-output SLOs")
    if not math.isfinite(max_wer) or max_wer < 0:
        raise ValueError("max_wer must be finite and nonnegative")
    samples, request_sources = repeat_corpus(samples, corpus_repeats)
    targets = {sample.sample_id: sample.ref_text for sample in samples}

    async def quality(results):
        details = {}
        for result in results:
            detail = SampleOutput(
                sample_id=result.request_id,
                target_text=targets.get(result.request_id, ""),
                audio_duration_s=result.audio_duration_s,
                latency_s=result.latency_s,
            )
            if result.is_success:
                apply_wer(detail, result.text, lang)
            else:
                detail.error = result.error or "ASR request failed"
            details[result.request_id] = detail
        return corpus_report(
            details,
            output=destination / "quality-detail.json",
            lang=lang,
            max_wer=max_wer,
        )

    def sender(url, audio_dir):
        (destination / "workload.json").write_text(
            json.dumps(
                {
                    "samples": [asdict(sample) for sample in samples],
                    **(
                        {
                            "corpus_repeats": corpus_repeats,
                            "request_sources": request_sources,
                        }
                        if corpus_repeats > 1
                        else {}
                    ),
                    **(
                        {"warmup_sample": asdict(warmup_sample)}
                        if warmup_sample is not None
                        else {}
                    ),
                    "lang": lang,
                    "stream": stream,
                    "max_wer": max_wer,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return make_asr_send_fn(
            model_path, f"{url}/v1/audio/transcriptions", lang=lang, stream=stream
        )

    return await execute_trial(
        config_path=config_path,
        model_path=model_path,
        samples=samples,
        send_factory=sender,
        quality=quality,
        slo=slo,
        rate=rate,
        destination=destination,
        port=port,
        warmup=warmup,
        warmup_sample=warmup_sample,
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
        arrival_seed=arrival_seed,
    )
