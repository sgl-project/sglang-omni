# SPDX-License-Identifier: Apache-2.0
"""TTS trial entry point with an ASR quality pass after serving stops."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.restage_corpus import repeat_corpus
from benchmarks.benchmarker.restage_quality import evaluate_tts_quality
from benchmarks.benchmarker.restage_trial import execute_trial
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.asr import run_asr_transcription
from benchmarks.tasks.tts import make_tts_send_fn
from sglang_omni.restage.evaluation import SLO, Evaluation

ASR_PORT_OFFSET = 100


async def execute_tts_trial(
    *,
    config_path: Path,
    model_path: str,
    asr_config_path: Path,
    asr_model_path: str,
    samples: list[SampleInput],
    slo: SLO,
    rate: float,
    destination: Path,
    port: int,
    lang: str,
    max_wer: float,
    sender_options: dict[str, Any] | None = None,
    warmup: int = 1,
    warmup_sample: SampleInput | None = None,
    startup_timeout_s: int = 1800,
    request_timeout_s: int = 300,
    asr_concurrency: int = 8,
    arrival_seed: int | None = None,
    corpus_repeats: int = 1,
) -> Evaluation:
    """Measure one admitted configuration, then transcribe its saved audio.

    Both services use the caller's visible devices and supplied configurations.
    The same port is reused sequentially. WER assesses transcript agreement,
    not speaker similarity or perceived audio quality.
    """
    if slo.max_ttfa_s is not None and not (sender_options or {}).get("stream", False):
        raise ValueError("First-audio SLO requires a streaming sender")
    if slo.max_underrun_s is not None and not (sender_options or {}).get(
        "stream", False
    ):
        raise ValueError("A playback SLO requires a streaming sender")
    samples, request_sources = repeat_corpus(samples, corpus_repeats)
    targets = {sample.sample_id: sample.target_text for sample in samples}

    async def transcribe(valid_samples):
        if not valid_samples:
            return []
        # Note (Jiaxin Deng): the TTS server just released ``port``; a fresh
        # one keeps the quality pass clear of its lingering connections.
        with managed_omni_server(
            model_path=asr_model_path,
            server_config=str(asr_config_path.resolve()),
            host="127.0.0.1",
            port=port + ASR_PORT_OFFSET,
            log_file=destination / "asr-server.log",
            timeout=startup_timeout_s,
            wait_for_gpu_release=False,
        ):
            results, _ = await run_asr_transcription(
                valid_samples,
                port=port + ASR_PORT_OFFSET,
                model_path=asr_model_path,
                lang=lang,
                concurrency=asr_concurrency,
                request_timeout_s=request_timeout_s,
            )
        return results

    async def quality(results):
        (destination / "quality-protocol.json").write_text(
            json.dumps(
                {
                    "asr_model_path": asr_model_path,
                    "asr_config_path": str(asr_config_path.resolve()),
                    "lang": lang,
                    "max_wer": max_wer,
                    "asr_concurrency": asr_concurrency,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return await evaluate_tts_quality(
            results,
            targets=targets,
            transcribe=transcribe,
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
                    "sender_options": sender_options or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return make_tts_send_fn(
            model_path,
            f"{url}/v1/audio/speech",
            save_audio_dir=str(audio_dir),
            **(sender_options or {}),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="TTS trial JSON spec")
    parser.add_argument(
        "--output", type=Path, required=True, help="New trial directory"
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec["samples"] = [SampleInput(**sample) for sample in spec["samples"]]
    spec["slo"] = SLO(**spec["slo"])
    for key in ("config_path", "asr_config_path"):
        path = Path(spec[key])
        spec[key] = path if path.is_absolute() else args.spec.resolve().parent / path
    result = asyncio.run(execute_tts_trial(destination=args.output, **spec))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
