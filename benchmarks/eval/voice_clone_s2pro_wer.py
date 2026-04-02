# SPDX-License-Identifier: Apache-2.0
"""S2 Pro Voice Cloning WER evaluation.

Generates speech via /v1/audio/speech, transcribes with Whisper (EN) or
FunASR (ZH), and computes corpus-level WER.

1. Launch server and download dataset

    python -m sglang_omni.cli.cli serve \
        --model-path fishaudio/s2-pro \
        --config examples/configs/s2pro_tts.yaml \
        --port 8001

    hf download zhaochenyang20/seed-tts-eval \
     --repo-type dataset --local-dir seedtts_testset

2. Evaluate

    python benchmarks/eval/voice_clone_s2pro_wer.py \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/s2pro_en \
        --port 8001 \
        --lang en --max-samples 5

    python benchmarks/eval/voice_clone_s2pro_wer.py \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/s2pro_en_c20 \
        --port 8001 \
        --lang en --max-samples 50 \
        --generation-concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp
import torch

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import get_wav_duration, wait_for_service
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.tasks.voice_clone import (
    SampleOutput,
    VoiceCloneTTS,
    _transcribe_and_compute_wer,
    calculate_wer_metrics,
    load_asr_model,
    print_wer_summary,
    save_wer_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    api_url: str
    model_name: str
    stream: bool
    max_new_tokens: int
    temperature: float
    seed: int | None
    audio_dir: str


async def _generate_entry(
    session: aiohttp.ClientSession,
    task: VoiceCloneTTS,
    sample: SampleInput,
    config: GenerationConfig,
) -> dict:
    wav_path = os.path.join(config.audio_dir, f"{sample.sample_id}.wav")
    entry: dict = {
        "sample_id": sample.sample_id,
        "target_text": sample.target_text,
        "wav_path": wav_path,
    }

    try:
        gen_fn = (
            task.generate_speech_streaming if config.stream else task.generate_speech
        )
        wav_bytes, latency = await gen_fn(
            session,
            config.api_url,
            config.model_name,
            sample,
            config.max_new_tokens,
            config.temperature,
            config.seed,
        )
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        entry["latency_s"] = round(latency, 4)
        entry["audio_duration_s"] = round(get_wav_duration(wav_bytes), 4)
        entry["is_success"] = True
    except Exception as exc:
        entry["is_success"] = False
        entry["error"] = str(exc)

    return entry


def _transcribe_entry(
    entry: dict,
    asr: dict,
    lang: str,
    device: str,
) -> SampleOutput:
    output = SampleOutput(
        sample_id=entry["sample_id"],
        target_text=entry["target_text"],
    )
    if not entry["is_success"]:
        output.error = f"Generation failed: {entry['error']}"
        return output

    output.latency_s = entry["latency_s"]
    output.audio_duration_s = entry["audio_duration_s"]
    return _transcribe_and_compute_wer(
        output,
        entry["wav_path"],
        asr,
        lang,
        device,
    )


async def generate_audio(args: argparse.Namespace) -> list[dict]:
    """Call TTS API and save WAV files using the shared benchmark dispatcher."""
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/audio/speech"

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from {args.meta}")

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    task = VoiceCloneTTS()
    config = GenerationConfig(
        api_url=api_url,
        model_name=args.model,
        stream=args.stream,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        audio_dir=audio_dir,
    )
    generated_by_id: dict[str, dict] = {}

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        entry = await _generate_entry(
            session,
            task,
            sample,
            config,
        )
        generated_by_id[sample.sample_id] = entry
        return RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text,
            is_success=entry["is_success"],
            latency_s=entry.get("latency_s", 0.0),
            audio_duration_s=entry.get("audio_duration_s", 0.0),
            wav_path=entry["wav_path"],
            error=entry.get("error", ""),
        )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.generation_concurrency,
            warmup=0,
            timeout_s=300,
        )
    )
    results = await runner.run(samples, send_fn)
    generated = [generated_by_id[result.request_id] for result in results]

    meta_path = os.path.join(args.output_dir, "generated.json")
    with open(meta_path, "w") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved generation metadata to {meta_path}")
    return generated


def _log_transcription_result(
    completed: int,
    total: int,
    entry: dict,
    output: SampleOutput,
) -> None:
    if output.is_success:
        logger.info(
            "[%d/%d] WER=%.3f  ref=%.50s  hyp=%.50s",
            completed,
            total,
            output.wer,
            output.ref_norm,
            output.hyp_norm,
        )
    else:
        logger.warning(
            "[%d/%d] Transcription failed: %s -- %s",
            completed,
            total,
            entry["sample_id"],
            output.error,
        )


def transcribe_audio(args: argparse.Namespace) -> None:
    """Load the ASR model, transcribe saved audio sequentially, and compute WER."""
    if "cuda" in args.device:
        torch.cuda.set_device(args.device)
        logger.info(f"Set default CUDA device to {args.device}")

    meta_path = os.path.join(args.output_dir, "generated.json")
    with open(meta_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {meta_path}")

    asr = load_asr_model(args.lang, args.device)

    final_outputs: list[SampleOutput] = []
    for completed, entry in enumerate(generated, start=1):
        output = _transcribe_entry(
            entry,
            asr,
            args.lang,
            args.device,
        )
        _log_transcription_result(completed, len(generated), entry, output)
        final_outputs.append(output)

    metrics = calculate_wer_metrics(final_outputs, args.lang)
    print_wer_summary(metrics, args.model)

    config = {
        "model": args.model,
        "meta": args.meta,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_samples": args.max_samples,
        "stream": args.stream,
        "generation_concurrency": args.generation_concurrency,
    }
    save_wer_results(final_outputs, metrics, config, args.output_dir)


async def main_async(args: argparse.Namespace) -> None:
    """Run local WER evaluation as generation first, then serialized ASR."""
    await generate_audio(args)
    transcribe_audio(args)


def main() -> None:
    p = argparse.ArgumentParser(
        description="WER evaluation for S2 Pro (Whisper for EN, FunASR for ZH)"
    )
    p.add_argument("--meta", default="seedtts_testset/en/meta.lst")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save generated audio and results",
    )
    p.add_argument(
        "--model",
        default="fishaudio/s2-pro",
        help="Model name for the API request",
    )
    p.add_argument(
        "--lang", choices=["en", "zh"], default="en", help="Language for ASR model"
    )
    p.add_argument("--host", type=str, default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    p.add_argument("--device", default="cuda:0", help="Device for ASR model")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    p.add_argument("--generation-concurrency", type=int, default=1)
    p.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming SSE for TTS generation",
    )

    mode = p.add_mutually_exclusive_group()

    # Note (chenyang):
    # On our CI, we use --generate-only to generate audio and --transcribe-only to
    # ensure that the SGLang Omni server is killed after the audio generation.
    # Thus the CI testset won't raise GPU OOM error.

    mode.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Only synthesize audio: call the TTS API and write WAVs under --output-dir."
        ),
    )
    mode.add_argument(
        "--transcribe-only",
        action="store_true",
        help=("Only run recognition and WER on existing output-dir)."),
    )
    args = p.parse_args()

    if args.generate_only:
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        wait_for_service(base_url)
        asyncio.run(generate_audio(args))
    elif args.transcribe_only:
        transcribe_audio(args)
    else:
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        wait_for_service(base_url)
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
