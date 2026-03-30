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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp
import soundfile as sf
import torch
from tqdm import tqdm

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.seedtts import load_seedtts_samples
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


async def generate_audio(args: argparse.Namespace) -> list[dict]:
    """Call TTS API for each sample and save WAV files to disk."""
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/audio/speech"

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from {args.meta}")

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    task = VoiceCloneTTS()
    timeout = aiohttp.ClientTimeout(total=300)
    generated: list[dict] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sample in enumerate(tqdm(samples, desc="Generating TTS audio")):
            wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
            entry: dict = {
                "sample_id": sample.sample_id,
                "target_text": sample.target_text,
                "wav_path": wav_path,
            }
            try:
                gen_fn = (
                    task.generate_speech_streaming
                    if args.stream
                    else task.generate_speech
                )
                wav_bytes, latency = await gen_fn(
                    session,
                    api_url,
                    args.model,
                    sample,
                    args.max_new_tokens,
                    args.temperature,
                    args.seed,
                )
                with open(wav_path, "wb") as f:
                    f.write(wav_bytes)
                wav_info = sf.info(wav_path)
                entry["latency_s"] = round(latency, 4)
                entry["audio_duration_s"] = round(wav_info.duration, 4)
                entry["is_success"] = True
                logger.info(
                    f"[{i + 1}/{len(samples)}] Generated {wav_info.duration:.1f}s audio "
                    f"for {sample.sample_id}"
                )
            except Exception as exc:
                entry["is_success"] = False
                entry["error"] = str(exc)
                logger.warning(
                    f"[{i + 1}/{len(samples)}] FAILED {sample.sample_id}: {exc}"
                )
            generated.append(entry)

    meta_path = os.path.join(args.output_dir, "generated.json")
    with open(meta_path, "w") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved generation metadata to {meta_path}")
    return generated


def transcribe_audio(args: argparse.Namespace) -> None:
    """Load ASR model, transcribe saved audio, compute and save WER."""
    if "cuda" in args.device:
        torch.cuda.set_device(args.device)
        logger.info(f"Set default CUDA device to {args.device}")

    meta_path = os.path.join(args.output_dir, "generated.json")
    with open(meta_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {meta_path}")

    asr = load_asr_model(args.lang, args.device)

    outputs: list[SampleOutput] = []
    for i, entry in enumerate(tqdm(generated, desc=f"Transcribing ({args.lang})")):
        output = SampleOutput(
            sample_id=entry["sample_id"],
            target_text=entry["target_text"],
        )
        if not entry.get("is_success", False):
            output.error = f"Generation failed: {entry.get('error', 'unknown')}"
            outputs.append(output)
            continue

        wav_path = entry["wav_path"]
        output.latency_s = entry.get("latency_s", 0.0)
        output.audio_duration_s = entry.get("audio_duration_s", 0.0)

        output = _transcribe_and_compute_wer(
            output,
            wav_path,
            asr,
            args.lang,
            args.device,
        )
        outputs.append(output)

        if output.is_success:
            logger.info(
                f"[{i + 1}/{len(generated)}] WER={output.wer:.3f}  "
                f"ref={output.ref_norm:.50}  hyp={output.hyp_norm:.50}"
            )
        else:
            logger.warning(
                f"[{i + 1}/{len(generated)}] Transcription failed: {entry['sample_id']} — "
                f"{output.error}"
            )

    metrics = calculate_wer_metrics(outputs, args.lang)
    print_wer_summary(metrics, args.model)

    config = {
        "model": args.model,
        "meta": args.meta,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_samples": args.max_samples,
        "stream": args.stream,
    }
    save_wer_results(outputs, metrics, config, args.output_dir)


async def main_async(args: argparse.Namespace) -> None:
    """Run both phases in one shot (original behavior).

    Note (chenyang):
    There is concurrency issue in S2 Pro voice clone WER eval.
    https://github.com/sgl-project/sglang-omni/issues/228
    """
    if "cuda" in args.device:
        torch.cuda.set_device(args.device)
        logger.info(f"Set default CUDA device to {args.device}")

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/audio/speech"

    asr = load_asr_model(args.lang, args.device)

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from {args.meta}")

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    task = VoiceCloneTTS()
    timeout = aiohttp.ClientTimeout(total=300)
    outputs = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating WER ({args.lang})")):
            result = await task.evaluate_sample(
                session,
                api_url,
                args.model,
                asr,
                sample,
                args.lang,
                args.device,
                audio_dir,
                args.max_new_tokens,
                args.temperature,
                args.seed,
                stream=args.stream,
            )
            outputs.append(result)

            if result.is_success:
                logger.info(
                    f"[{i + 1}/{len(samples)}] WER={result.wer:.3f}  "
                    f"target={result.ref_norm:.50}  whisper={result.hyp_norm:.50}"
                )
            else:
                logger.warning(
                    f"[{i + 1}/{len(samples)}] FAILED: {sample.sample_id} — {result.error}"
                )

    metrics = calculate_wer_metrics(outputs, args.lang)
    print_wer_summary(metrics, args.model)

    config = {
        "model": args.model,
        "meta": args.meta,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_samples": args.max_samples,
        "stream": args.stream,
    }
    save_wer_results(outputs, metrics, config, args.output_dir)


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
