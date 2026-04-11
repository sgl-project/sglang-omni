# SPDX-License-Identifier: Apache-2.0
"""Qwen3 Omni Voice Cloning WER evaluation.

Generates speech via /v1/chat/completions with modalities: ["text", "audio"],
then evaluates WER using Whisper (EN) or FunASR (ZH).

Usage:

    1. Generate audio and transcribe all at once

    python benchmarks/eval/voice_clone_omni_wer.py \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en \
        --lang en --max-samples 50

    2. Generate audio and then transcribe separately

    Note (Jingwen, Chenyang): we separate the generate and
    transcribe phases for CI to avoid OOM.

    python benchmarks/eval/voice_clone_omni_wer.py \
        --generate-only --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en --max-samples 10

    python benchmarks/eval/voice_clone_omni_wer.py \
        --transcribe-only --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en --lang en --device cuda:0
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
from benchmarks.dataset.prepare import download_dataset
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.voice_clone import (
    SampleOutput,
    VoiceCloneOmni,
    _transcribe_and_compute_wer,
    calculate_wer_metrics,
    load_asr_model,
    print_wer_summary,
    save_wer_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def generate_audio(args: argparse.Namespace) -> list[dict]:
    """Call Omni TTS API for each sample and save WAV files to disk."""
    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from {args.meta}")

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    task = VoiceCloneOmni()
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
                wav_bytes, latency, _usage = await task.generate_speech(
                    session,
                    api_url,
                    args.model,
                    sample,
                    args.lang,
                    speaker=args.speaker,
                    max_tokens=args.max_tokens,
                    voice_clone=args.voice_clone,
                )
                with open(wav_path, "wb") as f:
                    f.write(wav_bytes)
                wav_info = sf.info(wav_path)
                entry["latency_s"] = round(latency, 4)
                entry["audio_duration_s"] = round(wav_info.duration, 4)
                entry["is_success"] = True
                logger.info(
                    f"[{i + 1}/{len(samples)}] Generated {wav_info.duration:.1f}s "
                    f"audio for {sample.sample_id}",
                )
            except Exception as exc:
                entry["is_success"] = False
                entry["error"] = str(exc)
                logger.warning(
                    f"[{i + 1}/{len(samples)}] FAILED {sample.sample_id}: {exc}",
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
        logger.info(f"Set ASR CUDA device to {args.device}")

    meta_path = os.path.join(args.output_dir, "generated.json")
    with open(meta_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {meta_path}")

    asr = load_asr_model(args.lang, args.device)

    outputs: list[SampleOutput] = []
    for i, entry in enumerate(tqdm(generated, desc=f"Transcribing {args.lang}")):
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
                f"[{i + 1}/{len(generated)}] WER={output.wer:.3f}  ref={output.ref_norm[:50]}  hyp={output.hyp_norm[:50]}",
            )
        else:
            logger.warning(
                f"[{i + 1}/{len(generated)}] Transcription failed: {entry['sample_id']} -- {output.error}",
            )

    metrics = calculate_wer_metrics(outputs, args.lang)
    print_wer_summary(metrics, args.model)

    config = {
        "model": args.model,
        "speaker": args.speaker,
        "voice_clone": args.voice_clone,
        "meta": args.meta,
        "max_samples": args.max_samples,
    }
    save_wer_results(outputs, metrics, config, args.output_dir)


async def main_async(args: argparse.Namespace) -> None:
    """Run both phases in one shot (original behavior)."""
    if "cuda" in args.asr_device:
        torch.cuda.set_device(args.asr_device)
        logger.info(f"Set ASR CUDA device to {args.asr_device}")

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    if args.download_dataset:
        download_dataset("zhaochenyang20/seed-tts-eval", args.dataset_dir)

    asr = load_asr_model(args.lang, args.asr_device)

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples from {args.meta}")

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    task = VoiceCloneOmni()
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
                args.asr_device,
                audio_dir,
                args.speaker,
                args.max_tokens,
                voice_clone=args.voice_clone,
            )
            outputs.append(result)

            if result.is_success:
                logger.info(
                    f"[{i + 1}/{len(samples)}] WER={result.wer:.3f}  target={result.ref_norm[:50]}  whisper={result.hyp_norm[:50]}",
                )
            else:
                logger.warning(
                    f"[{i + 1}/{len(samples)}] FAILED: {sample.sample_id} -- {result.error}",
                )

    metrics = calculate_wer_metrics(outputs, args.lang)
    print_wer_summary(metrics, args.model)

    config = {
        "model": args.model,
        "speaker": args.speaker,
        "voice_clone": args.voice_clone,
        "server_url": base_url,
        "meta": args.meta,
        "max_samples": args.max_samples,
    }
    save_wer_results(outputs, metrics, config, args.output_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description="WER evaluation for Qwen3-Omni TTS via sglang-omni server",
    )
    p.add_argument("--meta", default="seedtts_testset/en/meta.lst")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save generated audio and results",
    )
    p.add_argument(
        "--model",
        default="qwen3-omni",
        help="Model name for the API request",
    )
    p.add_argument(
        "--speaker",
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Speaker voice for Qwen3-Omni TTS",
    )
    p.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language for ASR model",
    )
    p.add_argument("--host", type=str, default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument(
        "--asr-device",
        default="cuda:0",
        help="Device for ASR (Whisper) model",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Device for ASR model (transcribe-only mode)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--voice-clone",
        action="store_true",
        help="Pass ref_audio via 'audios' field for voice cloning",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens for thinker",
    )
    p.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds to wait for server readiness",
    )
    p.add_argument(
        "--download-dataset",
        action="store_true",
        help="Auto-download seed-tts-eval dataset before evaluation",
    )
    p.add_argument(
        "--dataset-dir",
        default="seedtts_testset",
        help="Local directory for the dataset",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Only synthesize audio: call the TTS API and write WAVs.",
    )
    mode.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only run recognition and WER on existing output-dir.",
    )
    args = p.parse_args()

    if args.generate_only:
        base_url = f"http://{args.host}:{args.port}"
        wait_for_service(base_url, timeout=args.server_timeout)
        asyncio.run(generate_audio(args))
    elif args.transcribe_only:
        transcribe_audio(args)
    else:
        base_url = f"http://{args.host}:{args.port}"
        wait_for_service(base_url, timeout=args.server_timeout)
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
