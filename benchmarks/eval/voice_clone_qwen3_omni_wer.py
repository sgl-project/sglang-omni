# SPDX-License-Identifier: Apache-2.0
"""Qwen3 Omni Voice Cloning WER evaluation.

Generates speech via /v1/chat/completions with modalities: ["text", "audio"],
then evaluates WER using Whisper (EN) or FunASR (ZH).

Usage:
    python benchmarks/eval/voice_clone_qwen3_omni_wer.py \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en \
        --lang en --max-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp
import torch
from tqdm import tqdm

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.prepare import download_dataset
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.voice_clone import (
    VoiceCloneOmni,
    calculate_wer_metrics,
    load_asr_model,
    print_wer_summary,
    save_wer_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main_async(args: argparse.Namespace) -> None:
    """Main async function for voice clone WER evaluation.

    TODO (chenyang):
    Current implementation of Qwen3 Omni on main branch is broken.
    Need to merge the changes from https://github.com/sgl-project/sglang-omni/pull/219
    """
    if "cuda" in args.asr_device:
        torch.cuda.set_device(args.asr_device)
        logger.info("Set ASR CUDA device to %s", args.asr_device)

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    if args.download_dataset:
        download_dataset("zhaochenyang20/seed-tts-eval", args.dataset_dir)

    asr = load_asr_model(args.lang, args.asr_device)

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

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
                    "[%d/%d] WER=%.3f  target=%.50s  whisper=%.50s",
                    i + 1,
                    len(samples),
                    result.wer,
                    result.ref_norm,
                    result.hyp_norm,
                )
            else:
                logger.warning(
                    "[%d/%d] FAILED: %s -- %s",
                    i + 1,
                    len(samples),
                    sample.sample_id,
                    result.error,
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
        description="WER evaluation for Qwen3-Omni TTS via sglang-omni server"
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
        "--lang", choices=["en", "zh"], default="en", help="Language for ASR model"
    )
    p.add_argument("--host", type=str, default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument(
        "--asr-device", default="cuda:0", help="Device for ASR (Whisper) model"
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
    args = p.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    wait_for_service(base_url, timeout=args.server_timeout)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
