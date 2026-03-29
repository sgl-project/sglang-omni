# SPDX-License-Identifier: Apache-2.0
"""S2 Pro Voice Cloning WER evaluation.

Generates speech via /v1/audio/speech, transcribes with Whisper (EN) or
FunASR (ZH), and computes corpus-level WER.

Usage:
    python benchmarks/eval/voice_clone_s2pro_wer.py \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/s2pro_en \
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
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.voice_clone import (
    VoiceCloneTTS,
    calculate_wer_metrics,
    load_asr_model,
    print_wer_summary,
    save_wer_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main_async(args: argparse.Namespace) -> None:
    """Main async function for S2 Pro voice clone WER evaluation.

    Note (chenyang):
    There is concurrency issue in S2 Pro voice clone WER eval.
    https://github.com/sgl-project/sglang-omni/issues/228
    """
    if "cuda" in args.device:
        torch.cuda.set_device(args.device)
        logger.info("Set default CUDA device to %s", args.device)

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/audio/speech"

    asr = load_asr_model(args.lang, args.device)

    samples = load_seedtts_samples(args.meta, args.max_samples)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

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
                    "[%d/%d] FAILED: %s — %s",
                    i + 1,
                    len(samples),
                    sample.sample_id,
                    result.error,
                )

    metrics = calculate_wer_metrics(outputs, args.lang)
    print_wer_summary(metrics, args.model)

    config = {
        "model": args.model,
        "meta": args.meta,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_samples": args.max_samples,
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
    args = p.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
