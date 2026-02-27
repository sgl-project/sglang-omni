#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FishAudio DualAR TTS via the sglang-omni pipeline.

Usage:
    # Text-only (no voice cloning):
    python examples/run_fishaudio_e2e.py --text "Hello, how are you today?"

    # With reference audio (voice cloning):
    python examples/run_fishaudio_e2e.py --text "Hello" \
        --reference-audio ref.wav --reference-text "Reference transcript."

    # Save output as wav:
    python examples/run_fishaudio_e2e.py --text "Hello" --output output.wav

    # With torch.compile:
    python examples/run_fishaudio_e2e.py --text "Hello" --compile --output output.wav

    # With voice cloning and radix cache:
    python examples/run_fishaudio_e2e.py --text "Hello" \
        --reference-audio ref.wav --reference-text "..." \
        --use-radix-cache --output output.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import soundfile as sf
import torch

from sglang_omni.config import compile_pipeline, PipelineRunner
from sglang_omni.models.fishaudio_s1 import create_tts_pipeline_config
from sglang_omni.models.fishaudio_s1.io import FishAudioState
from sglang_omni.proto import OmniRequest

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_e2e(args):
    # -- 1. Build pipeline config ------------------------------------------
    config = create_tts_pipeline_config(
        model_id=args.checkpoint,
        tts_device=args.device,
        vocoder_device=args.device,
        max_new_tokens=args.max_new_tokens,
        use_compile=args.compile,
        use_radix_cache=args.use_radix_cache,
    )

    # -- 2. Compile & start ------------------------------------------------
    coordinator, stages = compile_pipeline(config)
    runner = PipelineRunner(coordinator, stages)
    await runner.start()
    logger.info(
        "Pipeline '%s' started (%d stages)", config.name, len(stages)
    )

    try:
        if args.test_cache:
            await _run_cache_test(coordinator, args)
        else:
            await _run_single_request(coordinator, args)
    finally:
        await runner.stop()
        logger.info("Pipeline stopped.")


async def _run_single_request(coordinator, args):
    """Submit a single TTS request through the pipeline."""
    # Build inputs
    inputs: dict = {"text": args.text}
    if args.reference_audio:
        inputs["references"] = [
            {
                "audio_path": args.reference_audio,
                "text": args.reference_text,
            }
        ]

    request = OmniRequest(
        inputs=inputs,
        params={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )

    request_id = "req-0"
    logger.info("Submitting request '%s' …", request_id)
    t0 = time.perf_counter()

    result = await coordinator.submit(request_id, request)

    elapsed = time.perf_counter() - t0

    # Parse result
    state = FishAudioState.from_dict(result.data)

    if state.output_codes is not None:
        output_codes = state.output_codes
        if not isinstance(output_codes, torch.Tensor):
            output_codes = torch.tensor(output_codes)
        num_steps = output_codes.shape[1]
        logger.info(
            "Generated %d steps in %.3fs (%.1f steps/s)",
            num_steps,
            elapsed,
            num_steps / elapsed if elapsed > 0 else 0,
        )
        logger.info("Output codes shape: %s", output_codes.shape)

    if state.audio_samples is not None and args.output:
        audio = state.audio_samples
        if isinstance(audio, torch.Tensor):
            audio = audio.float().numpy()
        elif isinstance(audio, list):
            audio = torch.tensor(audio).float().numpy()
        sf.write(args.output, audio, state.sample_rate)
        logger.info(
            "Saved output audio to %s (%.2fs)",
            args.output,
            len(audio) / state.sample_rate,
        )


async def _run_cache_test(coordinator, args):
    """Send two requests with same voice ref but different text to test cache."""
    if not args.reference_audio:
        logger.warning("Cache test requires --reference-audio to be provided.")
        return

    texts = [
        "This is the first test sentence for cache validation.",
        "This is a completely different sentence to verify cache reuse.",
    ]

    for i, text in enumerate(texts):
        inputs: dict = {
            "text": text,
            "references": [
                {
                    "audio_path": args.reference_audio,
                    "text": args.reference_text,
                }
            ],
        }

        request = OmniRequest(
            inputs=inputs,
            params={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        )

        request_id = f"cache-test-{i}"
        t0 = time.perf_counter()

        result = await coordinator.submit(request_id, request)

        elapsed = time.perf_counter() - t0
        state = FishAudioState.from_dict(result.data)

        if state.output_codes is not None:
            output_codes = state.output_codes
            if not isinstance(output_codes, torch.Tensor):
                output_codes = torch.tensor(output_codes)
            logger.info(
                "Request %d: generated %d steps in %.3fs",
                i,
                output_codes.shape[1],
                elapsed,
            )


def main():
    parser = argparse.ArgumentParser(description="FishAudio DualAR TTS pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="fishaudio/openaudio-s1-mini",
        help="HF model ID or local path",
    )
    parser.add_argument("--text", type=str, default="Hello, how are you today?")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to reference wav for voice cloning",
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        default="",
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save output wav",
    )
    parser.add_argument(
        "--use-radix-cache",
        action="store_true",
        default=False,
        help="Enable radix-tree prefix cache for voice ref reuse",
    )
    parser.add_argument(
        "--test-cache",
        action="store_true",
        default=False,
        help="Run cache correctness test: 2 requests with same voice ref, different text",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile(mode='reduce-overhead') for decode steps",
    )
    args = parser.parse_args()

    asyncio.run(run_e2e(args))


if __name__ == "__main__":
    main()
