#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch TTS synthesis for the seed-tts-eval benchmark.

Reads a meta.lst file (seed-tts-eval format), synthesizes one WAV per line
using the sglang-omni FishAudio pipeline, and writes results to an output dir.

Usage:
    python examples/run_seed_tts_eval.py \
        --checkpoint /path/to/s2-pro \
        --meta /path/to/seedtts_testset/en/meta.lst \
        --output-dir /path/to/output_en \
        --max-new-tokens 2048
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.fishaudio_s1 import create_tts_pipeline_config
from sglang_omni.models.fishaudio_s1.io import FishAudioState
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def parse_meta(meta_path: str) -> list[dict]:
    """Parse seed-tts-eval meta.lst into a list of dicts.

    Format: utt_id|prompt_text|prompt_wav|infer_text[|ground_truth_wav]
    """
    entries = []
    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            utt_id, prompt_text, prompt_wav, infer_text = parts[:4]
            if not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(meta_dir, prompt_wav)
            entries.append(
                {
                    "utt_id": utt_id,
                    "prompt_text": prompt_text,
                    "prompt_wav": prompt_wav,
                    "infer_text": infer_text,
                }
            )
    return entries


async def run_batch(args):
    entries = parse_meta(args.meta)
    logger.info("Loaded %d entries from %s", len(entries), args.meta)

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip already synthesized
    if args.resume:
        remaining = []
        for e in entries:
            wav_path = os.path.join(args.output_dir, e["utt_id"] + ".wav")
            if os.path.exists(wav_path):
                continue
            remaining.append(e)
        logger.info(
            "Resuming: %d already done, %d remaining",
            len(entries) - len(remaining),
            len(remaining),
        )
        entries = remaining

    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
        logger.info("Limited to %d entries", len(entries))

    if not entries:
        logger.info("All samples already synthesized.")
        return

    config = create_tts_pipeline_config(
        model_id=args.checkpoint,
        tts_device=args.device,
        vocoder_device=args.device,
        max_new_tokens=args.max_new_tokens,
        use_compile=not args.no_compile,
        use_radix_cache=False,
    )

    coordinator, stages = compile_pipeline(config)
    runner = PipelineRunner(coordinator, stages)
    await runner.start()
    logger.info("Pipeline started (%d stages)", len(stages))

    total = len(entries)
    t_start = time.perf_counter()
    n_done = 0
    n_fail = 0

    try:
        for i, entry in enumerate(entries):
            utt_id = entry["utt_id"]
            wav_path = os.path.join(args.output_dir, utt_id + ".wav")

            inputs = {
                "text": entry["infer_text"],
                "references": [
                    {
                        "audio_path": entry["prompt_wav"],
                        "text": entry["prompt_text"],
                    }
                ],
            }

            request = OmniRequest(
                inputs=inputs,
                params={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                },
            )

            request_id = f"req-{i}"
            t0 = time.perf_counter()

            try:
                result = await coordinator.submit(request_id, request)
                state = FishAudioState.from_dict(result)

                if state.audio_samples is not None:
                    audio = state.audio_samples
                    if isinstance(audio, torch.Tensor):
                        audio = audio.float().numpy()
                    elif isinstance(audio, list):
                        audio = torch.tensor(audio).float().numpy()
                    sf.write(wav_path, audio, state.sample_rate)
                    elapsed = time.perf_counter() - t0
                    n_done += 1
                    audio_dur = len(audio) / state.sample_rate
                    logger.info(
                        "[%d/%d] %s -> %.2fs audio in %.1fs (RTF=%.2f)",
                        i + 1,
                        total,
                        utt_id,
                        audio_dur,
                        elapsed,
                        elapsed / audio_dur if audio_dur > 0 else 0,
                    )
                else:
                    n_fail += 1
                    logger.warning("[%d/%d] %s -> no audio output", i + 1, total, utt_id)
            except Exception:
                n_fail += 1
                logger.exception("[%d/%d] %s -> failed", i + 1, total, utt_id)

    finally:
        await runner.stop()
        total_time = time.perf_counter() - t_start
        logger.info(
            "Done: %d succeeded, %d failed, %.1fs total", n_done, n_fail, total_time
        )


def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS synthesis for seed-tts-eval"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/.cache/huggingface/s2-pro/s2-pro",
        help="HF model ID or local path",
    )
    parser.add_argument(
        "--meta", type=str, required=True, help="Path to meta.lst file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for output WAVs"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument(
        "--no-compile", action="store_true", default=False, help="Disable torch.compile"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already synthesized samples (default: True)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of samples (0=all)"
    )
    args = parser.parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
