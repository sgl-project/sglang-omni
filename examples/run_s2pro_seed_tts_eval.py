#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run seed-tts-eval benchmark for FishAudio S2-Pro through sglang-omni pipeline.

Usage:
    python examples/run_s2pro_seed_tts_eval.py \
        --checkpoint /root/.cache/huggingface/s2-pro/s2-pro \
        --meta /root/yitong/seed-tts-eval/seedtts_testset/en/meta.lst \
        --output-dir /root/yitong/seed-tts-eval/s2pro_sglang_omni \
        --tts-device cuda:5 --asr-device cuda:6 \
        --num-samples 0  # 0 = all samples

Runs the full 3-stage sglang-omni pipeline:
    preprocessing (tokenize + VQ encode)  ->  tts_engine (qwen3 generate)  ->  vocoder (VQGAN decode)
Then transcribes each output with Whisper-large-v3 and computes WER via jiwer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

os.environ.setdefault("FISH_BATCH_INVARIANT", "true")

import jiwer
import numpy as np
import scipy.signal
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/root/yitong/fish-speech")

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.fishaudio_s2_pro import create_tts_pipeline_config
from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_asr_model(device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("Loading Whisper-large-v3 on %s ...", device)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = (
        WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
        .to(device)
        .eval()
    )
    logger.info("ASR model ready")
    return processor, model


def asr_transcribe(
    processor, model, audio_np: np.ndarray, src_sr: int, device: str
) -> str:
    if src_sr != 16000:
        num_samples = int(len(audio_np) * 16000 / src_sr)
        audio_np = scipy.signal.resample(audio_np, num_samples)
    feats = processor(
        audio_np, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language="en", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def parse_meta(meta_path: str) -> list[dict]:
    base_dir = os.path.dirname(meta_path)
    samples = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            samples.append(
                {
                    "sid": parts[0],
                    "ref_text": parts[1],
                    "ref_path": os.path.join(base_dir, parts[2]),
                    "target_text": parts[3],
                }
            )
    return samples


async def run_eval(args):
    # -- Build pipeline config -----------------------------------------------
    fused = [[
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]]

    config = create_tts_pipeline_config(
        model_id=args.checkpoint,
        tts_device=args.tts_device,
        vocoder_device=args.tts_device,
        max_new_tokens=args.max_new_tokens,
        use_compile=not args.no_compile,
        fused_stages=fused,
    )

    # -- Compile & start pipeline --------------------------------------------
    coordinator, stages = compile_pipeline(config)
    runner = PipelineRunner(coordinator, stages)
    await runner.start()
    logger.info(
        "Pipeline '%s' started (%d stages)", config.name, len(stages)
    )

    # -- Load ASR model on separate GPU --------------------------------------
    asr_proc, asr_model = load_asr_model(args.asr_device)

    # -- Parse meta file -----------------------------------------------------
    samples = parse_meta(args.meta)
    if args.start > 0:
        samples = samples[args.start:]
    total = len(samples) if args.num_samples <= 0 else min(args.num_samples, len(samples))
    samples = samples[:total]
    start_idx = args.start
    logger.info("Will evaluate %d samples (start=%d)", total, start_idx)

    os.makedirs(args.output_dir, exist_ok=True)

    refs_list, hyps_list = [], []
    results_log = []
    eval_start = time.perf_counter()

    try:
        for idx_in_batch, sample in enumerate(samples):
            idx = start_idx + idx_in_batch
            sid = sample["sid"]
            ref_text = sample["ref_text"]
            ref_path = sample["ref_path"]
            target_text = sample["target_text"]

            logger.info(
                "[%d/%d] %s | target: %s",
                idx_in_batch + 1, total, sid, target_text[:80],
            )

            request = OmniRequest(
                inputs={
                    "text": target_text,
                    "references": [
                        {"audio_path": ref_path, "text": ref_text},
                    ],
                },
                params={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": 42 + idx,
                },
            )

            request_id = f"eval-{idx:04d}"
            t0 = time.perf_counter()

            try:
                result = await coordinator.submit(request_id, request)
            except Exception as e:
                logger.error("  FAILED: %s", e)
                results_log.append({"sid": sid, "error": str(e)})
                continue

            elapsed = time.perf_counter() - t0

            state = S2ProState.from_dict(result.get("s2pro_state", {}))

            if state.audio_samples is None:
                logger.warning("  No audio generated for %s", sid)
                results_log.append({"sid": sid, "error": "no_audio"})
                continue

            audio = state.audio_samples
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            elif isinstance(audio, torch.Tensor):
                audio = audio.float().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()

            sr = state.sample_rate
            duration = len(audio) / sr

            out_path = os.path.join(args.output_dir, f"{sid}.wav")
            sf.write(out_path, audio, sr)

            asr_text = asr_transcribe(asr_proc, asr_model, audio, sr, args.asr_device)

            ref_clean = target_text.strip().lower()
            hyp_clean = asr_text.strip().lower()
            wer = jiwer.wer(ref_clean, hyp_clean)

            refs_list.append(ref_clean)
            hyps_list.append(hyp_clean)

            entry = {
                "sid": sid,
                "target": target_text,
                "asr": asr_text,
                "wer": round(wer * 100, 1),
                "duration_s": round(duration, 2),
                "num_semantic": state.num_semantic_tokens,
                "gen_time_s": round(elapsed, 2),
            }
            results_log.append(entry)

            logger.info(
                "  dur=%.2fs  sem=%d  gen=%.2fs  WER=%.1f%%  ASR: %s",
                duration, state.num_semantic_tokens, elapsed, wer * 100, asr_text[:80],
            )

        # -- Summary ---------------------------------------------------------
        if refs_list:
            overall_wer = jiwer.wer(refs_list, hyps_list)
            logger.info(
                "=== Overall WER (%d samples): %.2f%% ===",
                len(refs_list), overall_wer * 100,
            )
        else:
            overall_wer = -1
            logger.warning("No successful samples to compute WER")

        total_time = time.perf_counter() - eval_start

        summary = {
            "num_samples": len(refs_list),
            "overall_wer_pct": round(overall_wer * 100, 2) if overall_wer >= 0 else None,
            "total_time_s": round(total_time, 1),
            "params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
            },
            "results": results_log,
        }

        summary_path = os.path.join(args.output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", summary_path)

    finally:
        await runner.stop()
        logger.info("Pipeline stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="seed-tts-eval for S2-Pro via sglang-omni pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/.cache/huggingface/s2-pro/s2-pro",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="/root/yitong/seed-tts-eval/seedtts_testset/en/meta.lst",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/yitong/seed-tts-eval/s2pro_sglang_omni",
    )
    parser.add_argument("--tts-device", type=str, default="cuda:5")
    parser.add_argument("--asr-device", type=str, default="cuda:6")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--start", type=int, default=0, help="Start index (skip first N samples)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Disable torch.compile",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
