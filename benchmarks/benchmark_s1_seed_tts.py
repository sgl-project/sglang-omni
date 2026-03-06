#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Seed-TTS-eval benchmark for FishAudio S1-Mini (sglang-omni runtime).

Generates audio for the seed-tts-eval English test set using the S1-Mini
engine with OmniEngine runtime.  Outputs per-sample WAV files and metrics
(latency, RTF, tok/s) for every sample, plus an aggregate summary.

Usage:

    CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark_s1_seed_tts.py \
        --checkpoint fishaudio/openaudio-s1-mini \
        --testset /tmp/seed-tts-eval/seedtts_testset/en/meta.lst \
        --output-dir results/s1_seed_tts \
        --max-samples 50 \
        --max-new-tokens 2048
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for third_party imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed-TTS-eval meta.lst parser
# ---------------------------------------------------------------------------


def parse_meta_lst(path: str) -> list[dict[str, Any]]:
    base_dir = os.path.dirname(path)
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            sample = {
                "id": parts[0],
                "ref_text": parts[1],
                "ref_audio": os.path.join(base_dir, parts[2]),
                "text": parts[3],
            }
            samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


async def run_benchmark(args):
    device = args.device

    # --- Load model + tokenizer -------------------------------------------
    logger.info("Loading S1-Mini model from %s ...", args.checkpoint)
    t0 = time.perf_counter()

    from third_party.fish_speech_s1.models.text2semantic.llama import DualARTransformer
    from third_party.fish_speech_s1.tokenizer import FishTokenizer

    # Resolve HF model ID to local path
    checkpoint = args.checkpoint
    if not os.path.isdir(checkpoint):
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download(checkpoint)
    args.checkpoint = checkpoint

    model = DualARTransformer.from_pretrained(checkpoint, load_weights=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FishTokenizer.from_pretrained(args.checkpoint)
    logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size

    # --- Load codec -------------------------------------------------------
    from sglang_omni.models.fishaudio_s1.pipeline.stages import _load_codec

    codec = _load_codec(args.checkpoint, device)

    # --- Create engine ----------------------------------------------------
    from sglang_omni.models.fishaudio_s1.factory import create_dual_ar_engine

    engine = create_dual_ar_engine(
        model=model,
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    # --- Tokenizer adapter ------------------------------------------------
    from sglang_omni.models.fishaudio_s1.runtime.dual_ar import DualARRequestData
    from sglang_omni.models.fishaudio_s1.tokenizer import (
        FishTokenizerAdapter,
        Reference,
    )

    adapter = FishTokenizerAdapter(tokenizer)

    # --- Parse test set ---------------------------------------------------
    samples = parse_meta_lst(args.testset)
    logger.info("Loaded %d samples from %s", len(samples), args.testset)

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
        logger.info("Truncated to %d samples", len(samples))

    # --- Output dirs ------------------------------------------------------
    out_dir = Path(args.output_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # --- Helpers ----------------------------------------------------------
    def encode_ref_audio(audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0).to(device)  # [1, T]
        audio_lengths = torch.tensor([audios.shape[1]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def vocode(all_codes: torch.Tensor) -> tuple[torch.Tensor, int]:
        """VQ codes [num_cb+1, T] -> (audio_1d, sample_rate)."""
        codebook_codes = all_codes[1:].to(device)  # [num_codebooks, T]
        with torch.no_grad():
            audio_out = codec.from_indices(codebook_codes[None])
        return audio_out[0, 0].float().cpu(), codec.sample_rate

    # --- Start engine -----------------------------------------------------
    await engine.start()
    logger.info("Engine started. Running %d samples...", len(samples))

    # --- Warmup -----------------------------------------------------------
    if args.warmup > 0 and len(samples) > 0:
        logger.info("Warming up with %d sample(s)...", args.warmup)
        for ws in samples[: args.warmup]:
            try:
                ref_codes = encode_ref_audio(ws["ref_audio"])
                refs = [
                    Reference(audio_bytes=b"", text=ws["ref_text"], vq_codes=ref_codes)
                ]
                values, audio_masks, audio_parts = adapter.build_prompt(
                    ws["text"],
                    references=refs,
                    num_codebooks=num_codebooks,
                )
                req = DualARRequestData(
                    input_values=values,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                    num_codebooks=num_codebooks,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                await engine.add_request(f"warmup-{ws['id']}", req)
                await engine.get_result(f"warmup-{ws['id']}")
            except Exception as e:
                logger.warning("Warmup failed for %s: %s", ws["id"], e)

    # --- Benchmark loop ---------------------------------------------------
    fieldnames = [
        "id",
        "text",
        "ref_audio",
        "output_audio",
        "latency_s",
        "audio_duration_s",
        "rtf",
        "gen_tokens",
        "tok_per_s",
        "error",
    ]
    rows: list[dict] = []

    for i, sample in enumerate(samples):
        sid = sample["id"]
        t_start = time.perf_counter()

        try:
            ref_codes = encode_ref_audio(sample["ref_audio"])
            refs = [
                Reference(audio_bytes=b"", text=sample["ref_text"], vq_codes=ref_codes)
            ]

            values, audio_masks, audio_parts = adapter.build_prompt(
                sample["text"],
                references=refs,
                num_codebooks=num_codebooks,
            )

            req = DualARRequestData(
                input_values=values,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                num_codebooks=num_codebooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            await engine.add_request(sid, req)
            result = await engine.get_result(sid)

            if (
                not isinstance(result, DualARRequestData)
                or len(result.output_codes) == 0
            ):
                raise RuntimeError("No output codes generated")

            all_codes = torch.cat(result.output_codes, dim=1)
            n_tokens = all_codes.shape[1]

            audio_out, sr = vocode(all_codes)
            latency = time.perf_counter() - t_start
            dur_s = audio_out.shape[0] / sr

            out_path = audio_dir / f"{sid}.wav"
            torchaudio.save(str(out_path), audio_out.unsqueeze(0), sr)

            rtf = latency / dur_s if dur_s > 0 else float("inf")
            tok_per_s = n_tokens / latency if latency > 0 else 0

            rows.append(
                {
                    "id": sid,
                    "text": sample["text"],
                    "ref_audio": sample["ref_audio"],
                    "output_audio": str(out_path),
                    "latency_s": f"{latency:.4f}",
                    "audio_duration_s": f"{dur_s:.4f}",
                    "rtf": f"{rtf:.4f}",
                    "gen_tokens": n_tokens,
                    "tok_per_s": f"{tok_per_s:.2f}",
                    "error": "",
                }
            )

            logger.info(
                "[%d/%d] %s: %.2fs latency | %.2fs audio | RTF=%.3f | %d tok @ %.1f tok/s",
                i + 1,
                len(samples),
                sid,
                latency,
                dur_s,
                rtf,
                n_tokens,
                tok_per_s,
            )

        except Exception as e:
            latency = time.perf_counter() - t_start
            logger.error("[%d/%d] %s FAILED: %s", i + 1, len(samples), sid, e)
            import traceback

            traceback.print_exc()
            rows.append(
                {
                    "id": sid,
                    "text": sample["text"],
                    "ref_audio": sample.get("ref_audio", ""),
                    "output_audio": "",
                    "latency_s": f"{latency:.4f}",
                    "audio_duration_s": "",
                    "rtf": "",
                    "gen_tokens": 0,
                    "tok_per_s": "",
                    "error": str(e),
                }
            )

    await engine.stop()

    # --- Write CSV --------------------------------------------------------
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Aggregate summary ------------------------------------------------
    ok_rows = [r for r in rows if r["error"] == ""]
    latencies = [float(r["latency_s"]) for r in ok_rows]
    rtfs = [float(r["rtf"]) for r in ok_rows]
    durations = [float(r["audio_duration_s"]) for r in ok_rows]
    tokens = [int(r["gen_tokens"]) for r in ok_rows]

    def _stats(vals):
        if not vals:
            return {}
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    summary = {
        "model": "s1-mini",
        "checkpoint": args.checkpoint,
        "device": device,
        "runtime": "sglang-omni OmniEngine (DualARBatchPlanner + DualAROutputProcessor)",
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": args.max_seq_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "total_samples": len(samples),
        "successful_samples": len(ok_rows),
        "failed_samples": len(samples) - len(ok_rows),
        "latency_s": _stats(latencies),
        "rtf": _stats(rtfs),
        "audio_duration_s": _stats(durations),
        "gen_tokens": _stats(tokens),
        "total_audio_s": sum(durations),
        "total_wall_time_s": sum(latencies),
        "avg_tok_per_s": sum(tokens) / sum(latencies) if latencies else 0,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print summary ----------------------------------------------------
    print("\n" + "=" * 64)
    print("  S1-Mini Seed-TTS-Eval Benchmark Results")
    print("=" * 64)
    print(f"  Samples:       {len(ok_rows)}/{len(samples)} successful")
    print(f"  Total audio:   {sum(durations):.1f}s")
    print(f"  Total wall:    {sum(latencies):.1f}s")
    if latencies:
        print(f"  Latency mean:  {np.mean(latencies):.3f}s")
        print(f"  Latency p95:   {np.percentile(latencies, 95):.3f}s")
        print(f"  RTF mean:      {np.mean(rtfs):.4f}")
        print(f"  RTF p95:       {np.percentile(rtfs, 95):.4f}")
        print(f"  Tok/s mean:    {summary['avg_tok_per_s']:.1f}")
    print(f"\n  Results:       {csv_path}")
    print(f"  Summary:       {summary_path}")
    print(f"  Audio:         {audio_dir}/")
    print("=" * 64 + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="S1-Mini seed-tts-eval benchmark (sglang-omni runtime)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="fishaudio/openaudio-s1-mini",
        help="S1-Mini checkpoint (HF model ID or local path)",
    )
    p.add_argument(
        "--testset",
        default="/tmp/seed-tts-eval/seedtts_testset/en/meta.lst",
        help="Path to seed-tts-eval meta.lst",
    )
    p.add_argument(
        "--output-dir", default="results/s1_seed_tts", help="Output directory"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--max-samples", type=int, default=None, help="Limit number of samples"
    )
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--warmup", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args))
