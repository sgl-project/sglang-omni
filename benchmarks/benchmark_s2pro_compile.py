#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S2-Pro decode acceleration benchmark: eager vs torch.compile vs CUDA graph.

Compares three decode variants using fish_speech's native generate() + decode_one_token()
with deterministic multinomial_with_seed sampling.  All variants use the same code path;
only the decode function wrapper differs.

Variants:
    eager      – bare decode_one_token (no compilation, no graphs)
    compile    – torch.compile(decode_one_token, mode="max-autotune", fullgraph=True)
    cudagraph  – CUDAGraphRunner with static buffer replay

Usage:
    CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark_s2pro_compile.py \
        --variant eager --max-samples 50 \
        --checkpoint /root/.cache/huggingface/s2-pro/s2-pro

    # Quick smoke test
    python benchmarks/benchmark_s2pro_compile.py --variant eager --max-samples 3
"""

from __future__ import annotations

import argparse
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
# Seed-TTS-eval meta.lst parser (same as existing benchmark)
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


def run_benchmark(args):
    device = args.device

    # --- Load model via fish_speech load_model ----------------------------
    logger.info(
        "Loading S2-Pro model [variant=%s] from %s ...",
        args.variant,
        args.checkpoint,
    )
    t0 = time.perf_counter()

    from fish_speech.models.text2semantic.qwen3 import generate, load_model

    model, tokenizer, decode_fn = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
        use_cuda_graph=(args.variant == "cudagraph"),
        use_torch_compile=(args.variant == "compile"),
    )

    logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    # --- Load codec -------------------------------------------------------
    from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import _load_codec

    codec = _load_codec(args.checkpoint, device)

    # --- Tokenizer adapter ------------------------------------------------
    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    adapter = S2ProTokenizerAdapter(tokenizer)

    num_codebooks = model.config.audio_decoder_config.num_codebooks

    # --- Helpers ----------------------------------------------------------
    def encode_ref_audio(audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0).to(device)
        audio_lengths = torch.tensor([audios.shape[1]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def vocode(vq_parts_out: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Generated VQ codes [num_semantic, num_codebooks] -> (audio_1d, sr)."""
        # vq_parts_out from generate is [num_semantic, num_codebooks]
        # codec.from_indices expects [batch, num_codebooks, T]
        codebook_codes = vq_parts_out.T.to(device).unsqueeze(0)  # [1, num_cb, T]
        with torch.no_grad():
            audio_out = codec.from_indices(codebook_codes)
        return audio_out[0, 0].float().cpu(), codec.sample_rate

    def build_prompt_tensors(sample: dict):
        """Build input_ids, vq_mask_tokens, vq_parts for a sample."""
        ref_codes = encode_ref_audio(sample["ref_audio"])
        refs = [Reference(audio_bytes=b"", text=sample["ref_text"], vq_codes=ref_codes)]
        prompt = adapter.build_prompt(
            sample["text"],
            references=refs,
            num_codebooks=num_codebooks,
        )
        input_ids = prompt["input_ids"].to(device)
        vq_mask = prompt["vq_mask_tokens"].to(device)
        # Flatten vq_parts list -> single tensor [num_codebooks, total_T]
        # then transpose to [total_T, num_codebooks] as generate() expects
        vq_parts = torch.cat(
            [p.to(device) for p in prompt["vq_parts"]], dim=1
        ).T  # [total_T, num_codebooks]
        return input_ids, vq_mask, vq_parts

    # --- Parse test set ---------------------------------------------------
    samples = parse_meta_lst(args.testset)
    logger.info("Loaded %d samples from %s", len(samples), args.testset)
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
        logger.info("Truncated to %d samples", len(samples))

    # --- Output dirs ------------------------------------------------------
    out_dir = Path(args.output_dir or f"results/s2pro_compile_{args.variant}")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # --- Warmup -----------------------------------------------------------
    if args.warmup > 0 and len(samples) > 0:
        logger.info("Warming up with %d sample(s)...", args.warmup)
        for ws in samples[: args.warmup]:
            try:
                input_ids, vq_mask, vq_parts = build_prompt_tensors(ws)
                generate(
                    model=model,
                    input_ids=input_ids,
                    vq_parts=vq_parts,
                    vq_mask_tokens=vq_mask,
                    max_new_tokens=args.max_new_tokens,
                    decode_one_token_fn=decode_fn,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    constrain_to_semantic=True,
                )
            except Exception as e:
                logger.warning("Warmup failed for %s: %s", ws["id"], e)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

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
        "peak_gpu_mb",
        "error",
    ]
    rows: list[dict] = []

    for i, sample in enumerate(samples):
        sid = sample["id"]
        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.perf_counter()

        try:
            input_ids, vq_mask, vq_parts = build_prompt_tensors(sample)

            torch.cuda.synchronize(device)
            t_gen_start = time.perf_counter()

            result = generate(
                model=model,
                input_ids=input_ids,
                vq_parts=vq_parts,
                vq_mask_tokens=vq_mask,
                max_new_tokens=args.max_new_tokens,
                decode_one_token_fn=decode_fn,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                constrain_to_semantic=True,
            )

            torch.cuda.synchronize(device)
            latency = time.perf_counter() - t_gen_start

            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

            gen_sample = result.samples[0]
            if gen_sample.vq_parts is None or gen_sample.vq_parts.numel() == 0:
                raise RuntimeError("No VQ codes generated")

            n_tokens = gen_sample.vq_parts.shape[0]

            # Vocode to get audio duration
            audio_out, sr = vocode(gen_sample.vq_parts)
            dur_s = audio_out.shape[0] / sr

            # Save WAV
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
                    "peak_gpu_mb": f"{peak_mb:.0f}",
                    "error": "",
                }
            )

            logger.info(
                "[%d/%d] %s: %.2fs lat | %.2fs audio | RTF=%.3f | %d tok @ %.1f tok/s | %.0f MB",
                i + 1,
                len(samples),
                sid,
                latency,
                dur_s,
                rtf,
                n_tokens,
                tok_per_s,
                peak_mb,
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
                    "peak_gpu_mb": "",
                    "error": str(e),
                }
            )

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
    tps_vals = [float(r["tok_per_s"]) for r in ok_rows]
    peak_mbs = [float(r["peak_gpu_mb"]) for r in ok_rows]

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
        "model": "s2-pro",
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "device": device,
        "runtime": f"fish_speech generate() + decode_one_token [{args.variant}]",
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": args.max_seq_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "total_samples": len(samples),
        "successful_samples": len(ok_rows),
        "failed_samples": len(samples) - len(ok_rows),
        "latency_s": _stats(latencies),
        "rtf": _stats(rtfs),
        "audio_duration_s": _stats(durations),
        "gen_tokens": _stats(tokens),
        "tok_per_s": _stats(tps_vals),
        "peak_gpu_mb": _stats(peak_mbs),
        "total_audio_s": sum(durations),
        "total_wall_time_s": sum(latencies),
        "avg_tok_per_s": sum(tokens) / sum(latencies) if latencies else 0,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print summary ----------------------------------------------------
    print("\n" + "=" * 64)
    print(f"  S2-Pro Compile Benchmark: {args.variant.upper()}")
    print("=" * 64)
    print(f"  Samples:       {len(ok_rows)}/{len(samples)} successful")
    print(f"  Total audio:   {sum(durations):.1f}s")
    print(f"  Total wall:    {sum(latencies):.1f}s")
    if latencies:
        print(f"  Latency mean:  {np.mean(latencies):.3f}s")
        print(f"  Latency p95:   {np.percentile(latencies, 95):.3f}s")
        print(f"  RTF mean:      {np.mean(rtfs):.4f}")
        print(f"  RTF p95:       {np.percentile(rtfs, 95):.4f}")
        print(f"  Tok/s mean:    {np.mean(tps_vals):.1f}")
        print(f"  Tok/s median:  {np.median(tps_vals):.1f}")
        print(f"  Peak GPU MB:   {np.max(peak_mbs):.0f}")
    print(f"\n  Results:       {csv_path}")
    print(f"  Summary:       {summary_path}")
    print(f"  Audio:         {audio_dir}/")
    print("=" * 64 + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="S2-Pro decode acceleration benchmark: eager vs compile vs cudagraph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        choices=["eager", "compile", "cudagraph"],
        required=True,
        help="Decode variant to benchmark",
    )
    p.add_argument(
        "--checkpoint",
        default="/root/.cache/huggingface/s2-pro/s2-pro",
        help="S2-Pro checkpoint path",
    )
    p.add_argument(
        "--testset",
        default="/tmp/seed-tts-eval/seedtts_testset/en/meta.lst",
        help="Path to seed-tts-eval meta.lst",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: results/s2pro_compile_{variant})",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
