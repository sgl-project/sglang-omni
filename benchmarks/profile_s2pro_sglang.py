#!/usr/bin/env python3
"""Performance profiling for S2-Pro SGLang paged-attention engine.

Measures:
  - TTFT (time to first token): from add_request to first stream output
  - tok/s: semantic tokens per second (generation throughput)
  - Total latency, RTF

Tests single-request and batched-request scenarios.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
        --checkpoint /root/.cache/huggingface/s2-pro/s2-pro \
        --testset /tmp/seed-tts-eval/seedtts_testset/en/meta.lst \
        --max-samples 10 --batch-sizes 1,2,4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT = "/root/.cache/huggingface/s2-pro/s2-pro"
TESTSET = "/tmp/seed-tts-eval/seedtts_testset/en/meta.lst"


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def parse_meta_lst(path: str, max_samples: int | None = None) -> list[dict]:
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
            samples.append(
                {
                    "id": parts[0],
                    "ref_text": parts[1],
                    "ref_audio": os.path.join(base_dir, parts[2]),
                    "text": parts[3],
                }
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_audio_decoder(checkpoint: str, device: str):
    from fish_speech.models.text2semantic.configuration import FishQwen3OmniConfig
    from fish_speech.models.text2semantic.modeling import FishQwen3OmniForCausalLM

    config = FishQwen3OmniConfig.from_pretrained(checkpoint)
    full_model = FishQwen3OmniForCausalLM.from_pretrained(checkpoint, config=config)
    full_model = full_model.to(device=device, dtype=torch.bfloat16).eval()

    audio_decoder = full_model.audio_decoder
    audio_decoder.setup_caches(max_batch_size=1, dtype=torch.bfloat16)
    audio_decoder._parent_ref = full_model
    return audio_decoder, config


def create_sglang_engine(
    checkpoint,
    audio_decoder,
    tokenizer,
    num_codebooks,
    codebook_size,
    max_new_tokens,
    top_k,
    use_torch_compile=False,
):
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.fishaudio_s2_pro.factory import (
        _patch_fish_config_for_sglang,
        create_s2pro_sglang_engine,
    )

    _patch_fish_config_for_sglang(checkpoint)
    server_args = ServerArgs(
        model_path=checkpoint,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        chunked_prefill_size=8192,
        max_running_requests=64,
        disable_cuda_graph=True,
    )
    return create_s2pro_sglang_engine(
        server_args=server_args,
        audio_decoder=audio_decoder,
        tokenizer=tokenizer,
        gpu_id=0,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        use_torch_compile=use_torch_compile,
    )


def build_request_data(
    sample,
    adapter,
    codec,
    tokenizer,
    num_codebooks,
    codebook_size,
    max_new_tokens,
    device,
):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
        S2ProSGLangRequestData,
    )
    from sglang_omni.models.fishaudio_s2_pro.tokenizer import Reference

    # Encode reference audio
    audio, sr = torchaudio.load(sample["ref_audio"])
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
    audios = audio.squeeze(0).unsqueeze(0).to(device)
    audio_lengths = torch.tensor([audios.shape[1]], device=device, dtype=torch.long)
    with torch.no_grad():
        indices, _ = codec.encode(audios, audio_lengths)
        if indices.ndim == 3:
            indices = indices[0]
    ref_codes = indices.cpu()

    refs = [Reference(audio_bytes=b"", text=sample["ref_text"], vq_codes=ref_codes)]
    prompt = adapter.build_prompt(
        sample["text"], references=refs, num_codebooks=num_codebooks
    )
    input_ids = prompt["input_ids"]

    sampling_params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.8)
    sampling_params.normalize(tokenizer)
    sampling_params.verify(adapter._tok.vocab_size)

    input_ids_list = (
        input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    )
    req = Req(
        rid=sample["id"],
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=adapter._tok.vocab_size,
    )

    return S2ProSGLangRequestData(
        input_ids=(
            torch.tensor(input_ids_list, dtype=torch.long)
            if not isinstance(input_ids, torch.Tensor)
            else input_ids
        ),
        req=req,
        vq_mask_tokens=prompt["vq_mask_tokens"],
        vq_parts=prompt["vq_parts"],
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
    )


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    request_id: str
    text: str
    prompt_tokens: int
    gen_tokens: int
    ttft_s: float  # time to first token
    ttfb_s: float  # time to first audio block (TTFT + 10 tokens + vocoder)
    total_s: float  # total generation time
    tok_per_s: float
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    ref_audio: str = ""
    output_audio: str = ""


async def profile_single_request(
    engine,
    sample,
    adapter,
    codec,
    tokenizer,
    num_codebooks,
    codebook_size,
    max_new_tokens,
    device,
    audio_dir=None,
) -> RequestMetrics:
    """Profile a single request with TTFT and TTFB measurement via streaming."""
    data = build_request_data(
        sample,
        adapter,
        codec,
        tokenizer,
        num_codebooks,
        codebook_size,
        max_new_tokens,
        device,
    )
    rid = f"prof-{sample['id']}"
    data.req.rid = rid
    prompt_len = len(data.input_ids)

    t_start = time.perf_counter()
    await engine.add_request(rid, data)

    # Stream to measure TTFT and TTFB (TTFT + 10 tokens + vocoder for first block)
    ttft = None
    t_10th_token = None
    token_count = 0
    async for codes in engine.stream(rid):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        token_count += 1
        if token_count == 10 and t_10th_token is None:
            t_10th_token = time.perf_counter() - t_start

    total = time.perf_counter() - t_start

    result = data
    n_codes = len(result.output_codes)
    tok_per_s = n_codes / total if total > 0 else 0

    # Vocode for audio duration + measure vocoder time for TTFB
    audio_dur = 0.0
    vocoder_time = 0.0
    output_audio_path = ""
    if n_codes > 0:
        all_codes = torch.cat(result.output_codes, dim=-1)
        codebook_codes = all_codes[1:].to(device)
        # Time the vocoder for first 10 tokens
        if n_codes >= 10:
            first_10_codes = torch.cat(result.output_codes[:10], dim=-1)
            first_10_cb = first_10_codes[1:].to(device)
            torch.cuda.synchronize()
            t_voc = time.perf_counter()
            with torch.no_grad():
                codec.from_indices(first_10_cb[None])
            torch.cuda.synchronize()
            vocoder_time = time.perf_counter() - t_voc

        with torch.no_grad():
            audio_out = codec.from_indices(codebook_codes[None])
        audio_dur = audio_out.shape[-1] / codec.sample_rate

        # Save audio if requested
        if audio_dir is not None:
            output_audio_path = str(Path(audio_dir) / f"{sample['id']}.wav")
            torchaudio.save(
                output_audio_path, audio_out.squeeze(0).cpu(), codec.sample_rate
            )

    # TTFB = time to 10th token + vocoder decode time for those 10 tokens
    if t_10th_token is not None:
        ttfb = t_10th_token + vocoder_time
    else:
        ttfb = (ttft or total) + vocoder_time

    return RequestMetrics(
        request_id=rid,
        text=sample["text"][:60],
        prompt_tokens=prompt_len,
        gen_tokens=n_codes,
        ttft_s=ttft if ttft is not None else total,
        ttfb_s=ttfb,
        total_s=total,
        tok_per_s=tok_per_s,
        audio_duration_s=audio_dur,
        rtf=total / audio_dur if audio_dur > 0 else float("inf"),
        ref_audio=sample.get("ref_audio", ""),
        output_audio=output_audio_path,
    )


async def profile_batch(
    engine,
    samples,
    adapter,
    codec,
    tokenizer,
    num_codebooks,
    codebook_size,
    max_new_tokens,
    device,
) -> list[RequestMetrics]:
    """Profile a batch of concurrent requests."""
    batch_size = len(samples)

    # Pre-build all request data
    requests = []
    for i, sample in enumerate(samples):
        data = build_request_data(
            sample,
            adapter,
            codec,
            tokenizer,
            num_codebooks,
            codebook_size,
            max_new_tokens,
            device,
        )
        rid = f"batch-{i}-{sample['id']}"
        data.req.rid = rid
        requests.append((rid, data, sample))

    t_batch_start = time.perf_counter()

    # Add all requests
    for rid, data, _ in requests:
        await engine.add_request(rid, data)

    # Collect results concurrently
    async def collect_one(rid, data, sample):
        t_start = time.perf_counter()
        ttft = None
        t_10th_token = None
        token_count = 0
        async for codes in engine.stream(rid):
            if ttft is None:
                ttft = time.perf_counter() - t_start
            token_count += 1
            if token_count == 10 and t_10th_token is None:
                t_10th_token = time.perf_counter() - t_start

        total = time.perf_counter() - t_start
        n_codes = len(data.output_codes)
        tok_per_s = n_codes / total if total > 0 else 0

        audio_dur = 0.0
        vocoder_time = 0.0
        if n_codes > 0:
            all_codes = torch.cat(data.output_codes, dim=-1)
            codebook_codes = all_codes[1:].to(device)
            if n_codes >= 10:
                first_10_codes = torch.cat(data.output_codes[:10], dim=-1)
                first_10_cb = first_10_codes[1:].to(device)
                torch.cuda.synchronize()
                t_voc = time.perf_counter()
                with torch.no_grad():
                    codec.from_indices(first_10_cb[None])
                torch.cuda.synchronize()
                vocoder_time = time.perf_counter() - t_voc
            with torch.no_grad():
                audio_out = codec.from_indices(codebook_codes[None])
            audio_dur = audio_out.shape[-1] / codec.sample_rate

        ttfb = (t_10th_token or ttft or total) + vocoder_time

        return RequestMetrics(
            request_id=rid,
            text=sample["text"][:60],
            prompt_tokens=len(data.input_ids),
            gen_tokens=n_codes,
            ttft_s=ttft if ttft is not None else total,
            ttfb_s=ttfb,
            total_s=total,
            tok_per_s=tok_per_s,
            audio_duration_s=audio_dur,
            rtf=total / audio_dur if audio_dur > 0 else float("inf"),
        )

    results = await asyncio.gather(
        *[collect_one(rid, data, sample) for rid, data, sample in requests]
    )
    return list(results)


def print_metrics(label: str, metrics: list[RequestMetrics]):
    if not metrics:
        print(f"\n{label}: No results")
        return

    ttfts = [m.ttft_s for m in metrics]
    ttfbs = [m.ttfb_s for m in metrics]
    toks = [m.tok_per_s for m in metrics]
    totals = [m.total_s for m in metrics]
    rtfs = [m.rtf for m in metrics if m.rtf < float("inf")]
    gen_tokens = [m.gen_tokens for m in metrics]

    total_tokens = sum(gen_tokens)
    total_wall = sum(totals)
    agg_tok_per_s = total_tokens / total_wall if total_wall > 0 else 0

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")
    print(f"  Requests:        {len(metrics)}")
    print(
        f"  TTFT mean:       {np.mean(ttfts)*1000:.1f}ms (median {np.median(ttfts)*1000:.1f}ms)"
    )
    print(
        f"  TTFB mean:       {np.mean(ttfbs)*1000:.1f}ms (median {np.median(ttfbs)*1000:.1f}ms)"
    )
    print(f"  TTFB p95:        {np.percentile(ttfbs, 95)*1000:.1f}ms")
    print(f"  Tok/s (per-req): {np.mean(toks):.1f} mean, {np.median(toks):.1f} median")
    print(f"  Tok/s (agg):     {agg_tok_per_s:.1f}")
    print(f"  Gen tokens:      {np.mean(gen_tokens):.0f} mean, {sum(gen_tokens)} total")
    print(f"  Latency mean:    {np.mean(totals):.3f}s")
    if rtfs:
        print(f"  RTF mean:        {np.mean(rtfs):.4f}")
    print(f"{'='*64}")

    return {
        "label": label,
        "n_requests": len(metrics),
        "ttft_mean_ms": float(np.mean(ttfts)) * 1000,
        "ttft_median_ms": float(np.median(ttfts)) * 1000,
        "ttfb_mean_ms": float(np.mean(ttfbs)) * 1000,
        "ttfb_median_ms": float(np.median(ttfbs)) * 1000,
        "ttfb_p95_ms": float(np.percentile(ttfbs, 95)) * 1000,
        "tok_per_s_mean": float(np.mean(toks)),
        "tok_per_s_agg": agg_tok_per_s,
        "gen_tokens_mean": float(np.mean(gen_tokens)),
        "latency_mean": float(np.mean(totals)),
        "rtf_mean": float(np.mean(rtfs)) if rtfs else None,
    }


async def run_profiling(args):
    device = "cuda"

    # Load model components
    logger.info("Loading audio decoder...")
    audio_decoder, config = load_audio_decoder(args.checkpoint, device)
    num_codebooks = config.audio_decoder_config.num_codebooks
    codebook_size = config.audio_decoder_config.vocab_size

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)

    from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import _load_codec

    codec = _load_codec(args.checkpoint, device)

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import S2ProTokenizerAdapter

    adapter = S2ProTokenizerAdapter(tokenizer)

    # Create engine
    logger.info("Creating SGLang engine (compile=%s)...", args.compile)
    engine = create_sglang_engine(
        args.checkpoint,
        audio_decoder,
        tokenizer,
        num_codebooks,
        codebook_size,
        args.max_new_tokens,
        args.top_k,
        use_torch_compile=args.compile,
    )
    await engine.start()

    # Load test samples
    samples = parse_meta_lst(args.testset, args.max_samples)
    logger.info("Loaded %d test samples", len(samples))

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    all_results = {}

    # Warmup
    logger.info("Warmup...")
    try:
        warmup_data = build_request_data(
            samples[0],
            adapter,
            codec,
            tokenizer,
            num_codebooks,
            codebook_size,
            args.max_new_tokens,
            device,
        )
        warmup_data.req.rid = "warmup-0"
        await engine.add_request("warmup-0", warmup_data)
        await asyncio.wait_for(engine.get_result("warmup-0"), timeout=120)
        logger.info("Warmup done")
    except Exception as e:
        logger.warning("Warmup failed: %s", e)

    # Set up audio output dir
    audio_dir = None
    if args.save_audio:
        audio_dir = str(Path(args.output_dir) / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    # Single-request profiling
    if 1 in batch_sizes:
        logger.info("Profiling single requests...")
        single_metrics = []
        for i, sample in enumerate(samples):
            try:
                m = await profile_single_request(
                    engine,
                    sample,
                    adapter,
                    codec,
                    tokenizer,
                    num_codebooks,
                    codebook_size,
                    args.max_new_tokens,
                    device,
                    audio_dir=audio_dir,
                )
                single_metrics.append(m)
                logger.info(
                    "[%d/%d] %s: TTFT=%.1fms TTFB=%.1fms %d tok %.1f tok/s %.3fs",
                    i + 1,
                    len(samples),
                    m.request_id,
                    m.ttft_s * 1000,
                    m.ttfb_s * 1000,
                    m.gen_tokens,
                    m.tok_per_s,
                    m.total_s,
                )
            except Exception as e:
                logger.error(
                    "[%d/%d] %s FAILED: %s", i + 1, len(samples), sample["id"], e
                )

        all_results["single"] = print_metrics("Single Request", single_metrics)

        # Radix cache hit test: re-run same samples (prefix should be cached)
        if args.test_cache_hit:
            logger.info("Testing radix cache hit (re-running same ref audio)...")
            cache_metrics = []
            for i, sample in enumerate(samples[: min(5, len(samples))]):
                try:
                    m = await profile_single_request(
                        engine,
                        sample,
                        adapter,
                        codec,
                        tokenizer,
                        num_codebooks,
                        codebook_size,
                        args.max_new_tokens,
                        device,
                        audio_dir=None,  # don't save audio for cache hit test
                    )
                    cache_metrics.append(m)
                    logger.info(
                        "[cache %d] %s: TTFT=%.1fms TTFB=%.1fms %d tok %.1f tok/s",
                        i + 1,
                        m.request_id,
                        m.ttft_s * 1000,
                        m.ttfb_s * 1000,
                        m.gen_tokens,
                        m.tok_per_s,
                    )
                except Exception as e:
                    logger.error("[cache %d] FAILED: %s", i + 1, e)
            all_results["cache_hit"] = print_metrics("Radix Cache Hit", cache_metrics)

        batch_sizes = [b for b in batch_sizes if b != 1]

    # Batched-request profiling
    for bs in batch_sizes:
        if bs <= 1:
            continue
        logger.info("Profiling batch_size=%d...", bs)
        batch_metrics = []

        # Run batches of `bs` samples
        for start in range(0, len(samples), bs):
            batch_samples = samples[start : start + bs]
            if len(batch_samples) < bs:
                break
            try:
                metrics = await profile_batch(
                    engine,
                    batch_samples,
                    adapter,
                    codec,
                    tokenizer,
                    num_codebooks,
                    codebook_size,
                    args.max_new_tokens,
                    device,
                )
                batch_metrics.extend(metrics)
                for m in metrics:
                    logger.info(
                        "  %s: TTFT=%.4fs, %d tok, %.1f tok/s",
                        m.request_id,
                        m.ttft_s,
                        m.gen_tokens,
                        m.tok_per_s,
                    )
            except Exception as e:
                logger.error("Batch starting at %d FAILED: %s", start, e)

        all_results[f"batch_{bs}"] = print_metrics(f"Batch Size {bs}", batch_metrics)

    await engine.stop()

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "profile_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save per-sample CSV (for WER eval compatibility)
    if "single" in all_results and single_metrics:
        import csv

        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "text",
                    "ref_audio",
                    "output_audio",
                    "latency_s",
                    "audio_duration_s",
                    "rtf",
                    "gen_tokens",
                    "tok_per_s",
                    "ttft_ms",
                    "ttfb_ms",
                    "error",
                ]
            )
            for m in single_metrics:
                w.writerow(
                    [
                        m.request_id.replace("prof-", ""),
                        m.text,
                        m.ref_audio,
                        m.output_audio,
                        f"{m.total_s:.4f}",
                        f"{m.audio_duration_s:.4f}",
                        f"{m.rtf:.4f}",
                        m.gen_tokens,
                        f"{m.tok_per_s:.2f}",
                        f"{m.ttft_s*1000:.1f}",
                        f"{m.ttfb_s*1000:.1f}",
                        "",
                    ]
                )
        logger.info("CSV saved to %s", csv_path)

    logger.info("Results saved to %s", out_dir / "profile_results.json")


def parse_args():
    p = argparse.ArgumentParser(description="S2-Pro SGLang performance profiling")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--testset", default=TESTSET)
    p.add_argument("--output-dir", default="results/s2pro_sglang_profile")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument(
        "--batch-sizes", default="1,2,4", help="Comma-separated batch sizes to test"
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on codebook loop (EXPERIMENTAL: may produce incorrect audio)",
    )
    p.add_argument(
        "--test-cache-hit", action="store_true", help="Test radix cache hit TTFB"
    )
    p.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated WAV files for WER eval",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_profiling(args))
