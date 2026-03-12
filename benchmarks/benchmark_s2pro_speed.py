#!/usr/bin/env python3
"""
S2-Pro TTS speed benchmark with engine-level metrics.
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

CHECKPOINT = os.environ.get("S2PRO_CKPT", "fishaudio/s2-pro")
TESTSET = os.environ.get("S2PRO_TESTSET", "")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_audio_decoder(checkpoint: str, device: str):
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
        FishQwen3OmniConfig,
    )
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3OmniForCausalLM,
    )

    config = FishQwen3OmniConfig.from_pretrained(checkpoint)
    model = FishQwen3OmniForCausalLM.from_pretrained(checkpoint, config=config)
    model = model.to(dtype=torch.bfloat16).eval()

    audio_decoder = model.audio_decoder
    audio_decoder.to(device=device)
    num_codebooks = config.audio_decoder_config.num_codebooks
    codebook_size = config.audio_decoder_config.vocab_size

    del model
    torch.cuda.empty_cache()
    return audio_decoder, num_codebooks, codebook_size


def load_codec(checkpoint: str, device: str):
    from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import _load_codec

    return _load_codec(checkpoint, device)


def create_engine(
    checkpoint: str,
    audio_decoder,
    tokenizer,
    num_codebooks: int,
    codebook_size: int,
    max_new_tokens: int = 2048,
    top_k: int = 30,
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
        disable_cuda_graph=False,
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
    )


# ---------------------------------------------------------------------------
# Test data
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

    refs = None
    if sample.get("ref_audio"):
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
        refs = [
            Reference(audio_bytes=b"", text=sample["ref_text"], vq_codes=indices.cpu())
        ]

    prompt = adapter.build_prompt(
        sample["text"], references=refs, num_codebooks=num_codebooks
    )
    input_ids = prompt["input_ids"]
    input_ids_list = (
        input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    )

    sampling_params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.8)
    sampling_params.normalize(tokenizer)
    sampling_params.verify(adapter._tok.vocab_size)

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
    ttft_s: float
    ttfb_s: float
    total_s: float
    tok_per_s: float
    audio_duration_s: float = 0.0
    rtf: float = 0.0


async def profile_request(
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

    # Stream to measure TTFT and time-to-10th-token
    ttft = None
    t_10th_token = None
    token_count = 0
    async for _codes in engine.stream(rid):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        token_count += 1
        if token_count == 10 and t_10th_token is None:
            t_10th_token = time.perf_counter() - t_start

    ar_time = time.perf_counter() - t_start
    n_codes = len(data.output_codes)
    tok_per_s = n_codes / ar_time if ar_time > 0 else 0

    # Vocode for audio duration + measure vocoder time for TTFB
    audio_dur = 0.0
    vocoder_time = 0.0
    if n_codes > 0:
        all_codes = torch.cat(data.output_codes, dim=-1)
        codebook_codes = all_codes[1:].to(device)

        # Time vocoder for first 10 tokens (for TTFB calculation)
        if n_codes >= 10:
            first_10_codes = torch.cat(data.output_codes[:10], dim=-1)
            first_10_cb = first_10_codes[1:].to(device)
            torch.cuda.synchronize()
            t_voc = time.perf_counter()
            with torch.no_grad():
                codec.from_indices(first_10_cb[None])
            torch.cuda.synchronize()
            vocoder_time = time.perf_counter() - t_voc

        torch.cuda.synchronize()
        t_full_voc = time.perf_counter()
        with torch.no_grad():
            audio_out = codec.from_indices(codebook_codes[None])
        torch.cuda.synchronize()
        full_vocoder_time = time.perf_counter() - t_full_voc
        audio_dur = audio_out.shape[-1] / codec.sample_rate

        if audio_dir is not None:
            out_path = str(Path(audio_dir) / f"{sample['id']}.wav")
            torchaudio.save(out_path, audio_out.squeeze(0).cpu(), codec.sample_rate)
    else:
        full_vocoder_time = 0.0

    total = ar_time + full_vocoder_time

    # TTFB = time to 10th token + vocoder for those 10 tokens
    ttfb = (t_10th_token or ttft or ar_time) + vocoder_time

    return RequestMetrics(
        request_id=rid,
        text=sample["text"][:60],
        prompt_tokens=prompt_len,
        gen_tokens=n_codes,
        ttft_s=ttft if ttft is not None else ar_time,
        ttfb_s=ttfb,
        total_s=total,
        tok_per_s=tok_per_s,
        audio_duration_s=audio_dur,
        rtf=total / audio_dur if audio_dur > 0 else float("inf"),
    )


def print_summary(label: str, metrics: list[RequestMetrics]) -> dict | None:
    if not metrics:
        print(f"\n{label}: No results")
        return None

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
        "ttft_mean_ms": round(float(np.mean(ttfts)) * 1000, 1),
        "ttft_median_ms": round(float(np.median(ttfts)) * 1000, 1),
        "ttfb_mean_ms": round(float(np.mean(ttfbs)) * 1000, 1),
        "ttfb_median_ms": round(float(np.median(ttfbs)) * 1000, 1),
        "ttfb_p95_ms": round(float(np.percentile(ttfbs, 95)) * 1000, 1),
        "tok_per_s_mean": round(float(np.mean(toks)), 1),
        "tok_per_s_agg": round(agg_tok_per_s, 1),
        "gen_tokens_mean": round(float(np.mean(gen_tokens)), 0),
        "latency_mean_s": round(float(np.mean(totals)), 3),
        "rtf_mean": round(float(np.mean(rtfs)), 4) if rtfs else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args):
    device = "cuda"

    # Load components
    logger.info("Loading audio decoder from %s ...", args.checkpoint)
    audio_decoder, num_codebooks, codebook_size = load_audio_decoder(
        args.checkpoint, device
    )

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)

    logger.info("Loading DAC codec...")
    codec = load_codec(args.checkpoint, device)

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import S2ProTokenizerAdapter

    adapter = S2ProTokenizerAdapter(tokenizer)

    # Create engine
    logger.info("Creating SGLang engine (unified decode, CUDA graph)...")
    engine = create_engine(
        args.checkpoint,
        audio_decoder,
        tokenizer,
        num_codebooks,
        codebook_size,
        args.max_new_tokens,
        args.top_k,
    )
    await engine.start()

    # Load samples
    if args.prompts:
        samples = [
            {"id": f"prompt-{i}", "text": t, "ref_audio": None, "ref_text": None}
            for i, t in enumerate(args.prompts)
        ]
        logger.info("Using %d text prompts (no ref audio)", len(samples))
    elif args.testset and os.path.isfile(args.testset):
        samples = parse_meta_lst(args.testset, args.max_samples)
        logger.info("Loaded %d samples from %s", len(samples), args.testset)
    else:
        logger.error("Provide --testset (voice cloning) or --prompts (no ref audio)")
        await engine.stop()
        return

    # Warmup
    logger.info("Warmup (%d requests)...", args.warmup)
    for i in range(args.warmup):
        s = samples[i % len(samples)]
        data = build_request_data(
            s,
            adapter,
            codec,
            tokenizer,
            num_codebooks,
            codebook_size,
            args.max_new_tokens,
            device,
        )
        rid = f"warmup-{i}"
        data.req.rid = rid
        await engine.add_request(rid, data)
        await asyncio.wait_for(engine.get_result(rid), timeout=120)
        # Warm up vocoder on first iteration
        if i == 0 and len(data.output_codes) > 0:
            codes = torch.cat(data.output_codes, dim=-1)
            with torch.no_grad():
                codec.from_indices(codes[1:].to(device)[None])
        logger.info("  warmup %d/%d done", i + 1, args.warmup)

    # Audio output dir
    audio_dir = None
    if args.save_audio:
        audio_dir = str(Path(args.output_dir) / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    # Profile
    logger.info("Profiling %d requests...", len(samples))
    metrics = []
    for i, sample in enumerate(samples):
        try:
            m = await profile_request(
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
            metrics.append(m)
            logger.info(
                "[%d/%d] TTFT=%.1fms TTFB=%.1fms %d tok %.1f tok/s RTF=%.3f  %s",
                i + 1,
                len(samples),
                m.ttft_s * 1000,
                m.ttfb_s * 1000,
                m.gen_tokens,
                m.tok_per_s,
                m.rtf,
                m.text[:50],
            )
        except Exception as e:
            logger.error("[%d/%d] %s FAILED: %s", i + 1, len(samples), sample["id"], e)

    await engine.stop()

    summary = print_summary("S2-Pro Speed Benchmark", metrics)

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "summary": summary,
            "config": {
                "checkpoint": args.checkpoint,
                "testset": args.testset,
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
                "top_k": args.top_k,
                "unified_decode": True,
                "warmup": args.warmup,
            },
            "per_request": [
                {
                    "id": m.request_id,
                    "text": m.text,
                    "prompt_tokens": m.prompt_tokens,
                    "gen_tokens": m.gen_tokens,
                    "ttft_ms": round(m.ttft_s * 1000, 1),
                    "ttfb_ms": round(m.ttfb_s * 1000, 1),
                    "total_s": round(m.total_s, 4),
                    "tok_per_s": round(m.tok_per_s, 1),
                    "audio_duration_s": round(m.audio_duration_s, 4),
                    "rtf": round(m.rtf, 4),
                }
                for m in metrics
            ],
        }
        out_path = out_dir / "speed_results.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("Results saved to %s", out_path)

        # CSV for downstream eval compatibility
        import csv

        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "text",
                    "latency_s",
                    "audio_duration_s",
                    "rtf",
                    "gen_tokens",
                    "tok_per_s",
                    "ttft_ms",
                    "ttfb_ms",
                ]
            )
            for m in metrics:
                w.writerow(
                    [
                        m.request_id,
                        m.text,
                        f"{m.total_s:.4f}",
                        f"{m.audio_duration_s:.4f}",
                        f"{m.rtf:.4f}",
                        m.gen_tokens,
                        f"{m.tok_per_s:.1f}",
                        f"{m.ttft_s*1000:.1f}",
                        f"{m.ttfb_s*1000:.1f}",
                    ]
                )
        logger.info("CSV saved to %s", csv_path)


def main():
    parser = argparse.ArgumentParser(description="S2-Pro TTS speed benchmark")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument(
        "--testset",
        default=TESTSET,
        help="seed-tts-eval meta.lst path (voice cloning mode)",
    )
    parser.add_argument(
        "--prompts", nargs="+", help="Plain text prompts (no ref audio mode)"
    )
    parser.add_argument("--output-dir", default="results/s2pro_speed")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--save-audio", action="store_true", help="Save generated WAV files"
    )
    args = parser.parse_args()

    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    asyncio.run(run(args))


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


if __name__ == "__main__":
    main()
