# SPDX-License-Identifier: Apache-2.0
"""Benchmark eager versus bucketed CUDA-graph Whisper encoder execution.

Example:

    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_whisper_encoder \
        --model-path openai/whisper-large-v3-turbo \
        --batch-sizes 1,2,4,8,16 --output whisper_encoder_graph.json

The benchmark loads real checkpoint encoder weights into SGLang-Omni's
Whisper encoder, requires every requested graph bucket to capture, and records
shape, dtype, device, exact equality, latency, and speedup for each batch size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Any, Callable

import torch
from transformers import WhisperForConditionalGeneration as HFWhisper

from sglang_omni.models.whisper_asr.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
)
from sglang_omni.models.whisper_asr.sglang_model import WhisperEncoder


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _batch_sizes(value: str) -> list[int]:
    sizes = [_positive_int(token.strip()) for token in value.split(",")]
    if not sizes:
        raise argparse.ArgumentTypeError("batch sizes must not be empty")
    return sorted(set(sizes))


def _dtype(value: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return choices[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"dtype must be one of {', '.join(choices)}"
        ) from exc


def _measure_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return (
        statistics.mean(samples),
        statistics.median(samples),
        samples[math.ceil(len(samples) * 0.95) - 1],
    )


def _load_encoder(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> WhisperEncoder:
    hf_model = HFWhisper.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    encoder = WhisperEncoder(hf_model.config).to(device=device, dtype=dtype).eval()
    encoder.load_state_dict(hf_model.model.encoder.state_dict(), strict=True)
    del hf_model
    return encoder


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_whisper_encoder requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encoder = _load_encoder(args.model_path, device, args.dtype)
    config = encoder.config
    feature_len = int(config.max_source_positions) * 2
    runner = WhisperEncoderCudaGraphRunner(
        encoder,
        num_mel_bins=int(config.num_mel_bins),
        input_feature_len=feature_len,
        min_free_gb=args.min_free_gb,
        warmup_iters=args.capture_warmup,
    )
    runner.capture(args.batch_sizes)
    captured = runner.captured_buckets
    if list(captured) != args.batch_sizes:
        raise RuntimeError(
            f"requested encoder graph buckets {args.batch_sizes}, captured {captured}"
        )

    results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_size)
        features = torch.randn(
            batch_size,
            int(config.num_mel_bins),
            feature_len,
            generator=generator,
            device=device,
            dtype=args.dtype,
        )
        with torch.inference_mode():
            eager_output = encoder(features)
            graph_output = runner.run(features)
        delta = (eager_output.float() - graph_output.float()).abs()
        eager_mean, eager_p50, eager_p95 = _measure_cuda_ms(
            lambda: encoder(features),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        graph_mean, graph_p50, graph_p95 = _measure_cuda_ms(
            lambda: runner.run(features),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(
            {
                "batch_size": batch_size,
                "input_shape": list(features.shape),
                "output_shape": list(graph_output.shape),
                "dtype": str(graph_output.dtype),
                "device": str(graph_output.device),
                "exact_equal": bool(torch.equal(eager_output, graph_output)),
                "max_abs_error": float(delta.max().item()),
                "mean_abs_error": float(delta.mean().item()),
                "eager_ms": {
                    "mean": eager_mean,
                    "p50": eager_p50,
                    "p95": eager_p95,
                },
                "cuda_graph_ms": {
                    "mean": graph_mean,
                    "p50": graph_p50,
                    "p95": graph_p95,
                },
                "speedup": eager_mean / graph_mean,
            }
        )
    return {
        "model_path": args.model_path,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "dtype": str(args.dtype),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "captured_buckets": captured,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    parser.add_argument("--batch-sizes", type=_batch_sizes, default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--capture-warmup", type=_positive_int, default=3)
    parser.add_argument("--iterations", type=_positive_int, default=50)
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="whisper_encoder_graph.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    output = os.path.abspath(args.output)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
