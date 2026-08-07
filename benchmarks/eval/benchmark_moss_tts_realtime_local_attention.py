# SPDX-License-Identifier: Apache-2.0
"""Benchmark MOSS-TTS-Realtime local attention backends."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.models.moss_tts_realtime.local_transformer import (
    MossTTSRealtimeLocalTransformer,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        rvq=16,
        audio_vocab_size=1027,
        audio_pad_token=1024,
    )


def _new_module(
    backend: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> MossTTSRealtimeLocalTransformer:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        torch.manual_seed(seed)
        with torch.device(device):
            module = MossTTSRealtimeLocalTransformer(
                _config(),
                attention_backend=backend,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    return module.eval()


def _frame_callable(
    module: MossTTSRealtimeLocalTransformer,
    hidden_states: torch.Tensor,
    codes: torch.Tensor,
):
    def run_frame() -> tuple[torch.Tensor, torch.Tensor]:
        values = hidden_states
        logits = values
        for position in range(16):
            values = module.step(values, position)
            logits = F.linear(
                values,
                module.local_lm_heads[position].weight,
            )
            if position + 1 < 16:
                values = F.embedding(
                    codes[:, position],
                    module.model.embed_tokens[position].weight,
                )
        return values, logits

    return run_frame


def _time_callable(
    fn,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000 / iterations)


def _run_backend(
    backend: str,
    *,
    fuse_qkv: bool,
    batch_size: int,
    hidden_states: torch.Tensor,
    codes: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor]]:
    module = _new_module(
        backend,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        seed=seed,
    )
    module.ensure_kv_cache(
        batch_size,
        hidden_states.device,
        hidden_states.dtype,
    )
    if fuse_qkv:
        for layer in module.model.layers:
            layer.self_attn.refresh_fused_qkv()
    run_frame = _frame_callable(module, hidden_states, codes)
    eager_samples_us = [
        _time_callable(
            run_frame,
            warmup=warmup,
            iterations=iterations,
        )
        for _ in range(repeats)
    ]
    expected = tuple(value.detach().clone() for value in run_frame())
    module.freeze_kv_cache()
    for _ in range(3):
        run_frame()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run_frame()
    graph_samples_us = [
        _time_callable(
            graph.replay,
            warmup=warmup,
            iterations=iterations,
        )
        for _ in range(repeats)
    ]
    actual = tuple(value.detach().clone() for value in graph_output)
    for graph_value, eager_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            graph_value,
            eager_value,
            rtol=0,
            atol=0,
        )
    resolved = module.model._resolved_attention_backend
    del module, graph
    torch.cuda.empty_cache()
    return (
        {
            "resolved_backend": resolved,
            "fused_qkv": fuse_qkv,
            "eager_us": statistics.median(eager_samples_us),
            "eager_samples_us": eager_samples_us,
            "cuda_graph_us": statistics.median(graph_samples_us),
            "cuda_graph_samples_us": graph_samples_us,
        },
        expected,
    )


def _comparison(
    baseline: tuple[torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    baseline_flat = torch.cat([value.float().flatten() for value in baseline])
    candidate_flat = torch.cat([value.float().flatten() for value in candidate])
    difference = (baseline_flat - candidate_flat).abs()
    cosine = F.cosine_similarity(
        baseline_flat,
        candidate_flat,
        dim=0,
    )
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine_similarity": float(cosine.item()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-affinity", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    if not args.batches or any(value < 1 for value in args.batches):
        parser.error("--batches must contain positive integers")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 1:
        parser.error("--warmup, --iterations, and --repeats must be positive")
    if args.cpu_affinity is not None and args.cpu_affinity < 0:
        parser.error("--cpu-affinity must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    if args.cpu_affinity is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpu-affinity is unsupported on this platform")
        os.sched_setaffinity(0, {args.cpu_affinity})
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    device_name = torch.cuda.get_device_name(device)
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": device_name,
        "torch": torch.__version__,
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cases": {},
    }
    for batch_size in args.batches:
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_size)
        hidden_states = torch.randn(
            batch_size,
            2048,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        codes = torch.randint(
            0,
            1024,
            (batch_size, 15),
            device=device,
            generator=generator,
        )
        sdpa, sdpa_output = _run_backend(
            "sdpa",
            fuse_qkv=False,
            batch_size=batch_size,
            hidden_states=hidden_states,
            codes=codes,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )
        sdpa_fused_qkv, sdpa_fused_qkv_output = _run_backend(
            "sdpa",
            fuse_qkv=True,
            batch_size=batch_size,
            hidden_states=hidden_states,
            codes=codes,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )
        fa3, fa3_output = _run_backend(
            "fa3",
            fuse_qkv=True,
            batch_size=batch_size,
            hidden_states=hidden_states,
            codes=codes,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )
        results["cases"][str(batch_size)] = {
            "sdpa": sdpa,
            "sdpa_fused_qkv": sdpa_fused_qkv,
            "fa3": fa3,
            "sdpa_fused_qkv_eager_speedup": (
                sdpa["eager_us"] / sdpa_fused_qkv["eager_us"]
            ),
            "sdpa_fused_qkv_cuda_graph_speedup": (
                sdpa["cuda_graph_us"] / sdpa_fused_qkv["cuda_graph_us"]
            ),
            "fa3_eager_speedup": sdpa["eager_us"] / fa3["eager_us"],
            "fa3_cuda_graph_speedup": (sdpa["cuda_graph_us"] / fa3["cuda_graph_us"]),
            "sdpa_fused_qkv_numerics": _comparison(
                sdpa_output,
                sdpa_fused_qkv_output,
            ),
            "fa3_numerics": _comparison(sdpa_output, fa3_output),
        }
    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
