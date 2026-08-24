# SPDX-License-Identifier: Apache-2.0
"""Validate Qwen3-Omni Code2Wav NPUGraph capture and replay on one NPU."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch_npu  # noqa: F401  # Registers the ``npu`` PyTorch backend.

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavNpuGraphRunner,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--memory-fraction", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--frames", type=int, nargs="+", default=[10, 20, 30, 35])
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _milliseconds_per_call(fn, *, iterations: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(3):
            fn()
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            fn()
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "npu" or device.index is None:
        raise ValueError("--device must be a concrete NPU device such as npu:0")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    torch.npu.set_device(device)
    model = load_code2wav_model(
        args.model_path,
        device=str(device),
        dtype=args.dtype,
    )
    num_quantizers = int(model.config.num_quantizers)
    graph_keys = tuple(GraphKey(batch_size=1, frames=frames) for frames in args.frames)
    runner = Code2WavNpuGraphRunner.build(
        model,
        device=device,
        num_quantizers=num_quantizers,
        total_gpu_memory_fraction=args.memory_fraction,
        graph_keys=graph_keys,
    )
    stats = runner.stats()
    if not stats["enabled"]:
        raise RuntimeError(
            "NPUGraph capture was disabled: "
            f"{json.dumps(stats, sort_keys=True, default=str)}"
        )

    parity: list[dict[str, Any]] = []
    inputs: dict[GraphKey, torch.Tensor] = {}
    for key in graph_keys:
        codes = torch.arange(
            num_quantizers * key.frames,
            dtype=torch.long,
            device=device,
        ).reshape(1, num_quantizers, key.frames)
        codes.remainder_(16)
        inputs[key] = codes
        with torch.inference_mode():
            eager = model(codes).detach().clone()
        graph_result = runner.run(codes)
        graph_output = graph_result.output.detach().clone()
        _synchronize(device)
        exact = bool(torch.equal(eager, graph_output))
        parity.append(
            {
                "batch_size": key.batch_size,
                "frames": key.frames,
                "execution_mode": graph_result.execution_mode,
                "exact_match": exact,
            }
        )
        if not exact:
            raise RuntimeError(f"Eager/NPUGraph mismatch for {key}")

    benchmark_key = max(graph_keys, key=lambda key: key.frames)
    benchmark_codes = inputs[benchmark_key]
    eager_ms = _milliseconds_per_call(
        lambda: model(benchmark_codes),
        iterations=args.iterations,
        device=device,
    )
    graph_ms = _milliseconds_per_call(
        lambda: runner.run(benchmark_codes),
        iterations=args.iterations,
        device=device,
    )
    result = {
        "status": "pass",
        "device": str(device),
        "parity": parity,
        "benchmark": {
            "frames": benchmark_key.frames,
            "iterations": args.iterations,
            "eager_ms": eager_ms,
            "npu_graph_ms": graph_ms,
            "speedup": eager_ms / graph_ms,
        },
        "runner_stats": runner.stats(),
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
