# SPDX-License-Identifier: Apache-2.0
"""Compare isolated eager and whole-solver graph execution with Flow weights."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import load_flow


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--frame-bucket", type=int, default=16)
    parser.add_argument("--short-frames", type=int, default=250)
    parser.add_argument("--long-frames", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    flow = load_flow(args.assets / "flow.yaml")
    flow.load_state_dict(
        torch.load(args.assets / "flow.pt", map_location="cpu", weights_only=True)
    )
    decoder = flow.decoder.to(device="cuda", dtype=dtype).eval()
    cases = [
        ("b1_short", 1, args.short_frames),
        ("b1_long", 1, args.long_frames),
        ("b4_mixed", 4, args.long_frames),
        ("b8_mixed", 8, args.long_frames),
    ]
    shapes = tuple(
        (
            batch,
            (frames + args.frame_bucket - 1) // args.frame_bucket * args.frame_bucket,
        )
        for _, batch, frames in cases
    )
    runner = FlowCudaGraphRunner(
        decoder,
        capture_shapes=shapes,
        frame_bucket=args.frame_bucket,
        n_timesteps=10,
        conditioning_dtype=torch.float32,
    )
    runner.capture_all()
    results = []
    for name, batch, frames in cases:
        mu = torch.randn(
            batch, decoder.out_channels, frames, device="cuda", dtype=torch.float32
        )
        noise = decoder.rand_noise[..., :frames].expand(batch, -1, -1).clone()
        lengths = (
            torch.linspace(frames // 2, frames, batch, device="cuda").long()
            if batch > 1
            else torch.tensor([frames], device="cuda")
        )
        mask = (
            torch.arange(frames, device="cuda")[None, None, :] < lengths[:, None, None]
        ).to(torch.float32)
        mu *= mask
        spks = torch.randn(batch, decoder.out_channels, device="cuda", dtype=dtype)
        cond = torch.randn_like(mu) * mask
        cond[..., frames // 4 :] = 0
        t_span = 1 - torch.cos(
            torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)
            * 0.5
            * torch.pi
        )
        inputs = (noise, t_span, mu, mask, spks, cond)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            eager = decoder.solve_euler(*inputs)
            graphed = runner.run(*inputs)
            if graphed is None:
                raise RuntimeError(f"Graph unavailable for {name}")
            assert (
                graphed.shape == eager.shape
                and torch.isfinite(graphed).all()
                and torch.isfinite(eager).all()
            )
            error = (eager.float() - graphed.float()).abs()
            correctness = {
                "max_abs_error": error.max().item(),
                "max_rel_error": (error / eager.float().abs().clamp_min(1e-6))
                .max()
                .item(),
                "bit_exact": torch.equal(eager, graphed),
            }
            torch.testing.assert_close(graphed, eager, atol=2e-3, rtol=2e-3)
            for mode in ("eager", "graph"):
                execute = decoder.solve_euler if mode == "eager" else runner.run
                for _ in range(3):
                    execute(*inputs)
                torch.cuda.synchronize()
                wall_ms = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    execute(*inputs)
                    torch.cuda.synchronize()
                    wall_ms.append((time.perf_counter() - start) * 1000)
                trace_path = args.output / f"{name}_{mode}.json"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    profile_start = time.perf_counter()
                    execute(*inputs)
                    torch.cuda.synchronize()
                    profiled_wall_ms = (time.perf_counter() - profile_start) * 1000
                profiler.export_chrome_trace(str(trace_path))
                events = json.loads(trace_path.read_text())["traceEvents"]
                kernels = [event for event in events if event.get("cat") == "kernel"]
                launches = [
                    event
                    for event in events
                    if event.get("cat") == "cuda_runtime"
                    and "Launch" in event.get("name", "")
                ]
                kernel_ms = sum(event["dur"] for event in kernels) / 1000
                result = {
                    "case": name,
                    "mode": mode,
                    "batch_size": batch,
                    "frames": frames,
                    "bucket_frames": runner.select(batch, frames).mel_frames_bucket,
                    "wall_ms_median": statistics.median(wall_ms),
                    "kernel_count": len(kernels),
                    "kernel_ms_profiled": kernel_ms,
                    "wall_ms_profiled": profiled_wall_ms,
                    "gpu_active_fraction_profiled": kernel_ms / profiled_wall_ms,
                    "gpu_kernel_gap_ms_profiled": (
                        max(event["ts"] + event["dur"] for event in kernels)
                        - min(event["ts"] for event in kernels)
                    )
                    / 1000
                    - kernel_ms,
                    "cuda_graph_launch_count": sum(
                        "cudaGraphLaunch" in event["name"] for event in launches
                    ),
                    "cuda_launch_api_ms_profiled": sum(
                        event["dur"] for event in launches
                    )
                    / 1000,
                    **correctness,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "dtype": args.dtype,
                "conditioning_dtype": "float32",
                "frame_bucket": args.frame_bucket,
                "iterations": args.iterations,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
