# SPDX-License-Identifier: Apache-2.0
"""Micro benchmark: HF audio encoder vs sglang native implementation.

Modes:
  - eager       : standard forward, vary batch and seq_len
  - cuda_graph  : capture + replay, measure against eager
  - tp          : (invoked via torchrun) TP=N sglang-only scaling

Usage:

  # eager, seq_lens 300/1000/3000, batch sizes 1/4/16
  python -m benchmarks.micro.bench_audio_encoder \\
      --model-path /fsx/enwei/models/Qwen3-Omni-30B-A3B-Instruct \\
      --device cuda:0 --mode eager \\
      --seq-lens 300,1000,3000 --batch-sizes 1,4,16 --iters 50

  # cuda graph (fixed shapes)
  python -m benchmarks.micro.bench_audio_encoder \\
      --model-path /fsx/enwei/models/Qwen3-Omni-30B-A3B-Instruct \\
      --device cuda:0 --mode cuda_graph \\
      --seq-lens 1000 --batch-sizes 1,4 --iters 100

  # TP=2 sglang-only (uses torchrun)
  torchrun --nproc_per_node=2 -m benchmarks.micro.bench_audio_encoder \\
      --model-path /fsx/enwei/models/Qwen3-Omni-30B-A3B-Instruct \\
      --device cuda --mode tp --tp-size 2 \\
      --seq-lens 1000 --batch-sizes 1,4 --iters 50
"""

from __future__ import annotations

import argparse
import gc
import inspect
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger("bench_audio_encoder")
logging.basicConfig(level=logging.INFO, format="%(message)s")


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


@dataclass
class BenchResult:
    impl: str
    mode: str
    batch: int
    seq_len: int
    latencies_ms: list[float]
    peak_mem_mb: float
    output_sample: torch.Tensor | None = None

    @property
    def p50(self) -> float:
        return float(np.percentile(self.latencies_ms, 50))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def mean(self) -> float:
        return float(np.mean(self.latencies_ms))


def _make_input(
    batch: int,
    seq_len: int,
    *,
    mel_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic mel input: (mel_bins, batch*seq_len) flat + per-sample lengths."""
    torch.manual_seed(seed)
    input_features = torch.randn(
        mel_bins, batch * seq_len, device=device, dtype=dtype
    )
    audio_feature_lengths = torch.tensor(
        [seq_len] * batch, device=device, dtype=torch.long
    )
    return input_features, audio_feature_lengths


def _run_once(encoder, input_features, audio_feature_lengths) -> torch.Tensor:
    with torch.inference_mode():
        out = encoder.forward(
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
        )
    return out["audio_embeds"]


def bench_eager(
    encoder,
    *,
    impl_name: str,
    batch: int,
    seq_len: int,
    mel_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> BenchResult:
    input_features, lengths = _make_input(
        batch, seq_len, mel_bins=mel_bins, device=device, dtype=dtype
    )
    for _ in range(warmup):
        _ = _run_once(encoder, input_features, lengths)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    out_sample = None
    for i in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = _run_once(encoder, input_features, lengths)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if i == 0:
            out_sample = out.detach().to(torch.float32).cpu()

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return BenchResult(
        impl=impl_name,
        mode="eager",
        batch=batch,
        seq_len=seq_len,
        latencies_ms=latencies_ms,
        peak_mem_mb=peak_mem_mb,
        output_sample=out_sample,
    )


def bench_cuda_graph(
    encoder,
    *,
    impl_name: str,
    batch: int,
    seq_len: int,
    mel_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> BenchResult | None:
    """Capture + replay. Returns None if capture fails."""
    input_features, lengths = _make_input(
        batch, seq_len, mel_bins=mel_bins, device=device, dtype=dtype
    )
    # Warmup on eager
    for _ in range(warmup):
        _ = _run_once(encoder, input_features, lengths)
    torch.cuda.synchronize(device)

    static_in = input_features.clone()
    static_lens = lengths.clone()

    graph_kwargs: dict = {"audio_feature_lengths": static_lens}
    if "skip_shape_check" in inspect.signature(encoder.forward).parameters:
        graph_kwargs["skip_shape_check"] = True

    g = torch.cuda.CUDAGraph()
    try:
        # Extra warmup on the inputs we'll capture against, to pre-allocate
        # any workspace buffers so graph capture sees only steady-state ops.
        for _ in range(3):
            with torch.inference_mode():
                _ = encoder.forward(input_features=static_in, **graph_kwargs)
        torch.cuda.synchronize(device)
        with torch.inference_mode():
            with torch.cuda.graph(g):
                static_out = encoder.forward(
                    input_features=static_in, **graph_kwargs
                )["audio_embeds"]
    except Exception as exc:
        logger.warning(f"[{impl_name}] cuda graph capture FAILED: {exc}")
        return None

    torch.cuda.reset_peak_memory_stats(device)
    latencies_ms = []
    out_sample = None
    # Reuse the single captured input across iters — identical methodology
    # to bench_eager, and keeps allocator churn out of the timing window.
    for i in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if i == 0:
            out_sample = static_out.detach().to(torch.float32).cpu()

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return BenchResult(
        impl=impl_name,
        mode="cuda_graph",
        batch=batch,
        seq_len=seq_len,
        latencies_ms=latencies_ms,
        peak_mem_mb=peak_mem_mb,
        output_sample=out_sample,
    )


# Acceptable numerical divergence between bf16 attention kernels.
# FA3 (sglang) and SDPA (HF) differ in softmax reduction order; accumulated
# error scales with sequence length. Empirically observed on H100 bf16:
#   B=1  L=1000: max ~1.5e-2, mean ~8e-4
#   B=4  L=3000: max ~1.6e-1, mean ~1.5e-3
#   B=16 L=3000: max ~2.1e-1, mean ~1.8e-3
# mean_diff is the more load-bearing threshold (a real regression shows up
# as a shift in many positions, not an extreme in one).
PARITY_MAX_ABS_DIFF = 3.0e-1
PARITY_MAX_MEAN_DIFF = 5.0e-3


def _run_parity_check(
    hf_encoder,
    r_eager: "BenchResult",
    r_graph: "BenchResult | None",
    *,
    batch: int,
    seq_len: int,
    mel_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Compare graphed eager + graph outputs against HF on the same input.

    ``bench_eager`` stores ``output_sample`` from its first iter (seed=0).
    ``bench_cuda_graph`` also uses seed=0 for iter 0 (see that function).
    We recompute HF at seed=0 and diff.
    """
    x0, lens0 = _make_input(
        batch, seq_len, mel_bins=mel_bins, device=device, dtype=dtype, seed=0
    )
    with torch.inference_mode():
        hf_ref = (
            hf_encoder.forward(input_features=x0, audio_feature_lengths=lens0)[
                "audio_embeds"
            ]
            .detach()
            .to(torch.float32)
            .cpu()
        )

    for label, r in (("graphed-eager", r_eager), ("graphed-cuda_graph", r_graph)):
        if r is None or r.output_sample is None:
            continue
        mx, mn = numerical_diff(hf_ref, r.output_sample)
        ok = mx < PARITY_MAX_ABS_DIFF and mn < PARITY_MAX_MEAN_DIFF
        status = "OK" if ok else "FAIL"
        logger.info(
            f"[parity {status}] B={batch} L={seq_len} {label} vs HF: "
            f"max_diff={mx:.3e}  mean_diff={mn:.3e}  "
            f"(threshold max<{PARITY_MAX_ABS_DIFF} mean<{PARITY_MAX_MEAN_DIFF})"
        )
        if not ok:
            raise RuntimeError(
                f"Parity check failed for {label} @ B={batch} L={seq_len}: "
                f"max_diff={mx:.3e} mean_diff={mn:.3e}"
            )


def numerical_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff). Squeeze trailing batch dims."""
    a = a.squeeze()
    b = b.squeeze()
    if a.shape != b.shape:
        return float("nan"), float("nan")
    diff = (a - b).abs()
    return float(diff.max()), float(diff.mean())


def _print_result(r: BenchResult) -> None:
    logger.info(
        f"[{r.impl:>7} {r.mode:>10}] B={r.batch:>3} L={r.seq_len:>5}  "
        f"mean={r.mean:7.2f}ms  p50={r.p50:7.2f}ms  p95={r.p95:7.2f}ms  "
        f"peak={r.peak_mem_mb:6.1f}MB"
    )


def _load_encoder(impl: str, model_path: str, device: str, dtype: torch.dtype):
    if impl == "hf":
        from sglang_omni.models.qwen3_omni.components.audio_encoder import (
            Qwen3OmniAudioEncoder,
        )

        return Qwen3OmniAudioEncoder(model_path, device=device, dtype=dtype)
    elif impl == "sglang":
        from sglang_omni.models.qwen3_omni.components.audio_encoder_native import (
            Qwen3OmniAudioEncoderNative,
        )

        return Qwen3OmniAudioEncoderNative(model_path, device=device, dtype=dtype)
    raise ValueError(f"unknown impl: {impl}")


def run_eager_mode(args) -> list[BenchResult]:
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    batches = [int(b) for b in args.batch_sizes.split(",")]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    results: list[BenchResult] = []
    for impl in ("hf", "sglang"):
        if impl in skip:
            continue
        logger.info(f"=== loading {impl} ===")
        enc = _load_encoder(impl, args.model_path, str(device), dtype)
        for B in batches:
            for L in seq_lens:
                r = bench_eager(
                    enc,
                    impl_name=impl,
                    batch=B,
                    seq_len=L,
                    mel_bins=args.mel_bins,
                    device=device,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                results.append(r)
                _print_result(r)
        del enc
        gc.collect()
        torch.cuda.empty_cache()
    return results


def run_cuda_graph_mode(args) -> list[BenchResult]:
    """Run both eager and cuda_graph for each (impl, B, L) so we can compare."""
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    batches = [int(b) for b in args.batch_sizes.split(",")]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    results: list[BenchResult] = []
    for impl in ("hf", "sglang"):
        if impl in skip:
            continue
        logger.info(f"=== loading {impl} ===")
        enc = _load_encoder(impl, args.model_path, str(device), dtype)
        for B in batches:
            for L in seq_lens:
                r_eager = bench_eager(
                    enc,
                    impl_name=impl,
                    batch=B,
                    seq_len=L,
                    mel_bins=args.mel_bins,
                    device=device,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                results.append(r_eager)
                _print_result(r_eager)

                r_graph = bench_cuda_graph(
                    enc,
                    impl_name=impl,
                    batch=B,
                    seq_len=L,
                    mel_bins=args.mel_bins,
                    device=device,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                if r_graph is not None:
                    results.append(r_graph)
                    _print_result(r_graph)
        del enc
        gc.collect()
        torch.cuda.empty_cache()
    return results


def run_graphed_mode(args) -> list[BenchResult]:
    """Phase 1: wrap sglang native encoder in GraphedAudioEncoder + CUDA graph.

    For each (B, L): (a) eager through GraphedAudioEncoder, (b) graph captured replay.
    """
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    batches = [int(b) for b in args.batch_sizes.split(",")]

    from sglang_omni.models.qwen3_omni.components.audio_encoder_native import (
        Qwen3OmniAudioEncoderNative,
    )
    from sglang_omni.models.qwen3_omni.components.audio_encoder_graphed import (
        GraphedAudioEncoder,
    )

    logger.info("=== loading sglang native (for graphed wrapper) ===")
    native = Qwen3OmniAudioEncoderNative(
        args.model_path, device=str(device), dtype=dtype
    )

    # Optional HF baseline for parity. Loaded once; reused for every (B, L).
    parity_hf = None
    if not args.no_parity:
        logger.info("=== loading HF baseline (for graphed-vs-hf parity) ===")
        from sglang_omni.models.qwen3_omni.components.audio_encoder import (
            Qwen3OmniAudioEncoder,
        )

        parity_hf = Qwen3OmniAudioEncoder(
            args.model_path, device=str(device), dtype=dtype
        )

    results: list[BenchResult] = []
    for B in batches:
        for L in seq_lens:
            logger.info(f"=== graphed B={B} L={L} ===")
            graphed = GraphedAudioEncoder(
                native.audio_tower,
                batch=B,
                seq_len=L,
                device=device,
            )
            r_eager = bench_eager(
                graphed,
                impl_name="graphed-eager",
                batch=B,
                seq_len=L,
                mel_bins=args.mel_bins,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(r_eager)
            _print_result(r_eager)

            r_graph = bench_cuda_graph(
                graphed,
                impl_name="graphed",
                batch=B,
                seq_len=L,
                mel_bins=args.mel_bins,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            if r_graph is not None:
                results.append(r_graph)
                _print_result(r_graph)

            # Parity check: graphed-eager and graphed-cuda_graph must agree
            # with the HF baseline within bf16 tolerance. Without this, a
            # silent miscompile in the refactor could show up as "5× speedup"
            # on meaningless output.
            if parity_hf is not None:
                _run_parity_check(
                    parity_hf,
                    r_eager,
                    r_graph,
                    batch=B,
                    seq_len=L,
                    mel_bins=args.mel_bins,
                    device=device,
                    dtype=dtype,
                )

            del graphed
            gc.collect()
            torch.cuda.empty_cache()
    return results


def run_tp_mode(args) -> list[BenchResult]:
    """TP=N sglang-only via torchrun. Returns bench results from rank 0."""
    # When launched via torchrun, RANK/WORLD_SIZE/LOCAL_RANK are set.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != args.tp_size:
        raise RuntimeError(
            f"tp_size={args.tp_size} but WORLD_SIZE={world_size}. "
            f"Launch with: torchrun --nproc_per_node={args.tp_size} ..."
        )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = _resolve_dtype(args.dtype)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    batches = [int(b) for b in args.batch_sizes.split(",")]

    # Load sglang native under TP. audio_encoder_native handles dist init
    # using env vars set by torchrun (WORLD_SIZE/RANK/LOCAL_RANK).
    from sglang_omni.models.qwen3_omni.components.audio_encoder_native import (
        Qwen3OmniAudioEncoderNative,
    )

    enc = Qwen3OmniAudioEncoderNative(
        args.model_path, device=str(device), dtype=dtype
    )

    results: list[BenchResult] = []
    for B in batches:
        for L in seq_lens:
            r = bench_eager(
                enc,
                impl_name=f"sglang-tp{args.tp_size}",
                batch=B,
                seq_len=L,
                mel_bins=args.mel_bins,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(r)
            if rank == 0:
                _print_result(r)
    return results if rank == 0 else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype", default="bfloat16", choices=sorted(_DTYPE_MAP.keys())
    )
    p.add_argument(
        "--mode",
        choices=["eager", "cuda_graph", "tp", "graphed"],
        default="eager",
    )
    p.add_argument("--seq-lens", default="300,1000,3000")
    p.add_argument("--batch-sizes", default="1,4,16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--mel-bins", type=int, default=128)
    p.add_argument("--skip", default="", help="Comma-sep impls to skip: hf|sglang")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip loading HF baseline for graphed-mode parity check (saves ~1GB GPU).",
    )
    args = p.parse_args()

    # The bench wraps every iter in torch.cuda.synchronize / max_memory_allocated
    # / CUDAGraph and has no meaningful non-CUDA path. Fail fast with a clear
    # message rather than letting the first .synchronize() blow up mid-bench.
    if args.mode != "tp":  # tp derives device from LOCAL_RANK internally
        if torch.device(args.device).type != "cuda":
            p.error(
                f"--device={args.device!r}: this bench requires a CUDA device "
                f"(the harness uses torch.cuda.synchronize / CUDAGraph)."
            )
        if not torch.cuda.is_available():
            p.error("CUDA is not available on this host.")

    if args.mode == "eager":
        results = run_eager_mode(args)
    elif args.mode == "cuda_graph":
        results = run_cuda_graph_mode(args)
    elif args.mode == "tp":
        results = run_tp_mode(args)
    elif args.mode == "graphed":
        results = run_graphed_mode(args)

    if not results:
        return

    # Summary table
    logger.info("\n=== Summary ===")
    logger.info(
        f"{'impl':>14} {'mode':>10} {'B':>3} {'L':>5}  {'p50 ms':>8}  "
        f"{'p95 ms':>8}  {'peak MB':>8}"
    )
    for r in results:
        logger.info(
            f"{r.impl:>14} {r.mode:>10} {r.batch:>3} {r.seq_len:>5}  "
            f"{r.p50:>8.2f}  {r.p95:>8.2f}  {r.peak_mem_mb:>8.1f}"
        )

    # Speedup + numerical parity (pair eager HF vs eager sglang if both present)
    by_key: dict[tuple[int, int, str], dict[str, BenchResult]] = {}
    for r in results:
        by_key.setdefault((r.batch, r.seq_len, r.mode), {})[r.impl] = r
    logger.info("\n=== Speedup (sglang vs hf) ===")
    logger.info(
        f"{'mode':>10} {'B':>3} {'L':>5}  {'hf p50':>8}  {'sg p50':>8}  "
        f"{'x':>6}  {'max_diff':>10}  {'mean_diff':>10}"
    )
    for (B, L, mode), pair in sorted(by_key.items()):
        if "hf" in pair and "sglang" in pair:
            hf_ms = pair["hf"].p50
            sg_ms = pair["sglang"].p50
            speedup = hf_ms / sg_ms if sg_ms > 0 else float("inf")
            if (
                pair["hf"].output_sample is not None
                and pair["sglang"].output_sample is not None
            ):
                mx, mn = numerical_diff(
                    pair["hf"].output_sample, pair["sglang"].output_sample
                )
            else:
                mx = mn = float("nan")
            logger.info(
                f"{mode:>10} {B:>3} {L:>5}  {hf_ms:>8.2f}  {sg_ms:>8.2f}  "
                f"{speedup:>6.2f}x  {mx:>10.4e}  {mn:>10.4e}"
            )

    # Eager vs cuda_graph speedup per impl
    by_impl: dict[tuple[str, int, int], dict[str, BenchResult]] = {}
    for r in results:
        by_impl.setdefault((r.impl, r.batch, r.seq_len), {})[r.mode] = r
    has_graph = any(r.mode == "cuda_graph" for r in results)
    if has_graph:
        logger.info("\n=== Eager vs CUDA graph (same impl) ===")
        logger.info(
            f"{'impl':>14} {'B':>3} {'L':>5}  {'eager p50':>10}  "
            f"{'graph p50':>10}  {'x':>6}"
        )
        for (impl, B, L), pair in sorted(by_impl.items()):
            if "eager" in pair and "cuda_graph" in pair:
                e_ms = pair["eager"].p50
                g_ms = pair["cuda_graph"].p50
                speedup = e_ms / g_ms if g_ms > 0 else float("inf")
                logger.info(
                    f"{impl:>14} {B:>3} {L:>5}  {e_ms:>10.2f}  "
                    f"{g_ms:>10.2f}  {speedup:>6.2f}x"
                )


if __name__ == "__main__":
    main()
