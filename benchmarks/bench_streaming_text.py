#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark Ming-Omni text streaming before/after fix for #600.

Measures the same metrics as the issue benchmark table:
  TTFT (mean/p95), TPOT (mean/p95), latency (mean), gen_toks (mean)

Usage:
    # Run against a single server:
    python benchmarks/bench_streaming_text.py --base-url http://localhost:30000

    # Compare two servers (main vs PR branch):
    python benchmarks/bench_streaming_text.py \
        --base-url http://localhost:30000 \
        --compare-url http://localhost:30001 \
        --label-a "main (before)" \
        --label-b "this PR (after)"
"""

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Optional

try:
    import openai
except ImportError:
    raise SystemExit("pip install openai")

PROMPTS = [
    "Explain the transformer attention mechanism in detail.",
    "What is PagedAttention and how does it reduce memory fragmentation?",
    "Compare FlashAttention and standard attention in terms of IO complexity.",
    "What is continuous batching and why is it better than static batching?",
    "Explain the difference between prefill and decode phases in LLM inference.",
    "What is speculative decoding and how does it speed up inference?",
    "Describe the KV cache and why it is important for autoregressive generation.",
    "What is tensor parallelism and how is it used in large model inference?",
]

MODEL = "ming-omni"


@dataclass
class StreamResult:
    ttft_ms: float = -1.0      # time to first non-empty delta.content (ms)
    e2e_ms: float = 0.0        # total time to [DONE] (ms)
    tpot_ms: float = -1.0      # (e2e - ttft) / (gen_toks - 1) (ms/tok)
    gen_toks: int = 0          # number of non-empty delta.content chunks
    has_duplication: bool = False
    error: Optional[str] = None


def bench_one(client: openai.OpenAI, prompt: str, model: str) -> StreamResult:
    r = StreamResult()
    chunks: list[str] = []
    t_start = time.perf_counter()
    t_first: Optional[float] = None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body={"modalities": ["text"]},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            content = getattr(delta, "content", None) if delta else None
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                chunks.append(content)

        t_end = time.perf_counter()
    except Exception as e:
        r.error = str(e)
        return r

    r.e2e_ms = (t_end - t_start) * 1000
    r.ttft_ms = (t_first - t_start) * 1000 if t_first else -1
    r.gen_toks = len(chunks)

    # TPOT = (e2e - ttft) / (gen_toks - 1)
    if r.gen_toks > 1 and r.ttft_ms > 0:
        r.tpot_ms = (r.e2e_ms - r.ttft_ms) / (r.gen_toks - 1)

    # Check text duplication: streamed text should not appear twice
    full_text = "".join(chunks)
    if len(full_text) > 10:
        half = len(full_text) // 2
        if full_text[:half] in full_text[half:]:
            r.has_duplication = True

    return r


def p95(values: list[float]) -> float:
    if not values:
        return -1.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.95)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def run_bench(base_url: str, label: str, model: str, n_warmup: int = 1) -> list[StreamResult]:
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")

    print(f"\n{'='*70}")
    print(f"  {label}  ({base_url})")
    print(f"{'='*70}")

    if n_warmup > 0:
        print(f"Warmup ({n_warmup} req)...")
        for i in range(n_warmup):
            bench_one(client, PROMPTS[i % len(PROMPTS)], model)
        print("Done.\n")

    results = []
    for i, prompt in enumerate(PROMPTS):
        r = bench_one(client, prompt, model)
        results.append(r)
        if r.error:
            print(f"  [{i+1}] ERROR: {r.error}")
        else:
            tpot_str = f"{r.tpot_ms:.1f} ms/tok" if r.tpot_ms > 0 else "n/a*"
            print(
                f"  [{i+1}] TTFT={r.ttft_ms:7.1f}ms  "
                f"E2E={r.e2e_ms:7.1f}ms  "
                f"TPOT={tpot_str:>12}  "
                f"chunks={r.gen_toks:4d}  "
                f"dup={'YES❌' if r.has_duplication else 'no✓'}"
            )
    return results


def print_table(label_a: str, results_a: list[StreamResult],
                label_b: Optional[str] = None,
                results_b: Optional[list[StreamResult]] = None) -> None:

    def stats(results: list[StreamResult]) -> dict:
        ok = [r for r in results if not r.error]
        if not ok:
            return {}
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
        e2es  = [r.e2e_ms for r in ok]
        toks  = [r.gen_toks for r in ok]
        return {
            "ttft_mean": statistics.mean(ttfts) if ttfts else -1,
            "ttft_p95":  p95(ttfts) if ttfts else -1,
            "tpot_mean": statistics.mean(tpots) if tpots else -1,
            "tpot_p95":  p95(tpots) if tpots else -1,
            "e2e_mean":  statistics.mean(e2es),
            "tok_mean":  statistics.mean(toks),
            "any_dup":   any(r.has_duplication for r in ok),
        }

    def fmt_ms(v: float) -> str:
        return f"{v:.3f} s" if v > 0 else "n/a*"

    def fmt_tpot(v: float) -> str:
        return f"{v:.1f} ms/tok" if v > 0 else "n/a*"

    sa = stats(results_a)
    sb = stats(results_b) if results_b else None

    rows = [label_a] + ([label_b] if label_b else [])
    stats_list = [sa] + ([sb] if sb else [])

    col_w = 28
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")

    header = f"{'backend':<{col_w}} {'TTFT(mean)':>12} {'TTFT(p95)':>10} {'TPOT(mean)':>14} {'TPOT(p95)':>12} {'latency(mean)':>14} {'gen_toks':>9}"
    print(header)
    print("-" * len(header))

    for label, s in zip(rows, stats_list):
        if not s:
            print(f"  {label}: no data")
            continue
        print(
            f"{label:<{col_w}} "
            f"{fmt_ms(s['ttft_mean']):>12} "
            f"{fmt_ms(s['ttft_p95']):>10} "
            f"{fmt_tpot(s['tpot_mean']):>14} "
            f"{fmt_tpot(s['tpot_p95']):>12} "
            f"{fmt_ms(s['e2e_mean']):>14} "
            f"{s['tok_mean']:>9.0f}"
        )
        if s.get("any_dup"):
            print(f"  ⚠️  {label}: text duplication detected!")

    print()
    if sb and sa.get("ttft_mean", -1) > 0 and sb.get("ttft_mean", -1) > 0:
        ratio = sa["ttft_mean"] / sb["ttft_mean"]
        print(f"  TTFT improvement ({label_b} vs {label_a}): {ratio:.1f}x faster")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--compare-url", default=None)
    parser.add_argument("--label-a", default="main (before fix)")
    parser.add_argument("--label-b", default="this PR (after fix)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    results_a = run_bench(args.base_url, args.label_a, args.model, args.warmup)
    results_b = None
    if args.compare_url:
        results_b = run_bench(args.compare_url, args.label_b, args.model, args.warmup)

    print_table(args.label_a, results_a, args.label_b, results_b)


if __name__ == "__main__":
    main()
