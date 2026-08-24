# SPDX-License-Identifier: Apache-2.0
"""ASR concurrency benchmark on the Pipecat STT benchmark set.

This script transcribes ``pipecat-ai/stt-benchmark-data`` (1000 English
utterances, 1 to 16 s, 16 kHz WAV, punctuated transcripts) through a running
ASR router and reports WER, request throughput, RTFx, RTF, latency, and worker
routing balance. It reuses the sweep loop, aggregation, table, and common
arguments of ``benchmark_asr_seedtts`` so the two result JSONs line up column
for column; the difference is the workload: longer, conversational voice-agent
turns instead of short SeedTTS prompts.

Usage:

    # Download the test set once:
    python -m benchmarks.dataset.prepare --dataset stt-benchmark

    # Launch Qwen3-ASR:
    sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000

    # Sweep the full set (3 repeats each):
    python -m benchmarks.eval.benchmark_asr_stt_benchmark \\
        --port 8000 \\
        --concurrencies 1,2,4,8,16,32,64 \\
        --repeats 3 --warmup

    # Quick local smoke on a 20-sample subset:
    python -m benchmarks.eval.benchmark_asr_stt_benchmark \\
        --port 8000 --max-samples 20 --concurrencies 2,32 --repeats 1

    # Run the same sweep against Fun-ASR-Nano with SSE streaming:
    python -m benchmarks.eval.benchmark_asr_stt_benchmark \\
        --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \\
        --concurrencies 32 --repeats 3 --warmup --stream

Pipecat's own benchmark reports Semantic WER and TTFS for this dataset; this
script does not reproduce those metrics, so its numbers are not comparable to
the Pipecat leaderboard.

The dataset is English only, so WER uses the Whisper English normalizer and
the request language is fixed to ``en``. Use ``--repo-id`` to point at a
re-upload with the same columns; the canonical repo loads at its pinned
revision unless you pass ``--dataset-revision``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from benchmarks.dataset.prepare import (
    STT_BENCHMARK_DATASET_ID,
    STT_BENCHMARK_DATASET_REVISION,
)
from benchmarks.dataset.stt_benchmark import (
    STT_BENCHMARK_LANG,
    STT_BENCHMARK_SPLIT,
    load_stt_benchmark_samples,
)
from benchmarks.eval.asr_profiling import (
    collect_environment_fingerprint,
    collect_server_identity,
)
from benchmarks.eval.benchmark_asr_seedtts import (
    _evaluation_input_sha256,
    _print_table,
    _sweep,
    add_common_args,
    finalize_args,
)
from benchmarks.runtime_metrics import collect_benchmark_provenance

RESULTS_FILE = "asr_stt_benchmark_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=STT_BENCHMARK_DATASET_ID,
        help=(
            "HF dataset repo with sample_id/audio/transcription columns "
            "and PCM WAV audio."
        ),
    )
    parser.add_argument(
        "--split",
        default=STT_BENCHMARK_SPLIT,
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0 = full set; 1000 for the canonical repo).",
    )
    add_common_args(parser, default_output=RESULTS_FILE)
    args = finalize_args(parser.parse_args())
    args.lang = STT_BENCHMARK_LANG
    return args


def main() -> None:
    args = parse_args()
    concurrencies = args.concurrencies
    max_samples = args.max_samples if args.max_samples > 0 else None
    if args.dataset_revision is not None:
        dataset_revision = args.dataset_revision
    elif args.repo_id == STT_BENCHMARK_DATASET_ID:
        dataset_revision = STT_BENCHMARK_DATASET_REVISION
    else:
        dataset_revision = None

    samples = load_stt_benchmark_samples(
        args.repo_id,
        max_samples=max_samples,
        split=args.split,
        revision=dataset_revision,
    )
    if not samples:
        raise RuntimeError(f"No STT benchmark samples loaded from {args.repo_id!r}")
    evaluation_input_sha256 = _evaluation_input_sha256(
        samples, namespace="stt-benchmark"
    )
    print(
        f"Loaded {len(samples)} STT benchmark samples ({args.repo_id}, "
        f"{args.split}); sweeping concurrency={concurrencies} x {args.repeats} "
        f"repeats against {args.host}:{args.port} ({args.model_path})"
    )

    aggregates = asyncio.run(_sweep(args, samples, concurrencies))
    _print_table(aggregates)

    server_config = {
        "dtype": args.dtype,
        "quantization": args.quantization,
        "attention_backend": args.attention_backend,
        "mm_attention_backend": args.mm_attention_backend,
        "cuda_graph": args.cuda_graph,
        "torch_compile": args.torch_compile,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
    }
    payload = {
        "schema_version": 2,
        "provenance": collect_benchmark_provenance(
            model_id=args.model_path,
            model_revision=args.model_revision,
            dataset_id=args.repo_id,
            dataset_revision=dataset_revision,
            launch_command=args.launch_command,
            server_config=server_config,
            evaluation_input_sha256=evaluation_input_sha256,
        ),
        "config": {
            "host": args.host,
            "port": args.port,
            "repo_id": args.repo_id,
            "split": args.split,
            "lang": args.lang,
            "model_path": args.model_path,
            "declared_model_revision": args.model_revision,
            "dataset_revision": dataset_revision,
            "num_samples": len(samples),
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "stream": args.stream,
            "declared_server": server_config,
            "resource_monitor": {
                "enabled": not args.disable_resource_monitor,
                "gpu_index": args.gpu_index,
                "interval_s": args.monitor_interval_s,
                "gpu_process_pids": args.gpu_process_pids or [],
            },
            "profile_events": args.profile_events,
            "sample_util": args.sample_util,
        },
        "results": aggregates,
    }
    if args.fingerprint:
        payload["environment_fingerprint"] = {
            "client": collect_environment_fingerprint(args.model_path),
            "server": collect_server_identity(f"http://{args.host}:{args.port}"),
        }
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
