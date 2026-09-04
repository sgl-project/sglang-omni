# SPDX-License-Identifier: Apache-2.0
# Author:
# Yuhao Chen: https://github.com/AkazaAkane
"""ASR concurrency benchmark on registered long-form English datasets.

The registered workloads are the 30 s and 60 s LongLibriHeavy test splits and
the complete Meanwhile test split. Audio is normalized to mono 16 kHz PCM WAV
before the timed sweep, then sent through the same request, WER, performance,
resource-monitoring, and reporting path as benchmark_asr_seedtts.

Usage:

    python -m benchmarks.dataset.prepare --dataset longlibriheavy-30
    python -m benchmarks.eval.benchmark_asr_longform \
        --dataset longlibriheavy-30 --port 8000 \
        --concurrencies 1,8,32 --repeats 3 --warmup

    python -m benchmarks.dataset.prepare --dataset longlibriheavy-60
    python -m benchmarks.eval.benchmark_asr_longform \
        --dataset longlibriheavy-60 --port 8000 \
        --concurrencies 1,8,32 --repeats 3 --warmup

    python -m benchmarks.dataset.prepare --dataset meanwhile
    python -m benchmarks.eval.benchmark_asr_longform \
        --dataset meanwhile --port 8000 \
        --concurrencies 1,8,32 --repeats 3 --warmup

--stream still uploads each complete file; it does not simulate real-time
audio arrival. The dataset is English only, so WER uses the Whisper English
normalizer and the request language is fixed to en.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from benchmarks.dataset.asr_longform import (
    ASR_LONGFORM_DATASETS,
    ASR_LONGFORM_LANG,
    load_asr_longform_samples,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(ASR_LONGFORM_DATASETS),
        default="longlibriheavy-30",
        help="Registered long-form ASR dataset to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0 = the complete registered split).",
    )
    add_common_args(parser, default_output="")
    args = finalize_args(parser.parse_args())
    args.lang = ASR_LONGFORM_LANG
    if not args.output:
        args.output = f"asr_{args.dataset.replace('-', '_')}_results.json"
    return args


def main() -> None:
    args = parse_args()
    config = ASR_LONGFORM_DATASETS[args.dataset]
    dataset_revision = args.dataset_revision or config.revision
    max_samples = args.max_samples if args.max_samples > 0 else None

    samples = load_asr_longform_samples(
        args.dataset,
        max_samples=max_samples,
        revision=dataset_revision,
    )
    if not samples:
        raise RuntimeError(f"No long-form ASR samples loaded for {args.dataset!r}")
    evaluation_input_sha256 = _evaluation_input_sha256(
        samples,
        namespace=f"asr-longform-{args.dataset}",
    )
    print(
        f"Loaded {len(samples)} {args.dataset} samples "
        f"({config.repo_id}, {config.split}); sweeping "
        f"concurrency={args.concurrencies} x {args.repeats} repeats against "
        f"{args.host}:{args.port} ({args.model_path})"
    )

    aggregates = asyncio.run(_sweep(args, samples, args.concurrencies))
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
            dataset_id=config.repo_id,
            dataset_revision=dataset_revision,
            launch_command=args.launch_command,
            server_config=server_config,
            evaluation_input_sha256=evaluation_input_sha256,
        ),
        "config": {
            "host": args.host,
            "port": args.port,
            "dataset": args.dataset,
            "repo_id": config.repo_id,
            "split": config.split,
            "lang": args.lang,
            "model_path": args.model_path,
            "declared_model_revision": args.model_revision,
            "dataset_revision": dataset_revision,
            "num_samples": len(samples),
            "expected_num_samples": config.expected_sample_count,
            "concurrencies": args.concurrencies,
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
