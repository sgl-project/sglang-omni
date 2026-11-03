# SPDX-License-Identifier: Apache-2.0
"""Sweep a speculative draft against no draft across concurrency.

Each leg starts a fresh server, with or without the draft config, runs one ASR benchmark client over every concurrency level, and stops the server. Legs run one after another in the order given, ABBA by default, so drift between runs shows as a gap between the two legs of one side. The client is any benchmark built on the benchmark_asr_seedtts sweep: benchmark_asr_seedtts, benchmark_asr_longform or benchmark_asr_stt_benchmark. Arguments after -- go to the client.

The summary gives, per concurrency, both sides' throughput, the gain, p95 latency and its change, WER and failed requests; then k, the loads where the draft is worse than no draft, and per leg the startup time, KV pool, peak token usage, retractions, acceptance length and peak GPU memory. sweep.json holds the same comparison.

Set CUDA_VISIBLE_DEVICES to one GPU that nothing else uses: after each leg the managed server ends every process still using it. Engine settings other than the draft belong in the configs, so both legs get them; --baseline-config takes the no-draft one.

The client sends the same prompts in every pass, so with the prefix cache on, almost every prompt token after the first pass comes from the cache. To measure prompts that do not repeat, set engine.disable_radix_cache: true in both configs. The summary shows each leg's cached share.

Usage:

    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.speculative_sweep \\
        --model-path Qwen/Qwen3-ASR-1.7B \\
        --draft-config examples/configs/qwen3_asr_1_7b_dflash.yaml \\
        --output-dir results/sweep/qwen3_asr \\
        -- --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.metrics.speculative import (
    K_GAIN,
    ClientPayload,
    Leg,
    Side,
    compare,
    read_server_log,
    render,
)

logger = logging.getLogger(__name__)

SIDE_BY_LETTER: dict[str, Side] = {"A": "baseline", "B": "draft"}
HOST = "127.0.0.1"
DRIVER_CLIENT_FLAGS = ("--host", "--port", "--model-path", "--output")


def run_leg(*, name: str, side: Side, port: int, args: argparse.Namespace) -> Leg:
    leg_dir = args.output_dir / name
    leg_dir.mkdir()
    server_log = leg_dir / "server.log"
    client_output = leg_dir / "client.json"
    started = time.monotonic()
    with managed_omni_server(
        model_path=args.model_path,
        port=port,
        host=HOST,
        log_file=server_log,
        server_config=args.draft_config if side == "draft" else args.baseline_config,
    ):
        startup_s = time.monotonic() - started
        logger.info(f"{name}: server ready in {startup_s:.1f} s")
        with open(leg_dir / "client.log", "w") as client_log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    args.client,
                    "--host",
                    HOST,
                    "--port",
                    str(port),
                    "--model-path",
                    args.model_path,
                    "--output",
                    str(client_output),
                    *args.client_args,
                ],
                stdout=client_log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    return Leg(
        name=name,
        side=side,
        startup_s=startup_s,
        server=read_server_log(server_log.read_text(errors="replace")),
        results=ClientPayload.model_validate_json(client_output.read_text()).results,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--draft-config", required=True, help="Server config that adds the draft."
    )
    parser.add_argument(
        "--baseline-config",
        default=None,
        help="Server config without the draft; the model's defaults when omitted.",
    )
    parser.add_argument(
        "--client",
        default="benchmarks.eval.benchmark_asr_seedtts",
        help="Client module built on the benchmark_asr_seedtts sweep.",
    )
    parser.add_argument(
        "--order", default="ABBA", help="A runs without the draft, B with it."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8700,
        help="The first leg's port; each later leg takes the next.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    if "--" in argv:
        separator = argv.index("--")
        args = parser.parse_args(argv[:separator])
        args.client_args = argv[separator + 1 :]
    else:
        args = parser.parse_args(argv)
        args.client_args = []
    visible_gpus = [
        gpu
        for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if gpu.strip()
    ]
    if len(visible_gpus) != 1:
        parser.error("set CUDA_VISIBLE_DEVICES to the one GPU the sweep owns")
    elif set(args.order) != set(SIDE_BY_LETTER):
        parser.error("--order needs at least one A and one B, and no other letters")
    elif any(
        argument.split("=")[0] in DRIVER_CLIENT_FLAGS for argument in args.client_args
    ):
        parser.error(
            f"the sweep sets {', '.join(DRIVER_CLIENT_FLAGS)} for the client itself"
        )
    else:
        pass
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = parse_args(sys.argv[1:])
    args.output_dir.mkdir(parents=True)
    legs = []
    for index, letter in enumerate(args.order):
        side = SIDE_BY_LETTER[letter]
        legs.append(
            run_leg(
                name=f"{index + 1}_{side}", side=side, port=args.port + index, args=args
            )
        )
    comparison = compare(legs, k_gain=K_GAIN)
    summary = render(comparison, legs, title=Path(args.draft_config).stem)
    (args.output_dir / "summary.md").write_text(summary)
    sweep = {
        "arguments": vars(args),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "legs": [
            {"name": leg.name, "startup_s": leg.startup_s, **asdict(leg.server)}
            for leg in legs
        ],
        "comparison": asdict(comparison),
    }
    (args.output_dir / "sweep.json").write_text(
        json.dumps(sweep, indent=2, default=str)
    )
    print(summary)


if __name__ == "__main__":
    main()
