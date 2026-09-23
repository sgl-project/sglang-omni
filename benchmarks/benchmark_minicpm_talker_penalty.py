# SPDX-License-Identifier: Apache-2.0
"""Compare MiniCPM-o window penalties without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest

BASELINE_REVISION = "12f7b6670cc236b63cca4f53a9a495b9aea93cf5"
PENALTY_WINDOW = 16
BENCHMARK_SEED = 2284


@dataclass(kw_only=True)
class CodecHistory:
    output_ids: list[int]


def dense_window_penalty(
    logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
) -> None:
    """Reproduce the dense baseline from the branch's main ancestor."""
    logits = logits_output.next_token_logits
    assert logits is not None and logits.ndim == 2
    vocabulary_size = logits.shape[1]
    row_indices: list[int] = []
    penalties: list[float] = []
    windows: list[list[int]] = []
    for row_index, request in enumerate(requests):
        penalty = float(request.data.talker_model_inputs.get("rep_penalty", 1.0))
        if penalty != 1.0:
            window = [
                token_id
                for token_id in map(int, request.data.req.output_ids[-PENALTY_WINDOW:])
                if 0 <= token_id < vocabulary_size
            ]
            if window:
                row_indices.append(row_index)
                penalties.append(penalty)
                windows.append(window)
    if windows:
        window_ids = torch.full(
            (len(windows), PENALTY_WINDOW), vocabulary_size, dtype=torch.long
        )
        for row_index, window in enumerate(windows):
            window_ids[row_index, : len(window)] = torch.tensor(
                window, dtype=torch.long
            )
        window_ids = window_ids.to(logits.device)
        counts = torch.zeros(
            len(windows), vocabulary_size + 1, dtype=torch.float32, device=logits.device
        )
        counts.scatter_add_(
            1, window_ids, torch.ones_like(window_ids, dtype=torch.float32)
        )
        counts = counts[:, :vocabulary_size]
        scaling_factors = (
            torch.tensor(penalties, dtype=torch.float32, device=logits.device)
            .unsqueeze(1)
            .pow(counts)
        )
        selected_rows = torch.tensor(
            row_indices, dtype=torch.long, device=logits.device
        )
        scores = logits[selected_rows].float()
        penalized = torch.where(
            scores < 0, scores * scaling_factors, scores / scaling_factors
        )
        logits[selected_rows] = torch.where(counts > 0, penalized, scores).to(
            logits.dtype
        )


def run_penalty(
    method: Literal["dense", "sparse"],
    runner: MiniCPMOTalkerModelRunner,
    output: LogitsProcessorOutput,
    requests: list[SchedulerRequest],
) -> None:
    if method == "dense":
        dense_window_penalty(output, requests)
    else:
        runner.process_sampling_logits(output, requests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Checkpoint tts_config.num_audio_tokens",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--steps-per-sample",
        type=int,
        default=1,
        help="One measures isolated latency; larger values queue reset+penalty steps.",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        arguments.vocab_size < 1
        or min(arguments.batch_sizes) < 1
        or arguments.samples < 20
        or arguments.warmup < 1
        or arguments.steps_per_sample < 1
    ):
        parser.error(
            "Require positive vocab-size/batches/warmup/steps, and samples >= 20"
        )
    if arguments.output.exists():
        parser.error("Output already exists; choose a new filename")
    if not arguments.output.parent.is_dir():
        parser.error("Output parent directory must exist")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(BENCHMARK_SEED)
    generator = random.Random(BENCHMARK_SEED)
    device = torch.device(arguments.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[arguments.dtype]
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    cases: list[dict[str, str | int | float | list[float]]] = []
    with torch.inference_mode():
        for batch_size in arguments.batch_sizes:
            for history_kind in ("empty", "repeated", "distinct", "mixed", "disabled"):
                requests: list[SchedulerRequest] = []
                for row_index in range(batch_size):
                    if history_kind == "empty":
                        history = []
                    elif history_kind == "repeated":
                        history = [row_index % arguments.vocab_size] * PENALTY_WINDOW
                    elif history_kind == "distinct":
                        history = [
                            (row_index + index) % arguments.vocab_size
                            for index in range(PENALTY_WINDOW)
                        ]
                    else:
                        history = [
                            generator.randrange(arguments.vocab_size)
                            for _ in range(row_index % 33)
                        ]
                    penalty = (
                        1.0
                        if history_kind == "disabled"
                        else [1.05, 1.1, 0.8][row_index % 3]
                    )
                    requests.append(
                        SchedulerRequest(
                            request_id=str(row_index),
                            data=SGLangARRequestData(
                                req=CodecHistory(output_ids=history),
                                talker_model_inputs={"rep_penalty": penalty},
                            ),
                        )
                    )
                original = torch.randn(
                    batch_size, arguments.vocab_size, dtype=dtype, device=device
                )
                baseline_output = LogitsProcessorOutput(
                    next_token_logits=original.clone()
                )
                candidate_output = LogitsProcessorOutput(
                    next_token_logits=original.clone()
                )
                dense_window_penalty(baseline_output, requests)
                runner.process_sampling_logits(candidate_output, requests)
                torch.testing.assert_close(
                    candidate_output.next_token_logits,
                    baseline_output.next_token_logits,
                    rtol=0,
                    atol=0,
                )
                output = LogitsProcessorOutput(next_token_logits=original.clone())
                samples: dict[str, list[float]] = {"dense": [], "sparse": []}
                peak_bytes: dict[str, int] = {}
                methods: list[Literal["dense", "sparse"]] = ["dense", "sparse"]
                for method in methods:
                    for _ in range(arguments.warmup):
                        output.next_token_logits.copy_(original)
                        run_penalty(method, runner, output, requests)
                for _ in range(arguments.samples):
                    order = methods.copy()
                    generator.shuffle(order)
                    for method in order:
                        output.next_token_logits.copy_(original)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        start = time.perf_counter_ns()
                        for step_index in range(arguments.steps_per_sample):
                            if step_index > 0:
                                output.next_token_logits.copy_(original)
                            run_penalty(method, runner, output, requests)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        samples[method].append(
                            (time.perf_counter_ns() - start)
                            / (1000 * arguments.steps_per_sample)
                        )
                if device.type == "cuda":
                    for method in methods:
                        output.next_token_logits.copy_(original)
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        allocated_bytes = torch.cuda.memory_allocated(device)
                        run_penalty(method, runner, output, requests)
                        torch.cuda.synchronize(device)
                        peak_bytes[method] = (
                            torch.cuda.max_memory_allocated(device) - allocated_bytes
                        )
                for method, timings in samples.items():
                    cases.append(
                        {
                            "batch_size": batch_size,
                            "history": history_kind,
                            "method": method,
                            "median_us_per_step": statistics.median(timings),
                            "p95_us_per_step": statistics.quantiles(
                                timings, n=100, method="inclusive"
                            )[94],
                            "peak_extra_allocated_bytes": peak_bytes.get(method, -1),
                            "wall_us_per_step": timings,
                        }
                    )
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    runner_path = repository / "sglang_omni/models/minicpm_o/talker_model_runner.py"
    result = {
        "revision": revision,
        "baseline_revision": BASELINE_REVISION,
        "runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": arguments.dtype,
        "vocab_size": arguments.vocab_size,
        "seed": BENCHMARK_SEED,
        "samples": arguments.samples,
        "warmup": arguments.warmup,
        "steps_per_sample": arguments.steps_per_sample,
        "measurement": (
            "Synchronized wall time per penalty call including Python packing and "
            "host-to-device copies. The first reset is excluded; subsequent "
            "queued resets are included. Not GPU kernel time or serving latency."
        ),
        "correctness": "Exact logits comparison passed for every generated case on this device.",
        "cases": cases,
    }
    with arguments.output.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
    print(f"Saved {len(cases)} measurements to {arguments.output}")


if __name__ == "__main__":
    main()
