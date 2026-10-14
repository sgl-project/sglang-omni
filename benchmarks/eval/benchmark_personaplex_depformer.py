# SPDX-License-Identifier: Apache-2.0
"""Measure complete eager and graphed Depformer frames with checkpoint weights."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

from benchmarks.eval.profile_personaplex_depformer import measure_frames, profile_frames
from sglang_omni.models.personaplex.architecture import DEPFORMER, MOSHI_WEIGHTS_NAME
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.components.depformer_cuda_graph import (
    DepformerCudaGraphRunner,
)
from sglang_omni.models.personaplex.sampling import (
    AudioSampling,
    SamplingRun,
    sample_runs,
)


@dataclass(kw_only=True)
class Measurement:
    batch_size: int
    temperature: float
    top_k: int
    capture_and_first_frame_milliseconds: float
    graph_allocated_bytes: int
    graph_reserved_bytes: int
    matching_frames: int
    checked_frames: int
    rng_matches: bool
    eager_median_milliseconds: float
    eager_mean_milliseconds: float
    eager_p95_milliseconds: float
    graph_median_milliseconds: float
    graph_mean_milliseconds: float
    graph_p95_milliseconds: float
    mean_speedup: float
    median_speedup: float
    eager_durations_milliseconds: list[float]
    graph_durations_milliseconds: list[float]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--parity-frames", type=int, default=20)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--input-snapshot", type=Path)
    parser.add_argument("--input-row", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--profile-frame-counts", type=int, nargs="+", default=[1, 150, 750]
    )
    parser.add_argument("--rollout-repeats", type=int, default=5)
    parser.add_argument(
        "--sampling-modes",
        choices=("greedy", "default", "mixed"),
        nargs="+",
        default=["greedy", "default", "mixed"],
    )
    arguments = parser.parse_args()
    if arguments.profile_dir is not None and (
        arguments.batch_sizes != [1] or arguments.sampling_modes != ["default"]
    ):
        parser.error(
            "Profiling requires --batch-sizes 1 --sampling-modes default; run other timings in a separate process"
        )
    else:
        pass
    if arguments.deterministic:
        enable_batch_invariant_mode()
    else:
        pass
    torch.manual_seed(17)
    model = Depformer(DEPFORMER)
    with safe_open(
        arguments.checkpoint / MOSHI_WEIGHTS_NAME, framework="pt"
    ) as checkpoint:
        weights = {
            name: checkpoint.get_tensor(name)
            for name in checkpoint.keys()
            if name.startswith(("depformer", "linears."))
        }
    model.load_reference_weights(weights)
    weights.clear()
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    measurements: list[Measurement] = []
    for sampling_name in arguments.sampling_modes:
        sampling = {
            "greedy": AudioSampling(0.0, 0),
            "default": AudioSampling(0.8, 250),
            "mixed": AudioSampling(1.0, 0),
        }[sampling_name]
        for batch_size in arguments.batch_sizes:
            runner = DepformerCudaGraphRunner(model, (1, 2, 4, 8, 16, 32))
            eager_generators = [
                torch.Generator(device="cuda").manual_seed(7 + row)
                for row in range(batch_size)
            ]
            graph_generators = [
                torch.Generator(device="cuda").manual_seed(7 + row)
                for row in range(batch_size)
            ]
            settings = [
                (
                    sampling
                    if sampling.top_k or sampling.greedy
                    else (AudioSampling(0.8, 250) if row % 2 else sampling)
                )
                for row in range(batch_size)
            ]
            eager_runs = [
                SamplingRun(
                    row_slice=slice(row, row + 1),
                    sampling=settings[row],
                    generators=[eager_generators[row]],
                )
                for row in range(batch_size)
            ]
            graph_runs = [
                SamplingRun(
                    row_slice=slice(row, row + 1),
                    sampling=settings[row],
                    generators=[graph_generators[row]],
                )
                for row in range(batch_size)
            ]
            if sampling.top_k or sampling.greedy:
                eager_runs = [
                    SamplingRun(
                        row_slice=slice(0, batch_size),
                        sampling=sampling,
                        generators=eager_generators,
                    )
                ]
                graph_runs = [
                    SamplingRun(
                        row_slice=slice(0, batch_size),
                        sampling=sampling,
                        generators=graph_generators,
                    )
                ]
            else:
                pass
            text_tokens = torch.arange(batch_size, device="cuda") + 3
            hidden_states = torch.randn(
                batch_size, DEPFORMER.input_dim, device="cuda", dtype=torch.bfloat16
            )
            forced_codes = torch.full((batch_size, DEPFORMER.steps), -1, device="cuda")
            if arguments.input_snapshot is not None:
                snapshot = torch.load(
                    arguments.input_snapshot, map_location="cpu", weights_only=True
                )
                saved_hidden_states = snapshot["hidden_states"]
                saved_text_tokens = snapshot["text_tokens"]
                assert isinstance(saved_hidden_states, torch.Tensor)
                assert isinstance(saved_text_tokens, torch.Tensor)
                hidden_states.copy_(saved_hidden_states[arguments.input_row])
                text_tokens.fill_(saved_text_tokens[arguments.input_row].item())
            else:
                pass
            original_hidden_states = hidden_states.clone()

            def generate_eager() -> torch.Tensor:
                return model.generate(
                    text_tokens,
                    hidden_states,
                    forced_codes,
                    lambda logits: sample_runs(logits, eager_runs),
                )

            def generate_graph() -> torch.Tensor:
                return runner.generate(
                    text_tokens, hidden_states, forced_codes, graph_runs
                )

            for _ in range(arguments.warmup):
                generate_eager()
            torch.cuda.synchronize()
            allocated_before_bytes = torch.cuda.memory_allocated()
            reserved_before_bytes = torch.cuda.memory_reserved()
            started_seconds = time.perf_counter()
            generate_graph()
            torch.cuda.synchronize()
            capture_milliseconds = (time.perf_counter() - started_seconds) * 1000
            allocated_bytes = torch.cuda.memory_allocated() - allocated_before_bytes
            reserved_bytes = torch.cuda.memory_reserved() - reserved_before_bytes
            assert runner.graphs and all(
                graph is not None for graph in runner.graphs.values()
            )
            for row, (eager_generator, graph_generator) in enumerate(
                zip(eager_generators, graph_generators, strict=True)
            ):
                eager_generator.manual_seed(7 + row)
                graph_generator.manual_seed(7 + row)
            matching_frames = 0
            rng_matches = True
            for frame_index in range(arguments.parity_frames):
                hidden_states.normal_()
                if frame_index % 2:
                    forced_codes[0, 0] = 17
                    forced_codes[-1, 2:4] = 23
                else:
                    forced_codes.fill_(-1)
                expected_codes = generate_eager()
                actual_codes = generate_graph()
                matching_frames += int(torch.equal(expected_codes, actual_codes))
                rng_matches = rng_matches and all(
                    torch.equal(
                        eager_generator.get_state(), graph_generator.get_state()
                    )
                    for eager_generator, graph_generator in zip(
                        eager_generators, graph_generators, strict=True
                    )
                )
            forced_codes.fill_(-1)
            hidden_states.copy_(original_hidden_states)
            for _ in range(arguments.warmup):
                generate_eager()
                generate_graph()
            eager_durations_milliseconds: list[float] = []
            graph_durations_milliseconds: list[float] = []
            for iteration in range(arguments.iterations):
                operations = (
                    (generate_eager, eager_durations_milliseconds),
                    (generate_graph, graph_durations_milliseconds),
                )
                for generate, durations_milliseconds in operations[
                    :: 1 if iteration % 2 else -1
                ]:
                    torch.cuda.synchronize()
                    started_seconds = time.perf_counter()
                    generate()
                    torch.cuda.synchronize()
                    durations_milliseconds.append(
                        (time.perf_counter() - started_seconds) * 1000
                    )
            eager_median_milliseconds = float(np.median(eager_durations_milliseconds))
            graph_median_milliseconds = float(np.median(graph_durations_milliseconds))
            measurement = Measurement(
                batch_size=batch_size,
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                capture_and_first_frame_milliseconds=capture_milliseconds,
                graph_allocated_bytes=allocated_bytes,
                graph_reserved_bytes=reserved_bytes,
                matching_frames=matching_frames,
                checked_frames=arguments.parity_frames,
                rng_matches=rng_matches,
                eager_median_milliseconds=eager_median_milliseconds,
                eager_mean_milliseconds=float(np.mean(eager_durations_milliseconds)),
                eager_p95_milliseconds=float(
                    np.percentile(eager_durations_milliseconds, 95)
                ),
                graph_median_milliseconds=graph_median_milliseconds,
                graph_mean_milliseconds=float(np.mean(graph_durations_milliseconds)),
                graph_p95_milliseconds=float(
                    np.percentile(graph_durations_milliseconds, 95)
                ),
                mean_speedup=float(np.mean(eager_durations_milliseconds))
                / float(np.mean(graph_durations_milliseconds)),
                median_speedup=eager_median_milliseconds / graph_median_milliseconds,
                eager_durations_milliseconds=eager_durations_milliseconds,
                graph_durations_milliseconds=graph_durations_milliseconds,
            )
            measurements.append(measurement)
            print(json.dumps(asdict(measurement)), flush=True)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "dtype": "bfloat16",
                        "input_source": (
                            str(arguments.input_snapshot)
                            if arguments.input_snapshot
                            else "synthetic hidden states with checkpoint weights"
                        ),
                        "deterministic": arguments.deterministic,
                        "iterations": arguments.iterations,
                        "measurements": [asdict(result) for result in measurements],
                    },
                    indent=2,
                )
                + "\n"
            )
            if (
                arguments.profile_dir is not None
                and batch_size == 1
                and sampling == AudioSampling(0.8, 250)
            ):
                arguments.profile_dir.mkdir(parents=True, exist_ok=True)
                eager_rollout_durations_seconds = measure_frames(
                    generate_eager,
                    arguments.profile_frame_counts,
                    arguments.rollout_repeats,
                )
                graph_rollout_durations_seconds = measure_frames(
                    generate_graph,
                    arguments.profile_frame_counts,
                    arguments.rollout_repeats,
                )
                (arguments.profile_dir / "unprofiled-rollouts.json").write_text(
                    json.dumps(
                        {
                            "eager": eager_rollout_durations_seconds,
                            "graph": graph_rollout_durations_seconds,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                profile_frames(
                    generate_eager,
                    arguments.profile_frame_counts,
                    arguments.profile_dir / "eager",
                    eager_rollout_durations_seconds,
                )
                profile_frames(
                    generate_graph,
                    arguments.profile_frame_counts,
                    arguments.profile_dir / "graph",
                    graph_rollout_durations_seconds,
                )
            else:
                pass
    assert all(
        result.matching_frames == result.checked_frames and result.rng_matches
        for result in measurements
    )


if __name__ == "__main__":
    main()
else:
    pass
