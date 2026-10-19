# SPDX-License-Identifier: Apache-2.0
"""Drama 3 decode CUDA graphs on a GPU: eager and graph engines, each in its own
process (graph_engines.py), must agree bit for bit on the synthetic checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    VALIDATED_COMPUTE_CAPABILITY,
)
from sglang_omni.models.fishaudio_drama3.tower_kernels import MAX_FUSED_ROWS
from tests.unit_test.fishaudio_drama3.graph_checks import (
    EnginePair,
    bit_problems,
    capture_break_problems,
    draw_problems,
    finite_problems,
    first_frame_path_problems,
    first_frame_split_problems,
    fused_layer_problems,
    generator_offset_problems,
    history_problems,
    lean_problems,
    lookahead_problems,
    outcome_problems,
    prefill_path_problems,
    prompt_row_counts,
    replay_problems,
    resampled_choice_count,
    sampling_step_problems,
    schedule_problems,
    stray_write_problems,
    stream_problems,
    wave_admission_problems,
)
from tests.unit_test.fishaudio_drama3.graph_fixtures import (
    CUDA_GRAPH_BATCH_SIZES,
    FP8_SCENARIOS,
    GREEDY_ABORT_REUSE,
    MAIN_SCENARIOS,
    PREFILL_CUDA_GRAPH_TOKEN_COUNTS,
    RANDOM_WAVES,
    RETRACT_INTERVAL,
    RETRACT_SCENARIOS,
    SMALL_BUDGET_SCENARIOS,
    EngineKind,
    GraphEngineRun,
    GraphEngineSpec,
    GraphScenario,
    bits,
)
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    TOTAL_GPU_MEMORY_FRACTION,
    write_expert_variant,
    write_synthetic_checkpoint,
)
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.scheduling.test_hybrid_gdn import (
    kill_process_group,
    require_free_gpu_memory,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_MODULE = "tests.unit_test.fishaudio_drama3.graph_engines"
ENGINE_TIMEOUT_SECONDS = 1200
RETRACT_ENVIRONMENT = {
    "SGLANG_TEST_RETRACT": "1",
    "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL),
}
# Requests whose last step never comes: the abort target.
PADDED_BATCH_SIZES = {3, 5, 6, 7}
SMALL_BUDGET_MAX_PREFILL_TOKENS = 32
SMALL_BUDGET_CONTEXT_LENGTH = 112
FIRST_FRAME_REQUEST_COUNTS = list(range(1, max(CUDA_GRAPH_BATCH_SIZES) + 1))
LOW_FIRST_FRAME_CAP = 3
LOOKAHEAD_PAIR_NAMES = {
    "lookahead main",
    "lookahead prefill main",
    "lookahead retract",
    "lookahead prefill retract",
}
# One line per problem is enough to locate the first divergence.
REPORTED_PROBLEM_COUNT = 20
UNFUSED_PAIR_NAMES = {"unfused main", "unfused prefill main"}


@dataclass(frozen=True, kw_only=True)
class GraphEngineRuns:
    main: EnginePair
    fp8: EnginePair
    retract: EnginePair
    prefill_main: EnginePair
    prefill_fp8: EnginePair
    prefill_retract: EnginePair
    prefill_small_budget: EnginePair
    first_frame_off: EnginePair
    first_frame_low_cap: EnginePair
    first_frame_on_against_off: EnginePair
    triton_main: EnginePair
    triton_prefill_main: EnginePair
    lookahead_main: EnginePair
    lookahead_prefill_main: EnginePair
    lookahead_retract: EnginePair
    lookahead_prefill_retract: EnginePair
    unfused_main: EnginePair
    unfused_prefill_main: EnginePair
    workdir: Path


def run_engine(
    name: str, spec: GraphEngineSpec, workdir: Path, extra_environment: dict[str, str]
) -> GraphEngineRun:
    """Run one engine process to completion and load what it recorded."""
    require_free_gpu_memory(spec.total_gpu_memory_fraction)
    spec_path = workdir / f"{name}-spec.json"
    output_path = workdir / f"{name}-run.pt"
    log_path = workdir / f"{name}.log"
    spec_path.write_text(spec.model_dump_json())
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                ENGINE_MODULE,
                "--spec",
                str(spec_path),
                "--output",
                str(output_path),
            ],
            cwd=REPO_ROOT,
            env={**os.environ, **extra_environment},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=ENGINE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(
                f"{name} engine timed out after {ENGINE_TIMEOUT_SECONDS}s; "
                f"log: {log_path}"
            ) from error
        finally:
            kill_process_group(process)
    if return_code != 0:
        log_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
        raise AssertionError(
            f"{name} engine exited with {return_code}; log: {log_path}\n{log_tail}"
        )
    else:
        pass
    run = torch.load(output_path, weights_only=False)
    assert isinstance(run, GraphEngineRun)
    return run


def engine_pairs(
    name: str,
    model_dir: Path,
    scenarios: list[GraphScenario],
    workdir: Path,
    extra_environment: dict[str, str],
    triton_sampling: bool,
    *,
    enable_async_decode: bool,
    tower_fused_layers: bool = True,
) -> tuple[EnginePair, EnginePair]:
    runs: dict[EngineKind, GraphEngineRun] = {}
    for kind in ("eager", "graph", "prefill_graph"):
        spec = GraphEngineSpec(
            model_dir=str(model_dir),
            kind=kind,
            triton_sampling=triton_sampling,
            tower_fused_layers=tower_fused_layers,
            enable_async_decode=enable_async_decode,
            scenarios=scenarios,
        )
        runs[kind] = run_engine(f"{name}-{kind}", spec, workdir, extra_environment)
    return (
        EnginePair(eager=runs["eager"], graph=runs["graph"], scenarios=scenarios),
        EnginePair(
            eager=runs["eager"], graph=runs["prefill_graph"], scenarios=scenarios
        ),
    )


@pytest.fixture(scope="module")
def engine_runs(tmp_path_factory: pytest.TempPathFactory) -> GraphEngineRuns:
    require_cuda()
    require_free_gpu_memory(TOTAL_GPU_MEMORY_FRACTION)
    workdir = tmp_path_factory.mktemp("drama3_graph")
    checkpoint_dir = workdir / "checkpoint"
    write_synthetic_checkpoint(checkpoint_dir)
    fp8_checkpoint_dir = workdir / "checkpoint-fp8"
    write_expert_variant(checkpoint_dir, fp8_checkpoint_dir, "fp8")
    # Serial on purpose: each engine owns its process and its share of the GPU.
    main, prefill_main = engine_pairs(
        "main",
        checkpoint_dir,
        MAIN_SCENARIOS,
        workdir,
        {},
        False,
        enable_async_decode=False,
    )
    fp8, prefill_fp8 = engine_pairs(
        "fp8",
        fp8_checkpoint_dir,
        FP8_SCENARIOS,
        workdir,
        {},
        False,
        enable_async_decode=False,
    )
    retract, prefill_retract = engine_pairs(
        "retract",
        checkpoint_dir,
        RETRACT_SCENARIOS,
        workdir,
        RETRACT_ENVIRONMENT,
        False,
        enable_async_decode=False,
    )
    triton_main, triton_prefill_main = engine_pairs(
        "triton-main",
        checkpoint_dir,
        MAIN_SCENARIOS,
        workdir,
        {},
        True,
        enable_async_decode=False,
    )
    lookahead_main, lookahead_prefill_main = engine_pairs(
        "lookahead-main",
        checkpoint_dir,
        MAIN_SCENARIOS,
        workdir,
        {},
        False,
        enable_async_decode=True,
    )
    lookahead_retract, lookahead_prefill_retract = engine_pairs(
        "lookahead-retract",
        checkpoint_dir,
        RETRACT_SCENARIOS,
        workdir,
        RETRACT_ENVIRONMENT,
        False,
        enable_async_decode=True,
    )
    unfused_main, unfused_prefill_main = engine_pairs(
        "unfused-main",
        checkpoint_dir,
        MAIN_SCENARIOS,
        workdir,
        {},
        False,
        enable_async_decode=False,
        tower_fused_layers=False,
    )
    small_budget_runs = {
        kind: run_engine(
            f"small-budget-{kind}",
            GraphEngineSpec(
                model_dir=str(checkpoint_dir),
                kind=kind,
                context_length=SMALL_BUDGET_CONTEXT_LENGTH,
                max_prefill_tokens=SMALL_BUDGET_MAX_PREFILL_TOKENS,
                scenarios=SMALL_BUDGET_SCENARIOS,
            ),
            workdir,
            {},
        )
        for kind in ("eager", "prefill_graph")
    }
    first_frame_off = run_engine(
        "main-graph-first-frame-off",
        GraphEngineSpec(
            model_dir=str(checkpoint_dir),
            kind="graph",
            first_frame_cuda_graphs=False,
            scenarios=MAIN_SCENARIOS,
        ),
        workdir,
        {},
    )
    first_frame_low_cap = run_engine(
        "main-graph-first-frame-low-cap",
        GraphEngineSpec(
            model_dir=str(checkpoint_dir),
            kind="graph",
            first_frame_cuda_graph_max_requests=LOW_FIRST_FRAME_CAP,
            scenarios=MAIN_SCENARIOS,
        ),
        workdir,
        {},
    )
    runs = GraphEngineRuns(
        main=main,
        fp8=fp8,
        retract=retract,
        prefill_main=prefill_main,
        prefill_fp8=prefill_fp8,
        prefill_retract=prefill_retract,
        prefill_small_budget=EnginePair(
            eager=small_budget_runs["eager"],
            graph=small_budget_runs["prefill_graph"],
            scenarios=SMALL_BUDGET_SCENARIOS,
        ),
        first_frame_off=EnginePair(
            eager=main.eager, graph=first_frame_off, scenarios=MAIN_SCENARIOS
        ),
        first_frame_low_cap=EnginePair(
            eager=main.eager, graph=first_frame_low_cap, scenarios=MAIN_SCENARIOS
        ),
        first_frame_on_against_off=EnginePair(
            eager=first_frame_off, graph=main.graph, scenarios=MAIN_SCENARIOS
        ),
        triton_main=triton_main,
        triton_prefill_main=triton_prefill_main,
        lookahead_main=lookahead_main,
        lookahead_prefill_main=lookahead_prefill_main,
        lookahead_retract=lookahead_retract,
        lookahead_prefill_retract=lookahead_prefill_retract,
        unfused_main=unfused_main,
        unfused_prefill_main=unfused_prefill_main,
        workdir=workdir,
    )
    print(f"Artifacts: {workdir}")
    return runs


def all_pairs(engine_runs: GraphEngineRuns) -> dict[str, EnginePair]:
    return {
        "main": engine_runs.main,
        "fp8": engine_runs.fp8,
        "retract": engine_runs.retract,
        "prefill main": engine_runs.prefill_main,
        "prefill fp8": engine_runs.prefill_fp8,
        "prefill retract": engine_runs.prefill_retract,
        "prefill small budget": engine_runs.prefill_small_budget,
        "first-frame off": engine_runs.first_frame_off,
        "first-frame low cap": engine_runs.first_frame_low_cap,
        "lookahead main": engine_runs.lookahead_main,
        "lookahead prefill main": engine_runs.lookahead_prefill_main,
        "lookahead retract": engine_runs.lookahead_retract,
        "lookahead prefill retract": engine_runs.lookahead_prefill_retract,
        "unfused main": engine_runs.unfused_main,
        "unfused prefill main": engine_runs.unfused_prefill_main,
    }


def first_problems(problems: list[str]) -> str:
    return "\n".join(problems[:REPORTED_PROBLEM_COUNT])


def is_on_validated_gpu(run: GraphEngineRun) -> bool:
    return run.facts.device_capability == VALIDATED_COMPUTE_CAPABILITY


def test_engines_start_as_specified(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        eager_facts = pair.eager.facts
        graph_facts = pair.graph.facts
        assert eager_facts.decode_graph_batch_sizes is None, name
        assert graph_facts.decode_graph_batch_sizes == CUDA_GRAPH_BATCH_SIZES, name
        assert eager_facts.prefill_runner == "EagerRunner", name
        assert eager_facts.prefill_capture_token_counts is None, name
        if pair.graph.kind == "prefill_graph":
            assert graph_facts.prefill_runner == "RequestIdPrefillCudaGraphRunner", name
            assert (
                graph_facts.prefill_capture_token_counts
                == PREFILL_CUDA_GRAPH_TOKEN_COUNTS
            ), name
        else:
            assert graph_facts.prefill_runner == "EagerRunner", name
            assert graph_facts.prefill_capture_token_counts is None, name
        for facts in (eager_facts, graph_facts):
            assert set(facts.tower_layer_ids) <= set(facts.recorded_kv_layer_ids), name
            assert facts.decode_buffer_rows >= facts.request_pool_size, name
            assert facts.decode_buffer_rows >= max(CUDA_GRAPH_BATCH_SIZES), name
            assert facts.tower_prefill_runs_outside_extend == 0, name
            assert facts.tower_prefill_rows >= max(
                facts.prefill_capture_token_counts or [0]
            ), name
        assert eager_facts.tower_prefill_runs_before_serving == 0, name
        if pair.graph.kind == "prefill_graph":
            assert graph_facts.tower_prefill_runs_before_serving >= len(
                PREFILL_CUDA_GRAPH_TOKEN_COUNTS
            ), name
        else:
            assert graph_facts.tower_prefill_runs_before_serving == 0, name
        # The split count saturating at its cap keeps padding out of the split.
        assert eager_facts.max_kv_splits == graph_facts.max_kv_splits == 8, name
        assert eager_facts.kv_pool_size == graph_facts.kv_pool_size, name
        assert eager_facts.expert_paths == graph_facts.expert_paths, name
        first_frame_request_counts = {
            "first-frame off": None,
            "first-frame low cap": list(range(1, LOW_FIRST_FRAME_CAP + 1)),
        }.get(name, FIRST_FRAME_REQUEST_COUNTS)
        assert eager_facts.first_frame_request_counts is None, name
        assert (
            graph_facts.first_frame_request_counts == first_frame_request_counts
        ), name
        assert eager_facts.first_frame_capture_written_kv_slots == [], name
        assert graph_facts.first_frame_capture_written_kv_slots == [], name
    for expert_path in engine_runs.fp8.graph.facts.expert_paths:
        assert expert_path.quant_method == "Fp8MoEMethod"
        assert expert_path.runner_backend == "TRITON"
        assert expert_path.uses_fp8_w8a8
        assert expert_path.block_shape == [128, 128]


def test_lookahead_engines_launch_each_decode_ahead_and_discard_only_ends(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        is_lookahead = name in LOOKAHEAD_PAIR_NAMES
        for run in (pair.eager, pair.graph):
            problems = lookahead_problems(run, is_lookahead)
            assert not problems, f"{name}:\n{first_problems(problems)}"
            assert lookahead_problems(run, not is_lookahead), name
            assert (
                any(
                    step.is_discarded
                    for record in run.forwards
                    for step in record.steps
                )
                == is_lookahead
            ), name
    run = engine_runs.lookahead_main.graph
    decode_records = [record for record in run.forwards if not record.is_extend]
    record, kept_step = next(
        (record, step)
        for record, next_record in zip(decode_records, decode_records[1:])
        for step in record.steps
        if not step.is_discarded
        and step.request_id in {later.request_id for later in next_record.steps}
    )
    wrongly_discarded = replace(
        run,
        forwards=[
            (
                replace(
                    candidate,
                    steps=[
                        replace(step, is_discarded=True) if step is kept_step else step
                        for step in candidate.steps
                    ],
                )
                if candidate is record
                else candidate
            )
            for candidate in run.forwards
        ],
    )
    assert lookahead_problems(wrongly_discarded, True)


def test_both_engines_run_the_same_batches(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        problems = (
            schedule_problems(pair)
            + wave_admission_problems(pair.eager, pair.scenarios)
            + wave_admission_problems(pair.graph, pair.scenarios)
        )
        assert not problems, f"{name}:\n{first_problems(problems)}"


def test_graph_engine_matches_the_eager_engine_bit_for_bit(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        assert not schedule_problems(pair), name
        problems = bit_problems(pair)
        assert not problems, f"{name}:\n{first_problems(problems)}"


def test_graph_engine_replays_every_decode_padded_and_unpadded(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        problems = replay_problems(pair)
        assert not problems, f"{name}:\n{first_problems(problems)}"
    replayed_sizes = {
        len(record.steps)
        for record in engine_runs.main.graph.forwards
        if record.is_graph_replay and not record.is_extend
    }
    assert replayed_sizes == set(range(1, 9))
    # Random sampling is compared only where both engines draw the same shapes.
    assert all(
        record.graph_batch_size == len(record.steps)
        for record in engine_runs.main.graph.forwards
        if record.scenario_name == RANDOM_WAVES.name
        and record.is_graph_replay
        and not record.is_extend
    )
    assert PADDED_BATCH_SIZES <= {
        len(record.steps)
        for record in engine_runs.fp8.graph.forwards
        if record.is_graph_replay and not record.is_extend
    }


def test_each_prefill_replays_its_bucket_or_runs_eager(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = prefill_path_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_each_prefill_graph_capture_breaks_at_every_attention_layer(
    engine_runs: GraphEngineRuns,
) -> None:
    assert engine_runs.prefill_main.graph.facts.tower_layer_ids
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = capture_break_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_prefill_graph_replays_a_bucket_past_its_prefill_budget_and_context(
    engine_runs: GraphEngineRuns,
) -> None:
    pair = engine_runs.prefill_small_budget
    largest_extend = max(SMALL_BUDGET_MAX_PREFILL_TOKENS, SMALL_BUDGET_CONTEXT_LENGTH)
    assert max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS) > largest_extend
    assert pair.graph.facts.tower_prefill_rows >= max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS)
    assert any(
        bucket > largest_extend
        for record in pair.graph.forwards
        for bucket in record.prefill_replay_buckets
    )
    assert any(
        record.is_extend and record.token_count > SMALL_BUDGET_MAX_PREFILL_TOKENS
        for record in pair.eager.forwards
    )


def test_prefill_graph_replays_every_admissible_shape(
    engine_runs: GraphEngineRuns,
) -> None:
    run = engine_runs.prefill_main.graph
    extends = [record for record in run.forwards if record.is_extend]
    replays = [record for record in extends if record.is_graph_replay]
    replayed_buckets = [
        bucket for record in replays for bucket in record.prefill_replay_buckets
    ]
    assert extends[0].is_graph_replay
    assert any(
        record.token_count in PREFILL_CUDA_GRAPH_TOKEN_COUNTS for record in replays
    )
    assert any(
        record.token_count not in PREFILL_CUDA_GRAPH_TOKEN_COUNTS for record in replays
    )
    assert any(len(record.steps) > 1 for record in replays)
    assert any(
        later < earlier
        for earlier, later in zip(replayed_buckets, replayed_buckets[1:])
    )
    assert any(
        earlier.is_graph_replay
        and later.prefill_replay_buckets == earlier.prefill_replay_buckets
        and later.token_count < earlier.token_count
        and len(later.steps) != len(earlier.steps)
        for earlier, later in zip(extends, extends[1:])
    )
    assert any(
        record.token_count > max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS)
        for record in extends
        if not record.is_graph_replay
    )
    assert any(
        record.token_count <= max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS)
        for record in extends
        if not record.is_graph_replay
    )


def test_prefill_graph_replays_beside_live_requests(
    engine_runs: GraphEngineRuns,
) -> None:
    run = engine_runs.prefill_retract.graph
    decode_forwards: dict[str, list[int]] = {}
    for record in run.forwards:
        for step in record.steps:
            if not record.is_extend:
                decode_forwards.setdefault(step.request_id, []).append(
                    record.forward_index
                )
            else:
                pass
    replays_beside_live_requests = [
        record
        for record in run.forwards
        if record.is_extend
        and record.is_graph_replay
        and any(
            min(forward_indices) < record.forward_index < max(forward_indices)
            for request_id, forward_indices in decode_forwards.items()
            if request_id not in {step.request_id for step in record.steps}
        )
    ]
    assert replays_beside_live_requests


def test_each_first_frame_step_replays_its_request_count_or_runs_eager(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = first_frame_path_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_first_frame_graphs_replay_every_request_count_and_resume_past_the_cap(
    engine_runs: GraphEngineRuns,
) -> None:
    for run in (engine_runs.main.graph, engine_runs.prefill_main.graph):
        assert {
            request_count
            for record in run.forwards
            for request_count in record.first_frame_replays
        } == set(FIRST_FRAME_REQUEST_COUNTS), run.kind
    extends = [
        record
        for record in engine_runs.first_frame_low_cap.graph.forwards
        if record.is_extend
    ]
    assert any(
        earlier.first_frame_eager_count == 1
        and len(earlier.steps) > LOW_FIRST_FRAME_CAP
        and later.first_frame_replays
        for earlier, later in zip(extends, extends[1:])
    )


def test_first_frame_replays_alternate_with_decode_replays_of_other_sizes(
    engine_runs: GraphEngineRuns,
) -> None:
    directions: set[str] = set()
    for pair in all_pairs(engine_runs).values():
        forwards = pair.graph.forwards
        for earlier, later in zip(forwards, forwards[1:]):
            if (
                earlier.first_frame_replays
                and not later.is_extend
                and later.is_graph_replay
                and later.graph_batch_size != len(earlier.steps)
            ):
                directions.add("first frame, then decode")
            elif (
                not earlier.is_extend
                and earlier.is_graph_replay
                and later.first_frame_replays
                and len(later.steps) != earlier.graph_batch_size
            ):
                directions.add("decode, then first frame")
            else:
                pass
    assert directions == {"first frame, then decode", "decode, then first frame"}


def test_first_frame_graphs_draw_and_split_like_the_eager_step(
    engine_runs: GraphEngineRuns,
) -> None:
    on_against_off = engine_runs.first_frame_on_against_off
    problems = (
        schedule_problems(on_against_off)
        + bit_problems(on_against_off)
        + first_frame_split_problems(on_against_off)
        + generator_offset_problems(on_against_off, None)
    )
    assert not problems, f"first-frame on against off:\n{first_problems(problems)}"
    for name, pair in all_pairs(engine_runs).items():
        problems = first_frame_split_problems(pair) + generator_offset_problems(
            pair, {RANDOM_WAVES.name}
        )
        assert not problems, f"{name}:\n{first_problems(problems)}"
    assert any(
        record.scenario_name == RANDOM_WAVES.name and record.first_frame_replays
        for record in engine_runs.main.graph.forwards
    )


def test_triton_sampling_engines_match_bit_for_bit_and_step_once_per_forward(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in {
        "triton main": engine_runs.triton_main,
        "triton prefill main": engine_runs.triton_prefill_main,
    }.items():
        problems = (
            schedule_problems(pair)
            + bit_problems(pair)
            + first_frame_split_problems(pair)
            + replay_problems(pair)
            + first_frame_path_problems(pair.graph)
        )
        for run in (pair.eager, pair.graph):
            problems += (
                history_problems(run, pair.scenarios)
                + finite_problems(run)
                + outcome_problems(run)
                + sampling_step_problems(run, is_on_validated_gpu(run))
            )
        assert not problems, f"{name}:\n{first_problems(problems)}"
    eager_forwards = engine_runs.triton_main.eager.forwards
    graph_forwards = engine_runs.triton_main.graph.forwards
    assert any(record.first_frame_replays for record in graph_forwards)
    assert any(
        record.is_graph_replay and not record.is_extend for record in graph_forwards
    )
    assert any(record.first_frame_eager_count for record in eager_forwards)
    assert any(not record.is_extend for record in eager_forwards)


def test_pytorch_sampling_engines_never_step(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = sampling_step_problems(run, False)
            assert not problems, f"{name} {run.kind}:\n{first_problems(problems)}"


def test_fused_layers_run_each_eager_decode_step_and_no_tower_prefill(
    engine_runs: GraphEngineRuns,
) -> None:
    pairs = {
        **all_pairs(engine_runs),
        "triton main": engine_runs.triton_main,
        "triton prefill main": engine_runs.triton_prefill_main,
    }
    for name, pair in pairs.items():
        for run in (pair.eager, pair.graph):
            is_fused = name not in UNFUSED_PAIR_NAMES and is_on_validated_gpu(run)
            problems = fused_layer_problems(run, is_fused)
            assert not problems, f"{name} {run.kind}:\n{first_problems(problems)}"
            assert fused_layer_problems(run, not is_fused), f"{name} {run.kind}"
    expected_calls = (
        len(engine_runs.main.eager.facts.tower_layer_ids)
        if is_on_validated_gpu(engine_runs.main.eager)
        else 0
    )
    eager_forwards = engine_runs.main.eager.forwards
    assert any(
        record.is_extend and record.fused_layer_calls == expected_calls
        for record in eager_forwards
    )
    assert any(
        not record.is_extend and record.fused_layer_calls == expected_calls
        for record in eager_forwards
    )


def test_tower_prefill_writes_the_unfused_bits_and_the_decode_step_fuses(
    engine_runs: GraphEngineRuns,
) -> None:
    fused_extends = [
        record for record in engine_runs.main.eager.forwards if record.is_extend
    ]
    unfused_extends = [
        record for record in engine_runs.unfused_main.eager.forwards if record.is_extend
    ]
    assert len(fused_extends) == len(unfused_extends) > 0
    assert any(record.token_count <= MAX_FUSED_ROWS for record in fused_extends)
    for fused_extend, unfused_extend in zip(
        fused_extends, unfused_extends, strict=True
    ):
        assert fused_extend.tower_prefill_runs == unfused_extend.tower_prefill_runs
        assert fused_extend.tower_prefill_fused_layer_calls == 0
        assert [step.request_id for step in fused_extend.steps] == [
            step.request_id for step in unfused_extend.steps
        ]
        for fused_step, unfused_step in zip(
            fused_extend.steps, unfused_extend.steps, strict=True
        ):
            assert torch.equal(
                bits(fused_step.written_cache[:, :-1]),
                bits(unfused_step.written_cache[:, :-1]),
            ), (fused_extend.forward_index, fused_step.request_id)
    expected_calls = (
        len(engine_runs.main.eager.facts.tower_layer_ids)
        if is_on_validated_gpu(engine_runs.main.eager)
        else 0
    )
    first_decode = next(
        record for record in engine_runs.main.eager.forwards if not record.is_extend
    )
    assert fused_extends[0].fused_layer_calls == expected_calls
    assert first_decode.fused_layer_calls == expected_calls


def test_decode_attention_keeps_the_standard_kernel(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = lean_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"
    assert set(FIRST_FRAME_REQUEST_COUNTS) <= {
        choice.batch_size
        for choice in engine_runs.main.graph.facts.startup_lean_choices
        if choice.policy == "capture"
    }
    assert {
        choice.policy
        for record in engine_runs.main.eager.forwards
        for choice in record.lean_choices
    } == {"sequence_length"}


def test_inputs_the_model_read_follow_each_history(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = history_problems(run, pair.scenarios)
            assert not problems, f"{name} {run.kind}:\n{first_problems(problems)}"


def test_ras_resampling_happens_and_picks_the_main_token(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = draw_problems(run)
            assert not problems, f"{name} {run.kind}:\n{first_problems(problems)}"
    for run in (engine_runs.main.eager, engine_runs.main.graph):
        assert resampled_choice_count(run, RANDOM_WAVES.name) > 0, run.kind


def test_every_recorded_value_is_finite(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = finite_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_decode_steps_write_only_their_own_slots(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = stray_write_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_streaming_requests_send_each_frame_once_a_whole_window_at_a_time(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = stream_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_every_request_finishes_once_and_pools_return(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = outcome_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


@pytest.mark.parametrize("pair_name", ["main", "prefill main"])
def test_rows_and_states_an_abort_freed_serve_the_next_wave(
    engine_runs: GraphEngineRuns, pair_name: str
) -> None:
    """The aborted request's row and state slot are reused by the next wave, whose
    key/value slots were written before; freed KV slots queue behind free ones."""
    run = all_pairs(engine_runs)[pair_name].graph
    aborted = [
        outcome for outcome in run.outcomes.values() if outcome.aborted_after_frames
    ]
    assert [outcome.aborted_after_frames for outcome in aborted] == [15]
    aborted_id = aborted[0].request_id
    scenario_records = [
        record
        for record in run.forwards
        if record.scenario_name == GREEDY_ABORT_REUSE.name
    ]
    aborted_steps = [
        step
        for record in scenario_records
        for step in record.steps
        if step.request_id == aborted_id
    ]
    later_steps = [
        step
        for record in scenario_records
        for step in record.steps
        if step.request_id.startswith(f"{GREEDY_ABORT_REUSE.name}-1-")
    ]
    assert aborted_steps and later_steps
    assert aborted_steps[0].request_row in {step.request_row for step in later_steps}
    assert aborted_steps[0].mamba_slot in {step.mamba_slot for step in later_steps}
    first_later_forward = min(
        record.forward_index
        for record in scenario_records
        if any(step in later_steps for step in record.steps)
    )
    earlier_slots = {
        slot
        for record in run.forwards[:first_later_forward]
        for step in record.steps
        for slot in step.written_slots
    }
    later_slots = {slot for step in later_steps for slot in step.written_slots}
    assert earlier_slots & later_slots
    assert run.forwards[first_later_forward].is_extend
    assert run.forwards[first_later_forward].is_graph_replay == (
        run.kind == "prefill_graph"
    )


def test_retracted_requests_reprefill_their_history_between_replays(
    engine_runs: GraphEngineRuns,
) -> None:
    run = engine_runs.retract.graph
    row_counts = prompt_row_counts(run)
    reprefills = [
        (record.forward_index, step.request_id)
        for record in run.forwards
        if record.is_extend
        for step in record.steps
        if len(step.positions) > row_counts[step.request_id]
    ]
    assert reprefills
    for forward_index, request_id in reprefills:
        steps_around = [
            record.forward_index < forward_index
            for record in run.forwards
            if record.is_graph_replay
            and not record.is_extend
            and any(step.request_id == request_id for step in record.steps)
        ]
        assert True in steps_around and False in steps_around, request_id
