# SPDX-License-Identifier: Apache-2.0
"""Checks of graph test engine runs, each returning one line per problem found:
eager against graph bit for bit, and each run against its own histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.fishaudio_drama3.config import Drama3VocoderFactoryArgs
from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)
from sglang_omni.models.fishaudio_drama3.stream_windows import (
    frames_in_complete_windows,
)
from tests.unit_test.fishaudio_drama3.graph_fixtures import (
    CUDA_GRAPH_BATCH_SIZES,
    PREFILL_CUDA_GRAPH_PADDING_FACTOR,
    PREFILL_CUDA_GRAPH_TOKEN_COUNTS,
    GraphEngineRun,
    GraphForwardRecord,
    GraphRequestStep,
    GraphScenario,
    bits,
)
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    EOS_TOKEN_ID,
    PROMPTS,
    SEMANTIC_END_TOKEN_ID,
    SEMANTIC_START_TOKEN_ID,
)


@dataclass(frozen=True, kw_only=True)
class EnginePair:
    eager: GraphEngineRun
    graph: GraphEngineRun
    scenarios: list[GraphScenario]


def same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(bits(left), bits(right))
    )


def forward_name(record: GraphForwardRecord) -> str:
    if record.is_extend:
        return (
            f"forward {record.forward_index} ({record.scenario_name}, extend of "
            f"{record.token_count} tokens)"
        )
    else:
        return (
            f"forward {record.forward_index} ({record.scenario_name}, decode of "
            f"{len(record.steps)} on graph batch {record.graph_batch_size})"
        )


def forward_layout(
    record: GraphForwardRecord,
) -> tuple[str, bool, list[tuple[str, list[int], list[int], int, int]]]:
    return (
        record.scenario_name,
        record.is_extend,
        [
            (
                step.request_id,
                step.positions,
                step.written_slots,
                step.request_row,
                step.mamba_slot,
            )
            for step in record.steps
        ],
    )


def schedule_problems(pair: EnginePair) -> list[str]:
    """The batches differ: a harness failure, not a graph result."""
    eager_forwards = pair.eager.forwards
    graph_forwards = pair.graph.forwards
    if len(eager_forwards) != len(graph_forwards):
        return [
            f"eager ran {len(eager_forwards)} forwards, graph {len(graph_forwards)}"
        ]
    else:
        return [
            f"forward {eager.forward_index}: eager ran {forward_layout(eager)}, "
            f"graph ran {forward_layout(graph)}"
            for eager, graph in zip(eager_forwards, graph_forwards, strict=True)
            if forward_layout(eager) != forward_layout(graph)
        ][:1]


def prompt_row_counts(run: GraphEngineRun) -> dict[str, int]:
    return {
        request_id: len(PROMPTS[outcome.request_spec.prompt_name])
        for request_id, outcome in run.outcomes.items()
    }


def wave_admission_problems(
    run: GraphEngineRun, scenarios: list[GraphScenario]
) -> list[str]:
    """Each wave's requests take their first prefill together, in one forward."""
    row_counts = prompt_row_counts(run)
    first_prefills: dict[str, list[int]] = {}
    for record in run.forwards:
        for step in record.steps:
            if record.is_extend and len(step.positions) == row_counts[step.request_id]:
                # Request ids are scenario-wave-position-prompt.
                wave_key = step.request_id.rsplit("-", 2)[0]
                first_prefills.setdefault(wave_key, []).append(record.forward_index)
            else:
                pass
    return [
        f"{run.kind} wave {scenario.name}-{wave_index} prefilled in forwards "
        f"{first_prefills.get(f'{scenario.name}-{wave_index}')}"
        for scenario in scenarios
        for wave_index, wave in enumerate(scenario.waves)
        if first_prefills.get(f"{scenario.name}-{wave_index}")
        != [first_prefills[f"{scenario.name}-{wave_index}"][0]] * len(wave)
    ]


def value_fields(step: GraphRequestStep) -> dict[str, str]:
    return {
        "sampled frame": repr(step.sampled_frame),
        "input frames": repr(step.input_frames),
        "staged RAS window": repr(step.staged_ras_window),
        "model RAS window": repr(step.model_ras_window),
        "sampling knobs": repr((step.temperature, step.top_k, step.top_p)),
        "draws": repr((step.first_draw, step.ras_draw)),
        "linear-attention state": step.state_digest,
        "discarded": repr(step.is_discarded),
    }


def tensor_fields(step: GraphRequestStep) -> dict[str, torch.Tensor]:
    final_state = step.final_state or []
    return {
        "written key/value cache": step.written_cache,
        "semantic logits": step.semantic_logits,
        "head logits": step.head_logits,
        **{
            f"final state tensor {index}": tensor
            for index, tensor in enumerate(final_state)
        },
    }


def bit_problems(pair: EnginePair) -> list[str]:
    """Every value the graph engine recorded that differs from the eager engine's."""
    problems = []
    for eager_record, graph_record in zip(
        pair.eager.forwards, pair.graph.forwards, strict=True
    ):
        for eager_step, graph_step in zip(
            eager_record.steps, graph_record.steps, strict=True
        ):
            where = f"{forward_name(graph_record)} {graph_step.request_id}"
            eager_values = value_fields(eager_step)
            graph_values = value_fields(graph_step)
            eager_tensors = tensor_fields(eager_step)
            graph_tensors = tensor_fields(graph_step)
            problems.extend(
                f"{where}: {name} eager {eager_values[name]} graph {graph_values[name]}"
                for name in eager_values
                if eager_values[name] != graph_values[name]
            )
            problems.extend(
                f"{where}: {name} differs"
                for name in eager_tensors.keys() | graph_tensors.keys()
                if name not in eager_tensors
                or name not in graph_tensors
                or not same_bits(eager_tensors[name], graph_tensors[name])
            )
    return problems


def finite_problems(run: GraphEngineRun) -> list[str]:
    """Non-finite outputs of one run, whatever the other engine recorded; the only
    exception is the eos logit the test masks with -inf."""
    problems = []
    for record in run.forwards:
        for step in record.steps:
            semantic_logits = step.semantic_logits
            is_masked_eos = (torch.arange(semantic_logits.numel()) == EOS_TOKEN_ID) & (
                semantic_logits == float("-inf")
            )
            is_finite = {
                "written key/value cache": bool(step.written_cache.isfinite().all()),
                "head logits": bool(step.head_logits.isfinite().all()),
                "semantic logits": bool(
                    (semantic_logits.isfinite() | is_masked_eos).all()
                ),
                "linear-attention state": step.is_state_finite,
                "final state": all(
                    bool(tensor.isfinite().all()) for tensor in step.final_state or []
                ),
            }
            problems.extend(
                f"{run.kind} {forward_name(record)} {step.request_id}: non-finite {name}"
                for name, is_field_finite in is_finite.items()
                if not is_field_finite
            )
    return problems


def expected_ras_window(history: list[list[int]]) -> list[int]:
    main_tokens = [row[0] for row in history[-RAS_WINDOW_FRAMES:]]
    return [EMPTY_RAS_SLOT] * (RAS_WINDOW_FRAMES - len(main_tokens)) + main_tokens


def as_float32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32))


def history_problems(run: GraphEngineRun, scenarios: list[GraphScenario]) -> list[str]:
    """Inputs the model read against each request's own frame history."""
    sampling_modes = {scenario.name: scenario.sampling_mode for scenario in scenarios}
    histories: dict[str, list[list[int]]] = {}
    problems = []
    for record in run.forwards:
        for step in record.steps:
            request_spec = run.outcomes[step.request_id].request_spec
            history = histories.setdefault(
                step.request_id,
                [list(row) for row in PROMPTS[request_spec.prompt_name]],
            )
            staged_window = expected_ras_window(history)
            expected = {
                "positions": repr(
                    list(range(len(history)))
                    if record.is_extend
                    else [len(history) - 1]
                ),
                "input frames": repr(history if record.is_extend else history[-1:]),
                "staged RAS window": repr(staged_window),
                "model RAS window": repr(
                    staged_window
                    if sampling_modes[record.scenario_name] == "random"
                    else [EMPTY_RAS_SLOT] * RAS_WINDOW_FRAMES
                ),
                "sampling knobs": repr(
                    (
                        as_float32(request_spec.temperature),
                        request_spec.top_k,
                        as_float32(request_spec.top_p),
                    )
                ),
            }
            actual = {
                "positions": repr(step.positions),
                "input frames": repr(step.input_frames),
                "staged RAS window": repr(step.staged_ras_window),
                "model RAS window": repr(step.model_ras_window),
                "sampling knobs": repr((step.temperature, step.top_k, step.top_p)),
            }
            problems.extend(
                f"{forward_name(record)} {step.request_id}: {name} {actual[name]} "
                f"is not {expected[name]}"
                for name in expected
                if actual[name] != expected[name]
            )
            if step.is_discarded:
                pass
            else:
                history.append(step.sampled_frame)
    return problems


def draw_problems(run: GraphEngineRun) -> list[str]:
    """The main token is the RAS draw exactly when a semantic first draw repeats the
    window."""
    return [
        f"{forward_name(record)} {step.request_id}: main token "
        f"{step.sampled_frame[0]} from draws {step.first_draw}, {step.ras_draw} "
        f"and window {step.model_ras_window}"
        for record in run.forwards
        for step in record.steps
        if step.sampled_frame[0]
        != (
            step.ras_draw
            if step.first_draw in step.model_ras_window
            and SEMANTIC_START_TOKEN_ID <= step.first_draw <= SEMANTIC_END_TOKEN_ID
            else step.first_draw
        )
    ]


def resampled_choice_count(run: GraphEngineRun, scenario_name: str) -> int:
    """Steps where a repeated semantic first draw was replaced by a different RAS
    draw."""
    return sum(
        step.first_draw in step.model_ras_window
        and SEMANTIC_START_TOKEN_ID <= step.first_draw <= SEMANTIC_END_TOKEN_ID
        and step.ras_draw != step.first_draw
        for record in run.forwards
        if record.scenario_name == scenario_name
        for step in record.steps
    )


def padded_graph_batch_size(batch_size: int) -> int:
    return min(size for size in CUDA_GRAPH_BATCH_SIZES if size >= batch_size)


def replay_problems(pair: EnginePair) -> list[str]:
    """Eager never replays; the graph engine replays every decode at its padded size."""
    return [
        f"eager {forward_name(record)} replayed a graph"
        for record in pair.eager.forwards
        if record.is_graph_replay
    ] + [
        f"graph {forward_name(record)} did not replay the graph for its batch"
        for record in pair.graph.forwards
        if not record.is_extend
        and not (
            record.is_graph_replay
            and record.graph_batch_size == padded_graph_batch_size(len(record.steps))
        )
    ]


def expected_prefill_bucket(token_count: int) -> int | None:
    fitting_buckets = [
        bucket for bucket in PREFILL_CUDA_GRAPH_TOKEN_COUNTS if bucket >= token_count
    ]
    if not fitting_buckets:
        return None
    elif min(fitting_buckets) > PREFILL_CUDA_GRAPH_PADDING_FACTOR * token_count:
        return None
    else:
        return min(fitting_buckets)


def prefill_path_problems(run: GraphEngineRun) -> list[str]:
    problems = []
    for record in run.forwards:
        expected_bucket = (
            expected_prefill_bucket(record.token_count)
            if record.is_extend and run.kind == "prefill_graph"
            else None
        )
        if not record.is_extend:
            expected = (0, 0, {}, 0)
        elif expected_bucket is None:
            expected = (0, 1, {}, 1)
        else:
            expected = (1, 0, {expected_bucket: 1}, 0)
        actual = (
            record.prefill_replay_count,
            record.prefill_eager_count,
            record.prefill_replay_buckets,
            record.tower_prefill_runs,
        )
        if actual != expected:
            problems.append(
                f"{run.kind} {forward_name(record)}: prefill replays, eager prefills, "
                f"replayed buckets and tower prefill runs {actual}, not {expected}"
            )
        elif record.is_extend and record.is_graph_replay != (
            expected_bucket is not None
        ):
            problems.append(
                f"{run.kind} {forward_name(record)}: graph replay "
                f"{record.is_graph_replay} disagrees with the prefill counters"
            )
        else:
            pass
    return problems


def capture_break_problems(run: GraphEngineRun) -> list[str]:
    facts = run.facts
    backbone_layer_ids = [
        layer_id
        for layer_id in facts.recorded_kv_layer_ids
        if layer_id not in facts.tower_layer_ids
    ]
    expected = {
        token_count: backbone_layer_ids + facts.tower_layer_ids
        for token_count in facts.prefill_capture_token_counts or []
    }
    actual = facts.prefill_capture_break_layer_ids
    return [
        f"{run.kind} prefill graph of {token_count} tokens: attention breaks at "
        f"layers {actual.get(token_count)}, not {expected.get(token_count)}"
        for token_count in sorted(expected.keys() | actual.keys())
        if actual.get(token_count) != expected.get(token_count)
    ]


def stray_write_problems(run: GraphEngineRun) -> list[str]:
    return [
        f"{run.kind} {forward_name(record)} changed KV slots {record.stray_kv_slots} "
        f"and state slots {record.stray_mamba_slots} outside its batch"
        for record in run.forwards
        if record.stray_kv_slots or record.stray_mamba_slots
    ]


def kept_sampled_frames(run: GraphEngineRun) -> dict[str, list[list[int]]]:
    sampled_frames: dict[str, list[list[int]]] = {}
    for record in run.forwards:
        for step in record.steps:
            if step.is_discarded:
                pass
            else:
                sampled_frames.setdefault(step.request_id, []).append(
                    step.sampled_frame
                )
    return sampled_frames


def outcome_problems(run: GraphEngineRun) -> list[str]:
    """One result per finished request, holding its frames; aborts end silently."""
    sampled_frames = kept_sampled_frames(run)
    problems = [
        f"{run.kind} scenario {scenario.name} left free pools at "
        f"{scenario.idle_free_counts}, not {run.baseline_free_counts}"
        for scenario in run.scenarios
        if scenario.idle_free_counts != run.baseline_free_counts
    ]
    for request_id, outcome in run.outcomes.items():
        request_frames = sampled_frames.get(request_id, [])
        returned_frames = (
            request_frames[:-1]
            if request_frames and request_frames[-1][0] == EOS_TOKEN_ID
            else request_frames
        )
        if outcome.aborted_after_frames is not None:
            expected_counts = (0, 0)
        else:
            expected_counts = (1, 0)
        if (len(outcome.result_frames), len(outcome.error_messages)) != (
            expected_counts
        ):
            problems.append(
                f"{run.kind} {request_id}: {len(outcome.result_frames)} results and "
                f"errors {outcome.error_messages}"
            )
        elif outcome.result_frames and outcome.result_frames[0] != returned_frames:
            problems.append(f"{run.kind} {request_id}: result is not its frames")
        else:
            pass
    return problems


def stream_problems(run: GraphEngineRun) -> list[str]:
    window = Drama3VocoderFactoryArgs()
    sampled_frames = kept_sampled_frames(run)
    problems = []
    for request_id, outcome in run.outcomes.items():
        request_frames = sampled_frames.get(request_id, [])
        audio_frames = (
            request_frames[:-1]
            if request_frames and request_frames[-1][0] == EOS_TOKEN_ID
            else request_frames
        )
        if outcome.aborted_after_frames is None:
            frame_count = len(audio_frames)
        else:
            frame_count = outcome.aborted_after_frames
        expected_chunk_ends = sorted(
            {
                (
                    sampled_count
                    if sampled_count <= window.stream_stride
                    else frames_in_complete_windows(
                        sampled_count,
                        stream_stride=window.stream_stride,
                        stream_followup_stride_ramp=window.stream_followup_stride_ramp,
                        stream_followup_stride=window.stream_followup_stride,
                    )
                )
                for sampled_count in range(1, frame_count + 1)
            }
        )
        chunk_ends = [
            sum(len(chunk) for chunk in outcome.streamed_frame_chunks[: index + 1])
            for index in range(len(outcome.streamed_frame_chunks))
        ]
        streamed = [frame for chunk in outcome.streamed_frame_chunks for frame in chunk]
        if not outcome.request_spec.is_streaming:
            if streamed:
                problems.append(f"{run.kind} {request_id}: streamed without stream")
            else:
                pass
        elif chunk_ends != expected_chunk_ends:
            problems.append(
                f"{run.kind} {request_id}: chunks end at {chunk_ends}, not at "
                f"{expected_chunk_ends} of {frame_count} frames"
            )
        elif streamed != audio_frames[: len(streamed)]:
            problems.append(
                f"{run.kind} {request_id}: streamed frames are not its first "
                f"{len(streamed)} sampled frames"
            )
        else:
            pass
    return problems


def first_frame_path_problems(run: GraphEngineRun) -> list[str]:
    captured_request_counts = set(run.facts.first_frame_request_counts or [])
    problems = []
    for record in run.forwards:
        request_count = len(record.steps)
        if not record.is_extend:
            expected = ({}, 0, 0)
        elif request_count in captured_request_counts:
            expected = ({request_count: 1}, 0, 0)
        else:
            expected = ({}, 1, 1)
        actual = (
            record.first_frame_replays,
            record.first_frame_eager_count,
            record.first_frame_sampler_calls,
        )
        if actual != expected:
            problems.append(
                f"{run.kind} {forward_name(record)}: first-frame replays, eager steps "
                f"and sampler calls {actual}, not {expected}"
            )
        else:
            pass
    return problems


def first_frame_split_problems(pair: EnginePair) -> list[str]:
    return [
        f"{forward_name(graph_record)}: first-frame KV splits eager "
        f"{eager_record.first_frame_kv_splits} graph "
        f"{graph_record.first_frame_kv_splits}"
        for eager_record, graph_record in zip(
            pair.eager.forwards, pair.graph.forwards, strict=True
        )
        if eager_record.first_frame_kv_splits != graph_record.first_frame_kv_splits
    ]


def generator_offset_problems(
    pair: EnginePair, scenario_names: set[str] | None
) -> list[str]:
    return [
        f"{forward_name(graph_record)}: generator offset eager "
        f"{eager_record.generator_offset} graph {graph_record.generator_offset}"
        for eager_record, graph_record in zip(
            pair.eager.forwards, pair.graph.forwards, strict=True
        )
        if (scenario_names is None or graph_record.scenario_name in scenario_names)
        and eager_record.generator_offset != graph_record.generator_offset
    ]


def sampling_step_problems(run: GraphEngineRun, is_triton_sampling: bool) -> list[str]:
    steps_per_forward = 1 if is_triton_sampling else 0
    sampler_problems = (
        []
        if run.facts.uses_triton_sampling == is_triton_sampling
        else [
            f"{run.kind} engine: uses_triton_sampling is "
            f"{run.facts.uses_triton_sampling}, expected {is_triton_sampling}"
        ]
    )
    return sampler_problems + [
        f"{forward_name(record)}: sampling step {record.sampling_step}, expected "
        f"{steps_per_forward * (record.forward_index + 1)}"
        for record in run.forwards
        if record.sampling_step != steps_per_forward * (record.forward_index + 1)
    ]


def fused_layer_problems(run: GraphEngineRun, is_fused: bool) -> list[str]:
    tower_layer_count = len(run.facts.tower_layer_ids)
    problems = (
        []
        if run.facts.uses_fused_tower_layers == is_fused
        else [
            f"{run.kind} engine: uses_fused_tower_layers is "
            f"{run.facts.uses_fused_tower_layers}, expected {is_fused}"
        ]
    )
    max_rows = run.facts.fused_max_rows
    for record in run.forwards:
        if record.is_extend:
            runs_decode_step_eagerly = record.first_frame_eager_count == 1
        else:
            runs_decode_step_eagerly = not record.is_graph_replay
        is_covered = max_rows is not None and len(record.steps) <= max_rows
        expected_calls = (
            tower_layer_count
            if is_fused and runs_decode_step_eagerly and is_covered
            else 0
        )
        if record.tower_prefill_fused_layer_calls != 0:
            problems.append(
                f"{forward_name(record)}: {record.tower_prefill_fused_layer_calls} "
                f"fused layer calls in the tower prefill"
            )
        elif record.fused_layer_calls != expected_calls:
            problems.append(
                f"{forward_name(record)}: {record.fused_layer_calls} fused layer "
                f"calls, expected {expected_calls}"
            )
        else:
            pass
    return problems


def lean_problems(run: GraphEngineRun) -> list[str]:
    facts = run.facts
    problems = (
        []
        if facts.lean_attention_override is None
        else [
            f"{run.kind} engine sets Lean attention to {facts.lean_attention_override}"
        ]
    )
    problems.extend(
        f"{run.kind} startup chose Lean for a batch of {choice.batch_size} by its "
        f"{choice.policy} policy"
        for choice in facts.startup_lean_choices
        if choice.is_lean
    )
    problems.extend(
        f"{run.kind} {forward_name(record)} chose Lean for a batch of "
        f"{choice.batch_size} by its {choice.policy} policy"
        for record in run.forwards
        for choice in record.lean_choices
        if choice.is_lean
    )
    return problems


def lookahead_problems(run: GraphEngineRun, is_lookahead: bool) -> list[str]:
    facts = run.facts
    switches = (
        facts.enable_async_decode,
        facts.async_decode_min_batch_size,
        facts.async_decode_resolve_before_prefill,
    )
    problems = []
    if switches != (is_lookahead, 1, True):
        problems.append(
            f"{run.kind} async decode switches {switches}, expected "
            f"{(is_lookahead, 1, True)}"
        )
    else:
        pass
    follows_a_discarded_step: dict[str, bool] = {}
    for record in run.forwards:
        if not record.is_extend and record.is_lookahead != is_lookahead:
            problems.append(
                f"{run.kind} {forward_name(record)}: lookahead is "
                f"{record.is_lookahead}, expected {is_lookahead}"
            )
        else:
            pass
        for step in record.steps:
            if step.is_discarded and not is_lookahead:
                problems.append(
                    f"{run.kind} {forward_name(record)} {step.request_id}: "
                    f"discarded without lookahead"
                )
            elif (
                follows_a_discarded_step.get(step.request_id, False)
                and not record.is_extend
            ):
                problems.append(
                    f"{run.kind} {forward_name(record)} {step.request_id}: a decode "
                    f"step follows a discarded one"
                )
            else:
                pass
            follows_a_discarded_step[step.request_id] = step.is_discarded
    return problems
