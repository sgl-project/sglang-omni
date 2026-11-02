# SPDX-License-Identifier: Apache-2.0
"""Drama 3 tower cache and four-stage steps on a GPU: paged cache, sampled codes,
backbone and scheduler bookkeeping against a dense fp32 recompute and single-request
runs."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.model_runner import ras_window
from sglang_omni.models.fishaudio_drama3.weights import load_audio_decoder_weights
from tests.unit_test.fishaudio_drama3.dense_attention import DenseAttention
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    ABORT_AFTER_FRAMES,
    ABORT_MAX_NEW_TOKENS,
    CHECKPOINT_CONFIG,
    CONTEXT_LENGTH,
    DECODER_CONFIG,
    EOS_TOKEN_ID,
    MAIN_SCENARIOS,
    MAX_RUNNING_REQUESTS,
    MAX_TOTAL_TOKENS,
    PROMPTS,
    SEMANTIC_END_TOKEN_ID,
    SEMANTIC_START_TOKEN_ID,
    TEST_RETRACT_SCENARIOS,
    TEXT_CONFIG,
    TOTAL_GPU_MEMORY_FRACTION,
    EngineRun,
    EngineSpec,
    RequestStep,
    ScenarioOutcome,
    decoder_checkpoint,
    write_synthetic_checkpoint,
)
from tests.unit_test.fishaudio_drama3.test_audio_decoder import torch_fused_op_backend
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.scheduling.test_hybrid_gdn import (
    kill_process_group,
    require_free_gpu_memory,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_MODULE = "tests.unit_test.fishaudio_drama3.mtp_engines"
ENGINE_TIMEOUT_SECONDS = 1200
TEST_RETRACT_INTERVAL = 5
# Measured on this fixture (one calibration run): twice the dense tower's
# bfloat16-vs-fp32 deviation over the single-request inputs, per cache layer
# and per head, and twice the torch backbone's bfloat16-vs-fp32 deviation.
TOWER_CACHE_TOLERANCES = (
    0.042,
    0.053,
    0.079,
    0.10,
    0.14,
    0.15,
    0.25,
    0.27,
    0.35,
    0.42,
    0.44,
    0.54,
    0.65,
)
HEAD_LOGIT_TOLERANCES = (0.52, 0.63, 0.89, 0.94, 1.3, 1.36, 1.34, 2.1, 2.6)
SEMANTIC_LOGIT_GAP = 3.4
HIDDEN_MEAN_TOLERANCE = 0.1


# How a run's first split from a reference run is judged: the choice must sit
# within the gap limit of the reference's best logit, and for top_two_within_gap
# also among the reference's two best.
DecisionRule = Literal["top_two_within_gap", "within_gap"]


@dataclass(frozen=True, kw_only=True)
class BackboneTolerances:
    """How far a run may sit from its reference runs, and how close a split decision."""

    hidden_mean: float
    semantic_gap: float
    head_gaps: tuple[float, ...]


SAME_ENGINE_TOLERANCES = BackboneTolerances(
    hidden_mean=HIDDEN_MEAN_TOLERANCE,
    semantic_gap=SEMANTIC_LOGIT_GAP,
    head_gaps=HEAD_LOGIT_TOLERANCES,
)


@dataclass(frozen=True, kw_only=True)
class EngineRuns:
    main: EngineRun
    test_retract: EngineRun
    is_lookahead: bool
    workdir: Path


def run_engine(
    name: str, spec: EngineSpec, workdir: Path, extra_environment: dict[str, str]
) -> EngineRun:
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
    assert isinstance(run, EngineRun)
    return run


@pytest.fixture(scope="module", params=[False, True], ids=["sync", "lookahead"])
def engine_runs(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> EngineRuns:
    require_cuda()
    is_lookahead = request.param
    assert isinstance(is_lookahead, bool)
    workdir = tmp_path_factory.mktemp("drama3_mtp")
    checkpoint_dir = workdir / "checkpoint"
    write_synthetic_checkpoint(checkpoint_dir)
    spec = EngineSpec(
        model_dir=str(checkpoint_dir),
        context_length=CONTEXT_LENGTH,
        max_running_requests=MAX_RUNNING_REQUESTS,
        max_prefill_tokens=CONTEXT_LENGTH,
        max_total_tokens=MAX_TOTAL_TOKENS,
        total_gpu_memory_fraction=TOTAL_GPU_MEMORY_FRACTION,
        prompts=PROMPTS,
        scenarios=MAIN_SCENARIOS,
        enable_async_decode=is_lookahead,
    )
    # Serial on purpose: each engine owns its process and its share of the GPU.
    main = run_engine("main", spec, workdir, {})
    test_retract = run_engine(
        "test_retract",
        spec.model_copy(update={"scenarios": TEST_RETRACT_SCENARIOS}),
        workdir,
        {
            "SGLANG_TEST_RETRACT": "1",
            "SGLANG_TEST_RETRACT_INTERVAL": str(TEST_RETRACT_INTERVAL),
        },
    )
    print(f"Artifacts: {workdir}")
    return EngineRuns(
        main=main,
        test_retract=test_retract,
        is_lookahead=is_lookahead,
        workdir=workdir,
    )


@pytest.fixture(scope="module")
def reference_decoder() -> AudioDecoder:
    """The tower in fp32 on CPU, loaded from the synthetic checkpoint's tensors."""
    decoder = AudioDecoder(FishOmniConfig(**CHECKPOINT_CONFIG))
    load_audio_decoder_weights(decoder, decoder_checkpoint().items())
    return decoder.float().eval()


@dataclass(frozen=True, kw_only=True)
class DenseTower:
    """A dense fp32 recompute of one request's tower rows 0..rows-1."""

    cache: torch.Tensor
    head_logits: torch.Tensor


@torch.no_grad()
@torch_fused_op_backend()
def dense_tower(
    decoder: AudioDecoder, hidden_states: torch.Tensor, frame_history: torch.Tensor
) -> DenseTower:
    """Row p sees rows <= p and is labelled with frame p + 1, as in the paged cache."""
    row_count = hidden_states.shape[0]
    positions = torch.arange(row_count)
    input_frames = frame_history[:row_count]
    labels = frame_history[1 : row_count + 1]
    attention = DenseAttention(
        head_count=int(DECODER_CONFIG["n_head"]),
        kv_head_count=int(DECODER_CONFIG["n_local_heads"]),
        head_dim=int(DECODER_CONFIG["head_dim"]),
    )
    attention.bind(["request"] * row_count, positions.tolist())
    label_is_semantic = decoder.is_semantic(labels[:, 0])
    tower_states = decoder.run_shared_layers(
        hidden_states,
        labels[:, 1],
        label_is_semantic,
        decoder.shared_input(input_frames),
        positions,
        attention,
    )
    head_logits = []
    shared_layer_count = len(decoder.shared_layers)
    for residual_index, (layer, head) in enumerate(
        zip(decoder.residual_layers, decoder.codebook_heads, strict=True)
    ):
        tower_states = layer(
            tower_states,
            labels[:, 1 + residual_index],
            label_is_semantic,
            None,
            positions,
            attention,
            shared_layer_count + residual_index,
        )
        head_logits.append(head(tower_states))
    tower_layer_count = shared_layer_count + len(decoder.residual_layers)
    cache = torch.stack(
        [
            torch.stack(
                [
                    torch.stack((key.flatten(), value.flatten()))
                    for _, key, value in attention.entries[(layer_index, "request")]
                ]
            )
            for layer_index in range(tower_layer_count)
        ]
    )
    return DenseTower(cache=cache, head_logits=torch.stack(head_logits, dim=1))


def request_steps(run: EngineRun) -> dict[str, list[RequestStep]]:
    discarded_steps = set(run.discarded_steps)
    steps: dict[str, list[RequestStep]] = {}
    for forward in run.forwards:
        for step in forward.steps:
            if (forward.forward_index, step.request_id) in discarded_steps:
                pass
            else:
                steps.setdefault(step.request_id, []).append(step)
    return steps


def frame_history(steps: list[RequestStep]) -> torch.Tensor:
    """Prompt rows, then every frame the request sampled, in order."""
    prompt_rows = steps[0].input_frames
    sampled = torch.tensor([step.sampled_frame for step in steps], dtype=torch.int64)
    return torch.cat((prompt_rows, sampled))


def epochs(steps: list[RequestStep]) -> list[list[RequestStep]]:
    """Steps between prefills; a retracted request starts a new epoch on re-prefill."""
    grouped: list[list[RequestStep]] = []
    for step in steps:
        if step.is_extend:
            grouped.append([step])
        else:
            grouped[-1].append(step)
    return grouped


def test_pool_holds_the_tower_layers_after_the_backbone(
    engine_runs: EngineRuns,
) -> None:
    config = FishOmniConfig(**CHECKPOINT_CONFIG)
    text_config = config.text_config
    tower_layer_count = config.text_config.tower_layer_count
    facts = engine_runs.main.facts

    backbone_full_ids = [
        layer_index
        for layer_index, layer_type in enumerate(TEXT_CONFIG["layer_types"])
        if layer_type == "full_attention"
    ]
    tower_ids = list(
        range(
            text_config.num_hidden_layers,
            text_config.num_hidden_layers + tower_layer_count,
        )
    )
    assert facts.full_attention_layer_ids == backbone_full_ids + tower_ids
    assert facts.mamba_layer_ids == text_config.linear_layer_ids
    assert facts.kv_layer_count == len(backbone_full_ids) + tower_layer_count
    assert (facts.kv_head_count, facts.kv_head_dim) == (
        text_config.num_key_value_heads,
        text_config.head_dim,
    )
    assert facts.kv_element_bytes == torch.bfloat16.itemsize
    assert facts.chunked_prefill_size is None
    assert facts.max_prefill_tokens == CONTEXT_LENGTH
    assert facts.max_running_requests == MAX_RUNNING_REQUESTS


def scenario_outcome(run: EngineRun, name: str) -> ScenarioOutcome:
    return next(scenario for scenario in run.scenarios if scenario.name == name)


def tower_problems(run: EngineRun, decoder: AudioDecoder) -> list[str]:
    """Every step's cache rows, head logits and frame against the dense tower."""
    problems = []
    candidate_token_ids = decoder.main_token_candidate_ids
    codebook_size = int(DECODER_CONFIG["vocab_size"])
    for request_id, steps in request_steps(run).items():
        history = frame_history(steps)
        for epoch in epochs(steps):
            positions = [position for step in epoch for position in step.positions]
            hidden_states = torch.cat([step.hidden_states for step in epoch]).float()
            if positions != list(range(len(positions))):
                problems.append(f"{request_id}: epoch rows {positions} skip a row")
                continue
            elif not bool(hidden_states.isfinite().all()):
                problems.append(f"{request_id}: non-finite backbone hidden states")
                continue
            else:
                pass
            reference = dense_tower(decoder, hidden_states, history)
            for step in epoch:
                newest_row = step.positions[-1]
                caches = [(step.written_cache, torch.tensor(step.positions))]
                if step.sequence_cache is not None:
                    caches.append((step.sequence_cache, torch.arange(newest_row + 1)))
                else:
                    pass
                for cache, rows in caches:
                    deviations = (
                        (cache.float() - reference.cache[:, rows])
                        .abs()
                        .amax(dim=(1, 2, 3))
                    )
                    for layer_index, deviation in enumerate(deviations.tolist()):
                        # Written this way round, a NaN deviation fails too.
                        if not deviation <= TOWER_CACHE_TOLERANCES[layer_index]:
                            problems.append(
                                f"{request_id} row {newest_row}: tower layer "
                                f"{layer_index} cache is {deviation:.3f} off"
                            )
                        else:
                            pass
                head_deviations = (
                    (step.head_logits - reference.head_logits[newest_row])
                    .abs()
                    .amax(dim=1)
                )
                for head_index, deviation in enumerate(head_deviations.tolist()):
                    if not deviation <= HEAD_LOGIT_TOLERANCES[head_index]:
                        problems.append(
                            f"{request_id} row {newest_row}: head {head_index} logits "
                            f"are {deviation:.3f} off"
                        )
                    else:
                        pass
                main_token, codebook_zero, *codes = step.sampled_frame
                semantic_logits = step.semantic_logits
                # The test masks eos with -inf; any other non-finite logit is a fault.
                is_masked_eos = (
                    torch.arange(semantic_logits.numel()) == EOS_TOKEN_ID
                ) & (semantic_logits == float("-inf"))
                expected_main_token = int(
                    candidate_token_ids[
                        torch.argmax(semantic_logits[candidate_token_ids])
                    ]
                )
                expected_codebook_zero = min(
                    max(main_token - SEMANTIC_START_TOKEN_ID, 0), codebook_size - 1
                )
                if not bool((semantic_logits.isfinite() | is_masked_eos).all()):
                    problems.append(
                        f"{request_id} row {newest_row}: non-finite semantic logits"
                    )
                elif main_token != expected_main_token:
                    problems.append(
                        f"{request_id} row {newest_row}: main token {main_token} is not "
                        f"the argmax {expected_main_token}"
                    )
                elif codebook_zero != expected_codebook_zero:
                    problems.append(
                        f"{request_id} row {newest_row}: code 0 {codebook_zero} does not "
                        f"follow main token {main_token}"
                    )
                elif codes != step.head_logits.argmax(dim=1).tolist():
                    problems.append(
                        f"{request_id} row {newest_row}: codes {codes} are not the "
                        f"argmax of the engine's head logits"
                    )
                else:
                    pass
    return problems


def reused_slot_count(run: EngineRun) -> int:
    """How many slot writes landed on a slot an earlier request had written."""
    slot_owners: dict[int, str] = {}
    reuse_count = 0
    for forward in run.forwards:
        for step in forward.steps:
            for slot in step.written_slots:
                if slot_owners.get(slot, step.request_id) != step.request_id:
                    reuse_count += 1
                else:
                    pass
                slot_owners[slot] = step.request_id
    return reuse_count


def slot_problems(run: EngineRun) -> list[str]:
    """Written slots are the request's slots at those positions; no slot is shared."""
    problems = []
    for forward in run.forwards:
        live_slots: list[int] = []
        for step in forward.steps:
            live_slots.extend(step.sequence_slots)
            expected = [step.sequence_slots[position] for position in step.positions]
            if step.written_slots != expected:
                problems.append(
                    f"forward {forward.forward_index} {step.request_id}: wrote slots "
                    f"{step.written_slots}, the sequence has {expected} there"
                )
            else:
                pass
        if len(set(live_slots)) != len(live_slots):
            problems.append(f"forward {forward.forward_index}: requests share KV slots")
        else:
            pass
    return problems


def ras_window_problems(run: EngineRun) -> list[str]:
    """The runner's window is the last 16 main tokens of the history it staged."""
    problems = []
    for request_id, steps in request_steps(run).items():
        history = frame_history(steps).tolist()
        for step in steps:
            expected = ras_window(history[: step.positions[-1] + 1])
            if step.ras_window != expected:
                problems.append(
                    f"{request_id} row {step.positions[-1]}: window {step.ras_window}, "
                    f"expected {expected}"
                )
            else:
                pass
    return problems


def outcome_problems(run: EngineRun) -> list[str]:
    """One result per finished request, holding every sampled frame once, eos out."""
    problems = []
    steps_by_request = request_steps(run)
    for request_id, outcome in run.outcomes.items():
        request_spec = outcome.request_spec
        steps = steps_by_request.get(request_id, [])
        prompt_row_count = len(PROMPTS[request_spec.prompt_name])
        sampled = frame_history(steps)[prompt_row_count:] if steps else torch.empty(0)
        if request_spec.end_at_frame is None:
            expected_frames = sampled.tolist()
            expected_count = request_spec.max_new_tokens
        else:
            expected_frames = sampled[: request_spec.end_at_frame].tolist()
            expected_count = request_spec.end_at_frame
        if outcome.error_messages:
            problems.append(f"{request_id}: errors {outcome.error_messages}")
        elif not steps:
            problems.append(f"{request_id}: no forward ever ran it")
        elif outcome.aborted:
            if outcome.result_frames:
                problems.append(f"{request_id}: aborted but got a result")
            else:
                pass
        elif len(outcome.result_frames) != 1:
            problems.append(f"{request_id}: {len(outcome.result_frames)} results")
        elif outcome.result_frames[0] != expected_frames:
            problems.append(
                f"{request_id}: the result is not the {len(expected_frames)} sampled "
                f"frames"
            )
        elif len(expected_frames) != expected_count:
            problems.append(
                f"{request_id}: {len(expected_frames)} frames, expected {expected_count}"
            )
        else:
            pass
    return problems


def single_request_runs(run: EngineRun) -> dict[str, list[RequestStep]]:
    """The steps of each prompt's run in the single scenario, by prompt name."""
    steps_by_request = request_steps(run)
    return {
        run.outcomes[request_id].request_spec.prompt_name: steps_by_request[request_id]
        for request_id in scenario_outcome(run, "single").request_ids
    }


@dataclass(frozen=True, kw_only=True)
class BackboneComparison:
    problems: list[str]
    compared_request_ids: list[str]


def backbone_comparison(
    run: EngineRun,
    single_runs: dict[str, list[RequestStep]],
    tolerances: BackboneTolerances,
    *,
    is_same_engine: bool,
    decision_rule: DecisionRule,
) -> BackboneComparison:
    """Each request against the single-request run of its prompt.

    Prefill rows see the same inputs, so their hidden states must agree. Frames
    are compared up to the first complete frame that differs: its history is
    still shared, so the first differing entry must be a near tie in the
    single run's own logits; after it the histories differ and comparing stops.
    A request without a single run to compare with is a problem, not a skip.
    When the single runs come from run's own engine, they are not compared with
    themselves; from another engine, every request is compared, same ids too.
    """
    problems = []
    compared_request_ids = []
    steps_by_request = request_steps(run)
    if is_same_engine:
        reference_request_ids = {steps[0].request_id for steps in single_runs.values()}
    else:
        reference_request_ids = set()
    semantic_bias = torch.full((int(TEXT_CONFIG["vocab_size"]),), float("-inf"))
    semantic_bias[SEMANTIC_START_TOKEN_ID : SEMANTIC_END_TOKEN_ID + 1] = 0.0
    semantic_bias[EOS_TOKEN_ID] = 0.0
    for request_id, steps in steps_by_request.items():
        request_spec = run.outcomes[request_id].request_spec
        single_steps = single_runs.get(request_spec.prompt_name)
        if request_id in reference_request_ids:
            continue
        elif single_steps is None:
            problems.append(
                f"{request_id}: no single-request run of {request_spec.prompt_name} "
                f"to compare with"
            )
            continue
        else:
            pass
        compared_request_ids.append(request_id)
        prompt_row_count = len(PROMPTS[request_spec.prompt_name])
        single_prefill = single_steps[0].hidden_states.float()
        for step in steps:
            if step.is_extend:
                deviation = (
                    (step.hidden_states[:prompt_row_count].float() - single_prefill)
                    .abs()
                    .mean()
                    .item()
                )
                if not deviation <= tolerances.hidden_mean:
                    problems.append(
                        f"{request_id}: prefill hidden states are {deviation:.4f} off "
                        f"the single run on average"
                    )
                else:
                    pass
            else:
                pass
        sampled = frame_history(steps)[prompt_row_count:]
        single_sampled = frame_history(single_steps)[prompt_row_count:]
        compared_count = min(len(sampled), len(single_sampled))
        if request_spec.end_at_frame is not None:
            compared_count = min(compared_count, request_spec.end_at_frame)
        else:
            pass
        for frame_index in range(compared_count):
            differing = (sampled[frame_index] != single_sampled[frame_index]).nonzero()
            if len(differing) == 0:
                continue
            else:
                pass
            entry = int(differing[0])
            chosen = int(sampled[frame_index][entry])
            single_step = single_steps[frame_index]
            if entry == 0:
                logits = single_step.semantic_logits + semantic_bias
                gap_limit = tolerances.semantic_gap
            elif entry >= 2:
                logits = single_step.head_logits[entry - 2]
                gap_limit = tolerances.head_gaps[entry - 2]
            else:
                problems.append(
                    f"{request_id} frame {frame_index}: code 0 alone differs"
                )
                break
            reason = split_decision_problem(logits, chosen, gap_limit, decision_rule)
            if reason is not None:
                problems.append(
                    f"{request_id} frame {frame_index} entry {entry}: {reason}"
                )
            else:
                pass
            break
    return BackboneComparison(
        problems=problems, compared_request_ids=compared_request_ids
    )


def split_decision_problem(
    logits: torch.Tensor, chosen: int, gap_limit: float, rule: DecisionRule
) -> str | None:
    """Why the reference's logits do not explain the chosen entry; None when they do."""
    if not 0 <= chosen < logits.numel():
        return f"chose {chosen}, outside the {logits.numel()} candidates"
    else:
        pass
    top = torch.topk(logits, 2)
    gap = float(logits.max() - logits[chosen])
    rank = int((logits > logits[chosen]).sum()) + 1
    if rule == "top_two_within_gap" and chosen not in top.indices.tolist():
        return (
            f"chose {chosen}, the single run's top-2 is {top.indices.tolist()} "
            f"{[round(value, 3) for value in top.values.tolist()]}"
        )
    # Written this way round, a NaN gap fails too.
    elif not gap <= gap_limit:
        return (
            f"chose {chosen} (rank {rank}), {gap:.4f} under the reference's best "
            f"logit; the limit is {gap_limit}"
        )
    else:
        return None


def backbone_problems(
    run: EngineRun, single_runs: dict[str, list[RequestStep]]
) -> list[str]:
    """Each request of run against its own engine's single-request runs."""
    return backbone_comparison(
        run,
        single_runs,
        SAME_ENGINE_TOLERANCES,
        is_same_engine=True,
        decision_rule="top_two_within_gap",
    ).problems


def test_tower_cache_and_codes_match_the_dense_recompute(
    engine_runs: EngineRuns, reference_decoder: AudioDecoder
) -> None:
    """Every written slot, every head and every frame of both engines, per request."""
    problems = tower_problems(engine_runs.main, reference_decoder) + tower_problems(
        engine_runs.test_retract, reference_decoder
    )
    assert not problems, "\n".join(problems[:40])


def test_written_slots_belong_to_their_request(engine_runs: EngineRuns) -> None:
    """Slots get reused across requests, so a stale row would be visible."""
    problems = slot_problems(engine_runs.main) + slot_problems(engine_runs.test_retract)
    assert not problems, "\n".join(problems[:40])
    assert reused_slot_count(engine_runs.main) > 0


def test_ras_windows_follow_the_frame_history(engine_runs: EngineRuns) -> None:
    """Including the re-prefills after retraction: rebuilt, never advanced twice."""
    problems = ras_window_problems(engine_runs.main) + ras_window_problems(
        engine_runs.test_retract
    )
    assert not problems, "\n".join(problems[:40])


def test_every_request_finishes_once_with_its_frames(engine_runs: EngineRuns) -> None:
    problems = outcome_problems(engine_runs.main) + outcome_problems(
        engine_runs.test_retract
    )
    assert not problems, "\n".join(problems[:40])


def lookahead_problems(run: EngineRun, is_lookahead: bool) -> list[str]:
    facts = run.facts
    switches = (
        facts.enable_async_decode,
        facts.async_decode_min_batch_size,
        facts.async_decode_resolve_before_prefill,
    )
    decode_forward_indices = [
        forward.forward_index for forward in run.forwards if not forward.is_extend
    ]
    problems = []
    if switches != (is_lookahead, 1, True):
        problems.append(
            f"async decode switches {switches}, expected {(is_lookahead, 1, True)}"
        )
    else:
        pass
    if run.lookahead_forward_indices != (
        decode_forward_indices if is_lookahead else []
    ):
        problems.append(
            f"{len(run.lookahead_forward_indices)} of {len(decode_forward_indices)} "
            f"decode forwards launched ahead"
        )
    else:
        pass
    discarded_steps = set(run.discarded_steps)
    follows_a_discarded_step: dict[str, bool] = {}
    for forward in run.forwards:
        for step in forward.steps:
            if (
                follows_a_discarded_step.get(step.request_id, False)
                and not step.is_extend
            ):
                problems.append(
                    f"forward {forward.forward_index} {step.request_id}: a decode "
                    f"step follows a discarded one"
                )
            else:
                pass
            follows_a_discarded_step[step.request_id] = (
                forward.forward_index,
                step.request_id,
            ) in discarded_steps
    return problems


def test_lookahead_engines_launch_each_decode_ahead_and_discard_only_ends(
    engine_runs: EngineRuns,
) -> None:
    for run in (engine_runs.main, engine_runs.test_retract):
        assert not lookahead_problems(run, engine_runs.is_lookahead)
        assert lookahead_problems(run, not engine_runs.is_lookahead)
    run = engine_runs.main
    assert bool(run.discarded_steps) == engine_runs.is_lookahead
    decode_forward_indices: dict[str, list[int]] = {}
    for forward in run.forwards:
        for step in forward.steps:
            if forward.is_extend:
                pass
            else:
                decode_forward_indices.setdefault(step.request_id, []).append(
                    forward.forward_index
                )
    request_id, forward_indices = max(
        decode_forward_indices.items(), key=lambda item: len(item[1])
    )
    wrongly_discarded = replace(
        run, discarded_steps=[*run.discarded_steps, (forward_indices[0], request_id)]
    )
    assert lookahead_problems(wrongly_discarded, engine_runs.is_lookahead)


def test_backbone_matches_single_request_runs(engine_runs: EngineRuns) -> None:
    """Both engines against the main engine's single runs of the same prompts."""
    single_runs = single_request_runs(engine_runs.main)
    problems = backbone_problems(engine_runs.main, single_runs) + backbone_problems(
        engine_runs.test_retract, single_runs
    )
    assert not problems, "\n".join(problems[:40])


def test_pools_return_to_baseline_and_full_waves_run_together(
    engine_runs: EngineRuns,
) -> None:
    for run in (engine_runs.main, engine_runs.test_retract):
        for scenario in run.scenarios:
            assert scenario.idle_free_counts == run.baseline_free_counts, scenario.name
    for name in ("full_wave_1", "full_wave_2", "after_abort"):
        scenario = scenario_outcome(engine_runs.main, name)
        decode_batch_sizes = [
            len(forward.steps)
            for forward in engine_runs.main.forwards[
                scenario.first_forward_index : scenario.end_forward_index
            ]
            if not forward.is_extend
        ]
        assert max(decode_batch_sizes) == MAX_RUNNING_REQUESTS, name


def test_abort_stops_one_request_and_frees_its_slots(engine_runs: EngineRuns) -> None:
    run = engine_runs.main
    abort = scenario_outcome(run, "abort")
    after_abort = scenario_outcome(run, "after_abort")
    target_id = abort.request_ids[0]
    target = run.outcomes[target_id]
    target_steps = request_steps(run)[target_id]
    assert target.aborted
    assert abort.frames_at_abort is not None
    assert abort.frames_at_abort >= ABORT_AFTER_FRAMES
    assert abort.results_at_abort == 0
    assert len(target_steps) < ABORT_MAX_NEW_TOKENS
    later_steps = [
        step
        for forward in run.forwards[abort.end_forward_index :]
        for step in forward.steps
    ]
    assert target_steps[-1].request_row in {
        step.request_row
        for step in later_steps
        if step.request_id in after_abort.request_ids
    }
    assert set(target_steps[-1].sequence_slots) & {
        slot for step in later_steps for slot in step.written_slots
    }


def test_retraction_reprefills_the_whole_history(engine_runs: EngineRuns) -> None:
    """Both entry points: an admin pause with retract, and the scheduler's own retract."""
    admin = scenario_outcome(engine_runs.main, "admin_retract")
    assert admin.retracted_request_count is not None
    assert admin.retracted_request_count >= 2
    for run, scenario_name in (
        (engine_runs.main, "admin_retract"),
        (engine_runs.test_retract, "test_retract"),
    ):
        scenario = scenario_outcome(run, scenario_name)
        reprefilled = 0
        for request_id in scenario.request_ids:
            steps = request_steps(run)[request_id]
            prompt_row_count = len(
                PROMPTS[run.outcomes[request_id].request_spec.prompt_name]
            )
            for step_index, step in enumerate(steps):
                if step.is_extend and step_index > 0:
                    assert step.positions == list(range(prompt_row_count + step_index))
                    reprefilled += 1
                else:
                    pass
        assert reprefilled > 0, scenario_name


def test_admission_defers_whole_prompts(engine_runs: EngineRuns) -> None:
    """Prompts past the prefill budget wait for the next batch instead of splitting."""
    run = engine_runs.main
    scenario = scenario_outcome(run, "admission")
    first_extend_forwards = []
    for request_id in scenario.request_ids:
        prompt_row_count = len(
            PROMPTS[run.outcomes[request_id].request_spec.prompt_name]
        )
        forward_index, first_step = next(
            (forward.forward_index, step)
            for forward in run.forwards
            for step in forward.steps
            if step.request_id == request_id
        )
        assert first_step.is_extend
        assert first_step.positions == list(range(prompt_row_count))
        first_extend_forwards.append(forward_index)
    assert len(set(first_extend_forwards)) > 1


def test_end_of_speech_frame_is_written_but_not_returned(
    engine_runs: EngineRuns,
) -> None:
    run = engine_runs.main
    scenario = scenario_outcome(run, "end_of_speech")
    for request_id in scenario.request_ids:
        end_at_frame = run.outcomes[request_id].request_spec.end_at_frame
        assert end_at_frame is not None
        steps = request_steps(run)[request_id]
        assert len(steps) == end_at_frame + 1
        last_step = steps[-1]
        assert last_step.sampled_frame[0] == EOS_TOKEN_ID
        # The cache of the end-of-speech step is checked against the dense tower
        # with every other step; here: it was kept for the whole sequence.
        assert last_step.sequence_cache is not None
        assert last_step.sequence_cache.shape[1] == last_step.positions[-1] + 1
        assert len(run.outcomes[request_id].result_frames[0]) == end_at_frame
