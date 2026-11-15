# SPDX-License-Identifier: Apache-2.0
"""Drama 3 with block-fp8 routed experts on a GPU, against those weights dequantized.

Loading, each expert call against a torch emulation, and engine outputs are checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
import triton
import triton.language as tl

from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.weights import load_audio_decoder_weights
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    CHECKPOINT_CONFIG,
    CONTEXT_LENGTH,
    FP8_BLOCK_SIZE,
    FP8_MAX,
    MAIN_SCENARIOS,
    MAX_RUNNING_REQUESTS,
    MAX_TOTAL_TOKENS,
    PROMPTS,
    TEXT_CONFIG,
    TOTAL_GPU_MEMORY_FRACTION,
    EngineRun,
    EngineSpec,
    LayerQuantMethod,
    MoeCall,
    decoder_checkpoint,
    dequantize_block_fp8,
    write_expert_variant,
    write_synthetic_checkpoint,
)
from tests.unit_test.fishaudio_drama3.test_mtp_engine import (
    BackboneTolerances,
    backbone_comparison,
    backbone_problems,
    outcome_problems,
    ras_window_problems,
    request_steps,
    reused_slot_count,
    run_engine,
    scenario_outcome,
    single_request_runs,
    slot_problems,
    tower_problems,
)
from tests.unit_test.fixtures.accelerator import require_cuda

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]

# bs=1, concurrent requests, and a full wave on the KV slots an abort freed.
FP8_SCENARIO_NAMES = ("single", "concurrent", "abort", "after_abort")
EXPERT_COUNT = int(TEXT_CONFIG["num_experts"])
EXPERT_WIDTH = int(TEXT_CONFIG["moe_intermediate_size"])
HIDDEN_SIZE = int(TEXT_CONFIG["hidden_size"])
TOP_K = int(TEXT_CONFIG["num_experts_per_tok"])
MOE_LAYER_INDICES = list(range(int(TEXT_CONFIG["num_hidden_layers"])))
EXPERT_CHECKPOINT_PREFIX = "model.language_model.layers"
# The activation quant kernel floors a 128-column group's fp32 amax here.
ACTIVATION_AMAX_FLOOR = 1e-10
DIVIDE_BLOCK_SIZE = 1024
# Twice the maximum per-row relative error of the emulation against the
# dequantized weights on the torch backbone's routed-expert inputs, measured on
# this fixture.
MOE_ACTIVATION_TOLERANCE = 0.139
# note (SunskyXH): Triton accumulates fp8 dot products inexactly on sm_90, so some
# fp8 steps land off this emulation. Bounds are about twice the maxima measured on
# H200: 3.5e-2 for a row, 2.8e-3 for the median row of a layer over the run or of a
# call of at least MEDIAN_MINIMUM_ROWS rows (fewer rows leave a median at hardware
# noise).
MOE_KERNEL_TOLERANCE = 7.1e-2
MOE_KERNEL_MEDIAN_TOLERANCE = 5.5e-3
MEDIAN_MINIMUM_ROWS = 32
# Twice the torch backbone's fp8-emulated vs dequantized deviation measured on
# shared greedy histories: prompt-row hidden mean, then the semantic band and
# each head's logits over every decision row.
FP8_ENGINE_TOLERANCES = BackboneTolerances(
    hidden_mean=0.127,
    semantic_gap=17.8,
    head_gaps=(2.08, 1.93, 1.67, 1.88, 1.83, 1.8, 2.15, 2.5, 3.74),
)


@dataclass(frozen=True, kw_only=True)
class Fp8EngineRuns:
    fp8: EngineRun
    dequantized: EngineRun
    # The expert tensors each checkpoint was written with, by checkpoint name.
    fp8_experts: dict[str, torch.Tensor]
    dequantized_experts: dict[str, torch.Tensor]


@pytest.fixture(scope="module")
def fp8_engine_runs(tmp_path_factory: pytest.TempPathFactory) -> Fp8EngineRuns:
    require_cuda()
    workdir = tmp_path_factory.mktemp("drama3_fp8")
    write_synthetic_checkpoint(workdir / "bf16")
    fp8_experts = write_expert_variant(workdir / "bf16", workdir / "fp8", "fp8")
    dequantized_experts = write_expert_variant(
        workdir / "bf16", workdir / "dequantized", "dequantized"
    )
    spec = EngineSpec(
        model_dir=str(workdir / "fp8"),
        context_length=CONTEXT_LENGTH,
        max_running_requests=MAX_RUNNING_REQUESTS,
        max_prefill_tokens=CONTEXT_LENGTH,
        max_total_tokens=MAX_TOTAL_TOKENS,
        total_gpu_memory_fraction=TOTAL_GPU_MEMORY_FRACTION,
        prompts=PROMPTS,
        scenarios=[
            scenario
            for scenario in MAIN_SCENARIOS
            if scenario.name in FP8_SCENARIO_NAMES
        ],
        record_moe_calls=True,
    )
    # Serial on purpose: each engine owns its process and its share of the GPU.
    fp8 = run_engine("fp8", spec, workdir, {})
    dequantized = run_engine(
        "dequantized",
        spec.model_copy(
            update={
                "model_dir": str(workdir / "dequantized"),
                "scenarios": [
                    scenario for scenario in MAIN_SCENARIOS if scenario.name == "single"
                ],
                "record_moe_calls": False,
            }
        ),
        workdir,
        {},
    )
    print(f"Artifacts: {workdir}")
    return Fp8EngineRuns(
        fp8=fp8,
        dequantized=dequantized,
        fp8_experts=fp8_experts,
        dequantized_experts=dequantized_experts,
    )


@pytest.fixture(scope="module")
def reference_decoder() -> AudioDecoder:
    """The tower in fp32 on CPU, loaded from the synthetic checkpoint's tensors."""
    decoder = AudioDecoder(FishOmniConfig(**CHECKPOINT_CONFIG))
    load_audio_decoder_weights(decoder, decoder_checkpoint().items())
    return decoder.float().eval()


@dataclass(frozen=True, kw_only=True)
class ExpertWeights:
    """One layer's routed experts in the engine's fused layout, [experts, ...] each."""

    w13_values: torch.Tensor
    w13_scale_inv: torch.Tensor
    w2_values: torch.Tensor
    w2_scale_inv: torch.Tensor
    w13_dequantized: torch.Tensor
    w2_dequantized: torch.Tensor


def fused_expert_tensor(
    experts: dict[str, torch.Tensor],
    layer_index: int,
    projection_names: tuple[str, ...],
    suffix: str,
) -> torch.Tensor:
    """Per-expert checkpoint tensors stacked as the engine fuses them, gate first."""
    stem = f"{EXPERT_CHECKPOINT_PREFIX}.{layer_index}.mlp.experts"
    return torch.stack(
        [
            torch.cat(
                [
                    experts[f"{stem}.{expert}.{projection_name}.{suffix}"]
                    for projection_name in projection_names
                ]
            )
            for expert in range(EXPERT_COUNT)
        ]
    )


def expert_weights(experts: dict[str, torch.Tensor], layer_index: int) -> ExpertWeights:
    w13_values = fused_expert_tensor(
        experts, layer_index, ("gate_proj", "up_proj"), "weight"
    )
    w13_scale_inv = fused_expert_tensor(
        experts, layer_index, ("gate_proj", "up_proj"), "weight_scale_inv"
    )
    w2_values = fused_expert_tensor(experts, layer_index, ("down_proj",), "weight")
    w2_scale_inv = fused_expert_tensor(
        experts, layer_index, ("down_proj",), "weight_scale_inv"
    )
    return ExpertWeights(
        w13_values=w13_values,
        w13_scale_inv=w13_scale_inv,
        w2_values=w2_values,
        w2_scale_inv=w2_scale_inv,
        w13_dequantized=torch.stack(
            [
                dequantize_block_fp8(values, scale_inv)
                for values, scale_inv in zip(w13_values, w13_scale_inv, strict=True)
            ]
        ),
        w2_dequantized=torch.stack(
            [
                dequantize_block_fp8(values, scale_inv)
                for values, scale_inv in zip(w2_values, w2_scale_inv, strict=True)
            ]
        ),
    )


@triton.jit
def approximate_divide_kernel(
    numerator: float,
    denominator_pointer: torch.Tensor,
    quotient_pointer: torch.Tensor,
    element_count: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    denominator = tl.load(denominator_pointer + offsets, mask=mask, other=1.0)
    numerators = tl.full((BLOCK_SIZE,), numerator, tl.float32)
    quotient = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [numerators, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(quotient_pointer + offsets, quotient, mask=mask)


def approximate_divide(numerator: float, denominator: torch.Tensor) -> torch.Tensor:
    """numerator / denominator the way nvcc --use_fast_math divides floats."""
    flat_denominator = denominator.contiguous().view(-1).cuda()
    quotient = torch.empty_like(flat_denominator)
    approximate_divide_kernel[
        (triton.cdiv(flat_denominator.numel(), DIVIDE_BLOCK_SIZE),)
    ](
        numerator,
        flat_denominator,
        quotient,
        flat_denominator.numel(),
        BLOCK_SIZE=DIVIDE_BLOCK_SIZE,
    )
    return quotient.view(denominator.shape).cpu()


def quantized_activations(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 values and fp32 dequant multipliers per row and 128 columns, as the kernel.

    A row of exactly 128 columns goes to the per-token kernel, which divides the
    other way round (on sm_90 its scale is amax * (1 / 448), within the bounds);
    the group kernel is built with --use_fast_math.
    """
    row_count, column_count = activations.shape
    groups = activations.float().reshape(
        row_count, column_count // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE
    )
    amax = groups.abs().amax(dim=-1)
    if column_count == FP8_BLOCK_SIZE:
        scale = amax / FP8_MAX
        multiplier = torch.where(scale == 0, 0.0, 1.0 / scale)
    else:
        amax = amax.clamp(min=ACTIVATION_AMAX_FLOOR)
        scale = amax * (torch.tensor(1.0) / torch.tensor(FP8_MAX))
        multiplier = approximate_divide(FP8_MAX, amax)
    values = (groups * multiplier[..., None]).clamp(-FP8_MAX, FP8_MAX)
    return values.to(torch.float8_e4m3fn).reshape(row_count, column_count), scale


def block_fp8_matmul(
    values: torch.Tensor,
    scale: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> torch.Tensor:
    """fp32 sum over 128-column groups of e4m3 dot products times both multipliers."""
    row_count, column_count = values.shape
    group_count = column_count // FP8_BLOCK_SIZE
    products = torch.einsum(
        "rgk,ngk->rng",
        values.float().reshape(row_count, group_count, FP8_BLOCK_SIZE),
        weight_values.float().reshape(-1, group_count, FP8_BLOCK_SIZE),
    )
    weight_scale = weight_scale_inv.repeat_interleave(FP8_BLOCK_SIZE, dim=0)
    # The kernel's N tile is one block wide, so it multiplies the two scales first.
    return (products * (scale[:, None, :] * weight_scale[None])).sum(dim=-1)


def routed_expert_output(
    call: MoeCall, weights: ExpertWeights, *, quantizes_activations: bool
) -> torch.Tensor:
    """The call's routed-expert output, rounded where the kernel rounds.

    gate/up and silu(gate) * up are stored in bfloat16; each route's down
    projection is scaled by its routing weight in fp32 and stored in bfloat16;
    the routes are added into bfloat16. Without quantized activations the
    weights are the dequantized fp32 ones.
    """
    hidden_states = call.hidden_states
    route_outputs = torch.zeros(
        hidden_states.shape[0], TOP_K, HIDDEN_SIZE, dtype=torch.bfloat16
    )
    for expert in range(EXPERT_COUNT):
        rows, routes = (call.topk_ids == expert).nonzero(as_tuple=True)
        if len(rows) == 0:
            continue
        elif quantizes_activations:
            # Each row is quantized on its own, so a subset of rows quantizes alike.
            hidden_values, hidden_scale = quantized_activations(hidden_states[rows])
            gate_up = block_fp8_matmul(
                hidden_values,
                hidden_scale,
                weights.w13_values[expert],
                weights.w13_scale_inv[expert],
            )
        else:
            gate_up = hidden_states[rows].float() @ weights.w13_dequantized[expert].T
        gate, up = gate_up.to(torch.bfloat16).float().split(EXPERT_WIDTH, dim=-1)
        activation = (gate / (1.0 + torch.exp(-gate)) * up).to(torch.bfloat16)
        if quantizes_activations:
            activation_values, activation_scale = quantized_activations(activation)
            down = block_fp8_matmul(
                activation_values,
                activation_scale,
                weights.w2_values[expert],
                weights.w2_scale_inv[expert],
            )
        else:
            down = activation.float() @ weights.w2_dequantized[expert].T
        routing_weights = call.topk_weights[rows, routes].float()
        route_outputs[rows, routes] = (down * routing_weights[:, None]).to(
            torch.bfloat16
        )
    return route_outputs.float().sum(dim=1).to(torch.bfloat16)


def row_relative_errors(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-row relative error; a zero reference row must be matched exactly."""
    difference_norms = (output.float() - reference.float()).norm(dim=-1)
    reference_norms = reference.float().norm(dim=-1)
    exact_or_infinite = torch.where(difference_norms == 0, 0.0, math.inf)
    return torch.where(
        reference_norms > 0, difference_norms / reference_norms, exact_or_infinite
    )


@dataclass(frozen=True, kw_only=True)
class MoeCallCheck:
    problems: list[str]
    checked_call_count: int
    largest_kernel_error: float
    largest_layer_median_error: float
    largest_call_median_error: float
    largest_activation_error: float


def moe_call_check(
    run: EngineRun, weights_by_layer: dict[int, ExpertWeights]
) -> MoeCallCheck:
    """Validate complete per-layer expert calls against emulated and dequantized
    outputs, with finite-value, per-row and grouped-median bounds."""
    problems = []
    checked_call_count = 0
    largest_kernel_error = 0.0
    largest_call_median_error = 0.0
    largest_activation_error = 0.0
    layer_row_errors: dict[int, list[torch.Tensor]] = {}
    for forward in run.forwards:
        row_count = sum(len(step.positions) for step in forward.steps)
        layer_indices = sorted(call.layer_index for call in forward.moe_calls)
        if layer_indices != MOE_LAYER_INDICES:
            problems.append(
                f"forward {forward.forward_index}: routed-expert calls for layers "
                f"{layer_indices}, expected {MOE_LAYER_INDICES}"
            )
            continue
        else:
            pass
        for call in forward.moe_calls:
            location = f"forward {forward.forward_index} layer {call.layer_index}"
            shapes = (
                tuple(call.hidden_states.shape),
                tuple(call.output.shape),
                tuple(call.topk_ids.shape),
                tuple(call.topk_weights.shape),
            )
            expected_shapes = (
                (row_count, HIDDEN_SIZE),
                (row_count, HIDDEN_SIZE),
                (row_count, TOP_K),
                (row_count, TOP_K),
            )
            if shapes != expected_shapes:
                problems.append(
                    f"{location}: shapes {shapes}, expected {expected_shapes}"
                )
                continue
            elif not all(
                bool(tensor.float().isfinite().all())
                for tensor in (call.hidden_states, call.output, call.topk_weights)
            ):
                problems.append(f"{location}: non-finite routed-expert input or output")
                continue
            else:
                pass
            weights = weights_by_layer[call.layer_index]
            for reference_name, quantizes_activations, tolerance in (
                ("the fp8 emulation", True, MOE_KERNEL_TOLERANCE),
                ("the dequantized weights", False, MOE_ACTIVATION_TOLERANCE),
            ):
                reference = routed_expert_output(
                    call, weights, quantizes_activations=quantizes_activations
                )
                row_errors = row_relative_errors(call.output, reference)
                error = row_errors.max().item()
                if quantizes_activations:
                    layer_row_errors.setdefault(call.layer_index, []).append(row_errors)
                else:
                    pass
                if quantizes_activations and row_count >= MEDIAN_MINIMUM_ROWS:
                    median_error = row_errors.median().item()
                    largest_call_median_error = max(
                        largest_call_median_error, median_error
                    )
                    if not median_error <= MOE_KERNEL_MEDIAN_TOLERANCE:
                        problems.append(
                            f"{location}: the median row is {median_error:.2e} off "
                            f"{reference_name} (relative)"
                        )
                    else:
                        pass
                else:
                    pass
                # Written this way round, a NaN error fails too.
                if not error <= tolerance:
                    problems.append(
                        f"{location}: a row is {error:.2e} off {reference_name} "
                        f"(relative)"
                    )
                elif quantizes_activations:
                    largest_kernel_error = max(largest_kernel_error, error)
                else:
                    largest_activation_error = max(largest_activation_error, error)
            checked_call_count += 1
    largest_layer_median_error = 0.0
    for layer_index, row_errors in sorted(layer_row_errors.items()):
        median_error = torch.cat(row_errors).median().item()
        largest_layer_median_error = max(largest_layer_median_error, median_error)
        if not median_error <= MOE_KERNEL_MEDIAN_TOLERANCE:
            problems.append(
                f"layer {layer_index}: the median row over the run is "
                f"{median_error:.2e} off the fp8 emulation (relative)"
            )
        else:
            pass
    return MoeCallCheck(
        problems=problems,
        checked_call_count=checked_call_count,
        largest_kernel_error=largest_kernel_error,
        largest_layer_median_error=largest_layer_median_error,
        largest_call_median_error=largest_call_median_error,
        largest_activation_error=largest_activation_error,
    )


def test_every_tensor_loads_and_only_the_routed_experts_are_fp8(
    fp8_engine_runs: Fp8EngineRuns,
) -> None:
    fp8_facts = fp8_engine_runs.fp8.load_facts
    dequantized_facts = fp8_engine_runs.dequantized.load_facts

    assert fp8_facts.unloaded_parameters == []
    assert dequantized_facts.unloaded_parameters == []
    for facts, expert_method in (
        (fp8_facts, "Fp8MoEMethod"),
        (dequantized_facts, "UnquantizedFusedMoEMethod"),
    ):
        assert set(facts.quant_methods.values()) == {
            LayerQuantMethod(
                layer_kind="linear", quant_method="UnquantizedLinearMethod"
            ),
            LayerQuantMethod(
                layer_kind="embedding", quant_method="UnquantizedEmbeddingMethod"
            ),
            LayerQuantMethod(layer_kind="routed_experts", quant_method=expert_method),
        }
        assert sorted(facts.expert_layers) == MOE_LAYER_INDICES
    for name, dtype in fp8_facts.parameter_dtypes.items():
        if name.endswith(("mlp.experts.w13_weight", "mlp.experts.w2_weight")):
            assert dtype == torch.float8_e4m3fn, name
            assert dequantized_facts.parameter_dtypes[name] == torch.bfloat16, name
        elif name.endswith("_weight_scale_inv"):
            assert dtype == torch.float32, name
        else:
            assert dtype == dequantized_facts.parameter_dtypes[name], name
    for layer_index in MOE_LAYER_INDICES:
        fp8_tensors = fp8_facts.expert_layers[layer_index].tensors
        weights = expert_weights(fp8_engine_runs.fp8_experts, layer_index)
        for name, expected in (
            ("w13_weight", weights.w13_values),
            ("w2_weight", weights.w2_values),
        ):
            assert torch.equal(
                fp8_tensors[name].view(torch.uint8), expected.view(torch.uint8)
            ), name
        for name, expected in (
            ("w13_weight_scale_inv", weights.w13_scale_inv),
            ("w2_weight_scale_inv", weights.w2_scale_inv),
        ):
            assert torch.equal(fp8_tensors[name], expected), name
        dequantized_tensors = dequantized_facts.expert_layers[layer_index].tensors
        for name, projection_names in (
            ("w13_weight", ("gate_proj", "up_proj")),
            ("w2_weight", ("down_proj",)),
        ):
            expected = fused_expert_tensor(
                fp8_engine_runs.dequantized_experts,
                layer_index,
                projection_names,
                "weight",
            )
            assert torch.equal(dequantized_tensors[name], expected), name


def test_fp8_experts_run_on_the_triton_block_kernel(
    fp8_engine_runs: Fp8EngineRuns,
) -> None:
    """The runner each layer holds; the global DeepGEMM switch is only reported."""
    facts = fp8_engine_runs.fp8.load_facts
    print(
        f"device capability {facts.device_capability}, "
        f"DeepGEMM enabled globally: {facts.is_deep_gemm_enabled}"
    )
    for layer_index, layer in facts.expert_layers.items():
        assert layer.quant_method == "Fp8MoEMethod", layer_index
        assert layer.runner_backend == "TRITON", layer_index
        assert layer.uses_fp8_w8a8, layer_index
        assert layer.block_shape == [FP8_BLOCK_SIZE, FP8_BLOCK_SIZE], layer_index
        assert layer.fused_shared_expert_count == 0, layer_index


@pytest.fixture(scope="module")
def weights_by_layer(fp8_engine_runs: Fp8EngineRuns) -> dict[int, ExpertWeights]:
    return {
        layer_index: expert_weights(fp8_engine_runs.fp8_experts, layer_index)
        for layer_index in MOE_LAYER_INDICES
    }


def test_every_routed_expert_call_matches_the_fp8_emulation(
    fp8_engine_runs: Fp8EngineRuns, weights_by_layer: dict[int, ExpertWeights]
) -> None:
    run = fp8_engine_runs.fp8
    check = moe_call_check(run, weights_by_layer)
    print(
        f"{check.checked_call_count} routed-expert calls; largest row error "
        f"{check.largest_kernel_error:.2e} off the fp8 emulation "
        f"(tolerance {MOE_KERNEL_TOLERANCE:.2e}), largest median row "
        f"{check.largest_layer_median_error:.2e} per layer and "
        f"{check.largest_call_median_error:.2e} per call of at least "
        f"{MEDIAN_MINIMUM_ROWS} rows (tolerance {MOE_KERNEL_MEDIAN_TOLERANCE:.2e}), "
        f"{check.largest_activation_error:.2e} off the dequantized weights "
        f"(tolerance {MOE_ACTIVATION_TOLERANCE:.2e})"
    )

    assert not check.problems, "\n".join(check.problems[:40])
    assert check.checked_call_count == len(run.forwards) * len(MOE_LAYER_INDICES)


def test_fp8_outputs_match_the_dequantized_engine(
    fp8_engine_runs: Fp8EngineRuns,
) -> None:
    """Every fp8 request against the dequantized engine's single run of its prompt."""
    fp8_run = fp8_engine_runs.fp8
    comparison = backbone_comparison(
        fp8_run,
        single_request_runs(fp8_engine_runs.dequantized),
        FP8_ENGINE_TOLERANCES,
        is_same_engine=False,
        decision_rule="within_gap",
    )
    fp8_single_ids = scenario_outcome(fp8_run, "single").request_ids

    assert not comparison.problems, "\n".join(comparison.problems[:40])
    assert len(fp8_single_ids) == len(PROMPTS)
    assert set(fp8_single_ids) <= set(comparison.compared_request_ids)
    assert set(comparison.compared_request_ids) == set(request_steps(fp8_run))


def test_fp8_tower_matches_the_dense_recompute(
    fp8_engine_runs: Fp8EngineRuns, reference_decoder: AudioDecoder
) -> None:
    """The tower is bfloat16 either way; it is fed the fp8 backbone's hidden states."""
    problems = tower_problems(fp8_engine_runs.fp8, reference_decoder)
    assert not problems, "\n".join(problems[:40])


def test_fp8_engine_bookkeeping(fp8_engine_runs: Fp8EngineRuns) -> None:
    """Slots, RAS windows, results and pools, as for the bfloat16 engine."""
    run = fp8_engine_runs.fp8
    problems = (
        slot_problems(run)
        + ras_window_problems(run)
        + outcome_problems(run)
        + backbone_problems(run, single_request_runs(run))
    )

    assert not problems, "\n".join(problems[:40])
    assert reused_slot_count(run) > 0
    for scenario in run.scenarios:
        assert scenario.idle_free_counts == run.baseline_free_counts, scenario.name
