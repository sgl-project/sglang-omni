# SPDX-License-Identifier: Apache-2.0
"""The decode CUDA graph test's scenarios, engine specs and records: what
graph_engines.py saves and graph_checks.py judges, without importing SGLang."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic import BaseModel

from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS,
)
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    CONTEXT_LENGTH,
    MAX_RUNNING_REQUESTS,
    MAX_TOTAL_TOKENS,
    PROMPTS,
    TOTAL_GPU_MEMORY_FRACTION,
    WAVE_PROMPTS,
    FreePoolCounts,
)

# Batch sizes 3, 5, 6 and 7 replay a graph padded to 4 or 8.
CUDA_GRAPH_BATCH_SIZES = [1, 2, 4, 8]
PREFILL_CUDA_GRAPH_TOKEN_COUNTS = [8, 32, 64, 128, 192]
PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2
# Enough for every wave to prefill in one batch.
MAX_PREFILL_TOKENS = 4096
RANDOM_SEED = 20261003
RETRACT_INTERVAL = 5
# Integer views of the same width, for comparing stored bits.
INTEGER_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# greedy: top_k 1 and every RAS window emptied before the model reads it, so a
# padded batch samples like an unpadded one; random: the request's own knobs.
SamplingMode = Literal["greedy", "random"]
EngineKind = Literal["eager", "graph", "prefill_graph"]
LeanPolicy = Literal["capture", "sequence_length"]


def bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(INTEGER_DTYPES[tensor.element_size()])


class GraphRequestSpec(BaseModel):
    prompt_name: str
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 1.0
    # Frame index whose end-of-speech logit the test forces; None blocks it.
    end_at_frame: int | None = None
    # The runner aborts the request once it has sampled this many frames.
    abort_after_frames: int | None = None
    is_streaming: bool = False


class GraphScenario(BaseModel):
    name: str
    sampling_mode: SamplingMode
    # Each wave is admitted whole while the scheduler is paused.
    waves: list[list[GraphRequestSpec]]


class GraphEngineSpec(BaseModel):
    model_dir: str
    kind: EngineKind
    context_length: int = CONTEXT_LENGTH
    max_running_requests: int = MAX_RUNNING_REQUESTS
    max_prefill_tokens: int = MAX_PREFILL_TOKENS
    max_total_tokens: int = MAX_TOTAL_TOKENS
    total_gpu_memory_fraction: float = TOTAL_GPU_MEMORY_FRACTION
    random_seed: int = RANDOM_SEED
    first_frame_cuda_graphs: bool = True
    first_frame_cuda_graph_max_requests: int = (
        DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS
    )
    triton_sampling: bool = False
    tower_fused_layers: bool = True
    enable_async_decode: bool = False
    async_decode_min_batch_size: int = 1
    prompts: dict[str, list[list[int]]] = PROMPTS
    scenarios: list[GraphScenario]


@dataclass(frozen=True, kw_only=True)
class GraphRequestStep:
    """One request's part of one forward, read after the forward finished."""

    request_id: str
    request_row: int
    mamba_slot: int
    positions: list[int]
    written_slots: list[int]
    # The input rows the model read: every extend row, or the decode row.
    input_frames: list[list[int]]
    # The window the runner built from the history, and the one the model read.
    staged_ras_window: list[int]
    model_ras_window: list[int]
    temperature: float
    top_k: int
    top_p: float
    first_draw: int
    ras_draw: int
    sampled_frame: list[int]
    # [full-attention layers, written slots, 2 (key, value), kv width], as stored.
    written_cache: torch.Tensor
    # sha256 of the request's linear-attention state after the forward.
    state_digest: str
    is_state_finite: bool
    # The whole state on the request's last step: conv tensors, then temporal.
    final_state: list[torch.Tensor] | None
    semantic_logits: torch.Tensor
    head_logits: torch.Tensor
    is_discarded: bool = False


@dataclass(frozen=True, kw_only=True)
class LeanChoice:
    policy: LeanPolicy
    batch_size: int
    is_lean: bool


@dataclass(frozen=True, kw_only=True)
class GraphForwardRecord:
    forward_index: int
    scenario_name: str
    is_extend: bool
    is_lookahead: bool
    is_graph_replay: bool
    graph_batch_size: int | None
    token_count: int
    prefill_replay_count: int
    prefill_eager_count: int
    prefill_replay_buckets: dict[int, int]
    tower_prefill_runs: int
    fused_layer_calls: int
    tower_prefill_fused_layer_calls: int
    first_frame_replays: dict[int, int]
    first_frame_eager_count: int
    first_frame_sampler_calls: int
    first_frame_kv_splits: list[int]
    generator_offset: int
    sampling_step: int
    lean_choices: list[LeanChoice]
    steps: list[GraphRequestStep]
    stray_kv_slots: list[int]
    stray_mamba_slots: list[int]


@dataclass(frozen=True, kw_only=True)
class ExpertPath:
    quant_method: str
    runner_backend: str
    uses_fp8_w8a8: bool
    block_shape: list[int] | None


@dataclass(frozen=True, kw_only=True)
class GraphEngineFacts:
    decode_graph_batch_sizes: list[int] | None
    prefill_runner: str
    prefill_capture_token_counts: list[int] | None
    recorded_kv_layer_ids: list[int]
    tower_layer_ids: list[int]
    decode_buffer_rows: int
    request_pool_size: int
    kv_pool_size: int
    max_kv_splits: int
    multiprocessor_count: int
    device_capability: tuple[int, int]
    expert_paths: list[ExpertPath]
    tower_prefill_runs_before_serving: int
    tower_prefill_runs_outside_extend: int
    tower_prefill_rows: int
    prefill_capture_break_layer_ids: dict[int, list[int]]
    first_frame_request_counts: list[int] | None
    first_frame_capture_written_kv_slots: list[int]
    lean_attention_override: bool | None
    startup_lean_choices: list[LeanChoice]
    uses_triton_sampling: bool
    uses_fused_tower_layers: bool
    fused_max_rows: int | None
    enable_async_decode: bool
    async_decode_min_batch_size: int
    async_decode_resolve_before_prefill: bool


@dataclass(kw_only=True)
class GraphRequestOutcome:
    request_id: str
    request_spec: GraphRequestSpec
    aborted_after_frames: int | None = None
    result_frames: list[list[list[int]]] = field(default_factory=list)
    streamed_frame_chunks: list[list[list[int]]] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class GraphScenarioOutcome:
    name: str
    request_ids: list[str]
    idle_free_counts: FreePoolCounts


@dataclass(frozen=True, kw_only=True)
class GraphEngineRun:
    kind: EngineKind
    facts: GraphEngineFacts
    baseline_free_counts: FreePoolCounts
    scenarios: list[GraphScenarioOutcome]
    outcomes: dict[str, GraphRequestOutcome]
    forwards: list[GraphForwardRecord]


TEMPERATURES = [0.7, 1.0, 1.3]
TOP_KS = [1, 5, 30, 64]
TOP_PS = [0.8, 0.95, 1.0]


def random_wave(prompt_names: list[str]) -> list[GraphRequestSpec]:
    """Mixed sampling knobs, no end of speech and one length, so the batch never shrinks."""
    return [
        GraphRequestSpec(
            prompt_name=prompt_name,
            max_new_tokens=16,
            temperature=TEMPERATURES[index % len(TEMPERATURES)],
            top_k=TOP_KS[index % len(TOP_KS)],
            top_p=TOP_PS[(index + 1) % len(TOP_PS)],
        )
        for index, prompt_name in enumerate(prompt_names)
    ]


# Random sampling only at batch sizes with their own graph, first, while both
# engines have drawn the same random numbers.
RANDOM_WAVES = GraphScenario(
    name="random_waves",
    sampling_mode="random",
    waves=[
        random_wave(["p40"]),
        random_wave(["p9", "p23"]),
        random_wave(["p2", "p13", "p31", "p60"]),
        random_wave(WAVE_PROMPTS),
    ],
)
GREEDY_SINGLE = GraphScenario(
    name="greedy_single",
    sampling_mode="greedy",
    waves=[
        [GraphRequestSpec(prompt_name=prompt_name, max_new_tokens=12)]
        for prompt_name in ["p2", "p23", "p60"]
    ],
)
# The batch shrinks from 8 to 1, through the padded sizes 7, 6, 5 and 3.
GREEDY_STAGGERED = GraphScenario(
    name="greedy_staggered",
    sampling_mode="greedy",
    waves=[
        [
            GraphRequestSpec(
                prompt_name=prompt_name,
                max_new_tokens=6 + 2 * index,
                end_at_frame=4 if prompt_name == "p13" else None,
                is_streaming=True,
            )
            for index, prompt_name in enumerate(WAVE_PROMPTS)
        ]
    ],
)
# A full wave after the abort reuses the slots the aborted request freed.
GREEDY_ABORT_REUSE = GraphScenario(
    name="greedy_abort_reuse",
    sampling_mode="greedy",
    waves=[
        [
            GraphRequestSpec(prompt_name="p2", max_new_tokens=10),
            GraphRequestSpec(prompt_name="p5", max_new_tokens=12),
            GraphRequestSpec(prompt_name="p9", max_new_tokens=14),
            GraphRequestSpec(prompt_name="p13", max_new_tokens=16),
            GraphRequestSpec(prompt_name="p23", max_new_tokens=18),
            GraphRequestSpec(
                prompt_name="p31",
                max_new_tokens=24,
                abort_after_frames=15,
                is_streaming=True,
            ),
        ],
        [
            GraphRequestSpec(prompt_name=prompt_name, max_new_tokens=12)
            for prompt_name in WAVE_PROMPTS
        ],
    ],
)
# Under SGLANG_TEST_RETRACT the scheduler retracts on every fifth forward; three
# requests keep at least two in the batch it retracts from.
GREEDY_RETRACT = GraphScenario(
    name="greedy_retract",
    sampling_mode="greedy",
    waves=[
        [
            GraphRequestSpec(
                prompt_name=prompt_name, max_new_tokens=40, is_streaming=True
            )
            for prompt_name in ["p9", "p23", "p31"]
        ]
    ],
)
GREEDY_LONG_WAVE = GraphScenario(
    name="greedy_long_wave",
    sampling_mode="greedy",
    waves=[
        [
            GraphRequestSpec(prompt_name=prompt_name, max_new_tokens=8)
            for prompt_name in ["a100", "b100"]
        ]
    ],
)
GREEDY_SAME_BUCKET = GraphScenario(
    name="greedy_same_bucket",
    sampling_mode="greedy",
    waves=[
        [
            GraphRequestSpec(prompt_name=prompt_name, max_new_tokens=8)
            for prompt_name in prompt_names
        ]
        for prompt_names in (
            ["p60", "p40", "p23"],
            ["p31", "p9", "p40", "p13"],
            ["p5", "p60"],
        )
    ],
)
GREEDY_ODD_WAVES = GraphScenario(
    name="greedy_odd_waves",
    sampling_mode="greedy",
    waves=[
        [
            GraphRequestSpec(prompt_name=prompt_name, max_new_tokens=8)
            for prompt_name in prompt_names
        ]
        for prompt_names in (
            ["p60", "p31", "p13", "p5", "p40"],
            ["p2", "p5", "p9", "p13", "p31", "p40", "p60"],
        )
    ],
)
MAIN_SCENARIOS = [
    RANDOM_WAVES,
    GREEDY_SINGLE,
    GREEDY_STAGGERED,
    GREEDY_ABORT_REUSE,
    GREEDY_LONG_WAVE,
    GREEDY_SAME_BUCKET,
    GREEDY_ODD_WAVES,
]
GREEDY_SMALL_BUDGET = GraphScenario(
    name="greedy_small_budget",
    sampling_mode="greedy",
    waves=[
        [GraphRequestSpec(prompt_name="c100", max_new_tokens=8)],
        [GraphRequestSpec(prompt_name="p60", max_new_tokens=8)],
        [
            GraphRequestSpec(prompt_name=prompt_name, max_new_tokens=8)
            for prompt_name in ["p13", "p9"]
        ],
    ],
)
FP8_SCENARIOS = [GREEDY_SINGLE, GREEDY_STAGGERED]
SMALL_BUDGET_SCENARIOS = [GREEDY_SMALL_BUDGET]
RETRACT_SCENARIOS = [GREEDY_RETRACT]
