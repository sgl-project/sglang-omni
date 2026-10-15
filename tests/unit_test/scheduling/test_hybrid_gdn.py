# SPDX-License-Identifier: Apache-2.0
"""Hybrid GDN (Qwen3.5) served by OmniScheduler, checked against upstream SGLang.

Needs one CUDA GPU with about 23 GiB free and the public checkpoint named by
SGLANG_OMNI_HYBRID_GDN_TEST_MODEL (default Qwen/Qwen3.5-4B). Each engine runs in
its own process (hybrid_gdn_engines.py); run with -s to see the agreement report.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from sglang_omni.models.weight_loader import resolve_model_path
from tests.unit_test.fixtures.accelerator import require_cuda

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]

MODEL_PATH = os.environ.get("SGLANG_OMNI_HYBRID_GDN_TEST_MODEL", "Qwen/Qwen3.5-4B")
REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_MODULE = "tests.unit_test.scheduling.hybrid_gdn_engines"
ENGINE_TIMEOUT_SECONDS = 1200
# Room for activations and the CUDA context beyond the static memory fraction.
GPU_HEADROOM_MIB = 3 * 1024
# The Omni engine sizes its pools from this stage budget, not from free memory.
OMNI_TOTAL_GPU_MEMORY_FRACTION = 0.6
MAX_NEW_TOKENS = 32
ABORT_MAX_NEW_TOKENS = 400
# One prefill token plus at least eight decode tokens before the abort lands.
ABORT_AFTER_OUTPUT_TOKEN_COUNT = 9
# A test tolerance for this bf16 setup, not a numerical guarantee across
# hardware. Top-2 logprob gaps sit on the bf16 logit grid (0.0625, 0.125, 0.25,
# ...); upstream's own batching flipped greedy picks only at gaps up to 0.125
# (measured on RTX 5090), and 0.3 admits one more step past float rounding.
NEAR_TIE_LOGPROB_GAP = 0.3


class EngineSettings(BaseModel):
    """Serving knobs both engines take verbatim."""

    dtype: str
    context_length: int
    mem_fraction_static: float
    max_running_requests: int
    chunked_prefill_size: int
    max_prefill_tokens: int
    page_size: int
    attention_backend: str
    random_seed: int
    speculative_algorithm: str | None


class Scenario(BaseModel):
    """Waves run one after another; a wave's requests are submitted together."""

    name: str
    waves: list[list[str]]
    abort_prompt_name: str | None = None


class EngineSpec(BaseModel):
    model_dir: str
    engine: EngineSettings
    prompts: dict[str, str]
    prompt_token_ids: dict[str, list[int]] = Field(default_factory=dict)
    max_new_tokens: int
    total_gpu_memory_fraction: float
    scenarios: list[Scenario]
    abort_max_new_tokens: int
    abort_after_output_token_count: int


class TopTwoCandidates(BaseModel):
    """Upstream bs=1's two likeliest tokens at one output position."""

    token_ids: tuple[int, int]
    logprobs: tuple[float, float]


class ReferenceRun(BaseModel):
    prompt_token_ids: dict[str, list[int]]
    single_output_ids: dict[str, list[int]]
    single_top_two: dict[str, list[TopTwoCandidates]]
    repeat_output_ids: dict[str, list[int]]
    batched_output_ids: dict[str, list[int]]


class RequestRun(BaseModel):
    """One request's outbox record; the per-step lists hold one entry per step."""

    request_id: str
    prompt_name: str
    max_new_tokens: int
    aborted: bool
    output_token_counts: list[int]
    occupied_mamba_slot_counts: list[int]
    mamba_slot_ids: list[int]
    result_output_ids: list[list[int]]
    error_messages: list[str]


class FreePoolCounts(BaseModel):
    request_row_count: int
    mamba_slot_count: int
    kv_token_count: int


class ScenarioRun(BaseModel):
    name: str
    requests: list[RequestRun]
    idle_free_counts: FreePoolCounts
    output_token_count_at_abort: int | None = None
    result_count_at_abort: int | None = None


class OmniRun(BaseModel):
    max_mamba_cache_size: int
    max_running_requests: int
    request_row_count: int
    mamba_slot_count: int
    max_total_num_tokens: int
    chunked_prefill_size: int
    is_hybrid_ssm: bool
    idle_used_mamba_slot_count: int
    baseline_free_counts: FreePoolCounts
    scenarios: list[ScenarioRun]


ENGINE_SETTINGS = EngineSettings(
    dtype="bfloat16",
    context_length=4096,
    mem_fraction_static=0.6,
    max_running_requests=8,
    # Not a multiple of the 64-token GDN chunk, so state crosses odd boundaries.
    chunked_prefill_size=100,
    max_prefill_tokens=16384,
    page_size=1,
    # Triton builds no CUDA-toolkit JIT and is a hybrid GDN backend on every GPU.
    attention_backend="triton",
    random_seed=123,
    speculative_algorithm=None,
)

PROMPTS = {
    "short_0": "The capital of France is",
    "short_1": "Here are the first ten prime numbers: 2, 3, 5,",
    "short_2": "At sea level, water boils at a temperature of",
    "short_3": "The three primary colors of light are",
    "short_4": "In Python, the built-in function that returns the length of a list is",
    "short_5": "Translate into French: Good morning, how are you today?\nFrench:",
    "short_6": "The largest planet in our solar system is",
    "short_7": "Photosynthesis is the process by which green plants",
    "short_8": (
        "Once upon a time, in a small village at the edge of a dark forest, "
        "there lived"
    ),
    "short_9": 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "long_0": (
        "The printing press, developed by Johannes Gutenberg around 1440, "
        "transformed the way information spread across Europe. Before its "
        "invention, books were copied by hand, usually by monks working in "
        "monasteries, and a single volume could take months to finish. Gutenberg "
        "combined several existing technologies: the screw press used for making "
        "wine and olive oil, oil-based inks, and a system of movable metal type "
        "cast from an alloy of lead, tin, and antimony. Each letter could be "
        "arranged into lines of text, inked, and pressed onto paper, then "
        "rearranged for the next page. Within a few decades, printing workshops "
        "appeared in hundreds of cities, and millions of books were produced by "
        "the year 1500. The lower cost of books encouraged literacy, helped "
        "standardize languages, and allowed scholars to share and compare ideas "
        "far more quickly than before. In summary, the most important "
        "consequence of the printing press was"
    ),
    "long_1": (
        "A hash table stores key-value pairs and supports fast insertion, "
        "deletion, and lookup. It applies a hash function to each key, which "
        "turns the key into an integer called a hash code. The hash code is then "
        "reduced, usually with a modulo operation, to an index into an array of "
        "buckets. When two different keys map to the same bucket, a collision "
        "occurs. In separate chaining, each bucket holds a small list of entries "
        "and colliding keys are appended to that list. In open addressing, the "
        "table probes other buckets, for example the next one in sequence, until "
        "it finds an empty slot. The load factor, the number of stored entries "
        "divided by the number of buckets, controls performance: as it grows, "
        "collisions become more frequent and operations slow down. Most "
        "implementations therefore double the array and reinsert every entry "
        "once the load factor passes a threshold such as 0.75.\nQuestion: why "
        "does resizing a hash table require reinserting every entry instead of "
        "copying the array?\nAnswer:"
    ),
    "long_2": (
        "To make a simple loaf of bread at home, you need flour, water, yeast, "
        "salt, and a little patience. Start by dissolving two teaspoons of dry "
        "yeast in a cup of warm water and letting it rest for about ten minutes, "
        "until a layer of foam forms on the surface. In a large bowl, mix four "
        "cups of flour with two teaspoons of salt. Pour in the yeast mixture and "
        "stir until a shaggy dough forms. Turn the dough onto a floured counter "
        "and knead it for eight to ten minutes, until it becomes smooth and "
        "elastic. Place the dough in an oiled bowl, cover it with a towel, and "
        "let it rise in a warm place for one to two hours, until it has doubled "
        "in size. Then press out the air, shape it into a loaf, and let it rise "
        "again for about forty minutes. Finally, bake it in a preheated oven at "
        "220 degrees Celsius for"
    ),
    "long_3": (
        "The water cycle describes how water moves continuously between the "
        "oceans, the atmosphere, and the land. It is driven mainly by energy "
        "from the sun and by gravity. When sunlight warms the surface of oceans, "
        "lakes, and rivers, some of the water evaporates and rises into the air "
        "as invisible water vapor. Plants also release water vapor through tiny "
        "pores in their leaves, a process called transpiration. As the warm, "
        "moist air rises, it expands and cools, and the vapor condenses around "
        "small particles of dust or salt to form droplets. Billions of these "
        "droplets together make up clouds. When the droplets collide and grow "
        "large enough, they fall back to the ground as precipitation, which can "
        "take the form of rain, snow, sleet, or hail depending on the "
        "temperature. Some of the precipitation flows over the surface as runoff "
        "and collects in streams and rivers that return it to the sea. Some "
        "soaks into the soil and becomes groundwater, which can stay underground "
        "for thousands of years before it reaches a spring or a well. Explain in "
        "one sentence why the water cycle matters for life on land:"
    ),
    "long_4": (
        "def merge_intervals(intervals):\n"
        '    """Merge overlapping closed intervals, sorted by start."""\n'
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals, key=lambda pair: pair[0])\n"
        "    merged = [list(ordered[0])]\n"
        "    for start, end in ordered[1:]:\n"
        "        last = merged[-1]\n"
        "        if start <= last[1]:\n"
        "            last[1] = max(last[1], end)\n"
        "        else:\n"
        "            merged.append([start, end])\n"
        "    return merged\n\n\n"
        "def find_gaps(intervals, lower, upper):\n"
        '    """Return the parts of [lower, upper] no interval covers."""\n'
        "    gaps = []\n"
        "    cursor = lower\n"
        "    for start, end in merge_intervals(intervals):\n"
        "        if start > cursor:\n"
        "            gaps.append([cursor, min(start, upper)])\n"
        "        cursor = max(cursor, end)\n"
        "        if cursor >= upper:\n"
        "            break\n"
        "    if cursor < upper:\n"
        "        gaps.append([cursor, upper])\n"
        "    return gaps\n\n\n"
        "# merge_intervals([[1, 3], [2, 6], [8, 10]]) returns"
    ),
    "long_5": (
        "Mara had worked as the lighthouse keeper on the northern island for "
        "eleven years, and in all that time the supply boat had never been late. "
        "It arrived every second Tuesday at nine in the morning, carrying flour, "
        "lamp oil, letters, and whatever news the mainland thought worth "
        "sharing. So when the second Tuesday of November came and the horizon "
        "stayed empty, she climbed the tower twice before noon to scan the water "
        "with her brass telescope. The sea was calm, the sky a pale and even "
        "gray. By evening there was still no sign of the boat. She lit the great "
        "lamp as usual, wound the clockwork that turned the lens, and wrote in "
        "the logbook that the weather was fair and the supply boat had not come. "
        "That night she could not sleep. Around two in the morning she heard "
        "something that did not belong to the familiar sounds of the island: a "
        "low, steady knocking from the direction of the old boathouse, which had "
        "been empty since before she arrived. She took the lantern, pulled on "
        "her heavy coat, and followed the path down the cliff. When she finally "
        "pushed open the boathouse door,"
    ),
}

SCENARIOS = [
    Scenario(name="bs1", waves=[["short_0"], ["short_0"], ["long_0"], ["long_2"]]),
    Scenario(name="concurrent4", waves=[["short_1", "long_1", "short_2", "long_4"]]),
    # Full waves fit one prefill chunk: a chunked request in flight makes the
    # scheduler hold back the last admission, so a slot would sit idle.
    Scenario(
        name="full_capacity_1",
        waves=[[f"short_{index}" for index in (0, 2, 3, 4, 5, 6, 7, 9)]],
    ),
    Scenario(
        name="full_capacity_2",
        waves=[[f"short_{index}" for index in (8, 1, 7, 6, 4, 3, 2, 0)]],
    ),
    Scenario(
        name="abort",
        waves=[["short_6", "long_5", "short_7", "long_3"]],
        abort_prompt_name="short_5",
    ),
    Scenario(
        name="after_abort",
        waves=[[f"short_{index}" for index in (8, 9, 0, 1, 2, 3, 4, 6)]],
    ),
]
FULL_CAPACITY_SCENARIOS = ("full_capacity_1", "full_capacity_2", "after_abort")


@dataclass(frozen=True, kw_only=True)
class EngineRuns:
    reference: ReferenceRun
    omni: OmniRun
    omni_log_text: str


def common_prefix_token_count(left_ids: list[int], right_ids: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        if left_id != right_id:
            return index
        else:
            pass
    return min(len(left_ids), len(right_ids))


def require_free_gpu_memory(memory_fraction: float) -> None:
    """Skip instead of crowding work that already holds the shared GPU."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.skip("nvidia-smi is unavailable")
    else:
        pass
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    query = subprocess.run(
        [
            nvidia_smi,
            f"--id={device}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    used_mib, total_mib = (int(value) for value in query.stdout.split(","))
    needed_mib = int(total_mib * memory_fraction) + GPU_HEADROOM_MIB
    if total_mib - used_mib < needed_mib:
        pytest.skip(
            f"GPU {device} has {total_mib - used_mib} MiB free; the engine needs "
            f"{needed_mib} MiB"
        )
    else:
        pass


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Reap the engine and every process it spawned, whatever its exit path."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The whole group already exited.
        pass
    process.wait(timeout=60)


def run_engine(
    engine: Literal["reference", "omni"], spec: EngineSpec, workdir: Path
) -> str:
    """Run one engine process to completion and return its JSON output."""
    require_free_gpu_memory(spec.engine.mem_fraction_static)
    spec_path = workdir / f"{engine}-spec.json"
    output_path = workdir / f"{engine}-output.json"
    log_path = workdir / f"{engine}.log"
    spec_path.write_text(spec.model_dump_json())
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                ENGINE_MODULE,
                engine,
                "--spec",
                str(spec_path),
                "--output",
                str(output_path),
            ],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=ENGINE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(
                f"{engine} engine timed out after {ENGINE_TIMEOUT_SECONDS}s; "
                f"log: {log_path}"
            ) from error
        finally:
            kill_process_group(process)
    if return_code != 0:
        log_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
        raise AssertionError(
            f"{engine} engine exited with {return_code}; log: {log_path}\n{log_tail}"
        )
    else:
        pass
    return output_path.read_text()


def scenario_by_name(runs: EngineRuns, name: str) -> ScenarioRun:
    return next(scenario for scenario in runs.omni.scenarios if scenario.name == name)


def mamba_slot_ids_used(scenarios: list[ScenarioRun]) -> set[int]:
    return {
        slot_id
        for scenario in scenarios
        for request in scenario.requests
        for slot_id in request.mamba_slot_ids
    }


def completed_requests(
    runs: EngineRuns, bucket: str
) -> list[tuple[RequestRun, list[int]]]:
    """Requests of a scenario, or of every long prompt, with their reference."""
    if bucket == "chunked_prefill":
        requests = [
            request
            for scenario in runs.omni.scenarios
            for request in scenario.requests
            if len(runs.reference.prompt_token_ids[request.prompt_name])
            > ENGINE_SETTINGS.chunked_prefill_size
        ]
    else:
        requests = scenario_by_name(runs, bucket).requests
    return [
        (request, runs.reference.single_output_ids[request.prompt_name])
        for request in requests
        if not request.aborted
    ]


def decisive_divergences(runs: EngineRuns, bucket: str) -> list[str]:
    """Requests that leave upstream bs=1 other than for its near-tied runner-up."""
    divergences = []
    for request, reference_ids in completed_requests(runs, bucket):
        output_ids = request.result_output_ids[0]
        position = common_prefix_token_count(output_ids, reference_ids)
        if position == len(reference_ids):
            continue
        elif position == len(output_ids):
            divergences.append(f"{request.request_id}: ended after {position} tokens")
            continue
        else:
            pass
        candidates = runs.reference.single_top_two[request.prompt_name][position]
        gap = candidates.logprobs[0] - candidates.logprobs[1]
        if output_ids[position] in candidates.token_ids and gap <= NEAR_TIE_LOGPROB_GAP:
            pass
        else:
            divergences.append(
                f"{request.request_id}@{position}: token {output_ids[position]}, "
                f"upstream top two {candidates.token_ids} gap {gap}"
            )
    return divergences


def divergence_summary(
    reference: ReferenceRun, prompt_name: str, output_ids: list[int]
) -> str:
    """Where output first leaves the bs=1 reference, and how close that pick was."""
    reference_ids = reference.single_output_ids[prompt_name]
    position = common_prefix_token_count(output_ids, reference_ids)
    if position == len(reference_ids) or position == len(output_ids):
        return f"{position}/{len(reference_ids)}"
    else:
        candidates = reference.single_top_two[prompt_name][position]
        gap = candidates.logprobs[0] - candidates.logprobs[1]
        return (
            f"{position}/{len(reference_ids)} token={output_ids[position]} "
            f"top_two={candidates.token_ids} gap={gap:.3f}"
        )


def agreement_report(runs: EngineRuns) -> str:
    """Per-request first divergence against upstream, plus upstream's own noise."""
    reference = runs.reference
    lines = ["OmniScheduler vs upstream bs=1 (common prefix / tokens):"]
    for scenario in runs.omni.scenarios:
        for request in scenario.requests:
            if request.aborted or len(request.result_output_ids) != 1:
                lines.append(f"  {request.request_id}: no single result")
                continue
            else:
                pass
            summary = divergence_summary(
                reference, request.prompt_name, request.result_output_ids[0]
            )
            lines.append(
                f"  {request.request_id}: {summary} "
                f"slot={sorted(set(request.mamba_slot_ids))} "
                f"peak_occupied={max(request.occupied_mamba_slot_counts)}"
            )
    lines.append("Upstream bs=1 vs its own repeat and batched runs:")
    for prompt_name in reference.single_output_ids:
        repeat = divergence_summary(
            reference, prompt_name, reference.repeat_output_ids[prompt_name]
        )
        batched = divergence_summary(
            reference, prompt_name, reference.batched_output_ids[prompt_name]
        )
        lines.append(
            f"  {prompt_name} ({len(reference.prompt_token_ids[prompt_name])} "
            f"prompt tokens): repeat={repeat} batched={batched}"
        )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def engine_runs(tmp_path_factory: pytest.TempPathFactory) -> EngineRuns:
    require_cuda()
    workdir = tmp_path_factory.mktemp("hybrid_gdn")
    spec = EngineSpec(
        model_dir=str(resolve_model_path(MODEL_PATH)),
        engine=ENGINE_SETTINGS,
        prompts=PROMPTS,
        max_new_tokens=MAX_NEW_TOKENS,
        total_gpu_memory_fraction=OMNI_TOTAL_GPU_MEMORY_FRACTION,
        scenarios=SCENARIOS,
        abort_max_new_tokens=ABORT_MAX_NEW_TOKENS,
        abort_after_output_token_count=ABORT_AFTER_OUTPUT_TOKEN_COUNT,
    )
    # Serial on purpose: each engine needs most of one GPU and owns its process.
    reference = ReferenceRun.model_validate_json(run_engine("reference", spec, workdir))
    omni = OmniRun.model_validate_json(
        run_engine(
            "omni",
            spec.model_copy(update={"prompt_token_ids": reference.prompt_token_ids}),
            workdir,
        )
    )
    runs = EngineRuns(
        reference=reference,
        omni=omni,
        omni_log_text=(workdir / "omni.log").read_text(errors="replace"),
    )
    report = agreement_report(runs)
    (workdir / "report.txt").write_text(report)
    print(f"{report}\nArtifacts: {workdir}")
    return runs


def test_stage_budget_sizes_the_hybrid_state_pool(engine_runs: EngineRuns) -> None:
    """The Omni engine took the stage-budget branch and carved state slots out."""
    omni = engine_runs.omni
    capacity = ENGINE_SETTINGS.max_running_requests
    assert "SGLang AR memory profile: hybrid_state=" in engine_runs.omni_log_text
    assert omni.max_mamba_cache_size == capacity
    assert omni.max_running_requests == capacity
    assert omni.request_row_count == capacity
    assert omni.mamba_slot_count == capacity
    assert omni.chunked_prefill_size == ENGINE_SETTINGS.chunked_prefill_size
    assert omni.is_hybrid_ssm
    assert omni.idle_used_mamba_slot_count == 0


def test_long_prompts_span_several_prefill_chunks(engine_runs: EngineRuns) -> None:
    token_counts = {
        prompt_name: len(token_ids)
        for prompt_name, token_ids in engine_runs.reference.prompt_token_ids.items()
        if prompt_name.startswith("long")
    }
    assert (
        min(token_counts.values()) > ENGINE_SETTINGS.chunked_prefill_size
    ), token_counts


def test_requests_finish_exactly_once_at_full_length(engine_runs: EngineRuns) -> None:
    for scenario in engine_runs.omni.scenarios:
        for request in scenario.requests:
            if request.aborted:
                continue
            else:
                pass
            assert request.error_messages == [], request.request_id
            assert len(request.result_output_ids) == 1, request.request_id
            assert (
                len(request.result_output_ids[0]) == request.max_new_tokens
            ), request.request_id


def test_bs1_matches_upstream_exactly(engine_runs: EngineRuns) -> None:
    """Upstream bs=1 greedy repeats bit for bit, so a lone request must match it."""
    reference = engine_runs.reference
    assert reference.single_output_ids == reference.repeat_output_ids
    mismatched = [
        request.request_id
        for request, reference_ids in completed_requests(engine_runs, "bs1")
        if request.result_output_ids[0] != reference_ids
    ]
    assert mismatched == []


@pytest.mark.parametrize(
    "bucket",
    [scenario.name for scenario in SCENARIOS if scenario.name != "bs1"]
    + ["chunked_prefill"],
)
def test_outputs_leave_upstream_only_at_near_ties(
    engine_runs: EngineRuns, bucket: str
) -> None:
    """Batch shape may flip a near-tie; stale or misplaced state flips clear picks."""
    assert completed_requests(engine_runs, bucket)
    assert decisive_divergences(engine_runs, bucket) == []


def test_waves_hold_their_requests_at_once(engine_runs: EngineRuns) -> None:
    """Peak state slots in use, read from the live pool at every scheduler step."""
    expected_peaks = {"concurrent4": 4} | dict.fromkeys(
        FULL_CAPACITY_SCENARIOS, ENGINE_SETTINGS.max_running_requests
    )
    peaks = {
        name: max(
            occupied_count
            for request in scenario_by_name(engine_runs, name).requests
            for occupied_count in request.occupied_mamba_slot_counts
        )
        for name in expected_peaks
    }
    assert peaks == expected_peaks


def test_full_capacity_waves_reuse_every_mamba_slot(engine_runs: EngineRuns) -> None:
    """Each full wave holds every slot at once; the second runs on used slots."""
    capacity = ENGINE_SETTINGS.max_running_requests
    for name in FULL_CAPACITY_SCENARIOS:
        (wave,) = next(
            scenario for scenario in SCENARIOS if scenario.name == name
        ).waves
        assert len(wave) == capacity, name
        assert (
            sum(
                len(engine_runs.reference.prompt_token_ids[prompt_name])
                for prompt_name in wave
            )
            <= ENGINE_SETTINGS.chunked_prefill_size
        ), name
        assert (
            len(mamba_slot_ids_used([scenario_by_name(engine_runs, name)])) == capacity
        ), name
    scenarios = engine_runs.omni.scenarios
    second_index = next(
        index
        for index, scenario in enumerate(scenarios)
        if scenario.name == "full_capacity_2"
    )
    assert mamba_slot_ids_used([scenarios[second_index]]) <= mamba_slot_ids_used(
        scenarios[:second_index]
    )


def test_abort_mid_decode_frees_its_slot(engine_runs: EngineRuns) -> None:
    abort = scenario_by_name(engine_runs, "abort")
    target = next(request for request in abort.requests if request.aborted)
    assert abort.output_token_count_at_abort >= ABORT_AFTER_OUTPUT_TOKEN_COUNT
    assert abort.result_count_at_abort == 0
    # Neither a fake completion nor an error may follow the abort.
    assert target.result_output_ids == []
    assert target.error_messages == []
    assert max(target.output_token_counts) < ABORT_MAX_NEW_TOKENS
    after_abort_slot_ids = mamba_slot_ids_used(
        [scenario_by_name(engine_runs, "after_abort")]
    )
    assert len(after_abort_slot_ids) == ENGINE_SETTINGS.max_running_requests
    assert set(target.mamba_slot_ids) <= after_abort_slot_ids


def test_pools_drain_after_every_scenario(engine_runs: EngineRuns) -> None:
    for scenario in engine_runs.omni.scenarios:
        assert (
            scenario.idle_free_counts == engine_runs.omni.baseline_free_counts
        ), scenario.name
