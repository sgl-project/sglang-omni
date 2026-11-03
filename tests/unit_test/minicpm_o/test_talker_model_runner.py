# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker sampling on per-slot device history against the host window and SGLang's sampler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams

from sglang_omni.model_runner.base import rank_shared_unseeded_sampling_seed
from sglang_omni.models.minicpm_o import talker_model_runner
from sglang_omni.models.minicpm_o.talker_model_runner import (
    MiniCPMOTalkerModelRunner,
    TalkerSlotState,
)
from tests.unit_test.fixtures.accelerator import require_cuda

VOCAB_SIZE = 64
CODEC_EOS_ID = VOCAB_SIZE - 1
MIN_NEW_TOKENS = 5
SEED = 7
SLOT_COUNT = 12
SUPPRESSED_ROW_AND_TOKEN = (1, 3)


def make_talker_request(
    index: int, penalty: float, sampling_params: SamplingParams
) -> SimpleNamespace:
    request = SimpleNamespace(
        output_ids=[], sampling_params=sampling_params, eos_token_ids={CODEC_EOS_ID}
    )
    inputs = {"rep_penalty": penalty, "min_new_tokens": MIN_NEW_TOKENS}
    data = SimpleNamespace(
        req=request, talker_model_inputs=inputs, return_logprob=False
    )
    return SimpleNamespace(data=data, request_id=f"request-{index}")


def suppress_one_token(
    logits_output: LogitsProcessorOutput, requests: list[SimpleNamespace]
) -> None:
    logits_output.next_token_logits[SUPPRESSED_ROW_AND_TOKEN] = float("-inf")


def build_device_sampling_runner(
    sglang_model_runner: SimpleNamespace, device: torch.device
) -> MiniCPMOTalkerModelRunner:
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(model_runner=sglang_model_runner)
    runner.model = SimpleNamespace(
        num_audio_tokens=VOCAB_SIZE,
        codec_eos_id=CODEC_EOS_ID,
        head_code=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
    )
    runner.device = device
    runner.apply_codec_suppress_tokens = suppress_one_token
    runner.enable_device_sampling()
    return runner


def build_batch_sampling_info(
    sampling_params: list[SamplingParams], device: torch.device
) -> SamplingBatchInfo:
    """The flags SGLang derives for a batch of these requests."""
    return SamplingBatchInfo(
        temperatures=torch.tensor(
            [[params.temperature] for params in sampling_params], device=device
        ),
        top_ps=torch.tensor(
            [params.top_p for params in sampling_params], device=device
        ),
        top_ks=torch.tensor(
            [params.top_k for params in sampling_params],
            dtype=torch.int32,
            device=device,
        ),
        min_ps=torch.tensor(
            [params.min_p for params in sampling_params], device=device
        ),
        is_all_greedy=all(params.top_k <= 1 for params in sampling_params),
        is_any_greedy=any(params.top_k <= 1 for params in sampling_params),
        need_top_p_sampling=any(params.top_p != 1.0 for params in sampling_params),
        need_top_k_sampling=any(
            params.top_k != TOP_K_ALL for params in sampling_params
        ),
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        grammars=[],
        penalizer_orchestrator=None,
        has_custom_logit_processor=False,
        custom_params=None,
        custom_logit_processor=None,
        sampling_seed=None,
        device=device.type,
        logit_bias=None,
    )


def test_device_history_matches_host_penalty_and_min_new_tokens() -> None:
    torch.manual_seed(0)
    requests = [
        make_talker_request(
            row,
            1.05 if row % 2 else 1.0,
            SamplingParams(sampling_seed=SEED if row % 3 else None),
        )
        for row in range(4)
    ]
    rows = torch.tensor([5, 0, 11, 2])
    runner = build_device_sampling_runner(
        SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(SLOT_COUNT, 1)),
            decode_cuda_graph_runner=None,
            sample=lambda logits_output, forward_batch: (
                logits_output.next_token_logits.argmax(dim=-1)
            ),
        ),
        torch.device("cpu"),
    )
    assert runner.slot_state is not None, "the step samples on the device"
    assert runner.sample_graphs is None, "no decode graphs to share"
    sampling_info = SimpleNamespace(
        sampling_seed=None,
        need_min_p_sampling=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        temperatures=torch.ones(len(requests), 1),
        top_ps=torch.ones(len(requests)),
        top_ks=torch.full((len(requests),), VOCAB_SIZE),
        min_ps=torch.zeros(len(requests)),
    )
    for step in range(12):
        logits = torch.randn(len(requests), VOCAB_SIZE) * 3
        logits[:, CODEC_EOS_ID] += 4.0
        expected = logits.clone()
        runner.process_sampling_logits(
            LogitsProcessorOutput(next_token_logits=expected, hidden_states=None),
            requests,
        )
        expected[SUPPRESSED_ROW_AND_TOKEN] = float("-inf")
        for row, scheduler_request in enumerate(requests):
            if len(scheduler_request.data.req.output_ids) < MIN_NEW_TOKENS:
                expected[row, CODEC_EOS_ID] = float("-inf")
            else:
                pass
        forward_batch = SimpleNamespace(
            req_pool_indices=rows,
            forward_mode=SimpleNamespace(is_extend=lambda first=step == 0: first),
            sampling_info=sampling_info,
        )

        next_token_ids = runner.sample_next_token_ids(
            LogitsProcessorOutput(next_token_logits=logits, hidden_states=None),
            forward_batch,
            None,
            requests,
        )

        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        for row, scheduler_request in enumerate(requests):
            scheduler_request.data.req.output_ids.append(int(next_token_ids[row]))
    assert sampling_info.sampling_seed.tolist() == [
        SEED if row % 3 else rank_shared_unseeded_sampling_seed(requests[row], row)
        for row in range(4)
    ]


@pytest.mark.parametrize(
    ("masks_chunk_boundary_eos", "masked_steps"),
    [(True, [25, 29, 50]), (False, [])],
    ids=["several-chunks", "one-chunk"],
)
def test_a_request_of_several_chunks_stops_only_at_an_eos_off_the_chunk_boundaries(
    masks_chunk_boundary_eos: bool, masked_steps: list[int]
) -> None:
    """A request that speaks several chunks of owed text ended at the codec EOS the talker samples at its first chunk boundary; it must still stop at an EOS between boundaries."""
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner.model = SimpleNamespace(codec_eos_id=CODEC_EOS_ID)
    steps = [0, 24, 25, 29, 30, 50]
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(output_ids=[0] * step),
                talker_model_inputs={
                    "rep_penalty": 1.0,
                    "masks_chunk_boundary_eos": masks_chunk_boundary_eos,
                },
            )
        )
        for step in steps
    ]
    logits = torch.zeros(len(steps), VOCAB_SIZE)

    runner.process_sampling_logits(
        LogitsProcessorOutput(next_token_logits=logits, hidden_states=None), requests
    )

    assert [
        step
        for step, row_logits in zip(steps, logits, strict=True)
        if row_logits[CODEC_EOS_ID] == float("-inf")
    ] == masked_steps


def test_a_failed_sampling_graph_capture_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_capture(*args: object) -> None:
        raise RuntimeError("no capture here")

    monkeypatch.setattr(talker_model_runner, "TalkerSampleGraphs", refuse_capture)
    monkeypatch.setattr(
        talker_model_runner.current_platform,
        "get_device_graph_backend",
        lambda device: SimpleNamespace(),
    )
    monkeypatch.setattr(
        talker_model_runner, "current_sglang_sampling_backend", lambda: "pytorch"
    )

    runner = build_device_sampling_runner(
        SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(SLOT_COUNT, 1)),
            decode_cuda_graph_runner=SimpleNamespace(capture_bs=[1, 2]),
            sampler=None,
        ),
        torch.device("cpu"),
    )

    assert runner.slot_state is not None and runner.sample_graphs is None


def test_npu_sampling_skips_capture_with_decode_graphs_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = build_device_sampling_runner(
        SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(SLOT_COUNT, 1)),
            decode_cuda_graph_runner=None,
            sampler=None,
        ),
        torch.device("cpu"),
    )
    runner.device = SimpleNamespace(type="npu")
    runner.tp_worker.model_runner.decode_cuda_graph_runner = SimpleNamespace(
        capture_bs=[1, 2]
    )
    capture = Mock()
    monkeypatch.setattr(talker_model_runner, "TalkerSampleGraphs", capture)
    monkeypatch.setattr(
        talker_model_runner.current_platform,
        "get_device_graph_backend",
        lambda device: SimpleNamespace(),
    )
    monkeypatch.setattr(
        talker_model_runner, "current_sglang_sampling_backend", lambda: "pytorch"
    )

    assert runner.capture_sample_graphs() is None
    capture.assert_not_called()


def test_reset_keeps_the_seeds_of_deterministic_inference() -> None:
    state = TalkerSlotState.allocate(
        SLOT_COUNT, VOCAB_SIZE, CODEC_EOS_ID, torch.device("cpu")
    )
    requests = [make_talker_request(row, 1.0, SamplingParams()) for row in range(2)]
    rows = torch.tensor([3, 8])
    sampling_info = SimpleNamespace(
        sampling_seed=torch.tensor([42, 42]),
        temperatures=torch.ones(len(requests), 1),
        top_ps=torch.ones(len(requests)),
        top_ks=torch.full((len(requests),), VOCAB_SIZE),
        min_ps=torch.zeros(len(requests)),
    )

    state.reset(
        rows,
        requests,
        sampling_info,
        torch.zeros(len(requests), VOCAB_SIZE, dtype=torch.bool),
    )

    assert state.seeds[rows].tolist() == [42, 42]


def test_a_reused_slot_penalizes_only_its_own_request_window() -> None:
    state = TalkerSlotState.allocate(
        SLOT_COUNT, VOCAB_SIZE, CODEC_EOS_ID, torch.device("cpu")
    )
    rows = torch.tensor([7])
    sampling_info = SimpleNamespace(
        sampling_seed=None,
        temperatures=torch.ones(1, 1),
        top_ps=torch.ones(1),
        top_ks=torch.full((1,), VOCAB_SIZE),
        min_ps=torch.zeros(1),
    )
    no_suppression = torch.zeros(1, VOCAB_SIZE, dtype=torch.bool)
    token_ids = [(step * 7) % (VOCAB_SIZE - 1) for step in range(20)]
    last_window = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    last_window[token_ids[-talker_model_runner.REP_PENALTY_WINDOW :]] = True

    def read_penalized_tokens_and_eos() -> tuple[torch.Tensor, bool]:
        logits = torch.ones(1, VOCAB_SIZE)
        state.apply(logits, rows)
        is_finite = torch.isfinite(logits[0])
        return (logits[0] < 1.0) & is_finite, bool(is_finite[CODEC_EOS_ID])

    first = make_talker_request(0, 1.05, SamplingParams())
    state.reset(rows, [first], sampling_info, no_suppression)
    for token_id in token_ids:
        state.append(rows, torch.tensor([token_id]))
    penalized, is_eos_open = read_penalized_tokens_and_eos()
    assert torch.equal(penalized, last_window) and is_eos_open

    second = make_talker_request(1, 1.0, SamplingParams())
    state.reset(rows, [second], sampling_info, no_suppression)
    penalized, is_eos_open = read_penalized_tokens_and_eos()
    assert not penalized.any() and not is_eos_open

    first.data.req.output_ids = list(token_ids)
    state.reset(rows, [first], sampling_info, no_suppression)
    penalized, is_eos_open = read_penalized_tokens_and_eos()
    assert torch.equal(penalized, last_window) and is_eos_open


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("row_sampling", "is_graph_used"),
    [
        ([{"temperature": 0.8, "top_k": 25, "top_p": 0.85}] * 3, True),
        ([{"temperature": 0.8, "top_k": -1, "top_p": 1.0}] * 3, False),
        ([{"temperature": 0.0}] * 3, False),
        (
            [
                {"temperature": 0.0},
                {"temperature": 0.8, "top_k": -1, "top_p": 1.0},
                {"temperature": 0.8, "top_k": 25, "top_p": 0.85},
            ],
            True,
        ),
    ],
    ids=["filtered", "unfiltered", "greedy", "mixed"],
)
def test_seeded_requests_draw_eager_tokens_when_batch_members_change(
    row_sampling: list[dict[str, float]],
    is_graph_used: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_cuda()
    device = torch.device("cuda")
    rows = torch.tensor([5, 0, 11], device=device)
    member_orders = ((0, 1, 2), (2, 0), (1, 2, 0), (1, 0))
    sampling_params = []
    for row, sampling in enumerate(row_sampling):
        params = SamplingParams(max_new_tokens=32, sampling_seed=SEED + row, **sampling)
        params.normalize(None)
        sampling_params.append(params)
    with (
        get_context().override_server_args(sampling_backend="pytorch"),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
    ):
        sampler = Sampler()

        def build_sglang_model_runner(capture_bs: list[int] | None) -> SimpleNamespace:
            return SimpleNamespace(
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.zeros(SLOT_COUNT, 1)
                ),
                decode_cuda_graph_runner=(
                    None
                    if capture_bs is None
                    else SimpleNamespace(capture_bs=capture_bs)
                ),
                sampler=sampler,
                sample=lambda logits_output, forward_batch: sampler(
                    logits_output,
                    forward_batch.sampling_info,
                    False,
                    [0] * len(forward_batch.req_pool_indices),
                    [[] for _ in forward_batch.req_pool_indices],
                    forward_batch.positions,
                ),
            )

        graph_runner = build_device_sampling_runner(
            build_sglang_model_runner([1, 2, 4, 8]), device
        )
        eager_runner = build_device_sampling_runner(
            build_sglang_model_runner(None), device
        )
        assert graph_runner.sample_graphs is not None
        assert eager_runner.sample_graphs is None
        replay = Mock(
            spec=graph_runner.sample_graphs.sample,
            wraps=graph_runner.sample_graphs.sample,
        )
        monkeypatch.setattr(graph_runner.sample_graphs, "sample", replay)
        runner_requests = [
            (
                runner,
                [
                    make_talker_request(row, 1.05, params)
                    for row, params in enumerate(sampling_params)
                ],
            )
            for runner in (graph_runner, eager_runner)
        ]
        torch.manual_seed(0)
        for step in range(8):
            members = member_orders[step % len(member_orders)]
            batch_rows = rows[list(members)]
            batch_params = [sampling_params[index] for index in members]
            logits = torch.randn(len(members), VOCAB_SIZE, device=device) * 3
            positions = torch.arange(len(members), device=device) + 10 * step
            draws = []
            for runner, requests in runner_requests:
                batch_requests = [requests[index] for index in members]
                forward_batch = SimpleNamespace(
                    req_pool_indices=batch_rows,
                    positions=positions,
                    forward_mode=SimpleNamespace(
                        is_extend=lambda first=step == 0: first
                    ),
                    sampling_info=build_batch_sampling_info(batch_params, device),
                )
                draws.append(
                    runner.sample_next_token_ids(
                        LogitsProcessorOutput(
                            next_token_logits=logits.clone(), hidden_states=None
                        ),
                        forward_batch,
                        None,
                        batch_requests,
                    ).tolist()
                )
                for request, token_id in zip(batch_requests, draws[-1], strict=True):
                    request.data.req.output_ids.append(token_id)
            assert draws[0] == draws[1], f"step {step}"
        assert bool(replay.call_count) is is_graph_used
        sampling_info = build_batch_sampling_info(sampling_params, device)
        assert graph_runner.samples_in_graph(len(rows), sampling_info) is is_graph_used
        assert not graph_runner.samples_in_graph(9, sampling_info), "above the top size"
