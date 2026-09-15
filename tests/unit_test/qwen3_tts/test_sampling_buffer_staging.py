# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS sampling buffer restage.

A batch change writes six per request sampling buffers that the predictor graph
and the layer 0 sampler read on the device. The values reach the device rows
from pinned host sources through non blocking copies: a restage does not wait
for the work queued on the stream, except that a source is rewritten only after
its own previous copy has landed, which two alternating sources keep off the
common path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

MAX_BS = 4
STREAM_CYCLES = 1_000_000_000


@pytest.fixture(autouse=True)
def _require_cuda_for_accelerator_tests(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("accelerator") and not torch.cuda.is_available():
        pytest.skip("pinned staging needs CUDA")


def _talker(device: torch.device) -> Qwen3TTSTalker:
    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=2048)
    )
    talker._sub_temperature_tensor = torch.zeros(
        MAX_BS, dtype=torch.float32, device=device
    )
    talker._sub_top_p_tensor = torch.zeros(MAX_BS, dtype=torch.float32, device=device)
    talker._sub_top_k_tensor = torch.zeros(MAX_BS, dtype=torch.long, device=device)
    talker._semantic_sampling_seed_tensor = torch.zeros(
        MAX_BS, dtype=torch.long, device=device
    )
    talker._sub_sampling_seed_tensor = torch.zeros(
        MAX_BS, dtype=torch.long, device=device
    )
    talker._sub_do_sample_tensor = torch.zeros(MAX_BS, dtype=torch.bool, device=device)
    return talker


def _request(
    request_id: str,
    temperature: float,
    *,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
    seeds: tuple[int, int] = (5, 7),
):
    return SimpleNamespace(
        request_id=request_id,
        data=Qwen3TTSSGLangRequestData(
            semantic_sampling_seed=seeds[0],
            subtalker_dosample=do_sample,
            subtalker_temperature=temperature,
            subtalker_top_p=top_p,
            subtalker_top_k=top_k,
            subtalker_sampling_seed=seeds[1],
        ),
    )


def _snapshot(talker: Qwen3TTSTalker, batch_size: int) -> list[torch.Tensor]:
    return [
        buffer[:batch_size].clone()
        for buffer in (
            talker._semantic_sampling_seed_tensor,
            talker._sub_temperature_tensor,
            talker._sub_top_p_tensor,
            talker._sub_top_k_tensor,
            talker._sub_sampling_seed_tensor,
            talker._sub_do_sample_tensor,
        )
    ]


def _columns(snapshot: list[torch.Tensor]) -> list[list]:
    return [column.tolist() for column in snapshot]


def _staged(talker: Qwen3TTSTalker, batch_size: int) -> dict[str, list]:
    return {
        "temperature": talker._sub_temperature_tensor[:batch_size].tolist(),
        "top_k": talker._sub_top_k_tensor[:batch_size].tolist(),
        "semantic_seed": talker._semantic_sampling_seed_tensor[:batch_size].tolist(),
        "do_sample": talker._sub_do_sample_tensor[:batch_size].tolist(),
    }


def test_restage_lands_the_second_batch_after_two_changes_in_a_row():
    talker = _talker(torch.device("cpu"))
    first = [_request("a", 0.8), _request("b", 0.6, top_k=20)]

    talker.prepare_decode_buffers(first)
    talker.prepare_decode_buffers(list(reversed(first)))

    assert _staged(talker, 2) == {
        "temperature": pytest.approx([0.6, 0.8]),
        "top_k": [20, 40],
        "semantic_seed": [5, 5],
        "do_sample": [True, True],
    }


@pytest.mark.accelerator
def test_restage_returns_while_the_stream_is_busy_and_lands_in_order():
    device = torch.device("cuda")
    talker = _talker(device)
    stream = torch.cuda.current_stream(device)

    torch.cuda._sleep(STREAM_CYCLES)
    talker.prepare_decode_buffers([_request("a", 0.8), _request("b", 0.6)])
    returned_early = not stream.query()
    torch.cuda.synchronize()

    host = [t for slot in talker._sampling_staging_slots for t in slot.host]

    assert returned_early
    assert talker._sub_temperature_tensor[:2].tolist() == pytest.approx([0.8, 0.6])
    assert all(t.is_pinned() for t in host)


@pytest.mark.accelerator
def test_two_restages_behind_a_busy_stream_leave_the_second_values():
    device = torch.device("cuda")
    talker = _talker(device)
    stream = torch.cuda.current_stream(device)

    torch.cuda._sleep(STREAM_CYCLES)
    talker.prepare_decode_buffers([_request("a", 0.8)])
    talker.prepare_decode_buffers([_request("b", 0.3)])
    both_returned_early = not stream.query()
    torch.cuda.synchronize()

    assert both_returned_early
    assert talker._sub_temperature_tensor[:1].tolist() == pytest.approx([0.3])


@pytest.mark.accelerator
def test_each_restage_lands_its_own_six_columns_behind_a_busy_stream():
    device = torch.device("cuda")
    talker = _talker(device)
    batches = [
        [_request("a", 0.8, seeds=(11, 12)), _request("b", 0.5, do_sample=False)],
        [
            _request("b", 0.5, do_sample=False),
            _request("c", 0.0, top_k=0, top_p=0.7, seeds=(31, 32)),
            _request("a", 0.8, seeds=(11, 12)),
        ],
        [_request("c", 0.0, top_k=0, top_p=0.7, seeds=(31, 32))],
    ]
    expected = [
        [[11, 5], [0.8, 1.0], [0.9, 1.0], [40, 1], [12, 7], [True, False]],
        [
            [5, 31, 11],
            [1.0, 1e-5, 0.8],
            [1.0, 0.7, 0.9],
            [1, 0, 40],
            [7, 32, 12],
            [False, True, True],
        ],
        [[31], [1e-5], [0.7], [0], [32], [True]],
    ]

    torch.cuda._sleep(STREAM_CYCLES)
    snapshots = []
    for batch in batches:
        talker.prepare_decode_buffers(batch)
        snapshots.append(_snapshot(talker, len(batch)))
    torch.cuda.synchronize()

    for snapshot, columns in zip(snapshots, expected):
        landed = _columns(snapshot)
        assert landed[0] == columns[0]
        assert landed[1] == pytest.approx(columns[1])
        assert landed[2] == pytest.approx(columns[2])
        assert landed[3:] == columns[3:]


@pytest.mark.accelerator
def test_third_restage_waits_for_the_copy_of_its_slot():
    device = torch.device("cuda")
    talker = _talker(device)
    stream = torch.cuda.current_stream(device)

    torch.cuda._sleep(STREAM_CYCLES)
    talker.prepare_decode_buffers([_request("a", 0.8)])
    talker.prepare_decode_buffers([_request("b", 0.3)])
    first_copy = talker._sampling_staging_slots[0].copied
    first_copy_pending = not first_copy.query()
    talker.prepare_decode_buffers([_request("c", 0.5)])
    first_copy_landed = first_copy.query()
    slot_recorded_again = talker._sampling_staging_slots[0].copied is not first_copy
    torch.cuda.synchronize()

    assert first_copy_pending
    assert first_copy_landed
    assert slot_recorded_again
    assert talker._sub_temperature_tensor[:1].tolist() == pytest.approx([0.5])
