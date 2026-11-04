# SPDX-License-Identifier: Apache-2.0
"""Real CPU flow steps preserve request ordering, feedback, and stop ownership."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.models.firered_tts3.request_builders import CONTROL_TOKEN_ID
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.firered_tts3 import (
    SmallFlowModel,
    make_flow_request,
    make_flow_result,
    make_flow_runner,
)


@pytest.mark.parametrize(
    "timesteps,stop_row", [([3], None), ([3, 3, 3], 1), ([3, 2, 3], None)]
)
def test_batched_decode_matches_serial_requests(
    timesteps: list[int], stop_row: int | None
) -> None:
    model = SmallFlowModel()
    generator = torch.Generator().manual_seed(88)
    hidden_steps = [
        torch.randn(len(timesteps), 8, generator=generator) for _ in range(2)
    ]
    if stop_row is not None:
        weight = model.stop_head.weight[0].detach()
        hidden_steps[0][stop_row] = weight * (
            (8.0 - model.stop_head.bias.item()) / weight.square().sum()
        )
    else:
        pass
    outcomes: list[list[SchedulerRequest]] = []
    forward_counts = []
    for batching in (False, True):
        runner = make_flow_runner(model, batching=batching)
        requests = [
            make_flow_request(index, steps, stopped=index == stop_row)
            for index, steps in enumerate(timesteps)
        ]
        with patch.object(model.dit, "forward", wraps=model.dit.forward) as forward:
            for hidden in hidden_steps:
                active = [
                    request for request in requests if not request.data.req.finished()
                ]
                indices = [requests.index(request) for request in active]
                result = make_flow_result(hidden[indices])
                runner.post_decode(result, None, None, active)
                assert result.next_token_ids.tolist() == [CONTROL_TOKEN_ID] * len(
                    active
                )
            forward_counts.append(forward.call_count)
        outcomes.append(requests)
    for index, (serial, batched) in enumerate(zip(*outcomes, strict=True)):
        assert len(batched.data.latent_patches) == (0 if index == stop_row else 2)
        assert len(batched.data.pending_feedback_queue) == len(
            batched.data.latent_patches
        )
        assert batched.data.req.finished() is (index == stop_row)
        for actual, expected in zip(
            batched.data.latent_patches, serial.data.latent_patches, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
        for actual, expected in zip(
            batched.data.pending_feedback_queue,
            serial.data.pending_feedback_queue,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
    assert forward_counts[1] <= forward_counts[0]
    if len(timesteps) > 1:
        assert forward_counts[1] < forward_counts[0]
    else:
        assert torch.equal(
            outcomes[0][0].data.latent_patches[0], outcomes[1][0].data.latent_patches[0]
        )


@pytest.mark.parametrize("status", ["active", "retracted", "finished"])
def test_async_stop_resolves_only_the_owned_request(status: str) -> None:
    runner = make_flow_runner(SmallFlowModel(), batching=True)
    request = make_flow_request(0, stopped=True)
    weight = runner.model.stop_head.weight[0].detach()
    hidden = (
        weight * ((8.0 - runner.model.stop_head.bias.item()) / weight.square().sum())
    ).unsqueeze(0)
    result = make_flow_result(hidden)
    runner.post_decode_launch(result, None, [request])
    assert request.data.stop_pending and request.data.req.finished_reason is None
    assert request.data.latent_patches == []
    request.data.req.is_retracted = status == "retracted"
    previous_finish = FINISH_MATCHED_TOKEN(99) if status == "finished" else None
    request.data.req.finished_reason = previous_finish
    runner.post_decode_resolve(None, result, None, None, [request])
    assert not request.data.stop_pending
    if status == "active":
        assert isinstance(request.data.req.finished_reason, FINISH_MATCHED_TOKEN)
    else:
        assert request.data.req.finished_reason is previous_finish
    runner.active_requests[request.request_id] = request.data
    runner.on_request_finished(request.request_id, request.data)
    assert runner.active_requests == {} and request.data.flow_state is None


@pytest.mark.parametrize("pause_steps", [(2, 3), (2, 2), (3, 3)])
def test_dual_batching_runs_only_patch_producing_requests(
    pause_steps: tuple[int, int]
) -> None:
    """Stop-check rows must not enter the flow batch or receive another request's patch."""
    model = SmallFlowModel()
    outcomes: list[list[SchedulerRequest]] = []
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
    for batching in (False, True):
        runner = make_flow_runner(model, batching=batching)
        requests = [
            make_flow_request(index, dual_stream=True, ratio=(2, 2))
            for index in range(2)
        ]
        for index, request in enumerate(requests):
            runner.post_prefill(
                make_flow_result(torch.zeros(3, 8), [123]), None, None, [request]
            )
            for _ in range(pause_steps[index]):
                runner.post_decode(
                    make_flow_result(hidden_states[index : index + 1]),
                    None,
                    None,
                    [request],
                )
        patch_counts = [len(request.data.latent_patches) for request in requests]
        with (
            patch.object(model.dit, "forward", wraps=model.dit.forward) as forward,
            patch.object(
                model.patch_encoder, "forward", wraps=model.patch_encoder.forward
            ) as encoder,
        ):
            result = make_flow_result(hidden_states)
            runner.post_decode(result, None, None, requests)
        producing_rows = sum(steps == 2 for steps in pause_steps)
        batches = int(producing_rows > 0) if batching else producing_rows
        assert forward.call_count == 3 * batches
        assert encoder.call_count == batches
        assert result.next_token_ids.tolist() == [
            CONTROL_TOKEN_ID if steps == 2 else 103 for steps in pause_steps
        ]
        for request, before, steps in zip(
            requests, patch_counts, pause_steps, strict=True
        ):
            assert len(request.data.latent_patches) == before + int(steps == 2)
            assert len(request.data.pending_feedback_queue) == len(
                request.data.latent_patches
            )
            assert request.data.req.finished_reason is None
        outcomes.append(requests)
    for serial, batched in zip(*outcomes, strict=True):
        for actual, expected in zip(
            batched.data.latent_patches, serial.data.latent_patches, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
        for actual, expected in zip(
            batched.data.pending_feedback_queue,
            serial.data.pending_feedback_queue,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
