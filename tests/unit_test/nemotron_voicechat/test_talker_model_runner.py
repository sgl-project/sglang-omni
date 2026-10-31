# SPDX-License-Identifier: Apache-2.0
"""Talker batch dispatch and request ownership across decode steps."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.talker import LevelScheduleEntry
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)


def make_runner(use_graph: bool) -> NemotronVoiceChatTalkerModelRunner:
    runner = object.__new__(NemotronVoiceChatTalkerModelRunner)
    runner.model = SimpleNamespace(
        hidden_out=torch.zeros(3, 2),
        fusion_buffer=torch.zeros(3, 2),
        fusion_mask=torch.zeros(3, dtype=torch.bool),
        talker=SimpleNamespace(generate_codes=Mock()),
        mog_head=Mock(),
    )
    runner.level_schedule = [LevelScheduleEntry(first_level=0, level_count=2)]
    runner.top_p = 0.9
    runner.noise_scale = 1.0
    runner.text_pad_id = 0
    if use_graph:
        runner.graph_code_generator = Mock(
            side_effect=lambda hidden_states: hidden_states.long().clone()
        )
    else:
        runner.graph_code_generator = None
        runner.model.talker.generate_codes.side_effect = (
            lambda hidden_states, mog_head, **parameters: hidden_states.long().clone()
        )
    runner.step_row = Mock(
        side_effect=lambda previous_codes, token: previous_codes.float() + token
    )
    return runner


def make_request(request_id: str, tokens: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(
            pending_text_queue=deque(tokens),
            talker_model_inputs={
                "prev_codes": torch.zeros(1, 2, dtype=torch.long),
                "codes_rows": [],
            },
        ),
    )


@pytest.mark.parametrize("use_graph", [False, True])
def test_decode_tracks_request_rows_after_reorder_and_removal(use_graph: bool) -> None:
    runner = make_runner(use_graph)
    first_request = make_request("first", [1, 2, 3])
    second_request = make_request("second", [10, 20])
    requests_by_step = [
        [first_request, second_request],
        [second_request, first_request],
        [first_request],
    ]
    expected_codes = {"first": [], "second": []}
    retained_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []

    for requests in requests_by_step:
        schedule_batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(to_finish=None, finished=lambda: False)
                for request in requests
            ]
        )
        expected_inputs = torch.cat(
            [
                request.data.talker_model_inputs["prev_codes"]
                + request.data.pending_text_queue[0]
                for request in requests
            ]
        )
        runner.before_decode(None, schedule_batch, requests)
        torch.testing.assert_close(
            runner.model.fusion_buffer[: len(requests)], expected_inputs.float()
        )
        runner.model.hidden_out[: len(requests)] = expected_inputs
        runner.model.fusion_mask[: len(requests)] = False
        runner.post_decode(None, None, schedule_batch, requests)

        for row, request in enumerate(requests):
            expected = expected_inputs[row : row + 1].clone()
            expected_codes[request.request_id].append(expected[0])
            inputs = request.data.talker_model_inputs
            torch.testing.assert_close(inputs["prev_codes"], expected)
            torch.testing.assert_close(inputs["stream_chunk"], expected)
            assert inputs["stream_chunk"].device.type == "cpu"
            retained_chunks.append((inputs["stream_chunk"], expected))

    for request in (first_request, second_request):
        torch.testing.assert_close(
            torch.stack(request.data.talker_model_inputs["codes_rows"]),
            torch.stack(expected_codes[request.request_id]),
        )
        assert not request.data.pending_text_queue
    for chunk, expected in retained_chunks:
        torch.testing.assert_close(chunk, expected)

    if use_graph:
        assert runner.graph_code_generator.call_count == 3
        runner.model.talker.generate_codes.assert_not_called()
    else:
        assert runner.model.talker.generate_codes.call_count == 3
        for generation_call in runner.model.talker.generate_codes.call_args_list:
            assert generation_call.kwargs == {
                "level_schedule": [LevelScheduleEntry(first_level=0, level_count=2)],
                "top_p": 0.9,
                "noise_scale": 1.0,
            }


@pytest.mark.parametrize("is_terminating", [False, True])
def test_decode_without_text_requires_termination(is_terminating: bool) -> None:
    runner = make_runner(False)
    request = make_request("empty", [])
    schedule_batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                omni_data=request.data,
                to_finish="aborted" if is_terminating else None,
                finished=lambda: False,
            )
        ]
    )
    assert runner.is_decode_batch_ready(schedule_batch) == is_terminating
    if is_terminating:
        runner.before_decode(None, schedule_batch, [request])
        assert runner.model.fusion_mask.tolist() == [True, False, False]
        assert not request.data.pending_text_queue
    else:
        with pytest.raises(RuntimeError, match="no text token and no finish reason"):
            runner.before_decode(None, schedule_batch, [request])
