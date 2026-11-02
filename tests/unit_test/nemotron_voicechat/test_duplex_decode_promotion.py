# SPDX-License-Identifier: Apache-2.0
"""One-position session units run as DECODE and replay the decode graph."""

from unittest.mock import Mock

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.nemotron_voicechat.duplex_ar import (
    DuplexTalkerRunner,
    DuplexThinkerRunner,
    FrameHistory,
    UnitDecodeRunner,
)
from sglang_omni.scheduling.types import SchedulerRequest


def unit_batch(**overrides) -> Mock:
    fields = {
        "forward_mode": ForwardMode.EXTEND,
        "extend_lens": [1, 1],
        "prefix_lens": [60, 7],
    }
    fields.update(overrides)
    return Mock(**fields)


class FakeBuildRunner(ModelRunner):
    def build_forward_batch(self, scheduler_output):
        batch = scheduler_output.batch_data
        forward_batch = Mock(forward_mode=batch.forward_mode)
        return forward_batch, batch, batch.forward_mode.is_extend()


class PromotingRunner(UnitDecodeRunner, FakeBuildRunner):
    pass


def test_promotion_builds_a_decode_forward_and_retires_an_extend() -> None:
    runner = object.__new__(PromotingRunner)
    batch = unit_batch()
    forward_batch, schedule_batch, is_prefill = runner.build_forward_batch(
        Mock(batch_data=batch)
    )
    assert forward_batch.forward_mode == ForwardMode.DECODE
    assert schedule_batch is batch
    assert batch.forward_mode == ForwardMode.EXTEND
    assert not is_prefill


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"extend_lens": [38], "prefix_lens": [0]}, id="opening-prefill"),
        pytest.param(
            {"extend_lens": [1, 38], "prefix_lens": [60, 0]}, id="unit-beside-opening"
        ),
        pytest.param({"prefix_lens": [60, 0]}, id="one-position-opening"),
    ],
)
def test_other_extends_are_built_unchanged(overrides: dict) -> None:
    runner = object.__new__(PromotingRunner)
    forward_batch, _, is_prefill = runner.build_forward_batch(
        Mock(batch_data=unit_batch(**overrides))
    )
    assert forward_batch.forward_mode == ForwardMode.EXTEND
    assert is_prefill


def unit_request(request_id: str, *rows: torch.Tensor) -> SchedulerRequest:
    history = FrameHistory(fusion_rows=list(rows), position_count=sum(map(len, rows)))
    data = Mock(talker_model_inputs={"duplex_history": history})
    return SchedulerRequest(request_id, data=data)


def test_thinker_decode_stages_each_unit_row_into_the_fusion_buffer() -> None:
    model = Mock(fusion_buffer=torch.zeros(3, 4), fusion_mask=torch.zeros(3).bool())
    runner = Mock(spec=DuplexThinkerRunner, model=model)
    requests = [
        unit_request("a", torch.ones(60, 4), torch.full((1, 4), 2.0)),
        unit_request("b", torch.full((2, 4), 3.0), torch.full((1, 4), 5.0)),
    ]
    DuplexThinkerRunner.before_decode(runner, Mock(), Mock(), requests)
    torch.testing.assert_close(
        model.fusion_buffer,
        torch.tensor([[2.0] * 4, [5.0] * 4, [0.0] * 4]),
    )
    assert model.fusion_mask.tolist() == [True, True, False]


def test_talker_decode_stages_rows_and_samples_codes_like_prefill() -> None:
    model = Mock(fusion_buffer=torch.zeros(2, 4), fusion_mask=torch.zeros(2).bool())
    runner = Mock(spec=DuplexTalkerRunner, model=model)
    runner.generate_codes.side_effect = lambda index: torch.full((1, 3), index)
    requests = [unit_request("a", torch.full((38, 4), 1.0), torch.full((1, 4), 9.0))]
    DuplexTalkerRunner.before_decode(runner, Mock(), Mock(), requests)
    torch.testing.assert_close(model.fusion_buffer[0], torch.full((4,), 9.0))
    assert model.fusion_mask.tolist() == [True, False]
    runner.post_prefill = lambda *args: DuplexTalkerRunner.post_prefill(runner, *args)
    DuplexTalkerRunner.post_decode(runner, Mock(), Mock(), Mock(), requests)
    assert requests[0].data.talker_model_inputs["duplex_codes"].tolist() == [[0, 0, 0]]
