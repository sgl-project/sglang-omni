# SPDX-License-Identifier: Apache-2.0
"""Speech hidden capture across asynchronous thinker launches."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


def _runner() -> ThinkerModelRunner:
    runner = object.__new__(ThinkerModelRunner)
    runner._capture_hidden_layers = [0, 24]
    runner._capture_hidden_width = 2
    runner._th_hidden_bufs = None
    runner._th_hidden_slot = 0
    return runner


def _result(embed: torch.Tensor | None) -> SimpleNamespace:
    packed = None if embed is None else torch.cat((embed, embed + 10), dim=-1)
    return SimpleNamespace(
        next_token_ids=torch.tensor([11, 22, 33]),
        logits_output=SimpleNamespace(hidden_states=packed),
        _captured_aux_hidden_states=None,
    )


def test_speech_capture_modes() -> None:
    runner = _runner()
    runner._should_capture_hidden = lambda request: request.request_id == "audio"
    text = SimpleNamespace(request_id="text")
    audio = SimpleNamespace(request_id="audio")

    assert runner.requested_capture_hidden_mode_decode(None, [text]).name == "NULL"
    assert runner.requested_capture_hidden_mode_decode(None, [audio]).name == "FULL"
    assert runner.requested_capture_hidden_mode_prefill(None, [audio]).name == "LAST"

    runner._async_host_buf = lambda like, n: torch.empty(n, dtype=like.dtype)
    text_result = _result(None)
    runner.post_decode_launch(text_result, None, [text])
    assert text_result._captured_aux_hidden_states is None

    runner._capture_hidden_layers = None
    assert runner.requested_capture_hidden_mode_decode(None, [audio]).name == "NULL"
    assert runner.requested_capture_hidden_mode_prefill(None, [audio]).name == "NULL"


def test_launch_snapshot_survives_replay_and_batch_changes() -> None:
    runner = _runner()
    embed = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = _result(embed)
    runner._stage_async_hidden_capture(result)

    result.logits_output.hidden_states.add_(100)
    next_result = _result(embed + 200)
    runner._stage_async_hidden_capture(next_result)
    assert torch.equal(next_result._captured_aux_hidden_states[0], embed + 200)

    processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        capture_hidden_width=2,
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="finished"),
            SchedulerRequest(request_id="retracted"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(reqs=[object()]),
    )

    outputs = processor.process(result, scheduler_output)
    hidden = outputs["audio"].extra["hidden_states"]
    assert torch.equal(hidden["embed"], torch.tensor([5.0, 6.0]))
    assert torch.equal(hidden[24], torch.tensor([15.0, 16.0]))

    reused_result = _result(embed + 300)
    runner._stage_async_hidden_capture(reused_result)
    assert torch.equal(reused_result._captured_aux_hidden_states[0], embed + 300)
    assert torch.equal(hidden["embed"], torch.tensor([5.0, 6.0]))


def test_missing_speech_capture_fails() -> None:
    with pytest.raises(RuntimeError, match="model produced no hidden states"):
        _runner()._stage_async_hidden_capture(_result(None))
