# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS one-step-ahead decode.

The launch of step N+1 reads its inputs without consuming them: the feedback row
of the step still in flight and the text row one past the queue head. Resolve
does the bookkeeping the synchronous step does, so a row that finished or was
retracted while its step was in flight leaves the queues and history untouched.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_tts.model_runner import (
    Qwen3TTSLaunchedStep,
    Qwen3TTSModelRunner,
)
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.scheduling.types import RequestOutput

EOS = 42


def make_runner() -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.has_pending_code_step = False
    runner.model = SimpleNamespace(config=SimpleNamespace(codec_eos_token_id=EOS))
    return runner


def make_request(name: str, *, repetition_penalty: float = 1.05) -> SimpleNamespace:
    req = SimpleNamespace(
        inflight_middle_chunks=0,
        is_retracted=False,
        finished=lambda: False,
        sampling_params=SimpleNamespace(repetition_penalty=repetition_penalty),
    )
    data = Qwen3TTSSGLangRequestData(req=req)
    data.pending_feedback_queue = deque()
    data.pending_text_queue = deque()
    data.thinker_chunks_done = True
    data.tts_pad_embed = torch.full((2,), -1.0)
    return SimpleNamespace(request_id=name, data=data)


def test_lookahead_inputs_read_the_inflight_frame_and_the_next_text_row():
    sched_req = make_request("r0")
    data = sched_req.data
    data.pending_feedback_queue.append(torch.full((2,), 0.0))
    data.pending_text_queue.extend([torch.full((2,), 1.0), torch.full((2,), 2.0)])

    feedback, text = Qwen3TTSModelRunner.peek_lookahead_decode_inputs(sched_req)
    assert feedback.tolist() == [0.0, 0.0] and text.tolist() == [1.0, 1.0]

    data.inflight_feedback = torch.full((2,), 5.0)
    feedback, text = Qwen3TTSModelRunner.peek_lookahead_decode_inputs(sched_req)
    assert feedback.tolist() == [5.0, 5.0] and text.tolist() == [2.0, 2.0]
    assert len(data.pending_feedback_queue) == 1
    assert len(data.pending_text_queue) == 2

    data.pending_text_queue.pop()
    feedback, text = Qwen3TTSModelRunner.peek_lookahead_decode_inputs(sched_req)
    assert text.tolist() == [-1.0, -1.0]

    data.thinker_chunks_done = False
    with pytest.raises(RuntimeError, match="no queued text"):
        Qwen3TTSModelRunner.peek_lookahead_decode_inputs(sched_req)


def test_emitted_codec_frames_counts_the_inflight_frame():
    runner = make_runner()
    data = make_request("r0").data
    data.output_codes.append(torch.zeros(3, dtype=torch.long))

    assert runner.emitted_codec_frames(data) == 1
    data.inflight_feedback = torch.zeros(2)
    assert runner.emitted_codec_frames(data) == 2


def test_resolve_does_the_synchronous_bookkeeping_and_skips_dropped_rows():
    runner = make_runner()
    requests = [make_request(name) for name in ("live", "eos", "finished", "retracted")]
    for sched_req in requests:
        sched_req.data.pending_feedback_queue.append(torch.full((2,), 9.0))
        sched_req.data.pending_text_queue.extend([torch.ones(2), torch.ones(2) * 2])
    requests[2].data.req.finished = lambda: True
    requests[3].data.req.is_retracted = True
    codes = torch.arange(12, dtype=torch.long).view(4, 3)
    feedback_rows = [torch.full((2,), float(row)) for row in range(4)]
    input_rows = [torch.full((2,), 10.0 + row) for row in range(4)]
    for row, sched_req in enumerate(requests):
        sched_req.data.inflight_feedback = feedback_rows[row]
        sched_req.data.inflight_codec_id = torch.tensor(row)
    resolved = Qwen3TTSLaunchedStep(
        ids=torch.tensor([7, EOS, 7, 7]),
        codes=codes,
        feedback_rows=feedback_rows,
        input_rows=input_rows,
        ready=None,
    )
    runner.resolved_step = resolved
    outputs = {
        "live": RequestOutput("live", data=7),
        "eos": RequestOutput("eos", data=EOS),
        "finished": RequestOutput("finished", data=7),
        "retracted": RequestOutput("retracted", data=7),
    }

    runner.post_process_outputs(None, SimpleNamespace(requests=requests), outputs)

    live, eos, finished, retracted = (sched_req.data for sched_req in requests)
    assert runner.resolved_step is None
    assert [chunk.tolist() for chunk in live.output_codes] == [[0, 1, 2]]
    assert live.latest_stream_code_chunk.tolist() == [0, 1, 2]
    assert [row.tolist() for row in live.pending_feedback_queue] == [[0.0, 0.0]]
    assert [
        row.tolist() for row in QwenTalkerModelRunner.decode_input_history(live)
    ] == [[10.0, 10.0]]
    assert len(live.pending_text_queue) == 1
    assert live.speech_output_tokens == 1
    assert eos.output_codes == [] and len(eos.pending_feedback_queue) == 0
    assert [
        row.tolist() for row in QwenTalkerModelRunner.decode_input_history(eos)
    ] == [[11.0, 11.0]]
    assert eos.speech_output_tokens == 1
    for dropped in (finished, retracted):
        assert dropped.output_codes == []
        assert [row.tolist() for row in dropped.pending_feedback_queue] == [[9.0, 9.0]]
        assert len(dropped.pending_text_queue) == 2
        assert QwenTalkerModelRunner.decode_input_history(dropped) == []
        assert dropped.speech_output_tokens == 0
    for data in (live, eos, finished, retracted):
        assert data.inflight_feedback is None and data.inflight_codec_id is None


def test_a_later_launch_keeps_its_inflight_marker_through_an_earlier_resolve():
    runner = make_runner()
    sched_req = make_request("r0")
    sched_req.data.pending_feedback_queue.append(torch.zeros(2))
    sched_req.data.pending_text_queue.append(torch.ones(2))
    later_row = torch.full((2,), 3.0)
    sched_req.data.inflight_feedback = later_row
    runner.resolved_step = Qwen3TTSLaunchedStep(
        ids=torch.tensor([7]),
        codes=torch.zeros((1, 3), dtype=torch.long),
        feedback_rows=[torch.full((2,), 2.0)],
        input_rows=[torch.zeros(2)],
        ready=None,
    )

    runner.post_process_outputs(
        None,
        SimpleNamespace(requests=[sched_req]),
        {"r0": RequestOutput("r0", data=7)},
    )

    assert sched_req.data.inflight_feedback is later_row


def test_inflight_repetition_penalty_scatters_into_the_launch_snapshot_only():
    runner = make_runner()
    requests = [
        make_request("a", repetition_penalty=1.05),
        make_request("b", repetition_penalty=1.2),
    ]
    requests[0].data.inflight_codec_id = torch.tensor(3)
    requests[1].data.inflight_codec_id = torch.tensor(5)
    scaling = torch.ones((2, 8))
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(acc_scaling_penalties=scaling)
    )

    runner.add_inflight_repetition_penalty(forward_batch, requests)

    expected = torch.ones((2, 8))
    expected[0, 3] = 1.05
    expected[1, 5] = 1.2
    assert torch.allclose(scaling, expected)


def test_inflight_repetition_penalty_skips_rows_without_an_inflight_step():
    runner = make_runner()
    requests = [make_request("a"), make_request("b")]
    requests[1].data.inflight_codec_id = torch.tensor(2)
    scaling = torch.ones((2, 4))
    runner.add_inflight_repetition_penalty(
        SimpleNamespace(sampling_info=SimpleNamespace(acc_scaling_penalties=scaling)),
        requests,
    )
    assert scaling[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert scaling[1].tolist() == pytest.approx([1.0, 1.0, 1.05, 1.0])

    untouched = SimpleNamespace(
        sampling_info=SimpleNamespace(acc_scaling_penalties=None)
    )
    runner.add_inflight_repetition_penalty(untouched, requests)


def test_step_ahead_decode_is_an_xpu_default_only():
    assert OmniPlatform.enable_tts_async_decode(object.__new__(OmniPlatform)) is False
    assert (
        XPUOmniPlatform.enable_tts_async_decode(object.__new__(XPUOmniPlatform)) is True
    )
