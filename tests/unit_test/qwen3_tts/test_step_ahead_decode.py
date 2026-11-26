# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS one-step-ahead decode.

The launch of step N+1 reads its inputs without consuming them: the feedback row
of the step still in flight and the text row one past the queue head. Resolve
does the bookkeeping the synchronous step does, so a row that finished or was
retracted while its step was in flight leaves the queues and history untouched.
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.penaltylib import (
    BatchedPenalizerOrchestrator,
    BatchedRepetitionPenalizer,
)

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


class ReusedBufferTalker:
    """Writes every predictor step into the same output buffers, as the graph does."""

    def __init__(self, rows: int) -> None:
        self.config = SimpleNamespace(codec_eos_token_id=EOS)
        self.output_codes = torch.zeros((rows, 3), dtype=torch.long)
        self.output_embeds = torch.zeros((rows, 2))
        self.decode_feedback_embedding = torch.nn.Embedding(rows, 2)

    def prepare_decode_buffers(self, requests: list[SimpleNamespace]) -> None:
        del requests

    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        hidden: torch.Tensor,
        *,
        semantic_positions: torch.Tensor,
    ) -> None:
        del hidden, semantic_positions
        rows = layer0_codes.shape[0]
        self.output_codes[:rows] = layer0_codes + torch.arange(3)
        self.output_embeds[:rows] = layer0_codes.float() + 0.5


def launch_lookahead_step(
    runner: Qwen3TTSModelRunner,
    requests: list[SimpleNamespace],
    tokens: list[int],
    scaling: torch.Tensor,
) -> tuple[Qwen3TTSLaunchedStep, SimpleNamespace]:
    rows = len(requests)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(rows, dtype=torch.long),
        positions=torch.zeros(rows, dtype=torch.long),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        sampling_info=SimpleNamespace(acc_scaling_penalties=scaling),
    )
    runner.before_decode(forward_batch, None, requests, is_lookahead=True)
    result = SimpleNamespace(
        next_token_ids=torch.tensor(tokens),
        logits_output=SimpleNamespace(hidden_states=torch.zeros((rows, 2))),
    )
    return runner.post_decode_launch(result, forward_batch, requests), result


def resolve_lookahead_step(
    runner: Qwen3TTSModelRunner,
    launched: Qwen3TTSLaunchedStep,
    result: SimpleNamespace,
    requests: list[SimpleNamespace],
) -> list[int]:
    runner.post_decode_resolve(launched, result, None, None, requests)
    tokens = result.next_token_ids.tolist()
    outputs = {
        sched_req.request_id: RequestOutput(sched_req.request_id, data=token)
        for sched_req, token in zip(requests, tokens, strict=True)
    }
    runner.post_process_outputs(None, SimpleNamespace(requests=requests), outputs)
    return tokens


def test_overlapped_launches_on_reused_buffers_resolve_their_own_step():
    runner = make_runner()
    runner.model = ReusedBufferTalker(rows=2)
    runner.row_ids_cache = None
    runner.host_staging_buffers = [torch.zeros(2, dtype=torch.long) for _ in range(2)]
    runner.staging_slot = 0
    requests = [make_request("r0"), make_request("r1")]
    for row, sched_req in enumerate(requests):
        sched_req.data.pending_feedback_queue.append(torch.full((2,), 100.0 + row))
        sched_req.data.pending_text_queue.extend(
            [torch.full((2,), 10.0 * step + row) for step in range(3)]
        )
    scaling = torch.ones((2, 16))

    first, first_result = launch_lookahead_step(runner, requests, [7, 8], scaling)
    second, second_result = launch_lookahead_step(runner, requests, [9, 10], scaling)

    assert scaling[0, 7].item() == pytest.approx(1.05)
    assert scaling[1, 8].item() == pytest.approx(1.05)
    assert resolve_lookahead_step(runner, first, first_result, requests) == [7, 8]
    for row, sched_req in enumerate(requests):
        data = sched_req.data
        assert [chunk.tolist() for chunk in data.output_codes] == [
            [7 + row, 8 + row, 9 + row]
        ]
        assert [entry.tolist() for entry in data.pending_feedback_queue] == [
            [7.5 + row] * 2
        ]
        assert [
            entry.tolist() for entry in QwenTalkerModelRunner.decode_input_history(data)
        ] == [[100.0 + 2 * row] * 2]
        assert data.inflight_feedback is second.feedback_rows[row]
        assert int(data.inflight_codec_id) == 9 + row

    assert resolve_lookahead_step(runner, second, second_result, requests) == [9, 10]
    for row, sched_req in enumerate(requests):
        data = sched_req.data
        assert [chunk.tolist() for chunk in data.output_codes] == [
            [7 + row, 8 + row, 9 + row],
            [9 + row, 10 + row, 11 + row],
        ]
        assert [entry.tolist() for entry in data.pending_feedback_queue] == [
            [9.5 + row] * 2
        ]
        assert [
            entry.tolist() for entry in QwenTalkerModelRunner.decode_input_history(data)
        ] == [[100.0 + 2 * row] * 2, [17.5 + 2 * row] * 2]
        assert [entry.tolist() for entry in data.pending_text_queue] == [
            [20.0 + row] * 2
        ]
        assert data.speech_output_tokens == 2
        assert data.inflight_feedback is None and data.inflight_codec_id is None


class PenaltyBatch:
    def __init__(self, reqs: list[SimpleNamespace]) -> None:
        self.reqs = reqs
        self.device = torch.device("cpu")


def test_a_drained_inflight_token_stays_in_the_live_repetition_penalty():
    """A drained step's token is never cumulated by prepare_for_decode, so launch must add it."""
    runner = make_runner()
    requests = [
        make_request("a", repetition_penalty=1.05),
        make_request("b", repetition_penalty=1.2),
    ]
    penalty_batch = PenaltyBatch([sched_req.data.req for sched_req in requests])
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=16, batch=penalty_batch, penalizers={BatchedRepetitionPenalizer}
    )
    runner.live_penalizer_orchestrator = orchestrator
    requests[0].data.inflight_codec_id = torch.tensor(3)
    requests[1].data.inflight_codec_id = torch.tensor(5)
    snapshot = orchestrator.accumulate_scaling_penalties()

    runner.add_inflight_repetition_penalty(
        SimpleNamespace(sampling_info=SimpleNamespace(acc_scaling_penalties=snapshot)),
        requests,
    )
    orchestrator.cumulate_output_tokens(torch.tensor([9, 10]))

    live = orchestrator.penalizers[BatchedRepetitionPenalizer]
    expected = torch.ones((2, 16))
    expected[0, [3, 9]] = 1.05
    expected[1, [5, 10]] = 1.2
    assert torch.allclose(live.cumulated_repetition_penalties, expected)
    assert snapshot[0, 3].item() == pytest.approx(1.05)
    assert snapshot[1, 5].item() == pytest.approx(1.2)


def test_launch_keeps_the_live_penalizer_that_sampling_isolation_hides():
    runner = make_runner()
    live_orchestrator = SimpleNamespace(penalizers={})
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=SimpleNamespace(penalizer_orchestrator=live_orchestrator),
    )

    @contextlib.contextmanager
    def isolated(batch, *, isolate_sampling: bool) -> Iterator[None]:
        live_sampling_info = batch.sampling_info
        batch.sampling_info = SimpleNamespace(penalizer_orchestrator=None)
        yield
        batch.sampling_info = live_sampling_info

    runner.execution_bridge = SimpleNamespace(forward_context=isolated)

    with runner.execution_context(batch, isolate_sampling=True):
        assert batch.sampling_info.penalizer_orchestrator is None
        assert runner.live_penalizer_orchestrator is live_orchestrator


def test_step_ahead_decode_is_an_xpu_default_only():
    assert OmniPlatform.enable_tts_async_decode(object.__new__(OmniPlatform)) is False
    assert (
        XPUOmniPlatform.enable_tts_async_decode(object.__new__(XPUOmniPlatform)) is True
    )
