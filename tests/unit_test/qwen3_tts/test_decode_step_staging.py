# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS decode step around the code predictor.

The layer 0 token id is the only value the host reads from a step, so its
pinned copy is enqueued before the predictor and the finalize wait returns
while the predictor still runs. The code and feedback rows keep coming from a
snapshot taken behind the predictor. The one-step-ahead path is covered in
test_step_ahead_decode.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.scheduling.types import RequestOutput

EOS = 42
PREDICTOR_CYCLES = 1_000_000_000


@pytest.fixture(autouse=True)
def require_cuda_for_accelerator_tests(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("accelerator") and not torch.cuda.is_available():
        pytest.skip("token id staging on a stream needs CUDA")


def make_runner(code_predictor_forward, device: torch.device) -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.has_pending_code_step = False
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.token_id_events = []
    runner.token_id_event_slot = 0
    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=EOS),
        code_predictor_forward=code_predictor_forward,
        output_codes=torch.zeros((4, 3), dtype=torch.long, device=device),
        output_embeds=torch.zeros((4, 2), device=device),
    )
    return runner


def step(ids: torch.Tensor):
    batch_size = ids.shape[0]
    result = SimpleNamespace(
        next_token_ids=ids,
        logits_output=SimpleNamespace(
            hidden_states=torch.ones((batch_size, 2), device=ids.device)
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        positions=torch.full((batch_size,), 3, dtype=torch.long, device=ids.device),
    )
    requests = [
        SimpleNamespace(
            request_id=f"r{row}",
            data=Qwen3TTSSGLangRequestData(
                req=SimpleNamespace(inflight_middle_chunks=0)
            ),
        )
        for row in range(batch_size)
    ]
    return result, forward_batch, requests


def test_collect_codes_stages_the_ids_before_the_predictor_runs():
    ids = torch.tensor([7, EOS, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    staged_when_called = []

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        staged_when_called.append(
            result._host_token_ids
        )  # noqa: leading-underscore  # production name
        runner.model.output_codes[:3] = layer0_codes + torch.arange(3)
        runner.model.output_embeds[:3] = layer0_codes.to(torch.float32)

    runner = make_runner(code_predictor_forward, torch.device("cpu"))

    runner.collect_codes(result, forward_batch, object(), requests)

    assert len(staged_when_called) == 1
    assert staged_when_called[0] is ids

    outputs = {
        req.request_id: RequestOutput(req.request_id, data=int(ids[row]))
        for row, req in enumerate(requests)
    }
    runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)
    runner.model.output_codes.zero_()
    runner.model.output_embeds.zero_()

    assert [c.tolist() for c in requests[0].data.output_codes] == [[7, 8, 9]]
    assert requests[0].data.pending_feedback_queue[0].tolist() == [7.0, 7.0]
    assert requests[1].data.output_codes == []
    assert len(requests[1].data.pending_feedback_queue) == 0
    assert [c.tolist() for c in requests[2].data.output_codes] == [[9, 10, 11]]
    assert requests[2].data.pending_feedback_queue[0].tolist() == [9.0, 9.0]
    assert [request.data.speech_output_tokens for request in requests] == [1, 1, 1]


def test_a_middle_prefill_chunk_adds_no_frame_feedback_or_stream_chunk():
    ids = torch.tensor([7, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    requests[0].data.req.inflight_middle_chunks = 1

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        runner.model.output_codes[:2] = layer0_codes + torch.arange(3)
        runner.model.output_embeds[:2] = layer0_codes.to(torch.float32)

    runner = make_runner(code_predictor_forward, torch.device("cpu"))
    runner.collect_codes(result, forward_batch, object(), requests)
    outputs = {
        req.request_id: RequestOutput(req.request_id, data=int(ids[row]))
        for row, req in enumerate(requests)
    }
    runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)

    assert requests[0].data.output_codes == []
    assert len(requests[0].data.pending_feedback_queue) == 0
    assert requests[0].data.latest_stream_code_chunk is None
    assert [c.tolist() for c in requests[1].data.output_codes] == [[9, 10, 11]]
    assert requests[1].data.latest_stream_code_chunk is not None
    assert [request.data.speech_output_tokens for request in requests] == [0, 1]


def test_a_middle_prefill_chunk_does_not_advance_its_generation_step():
    _, _, requests = step(torch.tensor([7, 9], dtype=torch.long))
    requests[1].data.req.inflight_middle_chunks = 2
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)

    assert runner.finalize_skip_rids(SimpleNamespace(requests=requests)) == {"r1"}


def lookahead_request(**overrides) -> SimpleNamespace:
    params = {
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "min_new_tokens": 0,
    }
    params.update(overrides)
    return SimpleNamespace(
        sampling_params=SimpleNamespace(**params), custom_logit_processor=None
    )


def decode_batch(*requests: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(reqs=list(requests), forward_mode=ForwardMode.DECODE)


def test_lookahead_admits_repetition_penalty_but_not_additive_history_terms():
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)

    assert runner.lookahead_eligible(decode_batch(lookahead_request())) is True
    assert runner.lookahead_eligible(decode_batch()) is False
    assert runner.lookahead_eligible(None) is False
    for blocked in (
        {"frequency_penalty": 0.5},
        {"presence_penalty": 0.5},
        {"min_new_tokens": 2},
    ):
        batch = decode_batch(lookahead_request(), lookahead_request(**blocked))
        assert runner.lookahead_eligible(batch) is False
    custom = lookahead_request()
    custom.custom_logit_processor = object()
    assert runner.lookahead_eligible(decode_batch(custom)) is False


@pytest.mark.parametrize(
    "forward_mode",
    [mode for mode in ForwardMode if not mode.is_decode()] + [None],
)
def test_lookahead_rejects_every_batch_that_is_not_a_decode(forward_mode):
    """An async prefill never stages step-ahead inputs, so it must take the synchronous path."""
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    batch = SimpleNamespace(reqs=[lookahead_request()], forward_mode=forward_mode)

    assert runner.lookahead_eligible(batch) is False


@pytest.mark.accelerator
def test_staged_ids_resolve_while_the_predictor_is_still_running():
    device = torch.device("cuda")
    ids = torch.tensor([7, 8], dtype=torch.long, device=device)
    result, forward_batch, requests = step(ids)

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        torch.cuda._sleep(PREDICTOR_CYCLES)  # noqa: leading-underscore  # upstream name

    runner = make_runner(code_predictor_forward, device)
    stream = torch.cuda.current_stream(device)

    runner.collect_codes(result, forward_batch, object(), requests)
    host_ids = runner.resolve_host_token_ids(result)
    predictor_done = stream.query()
    torch.cuda.synchronize()

    assert host_ids.tolist() == [7, 8]
    assert predictor_done is False


@pytest.mark.accelerator
def test_staged_ids_keep_the_sampled_values_when_later_stream_work_overwrites_them():
    device = torch.device("cuda")
    ids = torch.tensor([7, 8], dtype=torch.long, device=device)
    result, forward_batch, requests = step(ids)

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        layer0_codes.fill_(-1)

    runner = make_runner(code_predictor_forward, device)

    runner.collect_codes(result, forward_batch, object(), requests)

    assert runner.resolve_host_token_ids(result).tolist() == [7, 8]
    assert result.next_token_ids.tolist() == [-1, -1]
