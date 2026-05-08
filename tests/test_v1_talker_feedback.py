# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni_v1.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni_v1.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler


def _sched_req(**data_kwargs):
    data = SimpleNamespace(**data_kwargs)
    return SimpleNamespace(data=data)


def test_talker_sampling_path_applies_seed_buffer_before_shared_sampling() -> None:
    sampling_info = SimpleNamespace(
        temperatures=torch.empty(2, 1),
        sampling_seed=None,
    )
    forward_batch = SimpleNamespace(sampling_info=sampling_info)
    logits_output = SimpleNamespace(next_token_logits=torch.zeros(2, 4))
    captured = {}

    def sample(logits, batch):
        captured["logits"] = logits
        captured["sampling_seed"] = batch.sampling_info.sampling_seed
        return torch.tensor([1, 2])

    runner = object.__new__(QwenTalkerModelRunner)
    runner._sampling_seed_buffer = torch.empty(2, dtype=torch.int64)
    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sample=sample))
    requests = [
        _sched_req(
            suppress_tokens=[],
            req=SimpleNamespace(
                sampling_params=SimpleNamespace(sampling_seed=None),
                _codec_suppress_tokens=[],
            ),
        ),
        _sched_req(
            suppress_tokens=[],
            req=SimpleNamespace(
                sampling_params=SimpleNamespace(sampling_seed=7),
                _codec_suppress_tokens=[],
            ),
        ),
    ]

    next_token_ids = runner._sample_next_token_ids(
        logits_output,
        forward_batch,
        schedule_batch=None,
        requests=requests,
    )

    assert torch.equal(next_token_ids, torch.tensor([1, 2]))
    assert captured["logits"] is logits_output
    assert torch.equal(captured["sampling_seed"], torch.tensor([0, 7]))
    assert (
        captured["sampling_seed"].data_ptr() == runner._sampling_seed_buffer.data_ptr()
    )


def test_take_next_decode_input_embed_consumes_feedback_and_text_fifo() -> None:
    sched_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0])]),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    combined = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([21.0, 22.0]))
    assert len(sched_req.data.pending_feedback_queue) == 0
    assert len(sched_req.data.pending_text_queue) == 0


def test_take_next_decode_input_embed_falls_back_to_pad_when_stream_done() -> None:
    sched_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=True,
    )

    combined = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([8.0, 10.0]))
    assert len(sched_req.data.pending_feedback_queue) == 0
    assert len(sched_req.data.pending_text_queue) == 0


def test_take_next_decode_input_embed_preserves_feedback_until_text_arrives() -> None:
    sched_req = _sched_req(
        pending_feedback_queue=deque(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        ),
        pending_text_queue=deque(),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    combined = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert combined is None
    assert len(sched_req.data.pending_feedback_queue) == 2

    sched_req.data.pending_text_queue.append(torch.tensor([20.0, 20.0]))

    combined = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([21.0, 22.0]))
    assert len(sched_req.data.pending_feedback_queue) == 1
    assert torch.equal(
        sched_req.data.pending_feedback_queue[0], torch.tensor([3.0, 4.0])
    )


def test_data_has_next_decode_input_requires_feedback_and_text_or_pad() -> None:
    no_text = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    assert not QwenTalkerModelRunner._data_has_next_decode_input(no_text)

    with_text = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0])]),
        thinker_chunks_done=False,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_text)

    with_pad = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        thinker_chunks_done=True,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_pad)


def test_qwen_talker_scheduler_waits_for_complete_stream_before_build() -> None:
    scheduler = object.__new__(QwenTalkerScheduler)
    payload = SimpleNamespace(prefetched_chunks=[], prefetched_stream_done=False)

    assert not scheduler._is_request_build_ready(
        payload,
        pending_stream_done=False,
    )
    assert scheduler._is_request_build_ready(
        payload,
        pending_stream_done=True,
    )


def test_qwen_talker_scheduler_does_not_replay_prefetched_stream_after_build() -> None:
    scheduler = object.__new__(QwenTalkerScheduler)
    req_data = SimpleNamespace(
        pending_text_queue=deque([torch.tensor([11.0, 12.0])]),
        thinker_chunks_done=True,
    )
    payload = SimpleNamespace(
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([20.0, 20.0]))],
        prefetched_stream_done=True,
    )

    scheduler._initialize_request_stream_state(req_data, payload)

    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], torch.tensor([11.0, 12.0]))
