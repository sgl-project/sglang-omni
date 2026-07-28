# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.qwen3_omni import talker_model_runner
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner


def _fake_model(n: int, hidden: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_buffer=torch.zeros(n, hidden, dtype=torch.float32),
        _feedback_mask=torch.zeros(n, dtype=torch.bool),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    return runner


def _data(feedback: torch.Tensor | None, text: torch.Tensor | None) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_queue=deque([feedback]) if feedback is not None else deque(),
        pending_text_queue=deque([text]) if text is not None else deque(),
        thinker_chunks_done=False,
        tts_pad_embed=None,
    )


def _req_wrap(data: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(data=data)


def test_dense_write_skips_index_tensor(monkeypatch: Any) -> None:
    n, hidden = 3, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [_req_wrap(_data(feedbacks[i], texts[i])) for i in range(n)]

    calls: list = []
    real_tensor = torch.tensor

    def _spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(args)
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(talker_model_runner.torch, "tensor", _spy)

    runner._write_feedback_buffers(requests)

    assert calls == []
    assert torch.equal(model._feedback_mask, torch.ones(n, dtype=torch.bool))
    for i in range(n):
        assert torch.equal(model._feedback_buffer[i], feedbacks[i] + texts[i])


def test_sparse_write_leaves_starved_row_unwritten() -> None:
    n, hidden = 3, 3
    model = _fake_model(n, hidden)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [
        _req_wrap(_data(feedbacks[0], texts[0])),
        _req_wrap(_data(feedbacks[1], None)),
        _req_wrap(_data(feedbacks[2], texts[2])),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [True, False, True]
    assert torch.equal(model._feedback_buffer[0], feedbacks[0] + texts[0])
    assert torch.equal(model._feedback_buffer[1], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[2], feedbacks[2] + texts[2])
