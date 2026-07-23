# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder
from sglang_omni.models.ming_tts.model_runner import (
    MingTTSModelRunner,
    MingTTSTPStepUpdate,
)


def test_ming_tts_entry_tail_failure_is_published_before_reraise() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._tp_rank = 0
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4))
    )
    published = []

    def fail_tail(*_):
        raise RuntimeError("tail failed")

    runner._run_entry_tail_step = fail_tail
    runner._broadcast_tp_step_update = published.append
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(2, 1, 4))
    )

    with pytest.raises(RuntimeError, match="tail failed"):
        runner._collect_ming_tts_step(
            result,
            forward_batch=None,
            schedule_batch=SimpleNamespace(),
            requests=[SimpleNamespace(), SimpleNamespace()],
        )

    assert len(published) == 1
    assert published[0].tail_failed.tolist() == [1, 1]
    assert torch.count_nonzero(published[0].feedback_embeddings).item() == 0


def test_ming_tts_follower_rejects_tail_failure() -> None:
    runner = object.__new__(MingTTSModelRunner)
    update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )
    update.tail_failed.fill_(1)

    with pytest.raises(RuntimeError, match="acoustic tail failed"):
        runner._apply_follower_step_update(update, [SimpleNamespace()])


def test_ming_tts_abort_callback_resets_runner_state() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._request_states = {"req-ming-tts": object()}
    builder = object.__new__(MingTtsEngineBuilder)
    builder._model_runner = runner

    abort_callback = builder.make_abort_callback()
    abort_callback("req-ming-tts")
    abort_callback("req-ming-tts")

    assert runner._request_states == {}
