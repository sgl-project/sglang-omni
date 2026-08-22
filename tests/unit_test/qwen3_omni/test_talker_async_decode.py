# SPDX-License-Identifier: Apache-2.0
"""Talker post_decode split into the async-decode launch/resolve pair.

Under the one-step-lookahead loop the launch half runs right after the forward and
the resolve half runs a step later. The split follows two rules. Anything reading
the model's fixed row buffers (``_output_codes`` / ``_output_embeds``) belongs to
the launch, before the next forward overwrites them, and the launch must neither
sample (the forward already did) nor build an index tensor from a Python list
(that synchronizes the stream and serializes the launch against its own forward).
Shipping the codec frame belongs to the resolve, the first point at which a finish
from the previous step is visible.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni import talker_model_runner
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

POOL_SIZE = 8
POOL_IDS = [5, 2, 7]
HIDDEN = 3


def _model(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(POOL_SIZE, HIDDEN),
        _sampled_token_ids=torch.arange(n, dtype=torch.long) + 40,
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(n)]
        ),
        _output_embeds=torch.stack(
            [torch.full((HIDDEN,), float(i + 1)) for i in range(n)]
        ),
    )


def _runner(model: SimpleNamespace, *, feedback_enabled: bool = True):
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = feedback_enabled
    runner._code2wav_target = "code2wav"
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append

    def _never(*args, **kwargs):
        raise AssertionError(
            "the async launch must not sample: the forward already did"
        )

    runner._sample_next_token_ids = _never
    return runner


def _req(i: int, *, finished: bool = False, retracted: bool = False):
    return SimpleNamespace(
        rid=f"r{i}",
        req_pool_idx=POOL_IDS[i],
        is_retracted=retracted,
        finished=lambda: finished,
    )


def _requests(n: int, **kwargs) -> list:
    return [
        SimpleNamespace(
            data=SimpleNamespace(
                pending_feedback_count=0,
                feedback_slot_idx=None,
                stage_payload=None,
                req=_req(i, **kwargs),
            )
        )
        for i in range(n)
    ]


def _forward_batch(n: int, *, pad: int = 0) -> SimpleNamespace:
    # CUDA-graph padding appends rows past the real batch; they must stay out.
    return SimpleNamespace(
        req_pool_indices=torch.tensor(POOL_IDS[:n] + [0] * pad, dtype=torch.long)
    )


def _rids(runner) -> list[str]:
    return [msg.request_id for msg in runner._outbox.sent]


def test_launch_publishes_tokens_and_scatters_without_shipping() -> None:
    n = 3
    model = _model(n)
    runner = _runner(model)
    requests = _requests(n)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(result, _forward_batch(n), requests)

    assert torch.equal(result.next_token_ids, torch.tensor([40, 41, 42]))
    assert launch_buf[0] is result.next_token_ids
    assert runner._outbox.sent == []
    for i in range(n):
        assert requests[i].data.pending_feedback_count == 1
        assert requests[i].data.feedback_slot_idx == POOL_IDS[i]
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], model._output_embeds[i])


def test_launch_builds_no_index_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 3
    runner = _runner(_model(n))
    requests = _requests(n)
    forward_batch = _forward_batch(n)
    result = SimpleNamespace(next_token_ids=None)

    calls: list = []
    real_tensor = torch.tensor

    def _spy(*args, **kwargs):
        calls.append(args)
        return real_tensor(*args, **kwargs)

    # Installed after the fixture is built so only the runner's own calls land here.
    monkeypatch.setattr(talker_model_runner.torch, "tensor", _spy)

    runner.post_decode_launch(result, forward_batch, requests)

    assert calls == []


def test_pool_indices_track_batch_order_and_ignore_padding() -> None:
    requests = _requests(3)
    forward_batch = _forward_batch(3, pad=5)

    rows = QwenTalkerModelRunner._batch_pool_indices(forward_batch, len(requests))

    assert rows.tolist() == [req.data.req.req_pool_idx for req in requests]


def test_short_pool_index_row_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="fewer pool indices than requests"):
        QwenTalkerModelRunner._batch_pool_indices(_forward_batch(2), 3)


def test_launch_ids_survive_the_next_forward() -> None:
    # The next step's forward writes _sampled_token_ids in place before this step
    # resolves, so the launch payload has to be a private copy.
    model = _model(2)
    runner = _runner(model)
    requests = _requests(2)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(result, _forward_batch(2), requests)
    model._sampled_token_ids.copy_(torch.tensor([99, 99]))

    runner.post_decode_resolve(launch_buf, result, None, None, requests)
    assert torch.equal(result.next_token_ids, torch.tensor([40, 41]))


def test_resolve_ships_each_frame_once() -> None:
    model = _model(2)
    runner = _runner(model)
    requests = _requests(2)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(result, _forward_batch(2), requests)
    runner.post_decode_resolve(launch_buf, result, None, None, requests)

    assert _rids(runner) == ["r0", "r1"]
    assert [req.data.pending_feedback_count for req in requests] == [1, 1]
    for i, msg in enumerate(runner._outbox.sent):
        assert torch.equal(msg.data, model._output_codes[i])


def test_launch_plus_resolve_matches_the_sync_post_decode() -> None:
    requests_sync = _requests(2)
    runner_sync = _runner(_model(2))
    sync_result = SimpleNamespace(next_token_ids=None)
    runner_sync.post_decode(sync_result, _forward_batch(2), None, requests_sync)

    requests_async = _requests(2)
    runner_async = _runner(_model(2))
    async_result = SimpleNamespace(next_token_ids=None)
    launch_buf = runner_async.post_decode_launch(
        async_result, _forward_batch(2), requests_async
    )
    runner_async.post_decode_resolve(
        launch_buf, async_result, None, None, requests_async
    )

    assert torch.equal(sync_result.next_token_ids, async_result.next_token_ids)
    assert torch.equal(
        runner_sync.model._feedback_slots, runner_async.model._feedback_slots
    )
    assert _rids(runner_sync) == _rids(runner_async)
    assert [req.data.pending_feedback_count for req in requests_sync] == [
        req.data.pending_feedback_count for req in requests_async
    ]


@pytest.mark.parametrize("state", ["finished", "retracted"])
def test_rows_done_in_an_earlier_step_ship_no_frame(state: str) -> None:
    n = 2
    model = _model(n)
    runner = _runner(model)
    requests = _requests(n)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(result, _forward_batch(n), requests)

    # The finish lands between this step's launch and its resolve.
    done = requests[0].data.req
    if state == "finished":
        done.finished = lambda: True
    else:
        done.is_retracted = True
    runner.post_decode_resolve(launch_buf, result, None, None, requests)

    assert _rids(runner) == ["r1"]
    # The slot and the counter still moved: the retract snapshot reads both.
    assert torch.equal(model._feedback_slots[POOL_IDS[0]], model._output_embeds[0])
    assert requests[0].data.pending_feedback_count == 1
    assert requests[0].data.feedback_slot_idx == POOL_IDS[0]


def _frames(runner) -> list[tuple[str, list[int]]]:
    return [(msg.request_id, msg.data.tolist()) for msg in runner._outbox.sent]


def _sync_frames(steps: int, finish_after: int) -> list[tuple[str, list[int]]]:
    """Frames one request produces on the sync path, which stops at its finish."""
    runner = _runner(_model(1))
    requests = _requests(1)
    for step in range(steps):
        runner.model._output_codes[0] = torch.tensor([step, step + 100])
        runner.post_decode(
            SimpleNamespace(next_token_ids=None),
            _forward_batch(1),
            SimpleNamespace(output_ids=None),
            requests,
        )
        if step == finish_after:
            break
    return _frames(runner)


def _async_frames(steps: int, finish_after: int) -> list[tuple[str, list[int]]]:
    """The same request under launch-first lookahead.

    Mirrors ``_event_loop_async_decode``: each iteration launches step k then
    resolves step k-1, and the finish for step k-1 is only detected in the
    ``process_batch_result`` that follows that resolve — so the request is still
    in step ``finish_after + 1``'s already-launched batch.
    """
    runner = _runner(_model(1))
    requests = _requests(1)
    pending = None
    for step in range(steps):
        runner.model._output_codes[0] = torch.tensor([step, step + 100])
        launched = runner.post_decode_launch(
            SimpleNamespace(next_token_ids=None), _forward_batch(1), requests
        )
        if pending is not None:
            runner.post_decode_resolve(
                pending[0], SimpleNamespace(next_token_ids=None), None, None, requests
            )
            if pending[1] == finish_after:
                requests[0].data.req.finished = lambda: True
                break
        pending = (launched, step)
    runner.post_decode_resolve(
        launched, SimpleNamespace(next_token_ids=None), None, None, requests
    )
    return _frames(runner)


def test_finishing_request_ships_the_same_frames_as_sync() -> None:
    # The lookahead launches one step past the finish. That extra frame must not
    # reach code2wav, or the request gets audio the sync path never produced.
    finish_after = 3
    frames_sync = _sync_frames(steps=8, finish_after=finish_after)
    frames_async = _async_frames(steps=8, finish_after=finish_after)

    assert len(frames_sync) == finish_after + 1
    assert frames_async == frames_sync


def test_launch_is_inert_without_feedback() -> None:
    runner = _runner(_model(1), feedback_enabled=False)
    result = SimpleNamespace(next_token_ids=None)

    assert runner.post_decode_launch(result, _forward_batch(1), _requests(1)) is None
    assert result.next_token_ids is None
    assert runner._outbox.sent == []


def test_feedback_talker_is_lookahead_eligible_despite_penalties() -> None:
    # The base gate rejects repetition penalties because its launch samples on the
    # host against a lagged output history; the talker samples in the forward.
    runner = _runner(_model(1))
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    repetition_penalty=1.05,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    min_new_tokens=0,
                )
            )
        ]
    )

    assert runner.lookahead_eligible(batch) is True

    runner._feedback_enabled = False
    assert runner.lookahead_eligible(batch) is False


def test_launch_stages_the_ids_the_resolve_hands_upstream() -> None:
    # The resolve path gives upstream the staged host copy rather than the device
    # tensor, because .tolist() on the device tensor enqueues its copy behind the
    # forward the current launch already submitted. That only works if the launch
    # staged the copy first.
    n = 3
    runner = _runner(_model(n))
    result = SimpleNamespace(next_token_ids=None)

    runner.post_decode_launch(result, _forward_batch(n), _requests(n))

    staged = runner._resolve_host_token_ids(result)
    assert staged is not None
    assert staged.tolist() == [40, 41, 42]


def test_launch_never_materializes_ids_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 2
    runner = _runner(_model(n))
    requests = _requests(n)
    forward_batch = _forward_batch(n)
    result = SimpleNamespace(next_token_ids=None)

    calls: list = []
    real_tolist = torch.Tensor.tolist
    monkeypatch.setattr(
        torch.Tensor,
        "tolist",
        lambda self: calls.append(self) or real_tolist(self),
    )

    runner.post_decode_launch(result, forward_batch, requests)

    assert calls == []
