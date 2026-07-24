# SPDX-License-Identifier: Apache-2.0
"""``execute_resolve`` accepts the caller's precomputed skip set.

``_resolve_and_process`` already scans the lagged batch for finished/retracted
requests; ``execute_resolve`` must reuse that set instead of re-running the
same O(bs) ``finished()`` scan per step (and must keep computing it itself
when no set is passed).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import ModelRunner


class _StubRunner(ModelRunner):
    def post_decode_resolve(
        self, host_buf, result, forward_batch, schedule_batch, requests
    ):
        pass


def _make_pending(finished_calls):
    def _finished_probe(i):
        def _finished():
            finished_calls.append(i)
            return False

        return _finished

    reqs = [
        SimpleNamespace(
            request_id=f"req{i}",
            data=SimpleNamespace(
                req=SimpleNamespace(finished=_finished_probe(i), is_retracted=False),
                generation_steps=0,
                extra_model_outputs={},
            ),
        )
        for i in range(3)
    ]
    scheduler_output = SimpleNamespace(requests=reqs)
    batch_result = SimpleNamespace(
        next_token_ids=torch.zeros(3, dtype=torch.long),
        logits_output=None,
        can_run_cuda_graph=True,
    )
    return SimpleNamespace(
        event=SimpleNamespace(query=lambda: True),
        launch_buf=None,
        scheduler_output=scheduler_output,
        forward_batch=SimpleNamespace(batch_size=3),
        schedule_batch=SimpleNamespace(is_prefill_only=False),
        model_worker_batch=None,
        batch_result=batch_result,
    )


def _make_runner():
    outputs = {f"req{i}": SimpleNamespace(extra=None) for i in range(3)}
    output_processor = SimpleNamespace(process=lambda result, sched_out: outputs)
    tp_worker = SimpleNamespace(
        gpu_id=0, model_runner=SimpleNamespace(model=SimpleNamespace())
    )
    return _StubRunner(tp_worker=tp_worker, output_processor=output_processor)


def test_execute_resolve_uses_caller_skip_rids_without_rescan():
    finished_calls: list[int] = []
    runner = _make_runner()
    pending = _make_pending(finished_calls)
    out = runner.execute_resolve(pending, skip_rids={"req1"})
    assert finished_calls == [], "caller-provided skip_rids must skip the rescan"
    assert out is not None
    # req1 was skipped: its generation_steps did not advance.
    steps = [r.data.generation_steps for r in pending.scheduler_output.requests]
    assert steps == [1, 0, 1]


def test_execute_resolve_computes_skip_rids_when_not_passed():
    finished_calls: list[int] = []
    runner = _make_runner()
    pending = _make_pending(finished_calls)
    out = runner.execute_resolve(pending)
    assert sorted(finished_calls) == [0, 1, 2]
    assert out is not None
    steps = [r.data.generation_steps for r in pending.scheduler_output.requests]
    assert steps == [1, 1, 1]
