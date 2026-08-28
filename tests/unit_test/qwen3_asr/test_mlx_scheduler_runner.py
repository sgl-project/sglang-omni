# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


class _DecodeMode:
    @staticmethod
    def is_decode() -> bool:
        return True


class _Batch:
    def __init__(self, request_ids: list[str]):
        self.forward_mode = _DecodeMode()
        self.reqs = [SimpleNamespace(rid=request_id) for request_id in request_ids]

    def copy(self):
        return _Batch([req.rid for req in self.reqs])


class _Worker:
    def __init__(self, next_token_ids=None):
        self.calls = []
        self.next_token_ids = next_token_ids

    @staticmethod
    def _launch(lazy_tokens, decode):
        return SimpleNamespace(
            lazy_tokens=lazy_tokens,
            prefills=[],
            extends=[],
            decode=decode,
            mode="decode",
        )

    def async_forward_batch_generation_mlx(self, batch):
        self.calls.append(("fresh", [req.rid for req in batch.reqs]))
        return self._launch("lazy-1", "decode-1")

    def async_chained_decode_mlx(self, previous):
        self.calls.append(("chained", previous))
        return self._launch("lazy-2", "decode-2")

    def finalize_mlx_result(self, launch, reqs):
        self.calls.append(("finalize", launch.decode, [req.rid for req in reqs]))
        return SimpleNamespace(next_token_ids=self.next_token_ids)


class _Runner(MlxSchedulerModelRunner):
    def __init__(self, worker):
        self.tp_worker = worker
        self._last_mlx_pending = None
        self._execution_bridge = None
        self.finalized = []

    def _finalize(
        self,
        batch_result,
        forward_batch,
        schedule_batch,
        scheduler_output,
        skip_rids=None,
    ):
        del batch_result, forward_batch, schedule_batch, scheduler_output
        self.finalized.append(skip_rids or set())
        return "resolved"


def _scheduler_output(request_id: str) -> SchedulerOutput:
    req = SimpleNamespace(finished=lambda: False, is_retracted=False)
    return SchedulerOutput(
        requests=[
            SchedulerRequest(
                request_id=request_id,
                data=SimpleNamespace(req=req),
            )
        ],
        batch_data=_Batch([request_id]),
    )


def test_mlx_scheduler_runner_launches_then_chains_before_resolve() -> None:
    worker = _Worker()
    runner = _Runner(worker)
    scheduler_output = _scheduler_output("req")

    first = runner.execute_launch(scheduler_output)
    second = runner.execute_launch(scheduler_output)

    assert worker.calls[:2] == [("fresh", ["req"]), ("chained", "decode-1")]
    assert runner.execute_resolve(first) == "resolved"
    assert runner._last_mlx_pending is second
    assert runner.execute_resolve(second) == "resolved"
    assert runner._last_mlx_pending is None
    assert worker.calls[2:] == [
        ("finalize", "decode-1", ["req"]),
        ("finalize", "decode-2", ["req"]),
    ]


def test_mlx_scheduler_runner_rejects_changed_chained_batch() -> None:
    runner = _Runner(_Worker())
    previous = runner.execute_launch(_scheduler_output("req-a"))

    with pytest.raises(RuntimeError, match="unchanged request batch"):
        runner.execute_launch(_scheduler_output("req-b"))

    # A failed successor launch must not orphan the scheduler-owned lazy step.
    assert runner._last_mlx_pending is previous
    assert runner.execute_resolve(previous) == "resolved"
    assert runner._last_mlx_pending is None

    runner.execute_launch(_scheduler_output("req-b"))
    assert runner.tp_worker.calls[-1] == ("fresh", ["req-b"])


def test_mlx_scheduler_runner_drains_before_changed_chain() -> None:
    runner = _Runner(_Worker())
    runner.execute_launch(_scheduler_output("req-a"))
    sampling_params = SimpleNamespace(
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        min_new_tokens=0,
    )
    changed_batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid="req-b",
                sampling_params=sampling_params,
                custom_logit_processor=None,
            )
        ]
    )

    assert not runner.lookahead_eligible(changed_batch)


def test_mlx_scheduler_runner_clears_chain_after_resolve_failure() -> None:
    worker = _Worker()
    runner = _Runner(worker)
    pending = runner.execute_launch(_scheduler_output("req"))

    def fail_finalize(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("failed finalize")

    worker.finalize_mlx_result = fail_finalize
    with pytest.raises(RuntimeError, match="failed finalize"):
        runner.execute_resolve(pending)

    assert runner._last_mlx_pending is None


def test_mlx_scheduler_runner_limits_lookahead_to_concurrency_one() -> None:
    runner = _Runner(_Worker())
    sampling_params = SimpleNamespace(
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        min_new_tokens=0,
    )

    def request(rid: str):
        return SimpleNamespace(
            rid=rid,
            sampling_params=sampling_params,
            custom_logit_processor=None,
        )

    assert runner.lookahead_eligible(SimpleNamespace(reqs=[request("a")]))
    assert not runner.lookahead_eligible(
        SimpleNamespace(reqs=[request("a"), request("b")])
    )


def test_mlx_scheduler_runner_uses_future_map_bridge(monkeypatch) -> None:
    import sglang.srt.managers.overlap_utils as overlap_utils

    resolved = []
    bridge = SimpleNamespace(
        future_map=object(),
        published=[],
        publish_next_tokens=lambda batch, tokens: bridge.published.append(
            (batch, tokens)
        ),
    )
    monkeypatch.setattr(
        overlap_utils,
        "resolve_forward_inputs",
        lambda batch, future_map: resolved.append((batch, future_map)),
    )
    runner = _Runner(_Worker(next_token_ids="token-ids"))
    runner._execution_bridge = bridge
    scheduler_output = _scheduler_output("req")

    pending = runner.execute_launch(scheduler_output)
    runner.execute_resolve(pending)

    assert resolved == [(scheduler_output.batch_data, bridge.future_map)]
    assert bridge.published == [(pending.schedule_batch, "token-ids")]
