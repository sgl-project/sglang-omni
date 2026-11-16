# SPDX-License-Identifier: Apache-2.0
"""Queue order and model path events of MiniMax Music 3 CFG row pairs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minimax_music3.scheduler import (
    MiniMaxMusic3SchedulePolicy,
    MiniMaxMusic3Scheduler,
)
from sglang_omni.models.minimax_music3.sglang_request_builder import cfg_uncond_rid
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def queued_request(request_id: str) -> Req:
    return Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def cfg_pair(request_id: str) -> list[Req]:
    return [queued_request(request_id), queued_request(cfg_uncond_rid(request_id))]


def test_cfg_twin_is_sorted_with_its_request_under_priority_scheduling() -> None:
    """The twin had no priority or queue entry time, so sorting the queue raised or moved it ahead of its request."""
    earlier = queued_request("a")
    cond, uncond = cfg_pair("b")
    for entry_time, req in enumerate([earlier, cond], start=1):
        req.priority = 3
        req.time_stats.wait_queue_entry_time = float(entry_time)
    cond._coalesce_enqueue_t = 0.0  # noqa: leading-underscore  # upstream spelling
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
    scheduler.waiting_queue = [earlier, cond]
    scheduler.enqueue_cfg_uncond(SimpleNamespace(req=cond), SimpleNamespace(req=uncond))
    SchedulePolicy._sort_by_priority_and_fcfs(  # noqa: leading-underscore  # upstream spelling
        scheduler.waiting_queue, 1
    )
    assert [req.rid for req in scheduler.waiting_queue] == ["a", "b", "b-cfg"]


def test_cfg_pairs_stay_together_when_the_policy_reorders_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """schedule_policy=random placed each row alone, and the runner rejects a batch whose pairs are not adjacent."""
    queue = [*cfg_pair("a"), *cfg_pair("b")]

    def calc_priority(
        policy: SchedulePolicy,
        waiting_queue: list[Req],
        running_batch: ScheduleBatch | None,
        processed_tokens: int,
    ) -> None:
        # An order random.shuffle can produce.
        waiting_queue[:] = [waiting_queue[index] for index in (3, 0, 2, 1)]

    monkeypatch.setattr(SchedulePolicy, "calc_priority", calc_priority)
    policy = MiniMaxMusic3SchedulePolicy.__new__(MiniMaxMusic3SchedulePolicy)
    policy.calc_priority(queue, None, 0)
    assert [req.rid for req in queue] == ["a", "a-cfg", "b", "b-cfg"]


def test_finished_unconditioned_row_ends_its_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished unconditioned row emitted the start of its model path and never the end."""
    ended: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_end",
        lambda request_id, status: ended.append((request_id, status)),
    )
    monkeypatch.setattr(
        OmniScheduler,
        "stream_output",
        lambda scheduler, reqs, return_logprob=False, skip_req=None: None,
    )
    twin = SimpleNamespace(
        rid="a-cfg",
        omni_data=SimpleNamespace(is_cfg_uncond=True),
        finished=lambda: True,
    )
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
    scheduler.prefill_start_done = {"a-cfg"}
    # Closing a request forgets that its model path started.
    scheduler.close_completed_request = (
        lambda req: scheduler.prefill_start_done.discard(req.rid)
    )
    scheduler.stream_output([twin])
    scheduler.stream_output([twin])
    assert ended == [("a-cfg", "success")]
