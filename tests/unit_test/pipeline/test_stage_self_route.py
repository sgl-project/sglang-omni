# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeScheduler,
    make_stage_payload,
)
from tests.unit_test.pipeline.helpers import make_stage


def test_stage_self_route_preserves_reentered_request_state() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        scheduler = FakeScheduler()
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "thinker",
            endpoints={"thinker": "inproc://thinker"},
            scheduler=scheduler,
            same_process_targets={"thinker"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register(stage_obj)
        stage_obj._active_requests.add("req-reentry")

        await stage_obj._route_result(
            "req-reentry",
            make_stage_payload(request_id="req-reentry", data={"phase": 2}),
        )

        queued = scheduler.inbox.get_nowait()
        assert queued.request_id == "req-reentry"
        assert queued.data.data == {"phase": 2}
        assert "req-reentry" in stage_obj._active_requests

    asyncio.run(_run())


def test_stage_abort_during_self_route_suppresses_reentry() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()

        class _AbortBeforeDispatch(LocalStageDispatcher):
            async def send_payload(self, **kwargs) -> None:
                stage_obj._on_abort(kwargs["request_id"])
                await super().send_payload(**kwargs)

        dispatcher = _AbortBeforeDispatch()
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "thinker",
            endpoints={"thinker": "inproc://thinker"},
            scheduler=scheduler,
            same_process_targets={"thinker"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register(stage_obj)
        stage_obj._active_requests.add("req-abort-reentry")

        await stage_obj._route_result(
            "req-abort-reentry",
            make_stage_payload(request_id="req-abort-reentry", data={"phase": 2}),
        )

        assert scheduler.inbox.empty()
        assert scheduler.aborted == ["req-abort-reentry"]
        assert "req-abort-reentry" in stage_obj._aborted
        assert "req-abort-reentry" not in stage_obj._active_requests

    asyncio.run(_run())


def test_stage_self_route_preserves_reentry_with_multi_inflight_lifecycle() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        scheduler = FakeScheduler()
        scheduler.allow_multiple_inflight_per_request = True
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "thinker",
            endpoints={"thinker": "inproc://thinker"},
            scheduler=scheduler,
            same_process_targets={"thinker"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register(stage_obj)
        stage_obj._active_requests.add("req-multi-reentry")
        stage_obj._inflight_work_pending["req-multi-reentry"] = 2

        await stage_obj._route_result(
            "req-multi-reentry",
            make_stage_payload(request_id="req-multi-reentry", data={"phase": 2}),
        )

        queued = scheduler.inbox.get_nowait()
        assert queued.request_id == "req-multi-reentry"
        assert queued.data.data == {"phase": 2}
        assert "req-multi-reentry" in stage_obj._active_requests
        assert stage_obj._inflight_work_pending["req-multi-reentry"] == 2

    asyncio.run(_run())
