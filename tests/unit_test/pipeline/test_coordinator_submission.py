# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class SubmissionControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self, *, fail_abort: bool = False) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.error: Exception | None = None
        self.fail_abort = fail_abort

    async def submit_to_stage(self, stage, endpoint, msg) -> None:
        await super().submit_to_stage(stage, endpoint, msg)
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error

    async def broadcast_abort(self, msg) -> None:
        await super().broadcast_abort(msg)
        if self.fail_abort:
            raise RuntimeError("abort transport unavailable")


def make_coordinator(control_plane: SubmissionControlPlane) -> Coordinator:
    coordinator = Coordinator(
        "inproc://submission-complete",
        "inproc://submission-abort",
        entry_stage="orchestrator",
        terminal_stages=["orchestrator"],
        max_in_flight=1,
    )
    coordinator.control_plane = control_plane
    coordinator.register_stage("orchestrator", "inproc://submission-stage")
    return coordinator


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("fail_abort", [False, True])
def test_initial_submission_failure_releases_ownership_and_preserves_error(
    streaming, cancel, fail_abort
):
    async def run():
        control_plane = SubmissionControlPlane(fail_abort=fail_abort)
        coordinator = make_coordinator(control_plane)
        stream = coordinator.stream("failed", "draw and inspect") if streaming else None
        pending = asyncio.create_task(
            anext(stream)
            if stream
            else coordinator.submit("failed", "draw and inspect")
        )
        await asyncio.wait_for(control_plane.started.wait(), 1)
        future = coordinator._completion_futures["failed"]
        error = OSError("entry transport unavailable")
        if cancel:
            pending.cancel("caller cancelled during submission")
        else:
            control_plane.error = error
            control_plane.release.set()
        try:
            with pytest.raises(asyncio.CancelledError if cancel else OSError) as caught:
                await asyncio.wait_for(pending, 1)
            if cancel:
                assert str(caught.value) == "caller cancelled during submission"
            else:
                assert caught.value is error
            assert [msg.request_id for msg in control_plane.aborts] == ["failed"]
            assert future.cancelled()
            assert coordinator._requests == {}
            assert coordinator._completion_futures == {}
            assert coordinator._stream_queues == {}
            assert coordinator._partial_results == {}
            assert coordinator._abort_tasks == {}

            # A fresh request can use the released admission slot and complete.
            control_plane.error = None
            control_plane.started.clear()
            control_plane.release.set()
            recovered = asyncio.create_task(coordinator.submit("recovered", "hello"))
            await asyncio.wait_for(control_plane.started.wait(), 1)
            await coordinator._handle_completion(
                CompleteMessage("recovered", "orchestrator", True, result="done")
            )
            assert await asyncio.wait_for(recovered, 1) == "done"
            assert coordinator._requests == {}
            assert coordinator._completion_futures == {}
        finally:
            if stream is not None:
                await stream.aclose()
            for retained in coordinator._completion_futures.values():
                retained.cancel()

    asyncio.run(run())


@pytest.mark.parametrize("streaming_duplicate", [False, True])
def test_duplicate_submission_does_not_abort_or_clear_the_original_owner(
    streaming_duplicate,
):
    async def run():
        control_plane = SubmissionControlPlane()
        coordinator = make_coordinator(control_plane)
        original = coordinator.stream("shared", "original")
        pending = asyncio.create_task(anext(original))
        await asyncio.wait_for(control_plane.started.wait(), 1)
        info = coordinator._requests["shared"]
        future = coordinator._completion_futures["shared"]
        queue = coordinator._stream_queues["shared"]
        duplicate = (
            coordinator.stream("shared", "duplicate") if streaming_duplicate else None
        )
        try:
            with pytest.raises(ValueError, match="already exists"):
                if duplicate is not None:
                    await anext(duplicate)
                else:
                    await coordinator.submit("shared", "duplicate")
            assert coordinator._requests["shared"] is info
            assert coordinator._completion_futures["shared"] is future
            assert coordinator._stream_queues["shared"] is queue
            assert not future.done()
            assert control_plane.aborts == []
            assert len(control_plane.submitted) == 1

            control_plane.release.set()
            await coordinator._handle_completion(
                CompleteMessage("shared", "orchestrator", True, result="original")
            )
            assert (await asyncio.wait_for(pending, 1)).result == "original"
        finally:
            if duplicate is not None:
                await duplicate.aclose()
            await original.aclose()
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}
        assert control_plane.aborts == []

    asyncio.run(run())


def test_cancelled_submission_keeps_id_reserved_until_abort_finishes():
    async def run():
        class BlockingAbortControlPlane(SubmissionControlPlane):
            def __init__(self):
                super().__init__()
                self.abort_started = asyncio.Event()
                self.release_abort = asyncio.Event()

            async def broadcast_abort(self, msg):
                await super().broadcast_abort(msg)
                self.abort_started.set()
                await self.release_abort.wait()

        control_plane = BlockingAbortControlPlane()
        coordinator = make_coordinator(control_plane)
        pending = asyncio.create_task(coordinator.submit("held", "draw"))
        await asyncio.wait_for(control_plane.started.wait(), 1)
        future = coordinator._completion_futures["held"]
        pending.cancel("original cancellation")
        try:
            await asyncio.wait_for(control_plane.abort_started.wait(), 1)
            assert not pending.done()
            with pytest.raises(ValueError, match="already exists"):
                await coordinator.submit("held", "duplicate")
            pending.cancel("second cancellation while aborting")
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await asyncio.wait_for(pending, 1)
            assert future.cancelled()
            assert coordinator._requests == {}
            assert coordinator._completion_futures == {}
            assert "held" in coordinator._abort_tasks
            with pytest.raises(ValueError, match="already exists"):
                await coordinator.submit("held", "duplicate")
            assert len(control_plane.submitted) == 1
        finally:
            control_plane.release_abort.set()
            await asyncio.gather(*coordinator._abort_tasks.values())
            await asyncio.gather(pending, return_exceptions=True)
        assert coordinator._abort_tasks == {}
        assert [msg.request_id for msg in control_plane.aborts] == ["held"]

    asyncio.run(run())
