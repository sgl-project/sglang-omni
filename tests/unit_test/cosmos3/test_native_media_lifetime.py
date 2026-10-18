# SPDX-License-Identifier: Apache-2.0
"""Request-owned files remain valid until result routing settles."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.cosmos3.stages import NativeGenerationScheduler
from sglang_omni.pipeline.control_plane import PushSocket, StageControlPlane
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane
from tests.unit_test.pipeline.helpers import RecordingStageControlPlane, make_stage


class SavedGenerator:
    local_scheduler_process = None

    def generate(self, *, sampling_params_kwargs):
        # Native names the output after the request's output_file_name.
        name = sampling_params_kwargs["output_file_name"]
        target = Path(sampling_params_kwargs["output_path"]) / f"{name}.png"
        target.write_bytes(b"native result")
        return SimpleNamespace(
            output_file_path=str(target),
            size=(1, 1, 1),
            prompt="image",
            generation_time=1.0,
            peak_memory_mb=1.0,
            metrics={},
        )

    def shutdown(self):
        pass


class SavedControlPlane(RecordingStageControlPlane):
    async def send_complete(self, message, *, on_submitted=None):
        await super().send_complete(message)
        if on_submitted is not None:
            on_submitted()


def queued_result(tmp_path):
    keep = tmp_path / "preexisting.txt"
    keep.write_text("keep")
    scheduler = NativeGenerationScheduler(SavedGenerator(), str(tmp_path))
    payload = StagePayload("request", OmniRequest("image"), None)
    result = scheduler.generate(payload)
    path = Path(result.data["media"][0]["path"])
    scheduler.emit_result(payload.request_id, result, scheduler.outbox)
    stage = make_stage(
        name="generation",
        scheduler=scheduler,
        is_terminal=True,
        control_plane=SavedControlPlane(),
    )
    stage.active_requests.add(payload.request_id)
    # A drain routes what the scheduler published before this marker, then ends.
    stage.end_outbox_drain()
    return scheduler, stage, path, keep


@pytest.mark.parametrize("drop", ["abort", "inactive", "shutdown"])
def test_queued_result_drop_releases_only_its_owned_files(tmp_path, drop):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)
        assert path.read_bytes() == b"native result"
        if drop == "abort":
            stage.on_abort("request")
        elif drop == "inactive":
            stage.active_requests.clear()
        else:
            scheduler.stop()
        await asyncio.wait_for(stage.drain_outbox(), 2)
        assert stage.control_plane.completions == []
        assert not path.exists()
        assert list(tmp_path.iterdir()) == [keep]
        assert keep.read_text() == "keep"
        assert scheduler.native_requests == {}

    asyncio.run(run())


class BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self):
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()

    async def broadcast_abort(self, msg):
        await super().broadcast_abort(msg)
        self.abort_started.set()
        await self.release_abort.wait()


@pytest.mark.parametrize("order", [None, "completion_first", "abort_first"])
def test_delivered_media_belongs_to_whoever_receives_the_completion(tmp_path, order):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="generation",
            terminal_stages=["generation"],
        )
        coordinator.control_plane = BlockingAbortControlPlane()
        coordinator.register_stage("generation", "inproc://generation")
        await coordinator.submit_request("request", "image")
        future = coordinator.completion_futures["request"]
        await asyncio.wait_for(stage.drain_outbox(), 2)
        message = stage.control_plane.completions[0]
        assert scheduler.native_requests == {}
        if order is None:
            await coordinator.handle_completion(message)
        else:
            # The abort broadcast reaches the stage after it delivered.
            abort = asyncio.create_task(coordinator.abort("request"))
            await asyncio.wait_for(coordinator.control_plane.abort_started.wait(), 2)
            stage.on_abort("request")
            if order == "completion_first":
                await coordinator.handle_completion(message)
                coordinator.control_plane.release_abort.set()
                assert await abort is False
            else:
                coordinator.control_plane.release_abort.set()
                assert await abort is True
                await coordinator.handle_completion(message)
        scheduler.stop()
        if order == "abort_first":
            # The coordinator drops the completion, so no caller owns it.
            with pytest.raises(asyncio.CancelledError):
                await future
            assert not path.exists()
            assert list(tmp_path.iterdir()) == [keep]
        else:
            chunk = Client.default_result_builder("request", future.result())
            assert Path(chunk.media[0]["path"]) == path
            assert path.read_bytes() == b"native result"
        assert keep.read_text() == "keep"

    asyncio.run(run())


def test_failed_result_route_releases_owned_output(tmp_path):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)

        async def fail_send(message, *, on_submitted=None):
            raise RuntimeError("completion transport failed")

        stage.control_plane.send_complete = fail_send
        with pytest.raises(RuntimeError, match="completion transport failed"):
            await asyncio.wait_for(stage.drain_outbox(), 2)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == [keep]
        assert scheduler.native_requests == {}

    asyncio.run(run())


@pytest.mark.parametrize("interrupt", ["abort", "shutdown"])
@pytest.mark.parametrize("outcome", ["delivered", "failed", "cancelled"])
def test_in_flight_result_owns_files_until_routing_settles(
    tmp_path, interrupt, outcome
):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        send = stage.control_plane.send_complete

        async def held_send(message, *, on_submitted=None):
            entered.set()
            await release.wait()
            if outcome == "failed":
                raise RuntimeError("completion transport failed")
            await send(message, on_submitted=on_submitted)

        stage.control_plane.send_complete = held_send
        pending = asyncio.create_task(stage.drain_outbox())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if interrupt == "abort":
                stage.on_abort("request")
            else:
                scheduler.stop()
            assert path.read_bytes() == b"native result"
            if outcome == "cancelled":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                release.set()
                if outcome == "failed":
                    with pytest.raises(
                        RuntimeError, match="completion transport failed"
                    ):
                        await asyncio.wait_for(pending, 2)
                else:
                    await asyncio.wait_for(pending, 2)
                    result = stage.control_plane.completions[0].result
                    assert Path(result["media"][0]["path"]) == path
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        # Media the transport accepted may already be with a caller.
        assert path.exists() is (outcome == "delivered")
        assert scheduler.native_requests == {}
        assert keep.read_text() == "keep"

    asyncio.run(run())


def test_saved_paths_reject_nonterminal_delivery_before_sending(tmp_path):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)
        sent = []
        stage.get_next = lambda request_id, result: ["consumer"]
        stage.stream_targets = ["consumer"]

        async def send(*args, **kwargs):
            sent.append(args)

        stage.send_to_stage = send
        stage.send_stream_signal_to_target = send
        with pytest.raises(ValueError, match="requires terminal delivery"):
            await asyncio.wait_for(stage.drain_outbox(), 2)
        assert sent == []
        assert stage.control_plane.completions == []
        assert list(tmp_path.iterdir()) == [keep]
        assert scheduler.native_requests == {}

    asyncio.run(run())


def test_delivered_file_survives_later_local_cleanup_error(tmp_path):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)

        def fail_cleanup(*args, **kwargs):
            raise RuntimeError("local cleanup failed")

        stage.clear_request_state = fail_cleanup
        with pytest.raises(RuntimeError, match="local cleanup failed"):
            await asyncio.wait_for(stage.drain_outbox(), 2)
        assert len(stage.control_plane.completions) == 1
        scheduler.stop()
        assert path.read_bytes() == b"native result"
        assert keep.read_text() == "keep"
        assert scheduler.native_requests == {}

    asyncio.run(run())


@pytest.mark.parametrize("accepted", [False, True])
def test_cancellation_uses_raw_transport_outcome(tmp_path, accepted):
    async def run():
        scheduler, stage, path, keep = queued_result(tmp_path)
        entered = asyncio.Event()
        raw_send = asyncio.get_running_loop().create_future()

        class PendingSocket:
            def send(self, data):
                entered.set()
                return raw_send

        socket = PushSocket("unused")
        socket.socket = PendingSocket()
        control = StageControlPlane("generation", "", "", "")
        control.coordinator_socket = socket
        stage.control_plane = control
        pending = asyncio.create_task(stage.drain_outbox())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if accepted:
                # The wire send completes before the awaiting task resumes.
                raw_send.set_result(None)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        scheduler.stop()
        assert path.exists() is accepted
        assert raw_send.cancelled() is (not accepted)
        assert scheduler.native_requests == {}
        assert keep.read_text() == "keep"

    asyncio.run(run())


def test_old_result_release_cannot_delete_new_request_output(tmp_path):
    scheduler, stage, old_path, keep = queued_result(tmp_path)
    old = scheduler.outbox.get_nowait().data
    scheduler.abort("request")
    # The scheduler consumes its cancellation tombstone before admitting reuse.
    scheduler.emit_result("request", old, scheduler.outbox)
    newer = scheduler.generate(StagePayload("request", OmniRequest("new image"), None))
    new_path = Path(newer.data["media"][0]["path"])
    scheduler.release_result(old, delivered=False)
    assert not old_path.exists()
    assert new_path.read_bytes() == b"native result"
    scheduler.stop()
    assert list(tmp_path.iterdir()) == [keep]
