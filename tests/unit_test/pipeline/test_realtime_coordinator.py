# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sglang_omni.pipeline.realtime_coordinator import RealtimeCoordinator
from sglang_omni.proto import (
    CompleteMessage,
    InputUpdateMessage,
    OmniRequest,
    SubmitMessage,
)
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


def _input_update(
    seq_no: int,
    token_ids: tuple[int, ...] = (),
    byte_count: int = 0,
    input_done: bool = False,
    *,
    request_id: str = "request-1",
    session_id: str = "session-1",
    turn_id: str = "turn-1",
) -> InputUpdateMessage:
    return InputUpdateMessage(
        request_id=request_id,
        session_id=session_id,
        turn_id=turn_id,
        seq_no=seq_no,
        token_ids=token_ids,
        byte_count=byte_count,
        input_done=input_done,
    )


def _request() -> OmniRequest:
    return OmniRequest(inputs={"text": ""})


def _coordinator(
    control_plane: RecordingCoordinatorControlPlane | None = None,
    *,
    extra_input_stage: str | None = None,
) -> RealtimeCoordinator:
    coordinator = RealtimeCoordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocessing",
        terminal_stages=["vocoder"],
    )
    coordinator.control_plane = control_plane or RecordingCoordinatorControlPlane()
    coordinator.register_stage("preprocessing", "inproc://preprocessing")
    coordinator.register_stage("tts_engine", "inproc://tts_engine")
    if extra_input_stage is not None:
        coordinator.register_stage(extra_input_stage, f"inproc://{extra_input_stage}")
    return coordinator


def test_realtime_coordinator_accepts_full_mp_runner_construction_kwargs() -> None:
    """mp_runner hands custom coordinators the same kwargs as Coordinator."""
    from sglang_omni.config.topology import LogicalProcessPlan
    from sglang_omni.pipeline.replicas import ReplicaTopology

    coordinator = RealtimeCoordinator(
        completion_endpoint="inproc://complete",
        abort_endpoint="inproc://abort",
        entry_stage="preprocessing",
        terminal_stages=["vocoder"],
        terminal_stages_resolver=None,
        replica_topology=ReplicaTopology(),
        logical_process_plan=LogicalProcessPlan(processes=(), stage_to_process={}),
        binding_policy=None,
        max_in_flight=4,
    )

    assert coordinator.entry_stage == "preprocessing"
    assert coordinator.max_in_flight == 4


async def _open_realtime(
    coordinator: RealtimeCoordinator,
    request_id: str = "request-1",
    *,
    input_stage: str = "tts_engine",
):
    return await coordinator.open_realtime(
        request_id,
        _request(),
        session_id="session-1",
        turn_id="turn-1",
        input_stage=input_stage,
    )


def test_realtime_handle_routes_model_owned_messages_without_scheduler_state() -> None:
    async def _run() -> None:
        control_plane = RecordingCoordinatorControlPlane()
        coordinator = _coordinator(control_plane)
        handle = await _open_realtime(coordinator)

        message = _input_update(0, token_ids=(7, 8), byte_count=4)
        await handle.send_input(message)
        await handle.send_input(message)

        routed = [
            message
            for _, _, message in control_plane.submitted
            if isinstance(message, InputUpdateMessage)
        ]
        assert routed == [message, message]
        assert handle.request_id == "request-1"
        assert handle.input_stage == "tts_engine"

        await handle.aclose()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "message",
    [
        _input_update(0, request_id="other-request"),
        _input_update(0, session_id="other-session"),
        _input_update(0, turn_id="other-turn"),
    ],
)
def test_realtime_handle_rejects_mismatched_message_identity(
    message: InputUpdateMessage,
) -> None:
    async def _run() -> None:
        control_plane = RecordingCoordinatorControlPlane()
        coordinator = _coordinator(control_plane)
        handle = await _open_realtime(coordinator)

        with pytest.raises(ValueError, match="must match"):
            await handle.send_input(message)

        assert not any(
            isinstance(routed, InputUpdateMessage)
            for _, _, routed in control_plane.submitted
        )
        await handle.aclose()

    asyncio.run(_run())


def test_realtime_handle_routes_to_custom_registered_stage() -> None:
    async def _run() -> None:
        custom_stage = "custom_realtime_stage"
        control_plane = RecordingCoordinatorControlPlane()
        coordinator = _coordinator(
            control_plane,
            extra_input_stage=custom_stage,
        )
        handle = await _open_realtime(coordinator, input_stage=custom_stage)

        await handle.send_input(_input_update(0, token_ids=(7,), input_done=True))

        target, endpoint, routed = control_plane.submitted[-1]
        assert target == custom_stage
        assert endpoint == f"inproc://{custom_stage}"
        assert isinstance(routed, InputUpdateMessage)
        assert routed.request_id == "request-1"
        await handle.aclose()

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", ""),
        ("turn_id", " "),
        ("input_stage", 1),
        ("input_stage", "missing"),
    ],
)
def test_open_realtime_validates_identity_and_stage(field: str, value: Any) -> None:
    async def _run() -> None:
        coordinator = _coordinator()
        kwargs: dict[str, Any] = {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "input_stage": "tts_engine",
        }
        kwargs[field] = value

        with pytest.raises((TypeError, ValueError)):
            await coordinator.open_realtime("request-1", _request(), **kwargs)

        assert coordinator._realtime_requests == {}
        assert coordinator._requests == {}

    asyncio.run(_run())


class _FailNextInputControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_input = True

    async def submit_to_stage(self, stage: str, endpoint: str, msg: Any) -> None:
        if isinstance(msg, InputUpdateMessage) and self.fail_next_input:
            self.fail_next_input = False
            raise RuntimeError("input send failed")
        await super().submit_to_stage(stage, endpoint, msg)


def test_realtime_handle_retries_update_after_control_send_failure() -> None:
    async def _run() -> None:
        control_plane = _FailNextInputControlPlane()
        coordinator = _coordinator(control_plane)
        handle = await _open_realtime(coordinator)
        update = _input_update(0, token_ids=(7,))

        with pytest.raises(RuntimeError, match="input send failed"):
            await handle.send_input(update)

        await handle.send_input(update)
        await handle.aclose()

    asyncio.run(_run())


class _BlockingInputControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.first_input_started = asyncio.Event()
        self.release_first_input = asyncio.Event()
        self.input_calls = 0

    async def submit_to_stage(self, stage: str, endpoint: str, msg: Any) -> None:
        if isinstance(msg, InputUpdateMessage):
            self.input_calls += 1
            if self.input_calls == 1:
                self.first_input_started.set()
                await self.release_first_input.wait()
        await super().submit_to_stage(stage, endpoint, msg)


def test_realtime_handle_serializes_concurrent_updates() -> None:
    async def _run() -> None:
        control_plane = _BlockingInputControlPlane()
        coordinator = _coordinator(control_plane)
        handle = await _open_realtime(coordinator)

        first = asyncio.create_task(handle.send_input(_input_update(0)))
        await control_plane.first_input_started.wait()
        second = asyncio.create_task(handle.send_input(_input_update(1)))
        await asyncio.sleep(0)
        assert control_plane.input_calls == 1

        control_plane.release_first_input.set()
        await asyncio.gather(first, second)
        assert control_plane.input_calls == 2
        await handle.aclose()

    asyncio.run(_run())


class _BlockingSubmitControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.submit_started = asyncio.Event()
        self.release_submit = asyncio.Event()

    async def submit_to_stage(self, stage: str, endpoint: str, msg: Any) -> None:
        if isinstance(msg, SubmitMessage):
            self.submit_started.set()
            await self.release_submit.wait()
        await super().submit_to_stage(stage, endpoint, msg)


def test_open_realtime_returns_only_after_entry_stage_submission() -> None:
    async def _run() -> None:
        control_plane = _BlockingSubmitControlPlane()
        coordinator = _coordinator(control_plane)
        opening = asyncio.create_task(_open_realtime(coordinator))

        await control_plane.submit_started.wait()
        await asyncio.sleep(0)
        assert opening.done() is False
        assert coordinator._realtime_requests["request-1"].info.state.value == "running"
        assert coordinator._requests == {}

        control_plane.release_submit.set()
        handle = await opening
        await handle.send_input(_input_update(0, token_ids=(7,)))
        assert isinstance(control_plane.submitted[-1][2], InputUpdateMessage)
        await handle.aclose()

    asyncio.run(_run())


class _FailSubmitControlPlane(RecordingCoordinatorControlPlane):
    async def submit_to_stage(self, stage: str, endpoint: str, msg: Any) -> None:
        del stage, endpoint, msg
        raise RuntimeError("entry submit failed")


def test_open_realtime_propagates_submit_failure_and_cleans_state() -> None:
    async def _run() -> None:
        coordinator = _coordinator(_FailSubmitControlPlane())

        with pytest.raises(RuntimeError, match="entry submit failed"):
            await _open_realtime(coordinator)

        assert coordinator._realtime_requests == {}
        assert coordinator._requests == {}

    asyncio.run(_run())


class _CompletionDuringSubmitControlPlane(RecordingCoordinatorControlPlane):
    def __init__(
        self, coordinator: RealtimeCoordinator, completion: CompleteMessage
    ) -> None:
        super().__init__()
        self.coordinator = coordinator
        self.completion = completion

    async def submit_to_stage(self, stage: str, endpoint: str, msg: Any) -> None:
        await super().submit_to_stage(stage, endpoint, msg)
        if isinstance(msg, SubmitMessage):
            await self.coordinator._handle_completion(self.completion)


@pytest.mark.parametrize(
    ("success", "result", "error"),
    [
        (True, {"ok": True}, None),
        (False, None, "entry failed immediately"),
    ],
)
def test_open_realtime_preserves_fast_terminal_completion(
    success: bool,
    result: dict | None,
    error: str | None,
) -> None:
    async def _run() -> None:
        coordinator = _coordinator()
        coordinator.control_plane = _CompletionDuringSubmitControlPlane(
            coordinator,
            CompleteMessage(
                request_id="request-1",
                from_stage="vocoder" if success else "tts_engine",
                success=success,
                result=result,
                error=error,
            ),
        )

        handle = await _open_realtime(coordinator)
        with pytest.raises(ValueError, match="not active"):
            await handle.send_input(_input_update(0))

        if success:
            assert [message async for message in handle] == [
                CompleteMessage(
                    request_id="request-1",
                    from_stage="vocoder",
                    success=True,
                    result=result,
                )
            ]
        else:
            with pytest.raises(RuntimeError, match=error or ""):
                async for _ in handle:
                    pass
        assert coordinator._realtime_requests == {}
        assert coordinator._requests == {}

    asyncio.run(_run())


def test_realtime_handle_cleans_tracking_on_terminal_paths() -> None:
    async def _run() -> None:
        coordinator = _coordinator()

        completed = await _open_realtime(coordinator, "completed")
        await coordinator._handle_completion(
            CompleteMessage("completed", "vocoder", True, result={"ok": True})
        )
        assert len([message async for message in completed]) == 1

        aborted = await _open_realtime(coordinator, "aborted")
        await aborted.aclose()

        failed = await _open_realtime(coordinator, "failed")
        await coordinator._handle_completion(
            CompleteMessage("failed", "tts_engine", False, error="boom")
        )
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in failed:
                pass

        pending = await _open_realtime(coordinator, "pending")
        await coordinator.fail_pending_requests(RuntimeError("stage died"))
        with pytest.raises(RuntimeError, match="stage died"):
            async for _ in pending:
                pass

        assert coordinator._realtime_requests == {}
        assert coordinator._requests == {}

    asyncio.run(_run())


def test_ordinary_stream_path_does_not_require_realtime_configuration() -> None:
    async def _run() -> None:
        control_plane = RecordingCoordinatorControlPlane()
        coordinator = RealtimeCoordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocessing",
            terminal_stages=["vocoder"],
        )
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocessing", "inproc://preprocessing")

        messages = []

        async def consume() -> None:
            async for message in coordinator.stream(
                "ordinary", OmniRequest(inputs="hello")
            ):
                messages.append(message)

        task = asyncio.create_task(consume())
        for _ in range(10):
            if "ordinary" in coordinator._requests:
                break
            await asyncio.sleep(0)
        submitted = control_plane.submitted[0][2]
        assert isinstance(submitted, SubmitMessage)

        await coordinator._handle_completion(
            CompleteMessage("ordinary", "vocoder", True, result={"ok": True})
        )
        await asyncio.wait_for(task, timeout=1)
        assert len(messages) == 1
        assert coordinator._requests == {}
        assert coordinator._realtime_requests == {}

    asyncio.run(_run())


class _BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()

    async def broadcast_abort(self, msg: Any) -> None:
        self.aborts.append(msg)
        self.abort_started.set()
        await self.release_abort.wait()


def test_realtime_abort_reserves_request_id_until_broadcast_completes() -> None:
    async def _run() -> None:
        control_plane = _BlockingAbortControlPlane()
        coordinator = _coordinator(control_plane)
        handle = await _open_realtime(coordinator)

        close_task = asyncio.create_task(handle.aclose())
        await control_plane.abort_started.wait()

        assert "request-1" in coordinator._realtime_requests
        assert "request-1" in coordinator._abort_tasks
        with pytest.raises(ValueError, match="already exists"):
            await _open_realtime(coordinator)
        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("request-1", _request())

        control_plane.release_abort.set()
        await close_task
        assert coordinator._realtime_requests == {}
        assert coordinator._abort_tasks == {}

    asyncio.run(_run())


def test_realtime_close_cancellation_does_not_cancel_abort_broadcast() -> None:
    async def _run() -> None:
        control_plane = _BlockingAbortControlPlane()
        coordinator = _coordinator(control_plane)
        handle = await _open_realtime(coordinator)

        close_task = asyncio.create_task(handle.aclose())
        await control_plane.abort_started.wait()
        abort_task = coordinator._abort_tasks["request-1"]

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert abort_task.cancelled() is False
        assert "request-1" in coordinator._realtime_requests

        control_plane.release_abort.set()
        assert await abort_task is True
        await asyncio.sleep(0)
        assert coordinator._realtime_requests == {}
        assert coordinator._abort_tasks == {}
        assert [message.request_id for message in control_plane.aborts] == ["request-1"]

    asyncio.run(_run())


def test_completed_ordinary_stream_reserves_id_against_realtime_until_closed() -> None:
    async def _run() -> None:
        coordinator = _coordinator()
        stream = coordinator.stream("request-1", _request())
        terminal = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "request-1" in coordinator._requests:
                break
            await asyncio.sleep(0)

        await coordinator._handle_completion(
            CompleteMessage("request-1", "vocoder", True, result={"ok": True})
        )
        assert (await terminal).result == {"ok": True}
        assert "request-1" not in coordinator._requests
        assert "request-1" in coordinator._stream_queues

        with pytest.raises(ValueError, match="already exists"):
            await _open_realtime(coordinator)

        await stream.aclose()
        handle = await _open_realtime(coordinator)
        await handle.aclose()

    asyncio.run(_run())
