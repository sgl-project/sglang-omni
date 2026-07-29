# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64

import pytest

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import (
    MAX_INLINE_SESSION_COMMAND_BYTES,
    AbortMessage,
    CompleteMessage,
    OmniRequest,
    SessionCommandMessage,
    StreamMessage,
    SubmitMessage,
)
from sglang_omni.scheduling.messages import OutgoingMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeScheduler,
    RecordingCoordinatorControlPlane,
    RecordingStageControlPlane,
    make_stage_payload,
)
from tests.unit_test.pipeline.helpers import make_stage


def test_session_command_message_round_trip_and_strict_validation() -> None:
    message = SessionCommandMessage(
        session_id="session-1",
        generation=2,
        input_seq=3,
        response_epoch=1,
        command="playback_ack",
        data={"audio_end_ms": 1280.0},
    )

    restored = deserialize_message(serialize_message(message))

    assert restored == message
    assert restored.to_dict()["type"] == "session_command"

    valid = message.to_dict()
    with pytest.raises(ValueError, match="unknown fields"):
        SessionCommandMessage.from_dict({**valid, "extra": True})
    with pytest.raises(ValueError, match="missing fields"):
        SessionCommandMessage.from_dict(
            {key: value for key, value in valid.items() if key != "input_seq"}
        )
    with pytest.raises(TypeError, match="positive int"):
        SessionCommandMessage.from_dict({**valid, "generation": True})
    with pytest.raises(ValueError, match="command must be one of"):
        SessionCommandMessage.from_dict({**valid, "command": "unknown"})
    with pytest.raises(TypeError, match="data must be a dict"):
        SessionCommandMessage.from_dict({**valid, "data": b"audio"})
    with pytest.raises(ValueError, match="inline wire limit"):
        SessionCommandMessage(
            "session-1",
            1,
            1,
            0,
            "append",
            {"audio": b"x" * MAX_INLINE_SESSION_COMMAND_BYTES},
        ).to_dict()

    # The generic wire guard must accommodate a model-level 4 MiB decoded
    # media budget after base64 expansion plus the command envelope.
    encoded_media = base64.b64encode(b"x" * (4 * 1024 * 1024)).decode("ascii")
    SessionCommandMessage(
        "session-1",
        1,
        1,
        0,
        "append",
        {"audio_base64": encoded_media},
    ).to_dict()


def test_abort_message_generation_round_trip_is_backward_compatible() -> None:
    fenced = AbortMessage("session-1", generation=3)
    assert deserialize_message(serialize_message(fenced)) == fenced
    assert AbortMessage.from_dict({"type": "abort", "request_id": "req-1"}) == (
        AbortMessage("req-1")
    )


def test_session_output_generation_round_trip_is_backward_compatible() -> None:
    complete = CompleteMessage(
        "session-1", "duplex", True, result={"closed": True}, generation=3
    )
    stream = StreamMessage(
        "session-1",
        "duplex",
        {"type": "response.output.delta"},
        generation=3,
    )

    assert deserialize_message(serialize_message(complete)) == complete
    assert deserialize_message(serialize_message(stream)) == stream
    assert (
        CompleteMessage.from_dict(
            {
                "type": "complete",
                "request_id": "req-1",
                "from_stage": "stage",
                "success": True,
            }
        ).generation
        is None
    )
    assert (
        StreamMessage.from_dict(
            {
                "type": "stream",
                "request_id": "req-1",
                "from_stage": "stage",
                "chunk": {"ok": True},
            }
        ).generation
        is None
    )


def test_coordinator_orders_session_commands_and_reuses_generation() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")

        generation, events = await coordinator.open_session(
            "session-1",
            OmniRequest(inputs={"prompt": "hello"}),
        )
        assert generation == 1
        submit = control_plane.submitted[0][2]
        assert isinstance(submit, SubmitMessage)
        assert submit.data.request.metadata["duplex_session"] == {
            "session_id": "session-1",
            "generation": 1,
            "input_seq": 0,
            "response_epoch": 0,
        }

        append = await coordinator.send_session_command(
            "session-1", 1, "append", {"audio": [0.0]}
        )
        playback = await coordinator.send_session_command(
            "session-1", 1, "playback_ack", {"audio_end_ms": 20.0}
        )
        interrupt = await coordinator.send_session_command("session-1", 1, "interrupt")
        close = await coordinator.send_session_command("session-1", 1, "close")

        assert [
            append.input_seq,
            playback.input_seq,
            interrupt.input_seq,
            close.input_seq,
        ] == [
            1,
            2,
            3,
            4,
        ]
        assert [
            append.response_epoch,
            playback.response_epoch,
            interrupt.response_epoch,
            close.response_epoch,
        ] == [0, 0, 1, 2]
        with pytest.raises(RuntimeError, match="closing"):
            await coordinator.send_session_command("session-1", 1, "append", {})
        with pytest.raises(ValueError, match="Stale session generation"):
            await coordinator.send_session_command("session-1", 2, "append", {})

        await coordinator._handle_completion(
            CompleteMessage(
                request_id="session-1",
                from_stage="duplex",
                success=True,
                result={"closed": True},
                generation=1,
            )
        )
        terminal = await events.__anext__()
        assert isinstance(terminal, CompleteMessage)
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()

        generation, reopened = await coordinator.open_session(
            "session-1", OmniRequest(inputs={"prompt": "again"})
        )
        assert generation == 2
        assert (
            control_plane.submitted[-1][2].data.request.metadata["duplex_session"][
                "generation"
            ]
            == 2
        )
        assert await coordinator.abort("session-1") is True
        await reopened.aclose()

        next_generation, other = await coordinator.open_session("session-2", {})
        assert next_generation == 3
        assert await coordinator.abort("session-2") is True
        await other.aclose()

    asyncio.run(_run())


def test_coordinator_session_abort_cleans_state() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        assert coordinator.health()["active_sessions"] == 1
        assert await coordinator.abort("session-1") is True
        assert coordinator.health()["active_sessions"] == 0
        assert control_plane.aborts[-1].generation == 1
        terminal = await events.__anext__()
        assert terminal.success is False
        assert terminal.error == "aborted"
        await events.aclose()

    asyncio.run(_run())


def test_coordinator_unstarted_session_stream_aclose_releases_lease() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        await events.aclose()

        assert coordinator.health()["active_sessions"] == 0
        assert control_plane.aborts[-1].generation == 1
        assert "session-1" not in coordinator._session_stream_ids
        assert "session-1" not in coordinator._stream_queues
        assert "session-1" not in coordinator._completion_futures
        assert await coordinator.abort("session-1") is False

    asyncio.run(_run())


def test_coordinator_serializes_session_command_with_abort() -> None:
    class BlockingControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.command_started = asyncio.Event()
            self.release_command = asyncio.Event()

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            if isinstance(msg, SessionCommandMessage):
                self.command_started.set()
                await self.release_command.wait()
            await super().submit_to_stage(stage, endpoint, msg)

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        control_plane = BlockingControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        command_task = asyncio.create_task(
            coordinator.send_session_command("session-1", 1, "append", {})
        )
        await control_plane.command_started.wait()
        abort_task = asyncio.create_task(coordinator.abort("session-1"))
        await asyncio.sleep(0)
        assert not abort_task.done()

        control_plane.release_command.set()
        assert (await command_task).input_seq == 1
        assert await abort_task is True
        await events.aclose()

    asyncio.run(_run())


def test_coordinator_serializes_session_command_with_completion() -> None:
    class BlockingControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.command_started = asyncio.Event()
            self.release_command = asyncio.Event()

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            if isinstance(msg, SessionCommandMessage):
                self.command_started.set()
                await self.release_command.wait()
            await super().submit_to_stage(stage, endpoint, msg)

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        control_plane = BlockingControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        command_task = asyncio.create_task(
            coordinator.send_session_command("session-1", 1, "append", {})
        )
        await control_plane.command_started.wait()
        completion_task = asyncio.create_task(
            coordinator._handle_completion(
                CompleteMessage(
                    request_id="session-1",
                    from_stage="duplex",
                    success=True,
                    result={"closed": True},
                    generation=1,
                )
            )
        )
        await asyncio.sleep(0)
        assert not completion_task.done()

        control_plane.release_command.set()
        assert (await command_task).input_seq == 1
        await completion_task
        assert (await events.__anext__()).success is True

    asyncio.run(_run())


def test_coordinator_rejects_multi_terminal_duplex_session() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["audio", "text"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("duplex", "inproc://duplex")

        with pytest.raises(ValueError, match="exactly one active terminal"):
            await coordinator.open_session("session-1", {})
        assert coordinator.health()["active_sessions"] == 0

    asyncio.run(_run())


def test_coordinator_session_stream_fails_only_overflowing_session() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {}, output_queue_size=1)

        first = StreamMessage("session-1", "duplex", {"type": "first"}, generation=1)
        second = StreamMessage("session-1", "duplex", {"type": "second"}, generation=1)
        await coordinator._handle_stream(first)
        await asyncio.wait_for(coordinator._handle_stream(second), timeout=1)
        terminal = await events.__anext__()
        assert isinstance(terminal, CompleteMessage)
        assert terminal.success is False
        assert terminal.error == "duplex session output queue overflow"
        assert coordinator.health()["active_sessions"] == 0
        assert await coordinator.abort("session-1") is False
        await events.aclose()

    asyncio.run(_run())


def test_coordinator_successful_close_preserves_accepted_session_output() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {}, output_queue_size=1)

        output = StreamMessage(
            "session-1", "duplex", {"type": "last-output"}, generation=1
        )
        await coordinator._handle_stream(output)
        await coordinator._handle_completion(
            CompleteMessage(
                request_id="session-1",
                from_stage="duplex",
                success=True,
                result={"closed": True},
                generation=1,
            )
        )

        assert await events.__anext__() is output
        terminal = await events.__anext__()
        assert isinstance(terminal, CompleteMessage)
        assert terminal.success is True
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()

    asyncio.run(_run())


def test_coordinator_terminal_error_displaces_full_session_output_queue() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {}, output_queue_size=1)

        await coordinator._handle_stream(
            StreamMessage(
                "session-1",
                "duplex",
                {"type": "response.output.delta"},
                generation=1,
            )
        )
        terminal = StreamMessage(
            "session-1",
            "duplex",
            {
                "type": "session.error",
                "session_id": "session-1",
                "generation": 1,
                "input_seq": 0,
                "response_epoch": 0,
                "output_seq": 2,
                "error": "backend failed",
            },
            generation=1,
        )
        await coordinator._handle_stream(terminal)

        assert await events.__anext__() is terminal
        assert control_plane.aborts[-1] == AbortMessage("session-1", generation=1)
        await events.aclose()

    asyncio.run(_run())


def test_coordinator_drops_late_stream_after_session_terminal() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {}, output_queue_size=1)

        output = StreamMessage("session-1", "duplex", {"type": "output"}, generation=1)
        await coordinator._handle_stream(output)
        await coordinator._handle_completion(
            CompleteMessage(
                request_id="session-1",
                from_stage="duplex",
                success=True,
                result={"closed": True},
                generation=1,
            )
        )
        late = StreamMessage("session-1", "duplex", {"type": "late"}, generation=1)
        await asyncio.wait_for(coordinator._handle_stream(late), timeout=1)

        assert await events.__anext__() is output
        assert (await events.__anext__()).success is True
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()

    asyncio.run(_run())


def test_stage_delivers_only_active_matching_session_commands() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(
            name="duplex",
            scheduler=scheduler,
            control_plane=RecordingStageControlPlane(),
        )
        payload = make_stage_payload(request_id="session-1")
        payload.request.metadata["duplex_session"] = {
            "session_id": "session-1",
            "generation": 1,
            "input_seq": 0,
            "response_epoch": 0,
        }
        await stage._on_submit(SubmitMessage("session-1", payload))
        initial = scheduler.inbox.get_nowait()
        assert initial.type == "new_request"

        matching = SessionCommandMessage(
            "session-1", 1, 1, 0, "append", {"audio": [0.0]}
        )
        stale = SessionCommandMessage("session-1", 2, 2, 0, "append", {"audio": [1.0]})
        inactive = SessionCommandMessage(
            "session-2", 1, 1, 0, "append", {"audio": [2.0]}
        )
        stage._on_session_command(stale)
        stage._on_session_command(inactive)
        stage._on_session_command(matching)

        delivered = scheduler.inbox.get_nowait()
        assert delivered.type == "session_command"
        assert delivered.data is matching
        assert scheduler.inbox.empty()

        stage._on_abort("session-1", generation=1)
        payload.request.metadata["duplex_session"]["generation"] = 2
        await stage._on_submit(SubmitMessage("session-1", payload))
        assert scheduler.inbox.get_nowait().type == "new_request"
        stage._on_session_command(
            SessionCommandMessage("session-1", 2, 1, 0, "append", {})
        )
        assert scheduler.inbox.get_nowait().type == "session_command"

        scheduler.aborted.clear()
        stage._on_abort("session-1")
        assert stage._session_generations["session-1"] == 2
        assert scheduler.aborted == []
        stage._on_abort("session-1", generation=1)
        assert stage._session_generations["session-1"] == 2
        assert scheduler.aborted == []
        stage._on_abort("session-1", generation=2)
        assert "session-1" not in stage._active_requests
        assert scheduler.aborted == ["session-1"]

    asyncio.run(_run())


def test_stage_drops_stale_generation_outputs_after_reopen() -> None:
    async def _run() -> None:
        for role in ("single", "follower"):
            scheduler = FakeScheduler()
            control_plane = RecordingStageControlPlane()
            stage = make_stage(
                name="duplex",
                role=role,
                scheduler=scheduler,
                control_plane=control_plane,
            )
            stage._active_requests.add("session-1")
            stage._session_generations["session-1"] = 2
            stage._session_generation_watermarks["session-1"] = 2

            scheduler.outbox.put(
                OutgoingMessage(
                    "session-1",
                    "result",
                    {"generation": 1, "closed": True},
                )
            )
            await stage._drain_outbox()
            assert stage._session_generations["session-1"] == 2
            assert control_plane.completions == []

            scheduler.outbox.put(
                OutgoingMessage(
                    "session-1",
                    "result",
                    {"generation": 2, "closed": True},
                )
            )
            await stage._drain_outbox()
            assert "session-1" not in stage._active_requests
            if role == "single":
                assert control_plane.completions[-1].result == {
                    "generation": 2,
                    "closed": True,
                }

    asyncio.run(_run())


def test_stage_fences_session_abort_that_arrives_before_submit() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(name="duplex", scheduler=scheduler)
        payload = make_stage_payload(request_id="session-1")
        payload.request.metadata["duplex_session"] = {
            "session_id": "session-1",
            "generation": 1,
            "input_seq": 0,
            "response_epoch": 0,
        }

        stage._on_abort("session-1", generation=1)
        assert stage._session_generation_watermarks["session-1"] == 1
        assert scheduler.aborted == []

        await stage._on_submit(SubmitMessage("session-1", payload))
        assert scheduler.inbox.empty()
        assert "session-1" not in stage._active_requests

        payload.request.metadata["duplex_session"]["generation"] = 2
        await stage._on_submit(SubmitMessage("session-1", payload))
        assert scheduler.inbox.get_nowait().type == "new_request"

    asyncio.run(_run())


def test_stage_allows_legacy_request_id_reuse_after_session_completion() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage = make_stage(
            name="duplex",
            scheduler=scheduler,
            control_plane=control_plane,
        )
        payload = make_stage_payload(request_id="shared-id")
        payload.request.metadata["duplex_session"] = {
            "session_id": "shared-id",
            "generation": 1,
            "input_seq": 0,
            "response_epoch": 0,
        }
        await stage._on_submit(SubmitMessage("shared-id", payload))
        scheduler.inbox.get_nowait()
        scheduler.outbox.put(
            OutgoingMessage(
                "shared-id",
                "result",
                {"generation": 1, "closed": True},
                metadata={"generation": 1},
            )
        )
        await stage._drain_outbox()
        assert "shared-id" not in stage._active_requests

        legacy_payload = make_stage_payload(request_id="shared-id")
        await stage._on_submit(SubmitMessage("shared-id", legacy_payload))
        scheduler.inbox.get_nowait()
        scheduler.outbox.put(
            OutgoingMessage(
                "shared-id",
                "result",
                {"generation": "legacy-domain-value", "ok": True},
            )
        )
        await stage._drain_outbox()

        assert control_plane.completions[-1].result == {
            "generation": "legacy-domain-value",
            "ok": True,
        }

    asyncio.run(_run())


def test_stage_bounds_inactive_session_generation_watermarks() -> None:
    stage = make_stage(name="duplex")
    stage._active_requests.add("active-session")
    stage._record_session_generation_watermark("active-session", 1)
    stage._record_session_generation_watermark("reused-session", 1)

    for index in range(9998):
        stage._record_session_generation_watermark(f"finished-{index}", 1)
    # A newer generation refreshes the recency of a frequently reused id.
    stage._record_session_generation_watermark("reused-session", 2)
    stage._record_session_generation_watermark("trigger-prune", 1)

    assert "active-session" in stage._session_generation_watermarks
    assert stage._session_generation_watermarks["reused-session"] == 2
    assert len(stage._session_generation_watermarks) <= 5001


def test_tp_follower_drops_output_for_inactive_request() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(name="duplex", role="follower", scheduler=scheduler)
        scheduler.outbox.put(
            OutgoingMessage("finished-session", "error", RuntimeError("late error"))
        )

        await stage._drain_outbox()

    asyncio.run(_run())


def test_coordinator_fences_stale_outputs_after_session_reopen() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("duplex", "inproc://duplex")

        _, first_events = await coordinator.open_session("session-1", {})
        await coordinator._handle_completion(
            CompleteMessage(
                "session-1",
                "duplex",
                True,
                result={"closed": True},
                generation=1,
            )
        )
        assert (await first_events.__anext__()).generation == 1
        await first_events.aclose()

        generation, events = await coordinator.open_session("session-1", {})
        assert generation == 2
        await coordinator._handle_stream(
            StreamMessage(
                "session-1",
                "duplex",
                {"type": "stale"},
                generation=1,
            )
        )
        await coordinator._handle_completion(
            CompleteMessage(
                "session-1",
                "duplex",
                True,
                result={"closed": True},
                generation=1,
            )
        )

        assert coordinator.health()["active_sessions"] == 1
        assert coordinator._stream_queues["session-1"].empty()
        current = StreamMessage(
            "session-1",
            "duplex",
            {"type": "current"},
            generation=2,
        )
        await coordinator._handle_stream(current)
        assert await events.__anext__() is current
        assert await coordinator.abort("session-1") is True
        await events.aclose()

    asyncio.run(_run())


def test_stage_admits_downstream_session_generation_before_abort() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(name="downstream", scheduler=scheduler)
        payload = make_stage_payload(request_id="session-1")
        payload.request.metadata["duplex_session"] = {
            "session_id": "session-1",
            "generation": 7,
            "input_seq": 0,
            "response_epoch": 0,
        }

        await stage.receive_local_payload("session-1", "upstream", payload)

        assert stage._session_generations["session-1"] == 7
        assert scheduler.inbox.get_nowait().type == "new_request"
        stage._on_abort("session-1", generation=7)
        assert scheduler.aborted == ["session-1"]
        assert "session-1" not in stage._active_requests

    asyncio.run(_run())


def test_coordinator_fences_ambiguous_session_admission() -> None:
    class AmbiguousAdmissionControlPlane(RecordingCoordinatorControlPlane):
        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            await super().submit_to_stage(stage, endpoint, msg)
            if isinstance(msg, SubmitMessage):
                raise asyncio.CancelledError()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = AmbiguousAdmissionControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")

        with pytest.raises(asyncio.CancelledError):
            await coordinator.open_session("session-1", {})

        assert control_plane.aborts == [AbortMessage("session-1", generation=1)]
        assert coordinator.health()["active_sessions"] == 0
        assert "session-1" not in coordinator._requests
        assert "session-1" not in coordinator._stream_queues

    asyncio.run(_run())


def test_coordinator_aborts_session_after_ambiguous_command_send() -> None:
    class AmbiguousCommandControlPlane(RecordingCoordinatorControlPlane):
        fail_commands = False

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            await super().submit_to_stage(stage, endpoint, msg)
            if self.fail_commands and isinstance(msg, SessionCommandMessage):
                raise asyncio.CancelledError()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = AmbiguousCommandControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})
        control_plane.fail_commands = True

        with pytest.raises(asyncio.CancelledError):
            await coordinator.send_session_command(
                "session-1", 1, "append", {"audio": [0.0]}
            )

        assert control_plane.aborts[-1] == AbortMessage("session-1", generation=1)
        assert coordinator.health()["active_sessions"] == 0
        terminal = await events.__anext__()
        assert isinstance(terminal, CompleteMessage)
        assert terminal.success is False
        assert terminal.generation == 1
        await events.aclose()

    asyncio.run(_run())


def test_coordinator_uses_only_first_session_terminal() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
            terminal_stages=["duplex"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {}, output_queue_size=1)

        delta = StreamMessage(
            "session-1",
            "duplex",
            {"type": "response.output.delta"},
            generation=1,
        )
        closed = StreamMessage(
            "session-1",
            "duplex",
            {
                "type": "session.closed",
                "session_id": "session-1",
                "generation": 1,
            },
            generation=1,
        )
        await coordinator._handle_stream(delta)
        await coordinator._handle_stream(closed)
        await coordinator._handle_completion(
            CompleteMessage(
                "session-1",
                "duplex",
                True,
                result={"closed": True},
                generation=1,
            )
        )

        assert await events.__anext__() is delta
        assert await events.__anext__() is closed
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()
        assert coordinator.health()["active_sessions"] == 0

    asyncio.run(_run())


def test_coordinator_abort_transport_failure_releases_and_quarantines() -> None:
    class FailingAbortControlPlane(RecordingCoordinatorControlPlane):
        async def broadcast_abort(self, msg) -> None:
            self.aborts.append(msg)
            raise RuntimeError("abort transport failed")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = FailingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        with pytest.raises(RuntimeError, match="abort transport failed"):
            await coordinator.abort("session-1")

        assert coordinator.health()["active_sessions"] == 0
        assert coordinator.health()["uncertain_sessions"] == 1
        assert "session-1" not in coordinator._requests
        terminal = await events.__anext__()
        assert terminal.success is False
        await events.aclose()
        assert "session-1" not in coordinator._stream_queues
        with pytest.raises(RuntimeError, match="uncertain abort"):
            await coordinator.open_session("session-1", {})

    asyncio.run(_run())


def test_stage_abort_listener_failure_updates_background_health() -> None:
    class FailingAbortControlPlane(RecordingStageControlPlane):
        async def recv_abort(self):
            raise RuntimeError("abort socket failed")

    async def _run() -> None:
        control_plane = FailingAbortControlPlane()
        stage = make_stage(name="duplex", control_plane=control_plane)
        stage._running = True
        task = asyncio.create_task(stage._abort_listener())
        task.add_done_callback(
            lambda done: stage._on_background_task_done(done, "abort listener")
        )

        with pytest.raises(RuntimeError, match="abort socket failed"):
            await task
        await asyncio.sleep(0)
        assert stage._running is False
        assert isinstance(stage._background_task_error, RuntimeError)
        assert control_plane.closed is True

    asyncio.run(_run())


def test_coordinator_stop_terminalizes_active_session() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        await coordinator.stop()

        terminal = await events.__anext__()
        assert isinstance(terminal, CompleteMessage)
        assert terminal.success is False
        assert terminal.error == "coordinator stopped"
        assert terminal.generation == 1
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()
        assert coordinator.health()["active_sessions"] == 0
        assert control_plane.closed is True

    asyncio.run(_run())


def test_coordinator_stop_serializes_with_abort_and_emits_one_terminal() -> None:
    class BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.abort_started = asyncio.Event()
            self.release_abort = asyncio.Event()

        async def broadcast_abort(self, msg) -> None:
            self.abort_started.set()
            await self.release_abort.wait()
            await super().broadcast_abort(msg)

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="duplex",
        )
        control_plane = BlockingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("duplex", "inproc://duplex")
        _, events = await coordinator.open_session("session-1", {})

        abort_task = asyncio.create_task(coordinator.abort("session-1"))
        await control_plane.abort_started.wait()
        stop_task = asyncio.create_task(coordinator.stop())
        await asyncio.sleep(0)
        assert stop_task.done() is False

        control_plane.release_abort.set()
        assert await abort_task is True
        await stop_task

        terminal = await events.__anext__()
        assert isinstance(terminal, CompleteMessage)
        assert terminal.error == "aborted"
        with pytest.raises(StopAsyncIteration):
            await events.__anext__()
        assert len(control_plane.aborts) == 1

    asyncio.run(_run())
