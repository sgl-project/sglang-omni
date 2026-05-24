# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


def test_coordinator_multi_terminal_completion_and_abort_contracts() -> None:
    """Preserves multi-terminal completion and abort cancellation semantics."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", {"text": "hello"})
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["req-1"].done()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"})
        )
        assert coordinator._completion_futures["req-1"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

        await coordinator._submit_request("req-2", "hello")
        future = coordinator._completion_futures["req-2"]
        assert await coordinator.abort("req-2") is True
        assert control_plane.aborts[0].request_id == "req-2"
        with pytest.raises(asyncio.CancelledError):
            await future

    asyncio.run(_run())


def test_coordinator_resolves_active_terminal_subset_per_request() -> None:
    async def _run() -> None:
        def terminal_stages(request: OmniRequest) -> list[str]:
            assert isinstance(request, OmniRequest)
            if request.metadata.get("audio"):
                return ["decode", "code2wav"]
            return ["decode"]

        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=terminal_stages,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request(
            "text-req",
            OmniRequest(inputs="hello", metadata={"audio": False}),
        )
        await coordinator._handle_completion(
            CompleteMessage("text-req", "decode", True, result={"text": "hi"})
        )
        assert coordinator._completion_futures["text-req"].result() == {"text": "hi"}

        await coordinator._submit_request("raw-text-req", "hello")
        await coordinator._handle_completion(
            CompleteMessage("raw-text-req", "decode", True, result={"text": "raw"})
        )
        assert coordinator._completion_futures["raw-text-req"].result() == {
            "text": "raw"
        }

        await coordinator._submit_request(
            "audio-req",
            OmniRequest(inputs="hello", metadata={"audio": True}),
        )
        await coordinator._handle_completion(
            CompleteMessage("audio-req", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["audio-req"].done()
        await coordinator._handle_completion(
            CompleteMessage(
                "audio-req",
                "code2wav",
                True,
                result={"audio": "ok"},
            )
        )
        assert coordinator._completion_futures["audio-req"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

    asyncio.run(_run())


def test_coordinator_rejects_invalid_resolved_terminal_subset() -> None:
    async def _run() -> None:
        for resolved, error in (
            ([], "no terminal stages"),
            (["decode", "missing"], "outside the static terminal stages"),
            ("decode", "must return a sequence"),
        ):
            coordinator = Coordinator(
                "inproc://complete",
                "inproc://abort",
                entry_stage="preprocess",
                terminal_stages=["decode", "code2wav"],
                terminal_stages_resolver=lambda request, resolved=resolved: resolved,
            )
            coordinator.control_plane = RecordingCoordinatorControlPlane()
            coordinator.register_stage("preprocess", "inproc://preprocess")

            with pytest.raises(ValueError, match=error):
                await coordinator._submit_request("req-1", OmniRequest(inputs="hello"))
            assert coordinator._requests == {}
            assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_cleans_queue_when_terminal_resolver_rejects() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: [],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        with pytest.raises(ValueError, match="no terminal stages"):
            await stream.__anext__()
        await stream.aclose()

        assert coordinator._stream_queues == {}
        assert coordinator._completion_futures == {}
        assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_uses_request_terminal_subset_after_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: ["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        events = []

        async def _consume() -> None:
            async for event in coordinator.stream("req-1", OmniRequest(inputs="hello")):
                events.append(event)

        task = asyncio.create_task(_consume())
        for _ in range(10):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        await asyncio.wait_for(task, timeout=1)

        assert [event.from_stage for event in events] == ["decode"]

    asyncio.run(_run())


def test_coordinator_stream_received_event_pairs_terminal_chunk(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.coordinator._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        queue: asyncio.Queue = asyncio.Queue()
        coordinator._stream_queues["req-1"] = queue

        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hi"},
                modality="text",
                chunk_id=1,
            )
        )

        routed = queue.get_nowait()
        assert routed.chunk_id == 1

    asyncio.run(_run())

    receive_events = [
        event
        for event in events
        if event["event_name"] == "stage_stream_chunk_received"
    ]
    assert len(receive_events) == 1
    assert receive_events[0]["stage"] == "coordinator"
    assert receive_events[0]["metadata"] == {
        "from_stage": "decode",
        "chunk_id": 1,
        "modality": "text",
    }


def test_stream_message_round_trips_terminal_chunk_id() -> None:
    msg = StreamMessage(
        request_id="req-1",
        from_stage="decode",
        chunk={"text": "hi"},
        modality="text",
        chunk_id=3,
    )

    round_trip = StreamMessage.from_dict(msg.to_dict())

    assert round_trip.chunk_id == 3
    assert round_trip.modality == "text"


def test_coordinator_failure_completion_fails_fast_and_cleans_state() -> None:
    """Preserves fail-fast behavior and cleanup after any terminal failure."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "hi"}}

        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", False, error="boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await future
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._partial_results
        assert control_plane.aborts[-1].request_id == "req-1"

    asyncio.run(_run())


def test_coordinator_fail_pending_requests_resolves_waiters() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]

        await coordinator.fail_pending_requests(RuntimeError("stage died"))

        with pytest.raises(RuntimeError, match="stage died"):
            await future
        assert coordinator._requests == {}
        assert coordinator._partial_results == {}

    asyncio.run(_run())
