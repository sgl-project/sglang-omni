# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from sglang_omni.client import (
    Client,
    DuplexSession,
    DuplexSessionError,
    GenerateRequest,
)
from sglang_omni.proto import CompleteMessage, SessionCommandMessage, StreamMessage


class _DuplexCoordinator:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.commands: list[SessionCommandMessage] = []
        self.generation = 1
        self.input_seq = 0
        self.response_epoch = 0
        self.output_seq = 0
        self.aborted: list[str] = []
        self.output_queue_size: int | None = None

    async def open_session(
        self,
        session_id: str,
        request: Any,
        *,
        output_queue_size: int,
    ):
        assert request.params["stream"] is True
        self.output_queue_size = output_queue_size

        async def _events():
            while True:
                message = await self.queue.get()
                yield message
                if isinstance(message, CompleteMessage):
                    return

        await self.emit(
            session_id,
            {
                "type": "session.created",
                "session_id": session_id,
                "generation": self.generation,
                "input_seq": 0,
                "response_epoch": 0,
            },
        )
        return self.generation, _events()

    async def send_session_command(
        self,
        session_id: str,
        generation: int,
        command: str,
        data: dict[str, Any],
    ) -> SessionCommandMessage:
        self.input_seq += 1
        if command in {"interrupt", "close"}:
            self.response_epoch += 1
        message = SessionCommandMessage(
            session_id,
            generation,
            self.input_seq,
            self.response_epoch,
            command,
            data,
        )
        self.commands.append(message)
        if command == "append":
            await self.emit(
                session_id,
                {
                    "type": "session.input_processed",
                    "session_id": session_id,
                    "generation": generation,
                    "input_seq": self.input_seq,
                    "response_epoch": self.response_epoch,
                },
            )
            await asyncio.sleep(0)
        elif command == "close":
            await self.emit(
                session_id,
                {
                    "type": "session.closed",
                    "session_id": session_id,
                    "generation": generation,
                    "input_seq": self.input_seq,
                    "response_epoch": self.response_epoch,
                },
            )
        return message

    async def emit(self, session_id: str, event: dict[str, Any]) -> None:
        self.output_seq += 1
        event.setdefault("output_seq", self.output_seq)
        await self.queue.put(
            StreamMessage(
                request_id=session_id,
                from_stage="duplex",
                chunk=event,
            )
        )

    async def abort(self, session_id: str) -> bool:
        self.aborted.append(session_id)
        return True


def test_duplex_session_handshake_ack_fencing_and_playback() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = client.duplex_session(
            GenerateRequest(prompt="system prompt"),
            session_id="session-1",
            output_queue_size=1,
        )
        assert isinstance(session, DuplexSession)
        await session.start()
        assert coordinator.output_queue_size == 1
        event_iter = session.events()

        append_task = asyncio.create_task(session.append({"audio": [0.0]}))
        assert (await event_iter.__anext__())["type"] == "session.created"
        assert await append_task == 1
        assert (await event_iter.__anext__())["type"] == "session.input_processed"

        await coordinator.emit(
            "session-1",
            {
                "type": "response.output.delta",
                "session_id": "session-1",
                "generation": 1,
                "response_epoch": 0,
                "audio": [0.1],
            },
        )
        assert (await event_iter.__anext__())["audio"] == [0.1]

        assert await session.interrupt() == 1
        await coordinator.emit(
            "session-1",
            {
                "type": "response.output.delta",
                "session_id": "session-1",
                "generation": 1,
                "response_epoch": 0,
                "audio": ["stale"],
            },
        )
        await coordinator.emit(
            "session-1",
            {
                "type": "response.output.delta",
                "session_id": "session-1",
                "generation": 99,
                "response_epoch": 1,
                "audio": ["old-generation"],
            },
        )
        await coordinator.emit(
            "session-1",
            {
                "type": "response.output.delta",
                "session_id": "session-1",
                "generation": 1,
                "response_epoch": 1,
                "audio": [0.2],
            },
        )
        assert (await event_iter.__anext__())["audio"] == [0.2]

        assert await session.playback_ack(640.0) == 3
        assert coordinator.commands[-1].command == "playback_ack"
        assert coordinator.commands[-1].data == {"audio_end_ms": 640.0}

        await session.close()
        assert (await event_iter.__anext__())["type"] == "session.closed"
        try:
            await event_iter.__anext__()
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("duplex event stream did not terminate")

    asyncio.run(_run())


def test_duplex_session_event_stream_has_one_consumer() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"

        second = session.events()
        try:
            await second.__anext__()
        except DuplexSessionError as exc:
            assert "already has a consumer" in str(exc)
        else:
            raise AssertionError("a second duplex event consumer was accepted")

        await session.close()
        assert (await events.__anext__())["type"] == "session.closed"

    asyncio.run(_run())


def test_duplex_response_output_done_is_a_non_terminal_boundary() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"

        await coordinator.emit(
            "session-1",
            {
                "type": "response.output.done",
                "session_id": "session-1",
                "generation": 1,
                "input_seq": 0,
                "response_epoch": 0,
                "reason": "model_turn_end",
            },
        )
        done = await events.__anext__()
        assert done["type"] == "response.output.done"
        assert session.is_closed is False

        append_task = asyncio.create_task(
            session.append({"audio": [0.0]}, wait_processed=True)
        )
        assert (await events.__anext__())["type"] == "session.input_processed"
        assert await append_task == 1
        await session.close()
        assert (await events.__anext__())["type"] == "session.closed"

    asyncio.run(_run())


def test_duplex_response_output_done_uses_pending_epoch_fence() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        assert await session.interrupt() == 1

        for epoch in (0, 1):
            await coordinator.emit(
                "session-1",
                {
                    "type": "response.output.done",
                    "session_id": "session-1",
                    "generation": 1,
                    "input_seq": 1,
                    "response_epoch": epoch,
                    "reason": f"epoch-{epoch}",
                },
            )

        done = await events.__anext__()
        assert done["response_epoch"] == 1
        assert done["reason"] == "epoch-1"
        await session.close()

    asyncio.run(_run())


def test_duplex_session_rejects_duplicate_same_generation_output_seq() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        created = await events.__anext__()
        assert created["output_seq"] == 1

        delta = {
            "type": "response.output.delta",
            "session_id": "session-1",
            "generation": 1,
            "response_epoch": 0,
            "output_seq": 2,
            "kind": "text",
            "text": "one",
        }
        await coordinator.queue.put(StreamMessage("session-1", "duplex", dict(delta)))
        assert (await events.__anext__())["text"] == "one"
        await coordinator.queue.put(StreamMessage("session-1", "duplex", dict(delta)))
        error = await events.__anext__()
        assert error["type"] == "session.error"
        assert "strictly increasing" in error["error"]

    asyncio.run(_run())


def test_duplex_session_turns_invalid_payload_into_error_event() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        await coordinator.queue.put(
            StreamMessage(
                request_id="session-1",
                from_stage="duplex",
                chunk=b"not-a-dict",
            )
        )
        error = await events.__anext__()
        assert error["type"] == "session.error"
        assert "must be dicts" in error["error"]

    asyncio.run(_run())


def test_duplex_session_preserves_terminal_envelope_from_completion() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        await coordinator.queue.put(
            CompleteMessage(
                request_id="session-1",
                from_stage="duplex",
                success=True,
                result={
                    "type": "session.closed",
                    "session_id": "session-1",
                    "generation": 1,
                    "input_seq": 0,
                    "response_epoch": 0,
                    "output_seq": 2,
                    "reason": "ttl",
                },
            )
        )

        closed = await events.__anext__()
        assert closed["output_seq"] == 2
        assert closed["reason"] == "ttl"

    asyncio.run(_run())


def test_duplex_session_rejects_terminal_for_future_input() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        await coordinator.queue.put(
            CompleteMessage(
                request_id="session-1",
                from_stage="duplex",
                success=True,
                result={
                    "type": "session.closed",
                    "session_id": "session-1",
                    "generation": 1,
                    "input_seq": 1,
                    "response_epoch": 0,
                    "output_seq": 2,
                },
            )
        )

        error = await events.__anext__()
        assert error["type"] == "session.error"
        assert "terminal input_seq" in error["error"]

    asyncio.run(_run())


def test_duplex_close_finishes_without_draining_bounded_output_queue() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
            output_queue_size=1,
        )

        # session.created occupies the only user-visible regular event slot.
        # The terminal event and end marker must still be publishable.
        await asyncio.wait_for(session.close(), timeout=1)
        assert session._consumer_task is not None
        assert session._consumer_task.done()

        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        assert (await events.__anext__())["type"] == "session.closed"
        try:
            await events.__anext__()
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("duplex event stream did not terminate")

    asyncio.run(_run())


def test_duplex_session_rejects_duplicate_input_processed_ack() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        assert await session.append({"audio": [0.0]}, wait_processed=False) == 1
        assert (await events.__anext__())["type"] == "session.input_processed"

        await coordinator.emit(
            "session-1",
            {
                "type": "session.input_processed",
                "session_id": "session-1",
                "generation": 1,
                "input_seq": 1,
                "response_epoch": 0,
            },
        )
        error = await events.__anext__()
        assert error["type"] == "session.error"
        assert "strictly increasing" in error["error"]

    asyncio.run(_run())


def test_duplex_close_times_out_and_cancels_stuck_command() -> None:
    class StuckCloseCoordinator(_DuplexCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = asyncio.Event()

        async def send_session_command(
            self,
            session_id: str,
            generation: int,
            command: str,
            data: dict[str, Any],
        ) -> SessionCommandMessage:
            if command == "close":
                self.close_started.set()
                await asyncio.Future()
            return await super().send_session_command(
                session_id, generation, command, data
            )

    async def _run() -> None:
        coordinator = StuckCloseCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )

        try:
            await session.close(timeout_s=0.01)
        except TimeoutError:
            pass
        else:
            raise AssertionError("stuck close command did not time out")
        assert coordinator.close_started.is_set()
        assert coordinator.aborted == ["session-1"]
        assert session._close_command_task is not None
        assert session._close_command_task.cancelled()

    asyncio.run(_run())


def test_duplex_failed_completion_cannot_masquerade_as_closed() -> None:
    async def _run() -> None:
        coordinator = _DuplexCoordinator()
        client = Client(coordinator)  # type: ignore[arg-type]
        session = await client.open_duplex_session(
            GenerateRequest(prompt="hello"),
            session_id="session-1",
        )
        events = session.events()
        assert (await events.__anext__())["type"] == "session.created"
        await coordinator.queue.put(
            CompleteMessage(
                request_id="session-1",
                from_stage="duplex",
                success=False,
                error="backend failed",
                result={
                    "type": "session.closed",
                    "session_id": "session-1",
                    "generation": 1,
                    "input_seq": 0,
                    "response_epoch": 0,
                    "output_seq": 2,
                    "reason": "malformed-backend-envelope",
                },
            )
        )

        terminal = await events.__anext__()
        assert terminal["type"] == "session.error"
        assert terminal["error"] == "backend failed"
        assert terminal["output_seq"] == 2

    asyncio.run(_run())
