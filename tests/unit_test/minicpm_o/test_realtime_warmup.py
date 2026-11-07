# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o realtime startup warmup behavior."""
import asyncio

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexPipelineConfig
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from tests.unit_test.fixtures.session_client import FakeSessionClient


class WarmupSessionClient(FakeSessionClient):
    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        return await super().append_session(session_identity, chunk)


class FailingWarmupClient(WarmupSessionClient):
    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        assert session_identity == self.identity
        self.appended.append(chunk)
        self.fail(RuntimeError("warmup backend failed"))
        return chunk.seq


def test_warmup_uses_ephemeral_sampling_override_and_closes_session() -> None:
    client = WarmupSessionClient()
    config = MiniCPMODuplexPipelineConfig(model_path="unused")
    original_sampling = config.sampling.model_dump()
    deployment = build_realtime_deployment(client, config)
    assert deployment.warmup is not None
    asyncio.run(deployment.warmup(30.0))
    request, _, _ = client.opened[0]
    assert request.params["force_listen_count"] == 0
    assert request.params["listen_prob_scale"] == 0.0
    assert request.params["max_new_tokens_per_unit"] == 2
    assert config.sampling.model_dump() == original_sampling
    audio_chunks = [chunk for chunk in client.appended if chunk.payload]
    assert len(audio_chunks) == 4
    assert all(
        len(chunk.payload) == deployment.capabilities.native_unit_bytes
        for chunk in audio_chunks
    )
    assert client.appended[-1].eos
    assert client.closed.is_set()


def test_warmup_propagates_session_failure_and_closes_session() -> None:
    client = FailingWarmupClient()
    deployment = build_realtime_deployment(
        client, MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    assert deployment.warmup is not None
    try:
        asyncio.run(deployment.warmup(30.0))
    except RuntimeError as error:
        assert "warmup backend failed" in str(error)
    else:
        raise AssertionError("failed warmup unexpectedly returned")
    assert client.closed.is_set()


class WaitingWarmupClient(WarmupSessionClient):
    def __init__(self) -> None:
        super().__init__()
        self.auto_done = False
        self.appended_event = asyncio.Event()

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        sequence = await super().append_session(
            session_identity,
            chunk,
            ready_timestamp_ns=ready_timestamp_ns,
            ready_run_id=ready_run_id,
        )
        self.appended_event.set()
        return sequence


class CloseFailingWarmupClient(WarmupSessionClient):
    async def close_session(self, session_identity: SessionIdentity) -> None:
        assert session_identity == self.identity
        self.closed.set()
        raise RuntimeError("warmup close failed")


def test_warmup_cancellation_closes_session_and_drain_task() -> None:
    async def exercise_cancellation() -> None:
        client = WaitingWarmupClient()
        deployment = build_realtime_deployment(
            client, MiniCPMODuplexPipelineConfig(model_path="unused")
        )
        assert deployment.warmup is not None
        warmup_task = asyncio.create_task(deployment.warmup(30.0))
        await asyncio.wait_for(client.appended_event.wait(), timeout=5.0)
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled warmup unexpectedly returned")
        assert client.closed.is_set()
        pending_tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert pending_tasks == []

    asyncio.run(exercise_cancellation())


def test_warmup_close_failure_is_reported_as_failure() -> None:
    client = CloseFailingWarmupClient()
    deployment = build_realtime_deployment(
        client, MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    assert deployment.warmup is not None
    try:
        asyncio.run(deployment.warmup(30.0))
    except RuntimeError as error:
        assert "warmup close failed" in str(error) or "cleanup" in str(error)
    else:
        raise AssertionError("failed close unexpectedly returned")
    assert client.closed.is_set()
