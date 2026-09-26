# SPDX-License-Identifier: Apache-2.0
"""WebSocket to coordinator, stage, and model engine integration."""

from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.models.nemotron3_5_asr.realtime import create_realtime_deployment
from sglang_omni.models.nemotron3_5_asr.session import NemotronSessionScheduler
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, construct_stage
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.nemotron3_5_asr.test_session import make_engine
from tests.unit_test.nemotron3_5_asr.test_streaming import FakeRunner


def create_fake_scheduler() -> NemotronSessionScheduler:
    return NemotronSessionScheduler(make_engine(FakeRunner()), max_concurrency=8,
                                    max_open_sessions=64, max_state_bytes=1 << 30)


@pytest.mark.parametrize("sample_count", [0, 4040, 10000])
def test_websocket_stage_final_drain_and_cleanup(tmp_path: Path, sample_count: int) -> None:
    completion, abort, endpoint = [f"ipc://{tmp_path}/{name}" for name in ("done", "abort", "asr")]
    coordinator = Coordinator(completion, abort, "asr", ["asr"])
    client = Client(coordinator)
    app = create_app(client, model_name="nemotron-test",
                     realtime_deployment=create_realtime_deployment(client))
    owners = []

    @asynccontextmanager
    async def lifespan(application):
        await coordinator.start()
        stage = construct_stage(StageLaunchConfig(
            stage_name="asr", factory=f"{__name__}.create_fake_scheduler",
            factory_kwargs={}, next_stages=None, is_terminal=True,
            recv_endpoint=endpoint, coordinator_endpoint=completion,
            abort_endpoint=abort, stage_endpoints={"asr": endpoint},
        ), logging.getLogger(__name__))
        owners.append(stage)
        await stage.start()
        coordinator.register_stage("asr", endpoint)
        tasks = [asyncio.create_task(stage.run()), asyncio.create_task(coordinator.run_completion_loop())]
        try:
            yield
        finally:
            await coordinator.shutdown_stages()
            await coordinator.stop()
            await stage.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app.router.lifespan_context = lifespan
    with TestClient(app) as test_client, test_client.websocket_connect("/v1/realtime?model=nemotron-test") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json({"event_id": "update", "type": "session.update", "session": {"output_modalities": ["text"]}})
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated", updated
        if sample_count:
            websocket.send_json({
                "event_id": "append", "type": "input_audio_buffer.append", "audio": base64.b64encode(b"\0\0" * sample_count).decode(),
                "sglang": {"seq": 0},
            })
        websocket.send_json({"event_id": "end", "type": "sglang.input_audio.end"})
        events = []
        while not events or events[-1]["type"] != "sglang.input_audio.drained":
            event = websocket.receive_json()
            assert event["type"] != "error", event
            events.append(event)
        finals = [event for event in events if event["type"] == "response.output_text.done"]
        assert len(finals) == 1
        deltas = [event for event in events if event["type"] == "response.output_text.delta"]
        assert "".join(event["delta"] for event in deltas) == finals[0]["text"]
        assert len({event["response_id"] for event in deltas + finals}) == 1
        assert events[-1]["consumed_ms"] == sample_count / 16
        websocket.send_json({"event_id": "close", "type": "session.close"})
        while websocket.receive_json()["type"] != "session.closed":
            pass
    engine = owners[0].scheduler.engine
    assert not engine.states and not engine.tasks and not engine.thread.is_alive()
    if not sample_count:
        assert not engine.runner.batches
    assert not coordinator.sessions and not coordinator.requests
