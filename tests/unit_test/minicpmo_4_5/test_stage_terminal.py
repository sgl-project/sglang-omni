from __future__ import annotations

import asyncio

import torch

_original_torch_compile = torch.compile


def _identity_compile(model=None, *args, **kwargs):
    del args, kwargs
    if model is None:
        return lambda fn: fn
    return model


torch.compile = _identity_compile
try:
    from sglang_omni.scheduling.messages import OutgoingMessage
    from tests.unit_test.fixtures.pipeline_fakes import (
        FakeScheduler,
        RecordingStageControlPlane,
    )
    from tests.unit_test.pipeline.helpers import make_stage
finally:
    torch.compile = _original_torch_compile


def test_stage_preserves_scheduler_terminal_error_envelope() -> None:
    async def run() -> None:
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage = make_stage(
            name="minicpmo_duplex",
            scheduler=scheduler,
            control_plane=control_plane,
        )
        stage._active_requests.add("session-1")
        stage._session_generations["session-1"] = 4
        stage._session_generation_watermarks["session-1"] = 4
        terminal_event = {
            "type": "session.error",
            "session_id": "session-1",
            "generation": 4,
            "input_seq": 2,
            "response_epoch": 1,
            "output_seq": 7,
            "error": "generate failed",
        }
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="session-1",
                type="error",
                data=RuntimeError("generate failed"),
                metadata={
                    "generation": 4,
                    "terminal_event": terminal_event,
                },
            )
        )

        await stage._drain_outbox()

        assert len(control_plane.completions) == 1
        complete = control_plane.completions[0]
        assert complete.success is False
        assert complete.error == "generate failed"
        assert complete.result == terminal_event
        assert "session-1" not in stage._active_requests

    asyncio.run(run())
