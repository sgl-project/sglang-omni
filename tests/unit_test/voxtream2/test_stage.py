# SPDX-License-Identifier: Apache-2.0
"""Exercise Stage submission, worker thread, and Coordinator message routing."""

import asyncio

import numpy as np
import pytest

from sglang_omni.models.voxtream2.scheduler import VoXtreamScheduler
from sglang_omni.proto import OmniRequest, StagePayload, SubmitMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingStageControlPlane
from tests.unit_test.pipeline.helpers import make_stage


class Backend:
    sample_rate = 24000
    max_samples = 100

    def iter_audio(self, payload):
        yield np.array([0.1, 0.2], dtype=np.float32)
        yield np.array([0.3, 0.4], dtype=np.float32)


@pytest.mark.parametrize("streaming", [False, True])
def test_stage_submit_to_coordinator_completion(streaming):
    async def run():
        complete = asyncio.Event()

        class Control(RecordingStageControlPlane):
            def __init__(self):
                super().__init__()
                self.streams = []

            async def send_stream(self, msg):
                self.streams.append(msg)

            async def send_complete(self, msg):
                await super().send_complete(msg)
                complete.set()

        control = Control()
        scheduler = VoXtreamScheduler(Backend(), output_buffer_chunks=1)
        stage = make_stage(
            name="voxtream2",
            scheduler=scheduler,
            control_plane=control,
            is_terminal=True,
        )
        await stage.start()
        drainer = asyncio.create_task(stage.drain_outbox())
        try:
            payload = StagePayload(
                "a", OmniRequest("Hello", {"stream": streaming}), None
            )
            await stage.on_submit(SubmitMessage("a", payload))
            await asyncio.wait_for(complete.wait(), timeout=5)
            assert len(control.completions) == 1
            completion = control.completions[0]
            assert completion.request_id == "a"
            assert completion.success
            if streaming:
                assert [m.chunk_id for m in control.streams] == [0, 1]
                assert all(
                    m.modality == "audio" and m.from_stage == "voxtream2"
                    for m in control.streams
                )
                assert "audio_waveform" not in completion.result
                waveform = np.concatenate(
                    [
                        np.frombuffer(m.chunk["audio_waveform"], dtype=np.float32)
                        for m in control.streams
                    ]
                )
            else:
                assert not control.streams
                waveform = np.frombuffer(
                    completion.result["audio_waveform"], dtype=np.float32
                )
            np.testing.assert_array_equal(
                waveform, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            )
            assert not stage.active_requests
        finally:
            await stage.stop()
            await asyncio.wait_for(drainer, timeout=3)
        assert not scheduler.request_cancellations
        assert stage.scheduler_thread is None

    asyncio.run(run())
