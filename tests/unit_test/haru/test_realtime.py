# SPDX-License-Identifier: Apache-2.0
"""Test Haru admission, first-frame output and audio input through realtime."""

import asyncio
from contextlib import aclosing

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.haru.config import INPUT_FRAME_SAMPLES, HaruPipelineConfig
from sglang_omni.models.haru.realtime import create_realtime_deployment
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.serve.realtime.control import Accepted, Drained, Updated
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
)
from sglang_omni.serve.realtime.runtime import SessionRuntime
from tests.unit_test.fixtures.session_client import FakeSessionClient

FRAME_PCM = bytes(2 * INPUT_FRAME_SAMPLES)


class HaruSessionClient(FakeSessionClient):
    def __init__(self, delayed_sequence: int | None = None) -> None:
        super().__init__(auto_done=False)
        self.delayed_sequence = delayed_sequence
        self.delayed_output = asyncio.Event()

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        assert chunk.seq == len(self.appended)
        sequence = await super().append_session(session_identity, chunk)
        if sequence == self.delayed_sequence:
            self.delayed_output.set()
        else:
            self.done(sequence)
        return sequence

    def answer(self, chunk: TimedChunk) -> None:
        if chunk.payload or not chunk.eos:
            self.emit(
                "audio",
                "haru",
                {"pcm": FRAME_PCM, "text": "hello" if chunk.seq == 0 else ""},
                input_seq=chunk.seq,
            )
        else:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("speak_first", [False, True])
async def test_prompt_is_prefilled_before_session_updated(speak_first: bool) -> None:
    coordinator = HaruSessionClient()
    deployment = create_realtime_deployment(
        Client(coordinator), HaruPipelineConfig(model_path="Haru-4B")
    )
    runtime = SessionRuntime(
        "Haru-4B",
        deployment.capabilities,
        deployment.adapter_factory,
        deployment.limits,
    )
    try:
        async with aclosing(runtime.outputs()) as outputs:
            await runtime.update(
                {"sglang": {"sampling": {"speak_first": speak_first}}},
                "initial_update",
            )
            assert isinstance((await anext(outputs)).event, Updated)
            assert coordinator.appended == [
                TimedChunk("audio", 0, 0, 0, b"", format="pcm16")
            ]
            if not speak_first:
                await runtime.append(FRAME_PCM, 0, None, "first_audio")
                assert isinstance((await anext(outputs)).event, Accepted)
            else:
                pass
            async with asyncio.timeout(5):
                assert isinstance((await anext(outputs)).event, ResponseStarted)
                assert isinstance((await anext(outputs)).event, AudioDelta)
                transcript = (await anext(outputs)).event
            assert isinstance(transcript, TextDelta) and transcript.text == "hello"
            if speak_first:
                assert len(coordinator.appended) == 1
            else:
                assert coordinator.appended[1].seq == 1
                assert coordinator.appended[1].payload == FRAME_PCM
    finally:
        await runtime.close("test_finished")


@pytest.mark.asyncio
async def test_cancelling_an_in_flight_frame_only_ends_the_current_reply() -> None:
    coordinator = HaruSessionClient(delayed_sequence=1)
    deployment = create_realtime_deployment(
        Client(coordinator), HaruPipelineConfig(model_path="Haru-4B")
    )
    runtime = SessionRuntime(
        "Haru-4B",
        deployment.capabilities,
        deployment.adapter_factory,
        deployment.limits,
    )
    try:
        async with asyncio.timeout(5), aclosing(runtime.outputs()) as outputs:
            await runtime.update(
                {"sglang": {"sampling": {"speak_first": True}}}, "initial_update"
            )
            assert isinstance((await anext(outputs)).event, Updated)
            first_response = (await anext(outputs)).event
            assert isinstance(first_response, ResponseStarted)
            assert isinstance((await anext(outputs)).event, AudioDelta)
            assert isinstance((await anext(outputs)).event, TextDelta)

            await runtime.append(FRAME_PCM, 0, None, "first_input")
            await coordinator.delayed_output.wait()
            cancel_started = asyncio.Event()

            async def cancel() -> None:
                cancel_started.set()
                await runtime.cancel()

            cancel_task = asyncio.create_task(cancel())
            await cancel_started.wait()
            assert not cancel_task.done()
            coordinator.done(1)
            await cancel_task
            while not isinstance(
                (envelope := await anext(outputs)).event, ResponseFinished
            ):
                assert not isinstance(envelope.event, AudioDelta)
            cancelled = envelope.event
            assert cancelled.response_id == first_response.response_id
            assert cancelled.status == "cancelled"

            await runtime.append(FRAME_PCM, 1, None, "second_input")
            assert isinstance((await anext(outputs)).event, Accepted)
            second_response = (await anext(outputs)).event
            assert isinstance(second_response, ResponseStarted)
            assert second_response.response_id != first_response.response_id
            assert isinstance((await anext(outputs)).event, AudioDelta)

            await runtime.end("input_end")
            completed = None
            while not isinstance((envelope := await anext(outputs)).event, Drained):
                if isinstance(envelope.event, ResponseFinished):
                    completed = envelope.event
                else:
                    pass
            assert completed is not None
            assert completed.response_id == second_response.response_id
            assert (completed.status, completed.reason) == ("completed", "stop")
            assert len(coordinator.opened) == 1
            assert [chunk.seq for chunk in coordinator.appended] == [0, 1, 2, 3]
    finally:
        await runtime.close("test_finished")
