# SPDX-License-Identifier: Apache-2.0
import asyncio
import wave
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
import yaml

from examples.streaming_cascade.run import run
from sglang_omni.models.streaming_cascade.base import control_chunk
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionLimits, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    INTERRUPT,
    PCM16,
    PCM_F32,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)


@pytest.mark.asyncio
async def test_configured_runner_streams_two_utterances_and_shuts_down(
    backends, tmp_path
):
    config, _, words, _ = backends
    declaration = config.model_dump(mode="json")
    fields = ("model_path", "session_nodes", "session_publish", "timeout_s")
    launch = {key: declaration[key] for key in fields}
    config_path = tmp_path / "launch.yaml"
    config_path.write_text(
        yaml.safe_dump({"config_cls": "StreamingCascadePipelineConfig", **launch})
    )
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(640))
    output_path = tmp_path / "output.wav"
    await run(config_path, [audio_path, audio_path], output_path, commit=True)
    assert words == ["Hello", "world."] * 2
    with wave.open(str(output_path), "rb") as audio:
        assert audio.getnframes() == 12
        assert audio.getframerate() == 24000


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["failed_chat"], indirect=True)
async def test_run_raises_the_session_failure(backends, tmp_path):
    declaration = backends[0].model_dump(mode="json")
    fields = ("model_path", "session_nodes", "session_publish", "timeout_s")
    config_path = tmp_path / "launch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "StreamingCascadePipelineConfig",
                **{key: declaration[key] for key in fields},
            }
        )
    )
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(640 * 50))
    with pytest.raises(RuntimeError, match="abort"):
        await asyncio.wait_for(
            run(config_path, [audio_path], tmp_path / "output.wav", commit=True), 30
        )


class Conversation:
    """Drive the actual coordinator and stage processes, with bounded waits."""

    def __init__(self, coordinator, identity):
        self.coordinator, self.identity = coordinator, identity
        self.outputs = []
        self.changed = asyncio.Event()
        self.sequence = 0
        self.clock = 0.0
        self.error = None
        self.reader = asyncio.create_task(self.receive())

    async def receive(self):
        try:
            async for output in self.coordinator.session_outputs(self.identity):
                self.outputs.append(output)
                self.changed.set()
        except Exception as error:
            self.error = error
            self.changed.set()

    async def until(self, predicate):
        await asyncio.wait_for(self.settle(predicate), 15)

    async def settle(self, predicate):
        while not predicate():
            if self.error is not None:
                raise self.error
            else:
                self.changed.clear()
                await self.changed.wait()

    async def append(self, *chunks):
        """Append the chunks back to back, then wait until each is confirmed."""
        first = self.sequence
        for chunk in chunks:
            await self.coordinator.append_session(
                self.identity, replace(chunk, seq=self.sequence)
            )
            self.sequence += 1
        await self.until(
            lambda: set(range(first, self.sequence))
            <= {o.input_seq for o in self.outputs if o.kind == "input_done"}
        )

    async def audio(self, eos=False):
        await self.append(
            TimedChunk(AUDIO, self.clock, 20, 0, bytes(640), format=PCM16, eos=eos)
        )
        self.clock += 20

    async def command(self, event, **fields):
        await self.append(control_chunk(event, self.clock, **fields))

    async def close(self):
        await self.coordinator.close_session(self.identity)
        await asyncio.wait_for(self.reader, 5)


@asynccontextmanager
async def conversation(config):
    runner = MultiProcessPipelineRunner(config)
    await runner.start()
    session = None
    try:
        identity = await runner.coordinator.open_session(
            OmniRequest(None), limits=SessionLimits(operation_timeout_s=10)
        )
        session = Conversation(runner.coordinator, identity)
        yield session
    finally:
        if session is not None:
            await session.close()
        else:
            pass
        await runner.stop()


@pytest.mark.asyncio
async def test_eos_flushes_uncommitted_audio_and_history(backends):
    config, audio, words, messages = backends
    async with conversation(config) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.command(COMMIT)
        await session.audio(eos=True)
        assert len(audio) == 2
        assert words == ["Hello", "world."] * 2
        assert messages[1][-2:] == [
            {"role": "assistant", "content": "Hello world."},
            {"role": "user", "content": "test question"},
        ]
        events = [
            (control_type(o), o.turn) for o in session.outputs if o.kind == "data"
        ]
        assert (TURN_EMPTY, 2) in events
        assert (TURN_DONE, 3) in events
        assert session.outputs[-1].eos and session.outputs[-1].kind == "input_done"


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["slow_first"], indirect=True)
async def test_interrupt_cancels_old_reply_and_next_turn_completes(backends):
    config, _, _, messages = backends
    async with conversation(config) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(
            lambda: any(o.format == TEXT_DELTA for o in session.outputs)
        )
        await session.coordinator.control_session(
            session.identity, control_chunk(INTERRUPT, session.clock)
        )
        await session.until(
            lambda: any(control_type(o) == TURN_INTERRUPTED for o in session.outputs)
        )
        interrupted_at = len(session.outputs)
        await session.audio(eos=True)
        assert any(
            control_type(o) == TURN_DONE and o.turn == 2 for o in session.outputs
        )
        assert not any(
            o.turn == 1 and o.format in (TEXT_DELTA, PCM_F32)
            for o in session.outputs[interrupted_at:]
        )
        assert len(messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["slow_first"], indirect=True)
@pytest.mark.parametrize(
    "is_reply_open", [False, True], ids=["before_the_reply_opens", "before_its_audio"]
)
async def test_speech_that_resumes_before_the_reply_is_heard_interrupts_it(
    backends, is_reply_open
):
    config, _, _, messages = backends
    async with conversation(config) as session:
        await session.command(SPEECH_STARTED, t_ms=session.clock)
        await session.audio()
        stopped = control_chunk(SPEECH_STOPPED, session.clock, t_ms=session.clock)
        resumed = control_chunk(SPEECH_STARTED, session.clock, t_ms=session.clock)
        if is_reply_open:
            await session.append(stopped)
            await session.until(
                lambda: any(control_type(o) == TURN_STARTED for o in session.outputs)
            )
            await session.append(resumed)
        else:
            await session.append(stopped, resumed)
        await session.audio()
        await session.command(SPEECH_STOPPED, t_ms=session.clock)
        await session.audio(eos=True)
        assert [
            control_type(o)
            for o in session.outputs
            if o.turn == 1 and o.kind == "data" and o.format != TEXT_DELTA
        ] == [TURN_QUEUED, TURN_STARTED, TURN_INTERRUPTED]
        assert any(o.turn == 2 and o.format == PCM_F32 for o in session.outputs)
        assert any(
            control_type(o) == TURN_DONE and o.turn == 2 for o in session.outputs
        )
        assert messages[-1][-1] == {
            "role": "user",
            "content": "test question test question",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["slow_first"], indirect=True)
async def test_control_eos_during_reply_drains_its_feedback(backends):
    async with conversation(backends[0]) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(
            lambda: any(o.format == TEXT_DELTA for o in session.outputs)
        )
        await session.audio()
        await session.append(replace(control_chunk(COMMIT, session.clock), eos=True))
        events = [
            (control_type(o), o.turn) for o in session.outputs if o.kind == "data"
        ]
        assert (TURN_DONE, 1) in events and (TURN_DONE, 2) in events
        assert session.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["slow_first"], indirect=True)
async def test_configure_preserves_running_reply_and_updates_next_prompt(
    backends, backend_requests
):
    config, _, _, messages = backends
    async with conversation(config) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(
            lambda: any(o.format == TEXT_DELTA for o in session.outputs)
        )
        await session.command(CONFIGURE, sampling={"temperature": 0.1})
        await session.command(
            CONFIGURE,
            instructions="Answer with one word.",
            sampling={"max_new_tokens_per_unit": 64},
        )
        await session.until(
            lambda: any(control_type(o) == TURN_DONE for o in session.outputs)
        )
        await session.audio(eos=True)
        assert messages[0][0]["content"] != "Answer with one word."
        assert messages[1][0]["content"] == "Answer with one word."
        assert messages[1][-2]["content"] == "Hello world."
        assert backend_requests[-1]["temperature"] == 0.1
        assert backend_requests[-1]["max_tokens"] == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["slow_first"], indirect=True)
async def test_configure_during_a_reply_reaches_the_queued_turn(backends):
    config, _, _, messages = backends
    async with conversation(config) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(
            lambda: any(o.format == TEXT_DELTA for o in session.outputs)
        )
        await session.audio()
        await session.command(COMMIT)
        await session.command(CONFIGURE, instructions="Answer with one word.")
        await session.until(
            lambda: any(
                control_type(o) == TURN_DONE and o.turn == 2 for o in session.outputs
            )
        )
        assert messages[0][0]["content"] != "Answer with one word."
        assert messages[1][0]["content"] == "Answer with one word."


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["failed_chat"], indirect=True)
async def test_background_http_failure_reaches_reader_without_another_append(backends):
    async with conversation(backends[0]) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(lambda: session.error is not None)
        assert "abort" in str(session.error)


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["length_chat"], indirect=True)
async def test_reply_cut_at_max_tokens_ends_its_turn(backends):
    async with conversation(backends[0]) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(
            lambda: any(control_type(o) == TURN_DONE for o in session.outputs)
        )
        await session.audio(eos=True)
        assert session.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["long_audio"], indirect=True)
async def test_a_reply_still_playing_after_its_audio_left_is_interruptible(backends):
    """A reply whose audio had all left the server ended while the client still played it, so speaking over its tail interrupted nothing."""
    config = backends[0].model_copy(update={"max_output_lead_ms": 5000})
    async with conversation(config) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(
            lambda: sum(o.duration_ms for o in session.outputs if o.format == PCM_F32)
            == 2000
        )
        await session.coordinator.control_session(
            session.identity,
            control_chunk(INTERRUPT, session.clock),
            should_preempt=True,
        )
        await session.until(
            lambda: any(control_type(o) == TURN_INTERRUPTED for o in session.outputs)
        )
        assert not any(
            control_type(o) == TURN_DONE and o.turn == 1 for o in session.outputs
        )
        await session.audio(eos=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["long_audio"], indirect=True)
async def test_audio_stays_paced_and_held_turn_is_interruptible(backends):
    config = backends[0].model_copy(update={"max_output_lead_ms": 100})
    async with conversation(config) as session:
        await session.audio()
        await session.command(COMMIT)
        await session.until(lambda: any(o.format == PCM_F32 for o in session.outputs))
        audio_ms = sum(o.duration_ms for o in session.outputs if o.format == PCM_F32)
        assert audio_ms < 500
        assert not any(control_type(o) == TURN_DONE for o in session.outputs)
        await session.coordinator.control_session(
            session.identity,
            control_chunk(INTERRUPT, session.clock),
            should_preempt=True,
        )
        await session.until(
            lambda: any(control_type(o) == TURN_INTERRUPTED for o in session.outputs)
        )
        last = len(session.outputs)
        await session.command(TRUNCATE, t_ms=0)
        await session.audio(eos=True)
        assert backends[3][1][1:] == [
            {"role": "user", "content": "test question test question"}
        ]
        assert not any(
            o.turn == 1 and o.format == PCM_F32 for o in session.outputs[last:]
        )
        reply = [o for o in session.outputs if o.turn == 2 and o.kind == "data"]
        started = next(o for o in reply if control_type(o) == TURN_STARTED)
        done = next(o for o in reply if control_type(o) == TURN_DONE)
        audio = [o for o in reply if o.format == PCM_F32]
        assert audio[0].t_start_ms == started.t_start_ms
        for previous, current in zip(audio, audio[1:]):
            assert current.t_start_ms == pytest.approx(
                previous.t_start_ms + previous.duration_ms
            )
        assert done.t_start_ms == pytest.approx(
            audio[-1].t_start_ms + audio[-1].duration_ms
        )
