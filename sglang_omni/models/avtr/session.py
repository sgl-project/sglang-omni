# SPDX-License-Identifier: Apache-2.0
"""The avatar stage: one rendered stream per Session from the user's audio and the agent's speech, on the Session input timeline."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sglang_omni.models.avtr.formats import JPEG_FRAME, VIDEO, VIDEO_FPS
from sglang_omni.models.dyadic_audio.timeline import DyadicAudio, pcm16_samples
from sglang_omni.models.personaplex.formats import PERSONAPLEX_REPLY
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    INTERRUPT,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    TURN_DONE,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter, SessionContext, SessionHooks

# The timeline counts samples at the Session input's 16 kHz, the rate the AVTR-1 checkpoint reads.
FRAME_SAMPLES = SESSION_INPUT_SAMPLE_RATE_HZ // VIDEO_FPS
FRAME_MS = 1000 / VIDEO_FPS
# The checkpoint steps five frames per call and reads the chunk plus a look-ahead of five frames and 80 samples; the 120 ms before the chunk stay in the Session's state blob.
CHUNK_FRAMES = 5
LOOKAHEAD_FRAMES = 5
LOOKAHEAD_EXTRA_SAMPLES = 80
CHUNK_SAMPLES = CHUNK_FRAMES * FRAME_SAMPLES
WINDOW_SAMPLES = (
    CHUNK_FRAMES + LOOKAHEAD_FRAMES
) * FRAME_SAMPLES + LOOKAHEAD_EXTRA_SAMPLES
# How often a worker that failed asks its renderer whether it serves again.
RECOVERY_PROBE_S = 1.0


class RendererUnavailable(Exception):
    """A renderer worker could not render a chunk: it is down, restarting or overloaded, not the chunk's fault."""


class AvatarRenderer(Protocol):
    """One renderer worker; it renders one chunk at a time, of any Session, since each chunk carries its Session's state."""

    def render(
        self,
        state: bytes,
        speech: np.ndarray,
        listen: np.ndarray,
        frame: Callable[[bytes], None],
    ) -> bytes:
        """Render one chunk from float32 windows of WINDOW_SAMPLES at 16 kHz starting at the chunk, handing each frame's JPEG to `frame` as it comes; returns the Session's next state. An empty state starts a Session. Raises RendererUnavailable when the worker failed."""

    def is_healthy(self) -> bool:
        """Whether the worker serves again after a failure."""


@dataclass(kw_only=True)
class AvatarSession:
    emit: ChunkEmitter | None = None
    fail: Callable[[BaseException], None] | None = None
    # The renderer's state after the Session's last rendered chunk; any worker renders the next chunk from it.
    state: bytes = b""
    # A worker renders one of the Session's chunks; the next waits for its state.
    is_rendering: bool = False
    # Chunks in a row whose worker failed; as many as there are workers fail the Session.
    failed_chunks: int = 0
    # The user's audio a later chunk still listens to, and the agent's audio as replies or as units.
    audio: DyadicAudio = field(default_factory=DyadicAudio.at_rates)
    # The start of the next chunk to render; None until the user's audio arrives.
    cursor: int | None = None
    error: BaseException | None = None


@dataclass(frozen=True, kw_only=True)
class RenderWork:
    """One step of a render worker: the frame turns of the chunks it holds, then the chunk it renders with its windows."""

    session_identity: SessionIdentity
    session: AvatarSession
    held: dict[int, list[int | None]]
    chunk_start: int
    turns: list[int | None]
    speech: np.ndarray
    listen: np.ndarray


@dataclass(kw_only=True)
class RenderedFrames:
    """Publishes the frames of one chunk as the renderer hands them over, counting them."""

    hooks: AvatarHooks
    work: RenderWork
    shown: int = 0

    def publish(self, jpeg: bytes) -> None:
        if self.shown >= CHUNK_FRAMES:
            raise RendererUnavailable(
                f"the renderer sent more than the {CHUNK_FRAMES} frames of a chunk"
            )
        else:
            self.hooks.publish(
                self.work,
                self.work.chunk_start + self.shown * FRAME_SAMPLES,
                self.work.turns[self.shown],
                jpeg,
            )
            self.shown += 1


class AvatarHooks(SessionHooks):
    """Turns the user's audio and the agent's speech into frames on a pool of renderer workers shared by the stage's Sessions.

    A chunk is ready once the user's audio reaches its start, since it
    listens to the user WINDOW_SAMPLES earlier; a reply plays from a frame
    boundary at or after the next chunk to render, once WINDOW_SAMPLES of it
    arrived or its turn ended. Every frame carries the turn of the reply audio
    it shows; a held frame has an empty payload.

    Each renderer has a thread that takes the next ready chunk of any Session
    not already rendering, so a Session's chunks render in order on whichever
    worker is free. A chunk whose worker failed is published as held frames,
    the Session's state stays the one before it, and the worker rests until
    its renderer is healthy again.
    """

    def __init__(
        self,
        renderers: list[AvatarRenderer],
        speech_sample_rate_hz: int,
        *,
        input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
        reply_stream: bool = False,
        recovery_probe_s: float = RECOVERY_PROBE_S,
    ) -> None:
        if not renderers:
            raise ValueError("the avatar stage needs at least one renderer worker")
        else:
            pass
        self.renderers = renderers
        self.speech_sample_rate_hz = speech_sample_rate_hz
        self.input_sample_rate_hz = input_sample_rate_hz
        self.reply_stream = reply_stream
        self.recovery_probe_s = recovery_probe_s
        self.sessions: dict[SessionIdentity, AvatarSession] = {}
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.workers: list[threading.Thread | None] = [None] * len(renderers)

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        with self.lock:
            self.sessions[session_identity] = AvatarSession(
                audio=DyadicAudio.at_rates(
                    user_rate_hz=self.input_sample_rate_hz,
                    speech_rate_hz=self.speech_sample_rate_hz,
                    reply_stream=self.reply_stream,
                )
            )
            for index, worker in enumerate(self.workers):
                if worker is None:
                    self.workers[index] = threading.Thread(
                        target=self.render_sessions,
                        args=(index,),
                        name=f"avatar-render-{index}",
                        daemon=True,
                    )
                    self.workers[index].start()
                else:
                    pass

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        with self.lock:
            self.sessions[session_identity].emit = emit

    def attach_output_error_handler(
        self, session_identity: SessionIdentity, fail: Callable[[BaseException], None]
    ) -> None:
        with self.lock:
            self.sessions[session_identity].fail = fail

    def close(self, session_identity: SessionIdentity) -> None:
        with self.lock:
            self.sessions.pop(session_identity)
            self.wake.notify_all()

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        with self.lock:
            session = self.sessions[session_identity]
            return ResourceUsage(bytes=len(session.state) + session.audio.nbytes)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        with self.lock:
            session = self.sessions[context.session_identity]
            if chunk.format == PCM16 and isinstance(chunk.payload, bytes):
                self.append_partner(session, chunk, chunk.payload)
            elif chunk.format == PCM_F32 and isinstance(chunk.payload, bytes):
                self.append_speech(session, chunk, chunk.payload)
            elif chunk.format == PERSONAPLEX_REPLY and isinstance(chunk.payload, dict):
                session.audio.units.append(
                    chunk.t_start_ms, pcm16_samples(chunk.payload["pcm"]), eos=chunk.eos
                )
            elif control_type(chunk) == TURN_DONE:
                session.audio.replies.finish(chunk.turn)
            elif chunk.format == SESSION_CONTROL:
                pass
            else:
                raise ValueError(
                    "the avatar takes the user's pcm16 audio, the agent's pcm_f32le speech or personaplex_reply units, and session controls"
                )
            self.place(session)
            self.wake.notify_all()
        payload.data = {"accepted": True}
        return payload

    def append_partner(
        self, session: AvatarSession, chunk: TimedChunk, pcm: bytes
    ) -> None:
        session.audio.user.append(chunk.t_start_ms, pcm16_samples(pcm), eos=chunk.eos)
        if session.cursor is None:
            session.cursor = session.audio.user.end // CHUNK_SAMPLES * CHUNK_SAMPLES
        else:
            pass

    def append_speech(
        self, session: AvatarSession, chunk: TimedChunk, pcm: bytes
    ) -> None:
        session.audio.replies.append_pending(
            chunk.turn, np.frombuffer(pcm, dtype="<f4")
        )

    def place(self, session: AvatarSession) -> None:
        """Start each unplaced reply, in turn order, at the first frame boundary at or after the next chunk to render and the previous reply's end, once WINDOW_SAMPLES of it arrived or its turn ended."""
        if session.cursor is None:
            return
        else:
            session.audio.replies.place(
                cursor=session.cursor,
                boundary=FRAME_SAMPLES,
                min_length=WINDOW_SAMPLES,
            )

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        """An interrupt ends the replies of its turn, or every reply without one, at the next chunk to render, and drops their later audio."""
        if control_type(event) == INTERRUPT:
            with self.lock:
                session = self.sessions[session_identity]
                session.audio.replies.interrupt(
                    event.turn, at=0 if session.cursor is None else session.cursor
                )
                self.wake.notify_all()
        else:
            pass

    def ready_chunks(self, session: AvatarSession) -> range:
        """The starts of the chunks that may render: their listening window has arrived and, while the agent's units stream, so has their speech window; once the input ended, they show placed speech."""
        audio = session.audio
        if session.cursor is None or session.error is not None or session.is_rendering:
            return range(0)
        elif audio.user.is_ended:
            last = max(
                audio.user.end // CHUNK_SAMPLES * CHUNK_SAMPLES,
                math.ceil(audio.speech_end() / CHUNK_SAMPLES) * CHUNK_SAMPLES
                - CHUNK_SAMPLES,
            )
        else:
            last = audio.user.end // CHUNK_SAMPLES * CHUNK_SAMPLES
            while last >= session.cursor and audio.awaits_units(last + WINDOW_SAMPLES):
                last -= CHUNK_SAMPLES
        return range(session.cursor, last + 1, CHUNK_SAMPLES)

    def next_work(self) -> RenderWork | None:
        """Take the next chunk of the first Session with one ready and move that Session last; called with the lock held.

        Before the input ends the newest ready chunk renders and the older
        ready ones are held: they are late already, and rendering them would
        delay every later chunk as much.
        """
        for session_identity, session in list(self.sessions.items()):
            ready = self.ready_chunks(session)
            if not ready:
                continue
            elif session.audio.user.is_ended:
                held, chunk_start = [], ready[0]
            else:
                held, chunk_start = list(ready[:-1]), ready[-1]
            work = RenderWork(
                session_identity=session_identity,
                session=session,
                held={start: self.frame_turns(session, start) for start in held},
                chunk_start=chunk_start,
                turns=self.frame_turns(session, chunk_start),
                speech=session.audio.speech_window(chunk_start, WINDOW_SAMPLES),
                listen=session.audio.user.window(
                    chunk_start - WINDOW_SAMPLES, WINDOW_SAMPLES
                ),
            )
            self.sessions[session_identity] = self.sessions.pop(session_identity)
            session.is_rendering = True
            session.cursor = chunk_start + CHUNK_SAMPLES
            session.audio.user.drop_before(session.cursor - WINDOW_SAMPLES)
            session.audio.units.drop_before(session.cursor)
            session.audio.replies.drop_finished(session.cursor)
            self.place(session)
            return work
        return None

    def frame_turns(self, session: AvatarSession, chunk_start: int) -> list[int | None]:
        """The turn of the reply audio each frame of the chunk shows, None for a listening frame."""
        return [
            session.audio.replies.turn_at(frame_start, FRAME_SAMPLES)
            for frame_start in range(
                chunk_start, chunk_start + CHUNK_SAMPLES, FRAME_SAMPLES
            )
        ]

    def publish(
        self,
        work: RenderWork,
        frame_start: int,
        turn: int | None,
        jpeg: bytes,
    ) -> None:
        """Hand one frame to the Session's reader unless the Session closed meanwhile."""
        with self.lock:
            if self.sessions.get(work.session_identity) is work.session:
                assert work.session.emit is not None
                work.session.emit(
                    TimedChunk(
                        VIDEO,
                        frame_start * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ,
                        FRAME_MS,
                        0,
                        jpeg,
                        format=JPEG_FRAME,
                        turn=turn,
                    )
                )
            else:
                pass

    def take_work(self, index: int) -> RenderWork | None:
        """The worker's next chunk, waiting for one; None once the stage has no open Session, when the worker stops."""
        with self.lock:
            work = self.next_work()
            while work is None:
                if not self.sessions:
                    self.workers[index] = None
                    return None
                else:
                    self.wake.wait()
                    work = self.next_work()
            return work

    def recover(self, index: int) -> bool:
        """Wait until the worker's renderer is healthy again; False once the stage has no open Session, when the worker stops."""
        while True:
            with self.lock:
                if not self.sessions:
                    self.workers[index] = None
                    return False
                else:
                    pass
            time.sleep(self.recovery_probe_s)
            if self.renderers[index].is_healthy():
                return True
            else:
                pass

    def render_sessions(self, index: int) -> None:
        """The worker of one renderer; it stops once the stage has no open Session."""
        renderer = self.renderers[index]
        while (work := self.take_work(index)) is not None:
            for chunk_start, turns in work.held.items():
                for frame_index, turn in enumerate(turns):
                    self.publish(
                        work, chunk_start + frame_index * FRAME_SAMPLES, turn, b""
                    )
            frames = RenderedFrames(hooks=self, work=work)
            try:
                state = renderer.render(
                    work.session.state, work.speech, work.listen, frames.publish
                )
                if frames.shown != CHUNK_FRAMES:
                    raise RendererUnavailable(
                        f"the renderer sent {frames.shown} frames of a {CHUNK_FRAMES}-frame chunk"
                    )
                else:
                    pass
            except RendererUnavailable as error:
                for frame_index in range(frames.shown, CHUNK_FRAMES):
                    self.publish(
                        work,
                        work.chunk_start + frame_index * FRAME_SAMPLES,
                        work.turns[frame_index],
                        b"",
                    )
                with self.lock:
                    work.session.is_rendering = False
                    work.session.failed_chunks += 1
                    is_lost = work.session.failed_chunks >= len(self.renderers)
                    if is_lost:
                        work.session.error = error
                    else:
                        pass
                    self.wake.notify_all()
                if is_lost:
                    assert work.session.fail is not None
                    work.session.fail(error)
                else:
                    pass
                if not self.recover(index):
                    return
                else:
                    continue
            except Exception as error:
                with self.lock:
                    work.session.error = error
                    work.session.is_rendering = False
                assert work.session.fail is not None
                work.session.fail(error)
                continue
            with self.lock:
                work.session.state = state
                work.session.failed_chunks = 0
                work.session.is_rendering = False
                self.wake.notify_all()


__all__ = ["AvatarHooks", "AvatarRenderer", "RendererUnavailable"]
