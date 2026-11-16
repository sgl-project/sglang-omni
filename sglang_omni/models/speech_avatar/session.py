# SPDX-License-Identifier: Apache-2.0
"""Speech-driven avatars on the Session input timeline: a pool of chunk workers renders the agent's reply audio into pictures or motion.

The models behind these nodes (FlashHead, MuseTalk) hear only the
agent's speech. Each renders a chunk of `chunk_frames` from a window of that
speech, `context_samples` before the chunk and `lookahead_samples` after it,
and a state that carries the Session from one chunk to the next; the state
goes with every chunk, so any worker of the pool renders any Session's next
chunk.

- `SpeechVideoHooks` (talking heads): while the agent is silent the node
  publishes empty frames, which keep the picture, as the user's audio
  arrives; a reply starts `render_lead` after the frame the node has reached,
  once a chunk of it arrived or it ended, and its chunks render as soon as
  their audio is in. The Session's first frame carries the worker's idle
  picture.
- `SpeechMotionHooks` (gestures): every block renders, silent ones included,
  once the user's audio covers it, as ELLSA 2's motion node does; a reply's
  audio plays from the block the node has reached.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sglang_omni.models.dyadic_audio.timeline import (
    DyadicAudio,
    pcm16_samples,
    timeline_sample,
)
from sglang_omni.models.speech_avatar.formats import (
    JPEG_FRAME,
    MOTION,
    SMPLX_FEATURES_PER_FRAME,
    SMPLX_MOTION,
    VIDEO,
)
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

# The timeline counts samples at the Session input's 16 kHz, the rate the workers read.
TIMELINE_RATE_HZ = SESSION_INPUT_SAMPLE_RATE_HZ
# How often a worker that failed asks its service whether it serves again.
RECOVERY_PROBE_S = 1.0
# A motion block whose reply audio is still short of its end waits this long past it for the speech node to catch up, so the reply stays in one piece on the timeline.
AUDIO_GRACE_MS = 200.0
# A reply whose audio has not grown for this long is placed even when shorter than a chunk: the speech node marks a turn done only once its audio played out.
REPLY_QUIET_MS = 150.0
SMPLX_ROW_BYTES = 4 * SMPLX_FEATURES_PER_FRAME


@dataclass(frozen=True)
class ChunkGeometry:
    """What a worker renders per request, in frames at its rate and in timeline samples."""

    fps: int
    chunk_frames: int
    context_samples: int
    lookahead_samples: int

    def __post_init__(self) -> None:
        if self.chunk_frames * TIMELINE_RATE_HZ % self.fps:
            raise ValueError(
                f"{self.chunk_frames} frames at {self.fps} fps are not a whole number of 16 kHz samples"
            )
        else:
            pass

    @property
    def chunk_samples(self) -> int:
        return self.chunk_frames * TIMELINE_RATE_HZ // self.fps

    @property
    def window_samples(self) -> int:
        return self.context_samples + self.chunk_samples + self.lookahead_samples

    @property
    def chunk_ms(self) -> float:
        return self.chunk_frames * 1000 / self.fps

    @property
    def lookahead_ms(self) -> float:
        return self.lookahead_samples * 1000 / TIMELINE_RATE_HZ


class WorkerUnavailable(Exception):
    """A worker could not render a chunk: it is down, restarting or overloaded, not the chunk's fault."""


class ChunkWorker(Protocol):
    """One worker of a speech-avatar model; it renders one chunk at a time, of any Session, since each chunk carries its Session's state."""

    def render(self, state: bytes, speech: np.ndarray) -> tuple[bytes, list[bytes]]:
        """Render one chunk from a float32 window of the geometry's window_samples at 16 kHz starting context_samples before the chunk; returns the Session's next state and the chunk's payloads. An empty state starts a Session. Raises WorkerUnavailable when the worker failed."""

    def is_healthy(self) -> bool:
        """Whether the worker serves again after a failure."""


@dataclass(kw_only=True)
class SpeechSession:
    emit: ChunkEmitter | None = None
    fail: Callable[[BaseException], None] | None = None
    # The worker's state after the Session's last rendered chunk.
    state: bytes = b""
    # A worker renders one of the Session's chunks; the next waits for its state, and nothing else is published meanwhile, so the Session's output stays in timeline order.
    is_rendering: bool = False
    # Chunks in a row whose worker failed; as many as there are workers fail the Session.
    failed_chunks: int = 0
    audio: DyadicAudio = field(default_factory=DyadicAudio.at_rates)
    # The timeline sample the node publishes next; None until the user's audio arrives.
    cursor: int | None = None
    error: BaseException | None = None
    # Whether a picture went out (video), so the next held frame may be empty.
    shown: bool = False
    # The last motion frame published (motion), which a held block repeats.
    last_row: bytes | None = None
    # When each turn's reply audio last arrived (video), on the hooks' clock.
    speech_at: dict[int | None, float] = field(default_factory=dict)
    # The start of each chunk rendered ahead and the state it rendered from (video), so an interrupt can take back the chunks it cut.
    history: list[tuple[int, bytes]] = field(default_factory=list)
    # Interrupts that came while a chunk rendered (video), applied once it is published.
    pending_interrupts: list[int | None] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ChunkWork:
    session_identity: SessionIdentity
    session: SpeechSession
    start: int
    # The turn of the reply audio each frame shows (video), or the block shows (motion, one entry); None while listening.
    turns: list[int | None]
    speech: np.ndarray


class SpeechAvatarHooks(SessionHooks):
    """The Session bookkeeping and the worker pool both nodes share; a subclass decides which chunk renders next and what it publishes."""

    def __init__(
        self,
        workers: list[ChunkWorker],
        geometry: ChunkGeometry,
        speech_sample_rate_hz: int,
        *,
        input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
        recovery_probe_s: float = RECOVERY_PROBE_S,
    ) -> None:
        if not workers:
            raise ValueError("a speech avatar stage needs at least one worker")
        else:
            pass
        self.workers = workers
        self.geometry = geometry
        self.speech_sample_rate_hz = speech_sample_rate_hz
        self.input_sample_rate_hz = input_sample_rate_hz
        self.recovery_probe_s = recovery_probe_s
        self.sessions: dict[SessionIdentity, SpeechSession] = {}
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.threads: list[threading.Thread | None] = [None] * len(workers)

    # The cursor starts on a multiple of this.
    boundary: int

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        with self.lock:
            self.sessions[session_identity] = SpeechSession(
                audio=DyadicAudio.at_rates(
                    user_rate_hz=self.input_sample_rate_hz,
                    speech_rate_hz=self.speech_sample_rate_hz,
                )
            )
            for index, thread in enumerate(self.threads):
                if thread is None:
                    self.threads[index] = threading.Thread(
                        target=self.render_sessions,
                        args=(index,),
                        name=f"speech-avatar-{index}",
                        daemon=True,
                    )
                    self.threads[index].start()
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
                session.audio.user.append(
                    chunk.t_start_ms, pcm16_samples(chunk.payload), eos=chunk.eos
                )
                if session.cursor is None:
                    session.cursor = (
                        timeline_sample(chunk.t_start_ms)
                        // self.boundary
                        * self.boundary
                    )
                else:
                    pass
            elif chunk.format == PCM_F32 and isinstance(chunk.payload, bytes):
                self.append_speech(
                    session, chunk.turn, np.frombuffer(chunk.payload, dtype="<f4")
                )
            elif control_type(chunk) == TURN_DONE:
                session.audio.replies.finish(chunk.turn)
            elif chunk.format == SESSION_CONTROL:
                pass
            else:
                raise ValueError(
                    "a speech avatar takes the user's pcm16 audio (its clock), the agent's pcm_f32le speech, and session controls"
                )
            self.advance(session)
            self.wake.notify_all()
        payload.data = {"accepted": True}
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        """An interrupt cuts the replies of its turn, or every reply without one, at the next chunk to render, and drops their later audio; chunks already published stay, and the client drops their frames by response."""
        if control_type(event) == INTERRUPT:
            with self.lock:
                session = self.sessions[session_identity]
                self.interrupt(session, event.turn)
                self.advance(session)
                self.wake.notify_all()
        else:
            pass

    def append_speech(
        self, session: SpeechSession, turn: int | None, samples: np.ndarray
    ) -> None:
        raise NotImplementedError

    def interrupt(self, session: SpeechSession, turn: int | None) -> None:
        """Cut the replies of `turn`, or every reply when it is None, at the next chunk to render; called with the lock held."""
        session.audio.replies.interrupt(
            turn, at=0 if session.cursor is None else session.cursor
        )

    def advance(self, session: SpeechSession) -> None:
        """Place replies and publish what needs no worker; called with the lock held after anything changed."""

    def next_chunk(
        self, session_identity: SessionIdentity, session: SpeechSession
    ) -> ChunkWork | None:
        """The Session's next chunk to render, moving its cursor past it, or None; called with the lock held."""
        raise NotImplementedError

    def publish_rendered(self, work: ChunkWork, payloads: list[bytes]) -> None:
        """Publish a rendered chunk; raises WorkerUnavailable, publishing nothing, when the payloads do not make one. Called with the lock held while the Session is open."""
        raise NotImplementedError

    def publish_held(self, work: ChunkWork) -> None:
        """Publish a chunk whose worker failed as a hold of the last picture or pose. Called with the lock held while the Session is open."""
        raise NotImplementedError

    def emit(self, session: SpeechSession, chunk: TimedChunk) -> None:
        assert session.emit is not None
        session.emit(chunk)

    def is_open(self, work: ChunkWork) -> bool:
        return self.sessions.get(work.session_identity) is work.session

    def next_work(self) -> ChunkWork | None:
        """Take the next chunk of the first Session with one ready and move that Session last; called with the lock held."""
        for session_identity, session in list(self.sessions.items()):
            if (
                session.cursor is None
                or session.error is not None
                or session.is_rendering
            ):
                continue
            else:
                work = self.next_chunk(session_identity, session)
            if work is None:
                continue
            else:
                self.sessions[session_identity] = self.sessions.pop(session_identity)
                session.is_rendering = True
                return work
        return None

    def take_work(self, index: int) -> ChunkWork | None:
        """The worker's next chunk, waiting for one; None once the stage has no open Session, when the worker's thread stops."""
        with self.lock:
            work = self.next_work()
            while work is None:
                if not self.sessions:
                    self.threads[index] = None
                    return None
                else:
                    self.wake.wait()
                    work = self.next_work()
            return work

    def recover(self, index: int) -> bool:
        """Wait until the worker serves again; False once the stage has no open Session, when the worker's thread stops."""
        while True:
            with self.lock:
                if not self.sessions:
                    self.threads[index] = None
                    return False
                else:
                    pass
            time.sleep(self.recovery_probe_s)
            if self.workers[index].is_healthy():
                return True
            else:
                pass

    def render_sessions(self, index: int) -> None:
        """The thread of one worker; it stops once the stage has no open Session."""
        worker = self.workers[index]
        while (work := self.take_work(index)) is not None:
            try:
                state, payloads = worker.render(work.session.state, work.speech)
                with self.lock:
                    if self.is_open(work):
                        self.publish_rendered(work, payloads)
                    else:
                        pass
                    work.session.state = state
                    work.session.failed_chunks = 0
                    work.session.is_rendering = False
                    if self.is_open(work):
                        self.advance(work.session)
                    else:
                        pass
                    self.wake.notify_all()
            except WorkerUnavailable as error:
                with self.lock:
                    if self.is_open(work):
                        self.publish_held(work)
                    else:
                        pass
                    work.session.is_rendering = False
                    work.session.failed_chunks += 1
                    is_lost = work.session.failed_chunks >= len(self.workers)
                    if is_lost:
                        work.session.error = error
                    elif self.is_open(work):
                        self.advance(work.session)
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


class SpeechVideoHooks(SpeechAvatarHooks):
    """A talking head that moves only while the agent speaks.

    The node publishes an empty frame, which keeps the picture, for each
    frame the user's audio reaches while no reply plays there, the
    Session's first frame carrying the idle picture. A reply starts on the
    first frame at or after `render_lead` past the frame the node has
    reached and the previous reply's end, once a chunk of it (plus the
    look-ahead) arrived, its turn ended, or its audio stopped arriving for
    `reply_quiet`: the lead is the render's time budget, so its first frame
    comes on time. Chunks render from there, back to back, each once the
    audio its window reads is in or, at the latest, once the user's audio
    comes within `render_lead` of its start, until a chunk starts `tail`
    past every reply; the frames of a chunk the reply does not reach show
    the model's silence.
    """

    def __init__(
        self,
        workers: list[ChunkWorker],
        geometry: ChunkGeometry,
        speech_sample_rate_hz: int,
        *,
        idle_picture: bytes,
        render_lead_ms: float,
        tail_ms: float = 0.0,
        reply_quiet_ms: float = REPLY_QUIET_MS,
        input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
        recovery_probe_s: float = RECOVERY_PROBE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            workers,
            geometry,
            speech_sample_rate_hz,
            input_sample_rate_hz=input_sample_rate_hz,
            recovery_probe_s=recovery_probe_s,
        )
        if TIMELINE_RATE_HZ % geometry.fps:
            raise ValueError(
                f"a frame at {geometry.fps} fps is not a whole number of 16 kHz samples"
            )
        else:
            self.frame_samples = TIMELINE_RATE_HZ // geometry.fps
        self.boundary = self.frame_samples
        self.idle_picture = idle_picture
        self.render_lead = timeline_sample(render_lead_ms)
        self.tail = timeline_sample(tail_ms)
        self.reply_quiet_s = reply_quiet_ms / 1000
        self.clock = clock

    def append_speech(
        self, session: SpeechSession, turn: int | None, samples: np.ndarray
    ) -> None:
        session.audio.replies.append_pending(turn, samples)
        session.speech_at[turn] = self.clock()

    def place(self, session: SpeechSession) -> None:
        """Start each unplaced reply, in arrival order, at the first frame at or after the render lead past the cursor and the previous reply's end, once a chunk (and the look-ahead) of it arrived, it is done, or its audio went quiet."""
        assert session.cursor is not None
        replies = session.audio.replies
        now = self.clock()
        min_length = self.geometry.chunk_samples + self.geometry.lookahead_samples
        end = session.cursor + self.render_lead
        for reply in replies.replies:
            length = replies.length(reply)
            if reply.start is not None:
                end = max(end, reply.start + length)
            elif (
                length >= min_length
                or reply.is_done
                or now - session.speech_at.get(reply.turn, now) >= self.reply_quiet_s
            ):
                reply.start = math.ceil(end / self.frame_samples) * self.frame_samples
                end = reply.start + length
            else:
                break

    def interrupt(self, session: SpeechSession, turn: int | None) -> None:
        """Take back the chunks of the cut replies rendered ahead (see `rewind`), once no chunk of the Session renders."""
        session.pending_interrupts.append(turn)
        if not session.is_rendering:
            self.apply_interrupts(session)
        else:
            pass

    def apply_interrupts(self, session: SpeechSession) -> None:
        while session.pending_interrupts:
            self.rewind(session, session.pending_interrupts.pop(0))

    def rewind(self, session: SpeechSession, turn: int | None) -> None:
        """Cut the replies of `turn` (every reply when None) where the client has not shown them yet.

        Chunks render ahead of their frames, so the cursor may be far past
        the frame the client shows when the interrupt comes. The client drops
        the cut reply's frames not yet due (every frame of a chunk from the
        reply's start on carries its turn), so the node moves its cursor back
        to the first chunk it rendered from at or after the reply's start and
        `render_lead` past the user's audio, renders on from that chunk's
        state, and the next reply starts there instead of after the cut one.
        It does not when a reply of another turn played in the chunks it
        would take back.
        """
        replies = session.audio.replies
        if session.cursor is None:
            replies.interrupt(turn, at=0)
            return
        else:
            cut = [
                reply
                for reply in replies.replies
                if reply.start is not None and (turn is None or reply.turn == turn)
            ]
        floor = (
            math.ceil((session.audio.user.end + self.render_lead) / self.frame_samples)
            * self.frame_samples
        )
        earliest = max(
            floor, min((reply.start for reply in cut), default=session.cursor)
        )
        back = [
            (start, state)
            for start, state in session.history
            if earliest <= start < session.cursor
        ]
        if back:
            start, state = min(back)
            kept = [
                reply
                for reply in replies.replies
                if reply.start is not None
                and reply not in cut
                and reply.start < session.cursor
                and replies.end(reply) > start
            ]
        else:
            start, state, kept = session.cursor, session.state, []
        if not kept:
            session.cursor = start
            session.state = state
            session.history = [h for h in session.history if h[0] < start]
        else:
            pass
        replies.interrupt(turn, at=session.cursor)

    def speaks(self, session: SpeechSession, start: int) -> bool:
        """Whether the chunk from `start` renders: a reply plays in it or ended within the tail before it."""
        return (
            session.audio.replies.turn_at(
                start - self.tail, self.tail + self.geometry.chunk_samples
            )
            is not None
        )

    def has_audio_until(self, session: SpeechSession, until: int) -> bool:
        """Whether the replies a window up to `until` reads are in: the last one placed before it ended or reaches it."""
        replies = session.audio.replies
        last = None
        for reply in replies.replies:
            if reply.start is not None and reply.start < until:
                last = reply
            else:
                pass
        return last is None or last.is_done or replies.end(last) >= until

    def advance(self, session: SpeechSession) -> None:
        if session.cursor is None:
            return
        else:
            self.place(session)
        if session.error is not None or session.is_rendering or session.emit is None:
            return
        else:
            self.apply_interrupts(session)
        while session.cursor < session.audio.user.end and not self.speaks(
            session, session.cursor
        ):
            self.emit_frame(session, session.cursor, None, b"")
            session.cursor += self.frame_samples
        session.audio.user.drop_before(session.cursor)

    def emit_frame(
        self, session: SpeechSession, start: int, turn: int | None, jpeg: bytes
    ) -> None:
        """Publish one frame; the Session's first carries the idle picture when it has none."""
        if not jpeg and not session.shown:
            jpeg = self.idle_picture
        else:
            pass
        session.shown = session.shown or bool(jpeg)
        self.emit(
            session,
            TimedChunk(
                VIDEO,
                start * 1000 / TIMELINE_RATE_HZ,
                1000 / self.geometry.fps,
                0,
                jpeg,
                format=JPEG_FRAME,
                turn=turn,
            ),
        )

    def next_chunk(
        self, session_identity: SessionIdentity, session: SpeechSession
    ) -> ChunkWork | None:
        geometry = self.geometry
        start = session.cursor
        assert start is not None
        if not self.speaks(session, start):
            return None
        elif (
            not session.audio.user.is_ended
            and session.audio.user.end < start - self.render_lead
            and not self.has_audio_until(
                session, start + geometry.chunk_samples + geometry.lookahead_samples
            )
        ):
            return None
        else:
            replies = session.audio.replies
        # A frame after a reply in the chunk (its tail, or the silence after its end) carries that reply's turn, so a client dropping the reply drops it too; frames before the chunk's first reply listen.
        turns = []
        turn = replies.turn_at(start - self.tail, self.tail)
        for frame in range(start, start + geometry.chunk_samples, self.frame_samples):
            playing = replies.turn_at(frame, self.frame_samples)
            turn = turn if playing is None else playing
            turns.append(turn)
        session.history = [
            (old, state)
            for old, state in session.history
            if old + geometry.chunk_samples > session.audio.user.end
        ] + [(start, session.state)]
        work = ChunkWork(
            session_identity=session_identity,
            session=session,
            start=start,
            turns=turns,
            speech=replies.window(
                start - geometry.context_samples, geometry.window_samples
            ),
        )
        session.cursor = start + geometry.chunk_samples
        replies.drop_finished(session.cursor - geometry.context_samples)
        return work

    def publish_rendered(self, work: ChunkWork, payloads: list[bytes]) -> None:
        if len(payloads) != self.geometry.chunk_frames or not all(payloads):
            raise WorkerUnavailable(
                f"the worker sent {len(payloads)} pictures for a {self.geometry.chunk_frames}-frame chunk"
            )
        else:
            for index, jpeg in enumerate(payloads):
                self.emit_frame(
                    work.session,
                    work.start + index * self.frame_samples,
                    work.turns[index],
                    jpeg,
                )

    def publish_held(self, work: ChunkWork) -> None:
        for index, turn in enumerate(work.turns):
            self.emit_frame(
                work.session, work.start + index * self.frame_samples, turn, b""
            )


class SpeechMotionHooks(SpeechAvatarHooks):
    """Gestures from the agent's speech, one block of `chunk_frames` at a time, silent blocks included.

    A block is ready once the user's audio, the Session's clock, covers it
    and its look-ahead, and the reply playing at its end has audio there or
    the user's audio is `audio_grace` past it; after the input ended, while
    it holds the agent's audio. A reply's first audio starts at the block the node has reached,
    or the first block after the previous reply; its later audio follows on.
    Blocks render in order, each from the state of the one before.
    """

    def __init__(
        self,
        workers: list[ChunkWorker],
        geometry: ChunkGeometry,
        speech_sample_rate_hz: int,
        *,
        audio_grace_ms: float = AUDIO_GRACE_MS,
        input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
        recovery_probe_s: float = RECOVERY_PROBE_S,
    ) -> None:
        super().__init__(
            workers,
            geometry,
            speech_sample_rate_hz,
            input_sample_rate_hz=input_sample_rate_hz,
            recovery_probe_s=recovery_probe_s,
        )
        self.boundary = geometry.chunk_samples
        self.audio_grace = timeline_sample(audio_grace_ms)

    def append_speech(
        self, session: SpeechSession, turn: int | None, samples: np.ndarray
    ) -> None:
        session.audio.replies.append_following(
            turn,
            samples,
            cursor=0 if session.cursor is None else session.cursor,
            boundary=self.geometry.chunk_samples,
        )

    def next_chunk(
        self, session_identity: SessionIdentity, session: SpeechSession
    ) -> ChunkWork | None:
        geometry = self.geometry
        audio = session.audio
        start = session.cursor
        assert start is not None
        if audio.user.is_ended:
            is_ready = start < max(audio.user.end, audio.speech_end())
        else:
            until = start + geometry.chunk_samples + geometry.lookahead_samples
            replies = audio.replies
            last = next(
                (
                    reply
                    for reply in reversed(replies.replies)
                    if reply.start is not None and reply.start < until
                ),
                None,
            )
            is_short = (
                last is not None
                and not last.is_done
                and replies.end(last) < until
                and replies.end(last) > start
            )
            is_ready = audio.user.end >= until and (
                not is_short or audio.user.end >= until + self.audio_grace
            )
        if not is_ready:
            return None
        else:
            work = ChunkWork(
                session_identity=session_identity,
                session=session,
                start=start,
                turns=[audio.replies.turn_at(start, geometry.chunk_samples)],
                speech=audio.speech_window(
                    start - geometry.context_samples, geometry.window_samples
                ),
            )
        session.cursor = start + geometry.chunk_samples
        audio.user.drop_before(session.cursor)
        audio.replies.drop_ended(session.cursor - geometry.context_samples)
        return work

    def emit_block(self, work: ChunkWork, rows: bytes) -> None:
        self.emit(
            work.session,
            TimedChunk(
                MOTION,
                work.start * 1000 / TIMELINE_RATE_HZ,
                self.geometry.chunk_ms,
                0,
                rows,
                format=SMPLX_MOTION,
                turn=work.turns[0],
            ),
        )
        work.session.last_row = rows[-SMPLX_ROW_BYTES:]

    def publish_rendered(self, work: ChunkWork, payloads: list[bytes]) -> None:
        if (
            len(payloads) != 1
            or len(payloads[0]) != self.geometry.chunk_frames * SMPLX_ROW_BYTES
        ):
            raise WorkerUnavailable(
                f"the worker sent {[len(p) for p in payloads]} bytes for a {self.geometry.chunk_frames}-frame smplx_motion block"
            )
        else:
            self.emit_block(work, payloads[0])

    def publish_held(self, work: ChunkWork) -> None:
        row = work.session.last_row or bytes(SMPLX_ROW_BYTES)
        self.emit_block(work, row * self.geometry.chunk_frames)


__all__ = [
    "ChunkGeometry",
    "ChunkWorker",
    "SpeechAvatarHooks",
    "SpeechMotionHooks",
    "SpeechVideoHooks",
    "WorkerUnavailable",
]
