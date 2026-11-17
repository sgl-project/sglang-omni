# SPDX-License-Identifier: Apache-2.0
"""A talking head whose worker is a pipeline: several chunks of one Session are in flight at once and come back in order.

Live Avatar's worker spreads a block over five GPUs, one denoising step per
DiT rank and a decoder, so a block takes about five stages to come back
while a new one can start every stage. `PipelinedVideoHooks` keeps the
placement and the frame timeline of `SpeechVideoHooks` (a reply starts
`render_lead` past the frame the node has reached; listening frames are
empty and keep the picture) and changes one thing: up to `depth` chunks of
a Session render at once. Their state is a handle the node writes itself,
the Session's key and the chunk's index, because the worker keeps the
Session's caches on its GPUs; the worker takes a Session's chunks in index
order. Chunks are published in index order whatever order they come back
in, and nothing else of the Session is published while any is in flight.

With `prime_chunks`, a Session's first chunks are silent ones sent as soon as
the user's audio arrives, whose pictures are dropped: Live Avatar's first
block of a Session and of its second clip also fill the attention sink and
prime the decoder, so they take about twice as long, and that should happen
before the first reply rather than in it.
"""

from __future__ import annotations

import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sglang_omni.models.dyadic_audio.timeline import DyadicAudio
from sglang_omni.models.speech_avatar.session import (
    RECOVERY_PROBE_S,
    ChunkGeometry,
    ChunkWork,
    ChunkWorker,
    SpeechAvatarHooks,
    SpeechSession,
    SpeechVideoHooks,
    WorkerUnavailable,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ

# The state a pipelined worker reads: the Session's 16-byte key and the chunk's index.
HANDLE = struct.Struct("<16sI")


def handle(key: bytes, index: int) -> bytes:
    return HANDLE.pack(key, index)


@dataclass(kw_only=True)
class PipelinedSession(SpeechSession):
    key: bytes = field(default_factory=lambda: uuid.uuid4().bytes)
    # The index the Session's next chunk gets.
    next_index: int = 0
    # The chunks sent and not yet published, in index order.
    in_flight: deque = field(default_factory=deque)
    # What came back for chunks in flight: their payloads, or the WorkerUnavailable that failed them.
    results: dict = field(default_factory=dict)
    # Silent chunks sent at the Session's start whose pictures nobody sees.
    primed: int = 0


@dataclass(frozen=True, kw_only=True)
class IndexedWork(ChunkWork):
    index: int
    # A silent chunk that warms the worker's Session (its first blocks fill caches the later ones read) and publishes nothing.
    primer: bool = False


class PipelinedVideoHooks(SpeechVideoHooks):
    """`SpeechVideoHooks` with up to `depth` chunks of a Session in flight; each worker gets `depth` threads."""

    def __init__(
        self,
        workers: list[ChunkWorker],
        geometry: ChunkGeometry,
        speech_sample_rate_hz: int,
        *,
        idle_picture: bytes,
        render_lead_ms: float,
        depth: int,
        prime_chunks: int = 0,
        tail_ms: float = 0.0,
        input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
        recovery_probe_s: float = RECOVERY_PROBE_S,
    ) -> None:
        super().__init__(
            workers,
            geometry,
            speech_sample_rate_hz,
            idle_picture=idle_picture,
            render_lead_ms=render_lead_ms,
            tail_ms=tail_ms,
            input_sample_rate_hz=input_sample_rate_hz,
            recovery_probe_s=recovery_probe_s,
        )
        if depth < 1:
            raise ValueError("a pipelined worker takes at least one chunk at a time")
        else:
            self.depth = depth
        self.prime_chunks = prime_chunks
        self.threads = [None] * (len(workers) * depth)
        # Chunks in a row that failed before a Session counts as lost: every thread of the pool lost one, and one more.
        self.max_failed_chunks = len(workers) * depth + 1

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        with self.lock:
            self.sessions[session_identity] = PipelinedSession(
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
                        name=f"liveavatar-{index}",
                        daemon=True,
                    )
                    self.threads[index].start()
                else:
                    pass

    def interrupt(self, session: SpeechSession, turn: int | None) -> None:
        """Cut the replies of `turn` at the next chunk to render, as `SpeechAvatarHooks` does: a Live Avatar Session's state is a handle to the worker's caches, which moved on with the chunks in flight, so the chunks rendered ahead cannot be taken back."""
        SpeechAvatarHooks.interrupt(self, session, turn)

    def next_work(self) -> IndexedWork | None:
        """The next chunk of the first Session with one ready and room in its pipeline, moving that Session last; called with the lock held."""
        for session_identity, session in list(self.sessions.items()):
            if (
                session.cursor is None
                or session.error is not None
                or len(session.in_flight) >= self.depth
            ):
                continue
            elif session.primed < self.prime_chunks:
                session.primed += 1
                indexed = IndexedWork(
                    session_identity=session_identity,
                    session=session,
                    start=session.cursor,
                    turns=[],
                    speech=np.zeros(self.geometry.window_samples, dtype=np.float32),
                    index=session.next_index,
                    primer=True,
                )
            else:
                work = self.next_chunk(session_identity, session)
                if work is None:
                    continue
                else:
                    indexed = IndexedWork(
                        session_identity=work.session_identity,
                        session=work.session,
                        start=work.start,
                        turns=work.turns,
                        speech=work.speech,
                        index=session.next_index,
                    )
            session.next_index += 1
            session.in_flight.append(indexed)
            session.is_rendering = self.publishes_in_flight(session)
            self.sessions[session_identity] = self.sessions.pop(session_identity)
            return indexed
        return None

    @staticmethod
    def publishes_in_flight(session: PipelinedSession) -> bool:
        """Whether a chunk in flight will publish frames, so nothing else of the Session may go out before it."""
        return any(not work.primer for work in session.in_flight)

    def flush(self, session: PipelinedSession, is_open: bool) -> bool:
        """Publish the chunks at the head of the Session's pipeline that came back; True when the Session is lost. Called with the lock held."""
        while session.in_flight and session.in_flight[0].index in session.results:
            work = session.in_flight.popleft()
            result = session.results.pop(work.index)
            if work.primer or not is_open or session.error is not None:
                continue
            elif isinstance(result, WorkerUnavailable):
                self.publish_held(work)
                session.failed_chunks += 1
            else:
                try:
                    self.publish_rendered(work, result)
                    session.failed_chunks = 0
                except WorkerUnavailable:
                    self.publish_held(work)
                    session.failed_chunks += 1
        session.is_rendering = self.publishes_in_flight(session)
        if session.error is None and session.failed_chunks >= self.max_failed_chunks:
            session.error = WorkerUnavailable(
                f"{session.failed_chunks} chunks in a row found no Live Avatar worker"
            )
            return True
        elif is_open:
            self.advance(session)
            return False
        else:
            return False

    def recover(self, index: int) -> bool:
        """Wait until the thread's worker serves again; False once the stage has no open Session, when the thread stops."""
        while True:
            with self.lock:
                if not self.sessions:
                    self.threads[index] = None
                    return False
                else:
                    pass
            time.sleep(self.recovery_probe_s)
            if self.workers[index % len(self.workers)].is_healthy():
                return True
            else:
                pass

    def render_sessions(self, index: int) -> None:
        """One thread of a worker; it stops once the stage has no open Session."""
        worker = self.workers[index % len(self.workers)]
        while (work := self.take_work(index)) is not None:
            session = work.session
            try:
                _, payloads = worker.render(
                    handle(session.key, work.index), work.speech
                )
                result: list[bytes] | WorkerUnavailable = payloads
            except WorkerUnavailable as error:
                result = error
            except Exception as error:
                with self.lock:
                    session.error = error
                    session.in_flight.clear()
                    session.results.clear()
                    session.is_rendering = False
                    self.wake.notify_all()
                assert session.fail is not None
                session.fail(error)
                continue
            with self.lock:
                session.results[work.index] = result
                is_lost = self.flush(session, self.is_open(work))
                self.wake.notify_all()
            if is_lost:
                assert session.fail is not None and session.error is not None
                session.fail(session.error)
            else:
                pass
            if isinstance(result, WorkerUnavailable) and not self.recover(index):
                return
            else:
                pass


__all__ = [
    "HANDLE",
    "IndexedWork",
    "PipelinedSession",
    "PipelinedVideoHooks",
    "handle",
]
