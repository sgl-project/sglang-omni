# SPDX-License-Identifier: Apache-2.0
"""The ELLSA 2 motion stage: per Session, the user's audio and the agent's reply audio on the Session input timeline, generated into motion one 0.8 s block at a time."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.signal import resample_poly

from sglang_omni.models.ellsa2.batching import (
    GRAPH_BATCH_SIZES,
    BatchedBlocks,
    BlockRequest,
    SessionCache,
)
from sglang_omni.models.ellsa2.formats import MOTION, SMPLH_FLAME_MOTION
from sglang_omni.models.ellsa2.generator import (
    BLOCK_SAMPLES,
    MAX_HISTORY_BLOCKS,
    HistoryBlock,
    retained_history,
)
from sglang_omni.models.ellsa2.loading import MotionModels, load_motion_models
from sglang_omni.models.ellsa2.motion import MotionDecoder
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    INTERRUPT,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    control_type,
)
from sglang_omni.scheduling.session import (
    ChunkEmitter,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.utils.device import resolve_concrete_device

BLOCK_MS = BLOCK_SAMPLES * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ
PCM16_SCALE = 32768
# scipy's resample_poly filter reaches 10 * max(up, down) samples of the upsampled signal to each side.
RESAMPLE_FILTER_HALF_WIDTH = 10
# One Session keeps up on any GPU that runs the checkpoint faster than real time; a profile sets the capacity measured on its GPU.
DEFAULT_MAX_OPEN_SESSIONS = 1


@dataclass(kw_only=True)
class Speech:
    """Reply audio of one turn at the speech rate, playing from timeline sample `start` (16 kHz)."""

    turn: int | None
    start: int
    audio: np.ndarray


@dataclass(kw_only=True)
class MotionSession:
    decoder: MotionDecoder
    emit: ChunkEmitter | None = None
    fail: Callable[[BaseException], None] | None = None
    # (timeline sample, float32 samples) of the user's audio not yet in a generated block.
    partner: deque[tuple[int, np.ndarray]] = field(default_factory=deque)
    partner_end: int = 0
    is_input_ended: bool = False
    # The start of the next block to generate; None until the user's audio arrives.
    cursor: int | None = None
    speech: list[Speech] = field(default_factory=list)
    interrupted_turns: set[int | None] = field(default_factory=set)
    history: list[HistoryBlock] = field(default_factory=list)
    error: BaseException | None = None
    # The codes of the style this Session moves like; the stage's when None.
    style: torch.Tensor | None = None
    # Its slot of the batched path's caches, from its first block on.
    cache: SessionCache | None = None


@dataclass(frozen=True, kw_only=True)
class BlockWork:
    session_identity: SessionIdentity
    session: MotionSession
    start: int
    turn: int | None
    target_audio: np.ndarray
    partner_audio: np.ndarray


class MotionHooks(SessionHooks):
    """Turns each Session's audio into motion blocks on one worker shared by the stage's Sessions.

    The user's audio is ELLSA 2's partner stream and the reply audio its
    target stream. A reply starts on a block boundary, so a client that
    holds a reply's audio until the first motion chunk carrying its turn
    plays both together; a block is ready once the user's audio covers it,
    or, after the input ended, while it holds reply audio.
    The worker generates one ready block per Session in turn; every block
    runs, in order, since each conditions on the codes before it. With
    kv_batch it takes the ready block of every Session at once (up to
    max_batch) and generates them as one batch, each block's steps decoding
    from a KV cache; without it, one block at a time as the reference does,
    every step a full forward pass.
    """

    def __init__(
        self,
        models: MotionModels,
        speech_sample_rate_hz: int,
        *,
        kv_batch: bool = False,
        max_batch: int = 1,
        cuda_graphs: bool = True,
        reuse_history: bool = True,
    ) -> None:
        self.models = models
        self.max_batch = max_batch if kv_batch else 1
        self.batched = (
            BatchedBlocks(
                models.generator,
                models.filterbank,
                max_batch=max_batch,
                cuda_graphs=cuda_graphs,
                reuse_history=reuse_history,
            )
            if kv_batch
            else None
        )
        divisor = math.gcd(SESSION_INPUT_SAMPLE_RATE_HZ, speech_sample_rate_hz)
        self.resample_up = SESSION_INPUT_SAMPLE_RATE_HZ // divisor
        self.resample_down = speech_sample_rate_hz // divisor
        self.resample_context = RESAMPLE_FILTER_HALF_WIDTH * max(
            self.resample_up, self.resample_down
        )
        self.sessions: dict[SessionIdentity, MotionSession] = {}
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.warmed_up = threading.Event()
        self.worker: threading.Thread | None = None

    def warm_up_serving_thread(self) -> None:
        """Generate one silent block of every history length on the worker before the stage reports ready.

        A thread's first block of each sequence length costs about three
        blocks (cuBLAS and attention set-up on that thread); the worker lives
        as long as the stage, so it pays this once.
        """
        with self.lock:
            self.start_worker(warm_up=True)
        self.warmed_up.wait()

    def start_worker(self, *, warm_up: bool) -> None:
        """Called with the lock held."""
        if self.worker is None:
            self.worker = threading.Thread(
                target=self.generate_blocks,
                args=(warm_up,),
                name="ellsa2-motion",
                daemon=True,
            )
            self.worker.start()
        else:
            pass

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        decoder = self.models.new_decoder()
        with self.lock:
            self.sessions[session_identity] = MotionSession(decoder=decoder)
            self.start_worker(warm_up=False)

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
            self.release(self.sessions.pop(session_identity))
            self.wake.notify_all()

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        with self.lock:
            session = self.sessions[session_identity]
            return ResourceUsage(
                bytes=sum(samples.nbytes for _, samples in session.partner)
                + sum(speech.audio.nbytes for speech in session.speech)
                + sum(
                    block.target_audio.nbytes + block.partner_audio.nbytes
                    for block in session.history
                )
            )

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        with self.lock:
            session = self.sessions[context.session_identity]
            if chunk.format == PCM16 and isinstance(chunk.payload, bytes):
                self.append_partner(session, chunk, chunk.payload)
            elif chunk.format == PCM_F32 and isinstance(chunk.payload, bytes):
                self.append_speech(session, chunk.turn, chunk.payload)
            elif chunk.format == SESSION_CONTROL:
                pass
            else:
                raise ValueError(
                    "the ELLSA 2 motion node takes the user's pcm16 audio, the agent's pcm_f32le reply audio and session controls"
                )
            self.wake.notify_all()
        payload.data = {"accepted": True}
        return payload

    def append_partner(
        self, session: MotionSession, chunk: TimedChunk, pcm: bytes
    ) -> None:
        start = round(chunk.t_start_ms * SESSION_INPUT_SAMPLE_RATE_HZ / 1000)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / PCM16_SCALE
        if samples.size:
            session.partner.append((start, samples))
        else:
            pass
        session.partner_end = max(session.partner_end, start + samples.size)
        if session.cursor is None:
            session.cursor = start // BLOCK_SAMPLES * BLOCK_SAMPLES
        else:
            pass
        if chunk.eos:
            session.is_input_ended = True
        else:
            pass

    def append_speech(
        self, session: MotionSession, turn: int | None, pcm: bytes
    ) -> None:
        """A reply's first audio starts on a block boundary: the block still collecting the user's audio, or the first boundary after the previous reply. Its later audio follows on, unless the next block to generate is already past the reply's end, where it then starts."""
        samples = np.frombuffer(pcm, dtype="<f4")
        last = session.speech[-1] if session.speech else None
        cursor = 0 if session.cursor is None else session.cursor
        if turn in session.interrupted_turns:
            pass
        elif last is not None and last.turn == turn and self.end(last) >= cursor:
            last.audio = np.concatenate([last.audio, samples])
        elif last is not None and last.turn == turn:
            session.speech.append(Speech(turn=turn, start=cursor, audio=samples.copy()))
        else:
            previous_end = 0 if last is None else self.end(last)
            start = max(cursor, -(-previous_end // BLOCK_SAMPLES) * BLOCK_SAMPLES)
            session.speech.append(Speech(turn=turn, start=start, audio=samples.copy()))

    def end(self, speech: Speech) -> int:
        """The timeline sample after a reply's last sample."""
        return speech.start + speech.audio.size * self.resample_up // self.resample_down

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        """An interrupt cuts the replies of its turn, or every reply without one, at the next block to generate, as far as a client that plays reply audio with its motion has played them; their later audio is dropped."""
        if control_type(event) == INTERRUPT:
            with self.lock:
                session = self.sessions[session_identity]
                cut = 0 if session.cursor is None else session.cursor
                for speech in session.speech:
                    if event.turn is None or speech.turn == event.turn:
                        shown = max(cut - speech.start, 0)
                        speech.audio = speech.audio[
                            : shown * self.resample_down // self.resample_up
                        ]
                        session.interrupted_turns.add(speech.turn)
                    else:
                        pass
                session.interrupted_turns.add(event.turn)
                self.wake.notify_all()
        else:
            pass

    def is_ready(self, session: MotionSession) -> bool:
        if session.cursor is None or session.error is not None:
            return False
        elif session.partner_end >= session.cursor + BLOCK_SAMPLES:
            return True
        elif session.is_input_ended:
            return session.cursor < max(
                [session.partner_end] + [self.end(speech) for speech in session.speech]
            )
        else:
            return False

    def next_works(self) -> list[BlockWork]:
        """Take the next block of the first max_batch Sessions with one ready and move those Sessions last; called with the lock held."""
        works = []
        for session_identity, session in list(self.sessions.items()):
            if len(works) == self.max_batch:
                break
            elif self.is_ready(session):
                works.append(self.take_work(session_identity, session))
            else:
                pass
        return works

    def take_work(
        self, session_identity: SessionIdentity, session: MotionSession
    ) -> BlockWork:
        """Called with the lock held."""
        start = session.cursor
        work = BlockWork(
            session_identity=session_identity,
            session=session,
            start=start,
            turn=next(
                (
                    speech.turn
                    for speech in session.speech
                    if speech.start < start + BLOCK_SAMPLES
                    and self.end(speech) > start
                ),
                None,
            ),
            target_audio=self.speech_window(session, start),
            partner_audio=self.partner_window(session, start),
        )
        self.sessions[session_identity] = self.sessions.pop(session_identity)
        session.cursor = start + BLOCK_SAMPLES
        while (
            session.partner
            and session.partner[0][0] + session.partner[0][1].size <= session.cursor
        ):
            session.partner.popleft()
        session.speech = [
            speech
            for speech in session.speech
            if self.end(speech) > session.cursor or speech is session.speech[-1]
        ]
        return work

    def partner_window(self, session: MotionSession, start: int) -> np.ndarray:
        """The user's audio of the block from start, silence where none arrived."""
        window = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
        for begin, samples in session.partner:
            low = max(begin, start)
            high = min(begin + samples.size, start + BLOCK_SAMPLES)
            if high > low:
                window[low - start : high - start] = samples[low - begin : high - begin]
            else:
                pass
        return window

    def speech_window(self, session: MotionSession, start: int) -> np.ndarray:
        """The reply audio of the block from start at 16 kHz, silence where none plays."""
        window = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
        for speech in session.speech:
            low = max(start, speech.start) - speech.start
            high = min(start + BLOCK_SAMPLES, self.end(speech)) - speech.start
            if high > low:
                offset = speech.start + low - start
                window[offset : offset + high - low] = self.resampled(
                    speech.audio, low, high
                )
            else:
                pass
        return window

    def resampled(self, audio: np.ndarray, low: int, high: int) -> np.ndarray:
        """Samples [low, high) at 16 kHz of reply audio at the speech rate, with filter context on both sides, so windows join as one resampling of the whole reply would."""
        if self.resample_up == self.resample_down:
            return audio[low:high]
        else:
            first = (
                max(low - self.resample_context, 0)
                // self.resample_up
                * self.resample_up
            )
        native = audio[
            first
            * self.resample_down
            // self.resample_up : math.ceil(
                (high + self.resample_context) * self.resample_down / self.resample_up
            )
        ]
        return resample_poly(native, self.resample_up, self.resample_down)[
            low - first : high - first
        ].astype(np.float32)

    @torch.no_grad()
    def generate(
        self,
        session: MotionSession,
        target_audio: np.ndarray,
        partner_audio: np.ndarray,
    ) -> np.ndarray:
        """The smplh_flame_motion rows of the Session's next block; runs on the worker, which alone touches a Session's history."""
        models = self.models
        history = retained_history(session.history)
        block = HistoryBlock(target_audio=target_audio, partner_audio=partner_audio)
        blocks = history + [block]
        target = models.filterbank(
            torch.from_numpy(np.concatenate([b.target_audio for b in blocks]))
        )
        partner = models.filterbank(
            torch.from_numpy(np.concatenate([b.partner_audio for b in blocks]))
        )
        block.codes = models.generator.generate_block(
            models.prompt,
            target.unsqueeze(0).to(models.device),
            partner.unsqueeze(0).to(models.device),
            torch.cat([self.style(session)] + [b.codes for b in history]),
            self.style(session).shape[0],
            len(blocks),
        )
        session.history = blocks
        return session.decoder.decode(block.codes)

    @torch.no_grad()
    def generate_together(self, works: list[BlockWork]) -> list[np.ndarray]:
        """The rows of each work's block, the blocks generated as one batch; runs on the worker."""
        assert self.batched is not None
        models = self.models
        in_use = {
            work.session.cache.slot for work in works if work.session.cache is not None
        }
        requests, histories = [], []
        for work in works:
            session = work.session
            if session.cache is None:
                session.cache = self.batched.acquire(in_use)
            else:
                pass
            history = retained_history(session.history)
            blocks = history + [
                HistoryBlock(
                    target_audio=work.target_audio, partner_audio=work.partner_audio
                )
            ]
            histories.append(blocks)
            requests.append(
                BlockRequest(
                    target_audio=[b.target_audio for b in blocks],
                    partner_audio=[b.partner_audio for b in blocks],
                    known_codes=torch.cat(
                        [self.style(session)] + [b.codes for b in history]
                    ),
                    style_steps=self.style(session).shape[0],
                    num_blocks=len(blocks),
                    cache=session.cache,
                )
            )
        codes = self.batched.generate(models.prompt, requests)
        for work, blocks, block_codes in zip(works, histories, codes):
            blocks[-1].codes = block_codes
            work.session.history = blocks
        return MotionDecoder.decode_together(
            [work.session.decoder for work in works], codes
        )

    def release(self, session: MotionSession) -> None:
        """Give back the Session's slot of the batched path's caches; any thread."""
        if self.batched is not None and session.cache is not None:
            self.batched.release(session.cache)
            session.cache = None
        else:
            pass

    def style(self, session: MotionSession) -> torch.Tensor:
        return self.models.style if session.style is None else session.style

    def warm_up(self) -> None:
        silence = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
        if self.batched is None:
            session = MotionSession(decoder=self.models.new_decoder())
            for _ in range(MAX_HISTORY_BLOCKS + 1):
                self.generate(session, silence, silence)
        else:
            # Every history length once, then each batch size's graph at the longest history.
            sessions = [MotionSession(decoder=self.models.new_decoder())]
            for _ in range(MAX_HISTORY_BLOCKS):
                self.generate_together(
                    [self.silent_work(session, silence) for session in sessions]
                )
            for size in GRAPH_BATCH_SIZES:
                if size <= self.max_batch:
                    sessions += [
                        MotionSession(
                            decoder=self.models.new_decoder(),
                            history=list(sessions[0].history),
                        )
                        for _ in range(size - len(sessions))
                    ]
                    self.generate_together(
                        [self.silent_work(session, silence) for session in sessions]
                    )
                else:
                    pass
            for session in sessions:
                self.release(session)

    def silent_work(self, session: MotionSession, silence: np.ndarray) -> BlockWork:
        return BlockWork(
            session_identity=SessionIdentity("warm-up"),
            session=session,
            start=0,
            turn=None,
            target_audio=silence,
            partner_audio=silence,
        )

    def generate_blocks(self, warm_up: bool) -> None:
        """The worker, for the life of the stage."""
        if warm_up:
            self.warm_up()
        else:
            pass
        self.warmed_up.set()
        while True:
            with self.lock:
                works = self.next_works()
                while not works:
                    self.wake.wait()
                    works = self.next_works()
            try:
                if self.batched is None:
                    blocks = [
                        self.generate(
                            work.session, work.target_audio, work.partner_audio
                        )
                        for work in works
                    ]
                else:
                    blocks = self.generate_together(works)
            except Exception as error:
                for work in works:
                    with self.lock:
                        work.session.error = error
                    assert work.session.fail is not None
                    work.session.fail(error)
                continue
            with self.lock:
                for work, rows in zip(works, blocks):
                    if self.sessions.get(work.session_identity) is work.session:
                        assert work.session.emit is not None
                        work.session.emit(
                            TimedChunk(
                                MOTION,
                                work.start * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ,
                                BLOCK_MS,
                                0,
                                rows.tobytes(),
                                format=SMPLH_FLAME_MOTION,
                                turn=work.turn,
                            )
                        )
                    else:
                        pass


def create_scheduler(
    *,
    model_path: str,
    data_path: str,
    style: str,
    speech_sample_rate_hz: int,
    encoder_attn_implementation: str,
    cudnn_attention: bool = False,
    kv_batch: bool = True,
    cuda_graphs: bool = True,
    max_open_sessions: int = DEFAULT_MAX_OPEN_SESSIONS,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SessionScheduler:
    models = load_motion_models(
        model_path=model_path,
        data_path=data_path,
        style=style,
        encoder_attn_implementation=encoder_attn_implementation,
        device=resolve_concrete_device(device, gpu_id),
        cudnn_attention=cudnn_attention,
    )
    return SessionScheduler(
        MotionHooks(
            models,
            speech_sample_rate_hz,
            kv_batch=kv_batch,
            max_batch=max_open_sessions,
            cuda_graphs=cuda_graphs,
        ),
        max_open_sessions=max_open_sessions,
    )


__all__ = ["MotionHooks", "create_scheduler"]
