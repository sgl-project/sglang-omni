# SPDX-License-Identifier: Apache-2.0
"""Turns of a Qwen session: utterances in, one reply unit per turn out, interruption and truncation forwarded."""

from collections import deque
from dataclasses import dataclass, field

from sglang_omni.audio.segmentation import (
    EMPTY_UTTERANCE,
    FINISH_UTTERANCE,
    OPEN_UTTERANCE,
    UtteranceSegmenter,
)
from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.thinker_session import TRUNCATE_FORMAT
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONTROL,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter, SessionContext, SessionHooks

INPUT_BYTES_PER_SECOND = 32000


@dataclass(kw_only=True)
class PendingUtterance:
    turn: int
    pcm: bytes
    frames: tuple[bytes, ...]


@dataclass(kw_only=True)
class QwenInputState:
    segmenter: UtteranceSegmenter = field(default_factory=UtteranceSegmenter)
    utterance: bytearray | None = None
    utterance_start_ms: float | None = None
    frames: deque[tuple[float, bytes]] = field(default_factory=deque)
    waiting: deque[PendingUtterance] = field(default_factory=deque)
    turn_count: int = 0
    answering: PendingUtterance | None = None
    # Note (Junnan Li): Utterances interrupted before any of their reply reached the client; the next unit asks them again.
    carried: list[PendingUtterance] = field(default_factory=list)
    # The reply a truncate refers to: the latest one whose turn.started the model node published.
    reply_turn: int | None = None
    input_end_ms: float = 0
    emit: ChunkEmitter | None = None


class QwenInputHooks(SessionHooks):
    def __init__(
        self,
        *,
        max_utterance_seconds: float,
        max_queued_turns: int,
        max_utterance_frames: int,
    ) -> None:
        self.max_utterance_bytes: int = int(
            max_utterance_seconds * INPUT_BYTES_PER_SECOND
        )
        self.max_queued_turns: int = max_queued_turns
        self.max_utterance_frames: int = max_utterance_frames
        self.states: dict[SessionIdentity, QwenInputState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = QwenInputState()

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        self.states[session_identity].emit = emit

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states[session_identity]
        held = [*state.waiting, *state.carried]
        if state.answering is not None:
            held.append(state.answering)
        else:
            pass
        return ResourceUsage(
            bytes=len(state.utterance or b"")
            + sum(len(frame) for _, frame in state.frames)
            + sum(len(turn.pcm) + sum(map(len, turn.frames)) for turn in held)
        )

    def emit_control(self, state: QwenInputState, event: str, turn: int) -> None:
        state.emit(
            TimedChunk(
                CONTROL,
                state.input_end_ms,
                0,
                0,
                {"type": event},
                format=SESSION_CONTROL,
                turn=turn,
            )
        )

    def drop_unreachable_frames(self, state: QwenInputState) -> None:
        """Keep the frames an utterance can still take: from the open one's start, else from the oldest audio the segmenter holds."""
        if state.utterance is not None and state.utterance_start_ms is not None:
            horizon_ms = state.utterance_start_ms
        elif state.segmenter.pending_audio:
            horizon_ms = state.segmenter.pending_audio[0].t_start_ms
        else:
            horizon_ms = state.input_end_ms
        while state.frames and state.frames[0][0] < horizon_ms:
            state.frames.popleft()

    def start_next(self, state: QwenInputState) -> None:
        if state.answering is not None or not state.waiting:
            return
        else:
            utterance = state.waiting.popleft()
        state.answering = utterance
        asked = [*state.carried, utterance]
        state.carried = []
        state.emit(
            TimedChunk(
                CONTROL,
                state.input_end_ms,
                0,
                0,
                {
                    "pcm": b"".join(turn.pcm for turn in asked),
                    "images": [frame for turn in asked for frame in turn.frames],
                    "carried_turns": [turn.turn for turn in asked[:-1]],
                },
                format=UNIT_FORMAT,
                turn=utterance.turn,
            )
        )

    def interrupt(self, state: QwenInputState) -> None:
        utterance = state.answering
        assert utterance is not None
        self.emit_control(state, INTERRUPT, utterance.turn)
        self.emit_control(state, TURN_INTERRUPTED, utterance.turn)
        if state.reply_turn != utterance.turn:
            state.carried.append(utterance)
        else:
            pass
        state.answering = None
        self.start_next(state)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        state = self.states[session_identity]
        if (
            control_type(event) == INTERRUPT
            and state.answering is not None
            and event.turn in (None, state.answering.turn)
        ):
            self.interrupt(state)
        else:
            pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        event_type = control_type(chunk)
        if event_type == TURN_DONE:
            if state.answering is not None and chunk.turn == state.answering.turn:
                state.answering = None
                self.start_next(state)
            else:
                pass
            return payload
        elif event_type == TURN_STARTED:
            state.reply_turn = chunk.turn
            return payload
        elif event_type == TRUNCATE:
            if state.reply_turn is not None:
                # Note (Junnan Li): The truncate names its reply in the payload; a chunk of an interrupted turn would not pass the model node's edge.
                state.emit(
                    TimedChunk(
                        CONTROL,
                        state.input_end_ms,
                        0,
                        0,
                        {
                            "turn": state.reply_turn,
                            "t_ms": float(chunk.payload["t_ms"]),
                        },
                        format=TRUNCATE_FORMAT,
                    )
                )
            else:
                pass
            return payload
        elif event_type == SPEECH_STARTED and state.answering is not None:
            self.interrupt(state)
        else:
            pass
        if chunk.format == IMAGE_FRAME:
            if not isinstance(chunk.payload, bytes):
                raise ValueError("camera frames must be encoded image bytes")
            else:
                state.frames.append((chunk.t_start_ms, chunk.payload))
                self.drop_unreachable_frames(state)
            return payload
        elif chunk.modality == AUDIO:
            if (
                chunk.format != PCM16
                or not isinstance(chunk.payload, bytes)
                or len(chunk.payload) % 2
            ):
                raise ValueError("audio must be 16 kHz mono pcm16 bytes")
            else:
                state.input_end_ms = chunk.t_start_ms + chunk.duration_ms
                self.drop_unreachable_frames(state)
        elif event_type not in (COMMIT, SPEECH_STARTED, SPEECH_STOPPED):
            return payload
        else:
            pass
        actions = state.segmenter.step(chunk)
        if chunk.eos:
            actions.extend(state.segmenter.finish())
        else:
            pass
        for action in actions:
            if action == OPEN_UTTERANCE:
                state.utterance = bytearray()
                state.utterance_start_ms = None
            elif action == EMPTY_UTTERANCE:
                state.turn_count += 1
                self.emit_control(state, TURN_QUEUED, state.turn_count)
                self.emit_control(state, TURN_EMPTY, state.turn_count)
            elif action == FINISH_UTTERANCE:
                if len(state.waiting) >= self.max_queued_turns:
                    raise ValueError("queued utterance limit exhausted")
                else:
                    pass
                state.turn_count += 1
                frames = [
                    frame
                    for time_ms, frame in state.frames
                    if state.utterance_start_ms is not None
                    and time_ms >= state.utterance_start_ms
                ]
                state.waiting.append(
                    PendingUtterance(
                        turn=state.turn_count,
                        pcm=bytes(state.utterance),
                        frames=tuple(
                            frames[max(len(frames) - self.max_utterance_frames, 0) :]
                        ),
                    )
                )
                state.utterance = None
                state.frames.clear()
                self.emit_control(state, TURN_QUEUED, state.turn_count)
                self.start_next(state)
            else:
                if (
                    len(state.utterance) + len(action.payload)
                    > self.max_utterance_bytes
                ):
                    raise ValueError("utterance exceeds max_utterance_seconds")
                else:
                    state.utterance.extend(action.payload)
                if state.utterance_start_ms is None:
                    state.utterance_start_ms = action.t_start_ms
                else:
                    pass
        return payload
