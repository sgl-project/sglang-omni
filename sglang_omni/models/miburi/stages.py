# SPDX-License-Identifier: Apache-2.0
"""Cross-session gesture batching with independent sampling and motion clocks."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch

from sglang_omni.models.miburi.batching import GestureBatch
from sglang_omni.models.miburi.formats import MOTION, SMPLX_MOTION
from sglang_omni.models.miburi.loading import GestureRuntime, load_gesture_runtime
from sglang_omni.models.miburi.motion import ROOT_START_POSITION_M, smplx_frames
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.sampling.seed import derive_sampling_seed, new_random_sampling_seed
from sglang_omni.scheduling.session import (
    BatchedSessionHooks,
    SessionAppend,
    SessionContext,
    SessionScheduler,
)
from sglang_omni.utils.device import resolve_concrete_device

SEED_NAMESPACE = "miburi"
# The checkpoint demo's idle test: below this L2 norm of the agent's last frames of float audio, upper body and face play the idle slices; while speaking, a slice with fewer unread codes than the rewind count restarts.
IDLE_MAX_AUDIO_NORM = 0.5
IDLE_WINDOW_FRAMES = 3
IDLE_REWIND_CODES = 12


@dataclass(kw_only=True)
class GestureSession:
    slot: int
    root_position_m: torch.Tensor
    audio_energies: deque[float] = field(
        default_factory=lambda: deque(
            [0.0] * IDLE_WINDOW_FRAMES, maxlen=IDLE_WINDOW_FRAMES
        )
    )
    upper_idle_cursor: int = 0
    face_idle_cursor: int = 0
    motion_frames: int = 0


class GestureHooks(BatchedSessionHooks):
    """Batch one frame per ready session while preserving the append contract."""

    def __init__(self, gesture: GestureRuntime, max_open_sessions: int) -> None:
        self.gesture = gesture
        self.sessions: dict[SessionIdentity, GestureSession] = {}
        self.batch = GestureBatch(gesture, max_open_sessions)
        self.idle_upper_length = gesture.idle_upper_codes.shape[-1]
        self.idle_face_length = gesture.idle_face_codes.shape[-1]
        self.frame_ms = 1000.0 / gesture.motion_fps

    def warm_up_serving_thread(self) -> None:
        """Capture every admitted batch size on the serving thread before readiness."""
        self.batch.warm_up()

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        public_seed = request.params.get("seed")
        generator = torch.Generator(device=self.gesture.device)
        generator.manual_seed(
            new_random_sampling_seed()
            if public_seed is None
            else derive_sampling_seed(SEED_NAMESPACE, public_seed, "gesture")
        )
        self.sessions[session_identity] = GestureSession(
            slot=self.batch.open(generator),
            root_position_m=torch.tensor(ROOT_START_POSITION_M),
        )

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        return self.append_batch(
            [SessionAppend(chunk=chunk, payload=payload, context=context)]
        )[0]

    def idle_slices(
        self, session: GestureSession, audio_energy: float
    ) -> tuple[bool, torch.Tensor, torch.Tensor]:
        """Whether the frame plays the idle slices, and the upper and face idle codes [1, codebooks, 1] it plays if so."""
        gesture = self.gesture
        session.audio_energies.append(audio_energy)
        if math.sqrt(sum(session.audio_energies)) <= IDLE_MAX_AUDIO_NORM:
            session.upper_idle_cursor %= self.idle_upper_length
            session.face_idle_cursor %= self.idle_face_length
            upper = gesture.idle_upper_codes[
                :, :, session.upper_idle_cursor : session.upper_idle_cursor + 1
            ]
            face = gesture.idle_face_codes[
                :, :, session.face_idle_cursor : session.face_idle_cursor + 1
            ]
            session.upper_idle_cursor += 1
            session.face_idle_cursor += 1
            return True, upper, face
        else:
            if self.idle_upper_length - session.upper_idle_cursor < IDLE_REWIND_CODES:
                session.upper_idle_cursor = 0
            else:
                pass
            if self.idle_face_length - session.face_idle_cursor < IDLE_REWIND_CODES:
                session.face_idle_cursor = 0
            else:
                pass
            return (
                False,
                gesture.idle_upper_codes[:, :, :1],
                gesture.idle_face_codes[:, :, :1],
            )

    def append_frame(
        self,
        frame_index: int,
        ready: list[SessionAppend],
        sessions: list[GestureSession],
    ) -> None:
        """Step frame frame_index of every ready append together and emit each one's motion chunk."""
        gesture = self.gesture
        conditioning: list[list[int]] = []
        idle_mask: list[bool] = []
        upper_idle: list[torch.Tensor] = []
        face_idle: list[torch.Tensor] = []
        for append, session in zip(ready, sessions, strict=True):
            frames = append.chunk.payload
            conditioning.append(
                [frames["text_ids"][frame_index], *frames["audio_codes"][frame_index]]
            )
            is_idle, upper, face = self.idle_slices(
                session, frames["audio_energy"][frame_index]
            )
            idle_mask.append(is_idle)
            upper_idle.append(upper)
            face_idle.append(face)
        _, upper, lower, face = self.batch.step(
            [session.slot for session in sessions],
            torch.tensor(conditioning, device=gesture.device).unsqueeze(-1),
            torch.tensor(idle_mask, device=gesture.device).view(-1, 1, 1),
            torch.cat(upper_idle),
            torch.cat(face_idle),
        )
        upper, lower, face = upper.cpu(), lower.cpu(), face.cpu()
        for row, (append, session) in enumerate(zip(ready, sessions, strict=True)):
            motion, session.root_position_m = smplx_frames(
                upper[row : row + 1],
                lower[row : row + 1],
                face[row : row + 1],
                gesture.masks,
                session.root_position_m,
                self.frame_ms / 1000.0,
            )
            if not append.context.cancelled.is_set():
                append.context.emit(
                    TimedChunk(
                        MOTION,
                        session.motion_frames * self.frame_ms,
                        motion.shape[0] * self.frame_ms,
                        append.chunk.seq,
                        motion.tobytes(),
                        format=SMPLX_MOTION,
                        eos=append.chunk.eos
                        and frame_index == len(append.chunk.payload["text_ids"]) - 1,
                    )
                )
            else:
                pass
            session.motion_frames += motion.shape[0]

    @torch.no_grad()
    def append_batch(self, appends: list[SessionAppend]) -> list[StagePayload]:
        session_by_identity = {
            append.context.session_identity: self.sessions[
                append.context.session_identity
            ]
            for append in appends
        }
        for frame_index in range(
            max(
                (len(append.chunk.payload["text_ids"]) for append in appends), default=0
            )
        ):
            ready = [
                append
                for append in appends
                if frame_index < len(append.chunk.payload["text_ids"])
                and not append.context.cancelled.is_set()
            ]
            if ready:
                self.append_frame(
                    frame_index,
                    ready,
                    [
                        session_by_identity[append.context.session_identity]
                        for append in ready
                    ],
                )
            else:
                pass
        for append in appends:
            session = session_by_identity[append.context.session_identity]
            if (
                append.chunk.eos
                and not append.chunk.payload["text_ids"]
                and not append.context.cancelled.is_set()
            ):
                append.context.emit(
                    TimedChunk(
                        MOTION,
                        session.motion_frames * self.frame_ms,
                        0.0,
                        append.chunk.seq,
                        b"",
                        format=SMPLX_MOTION,
                        eos=True,
                    )
                )
            else:
                pass
            append.payload.data = {}
        return [append.payload for append in appends]

    def close(self, session_identity: SessionIdentity) -> None:
        self.batch.close(self.sessions.pop(session_identity).slot)


def create_gesture_executor(
    model_path: str,
    *,
    experiment: str,
    speaker_id: int,
    cfg_coef: float,
    temporal_temperature: float,
    temporal_top_p: float,
    depth_temperature: float,
    depth_top_p: float,
    max_open_sessions: int,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SessionScheduler:
    """Share model weights and batch the frames already waiting on the serving worker."""
    gesture = load_gesture_runtime(
        checkpoint=model_path,
        experiment=experiment,
        speaker_id=speaker_id,
        cfg_coef=cfg_coef,
        temporal_temperature=temporal_temperature,
        temporal_top_p=temporal_top_p,
        depth_temperature=depth_temperature,
        depth_top_p=depth_top_p,
        device=resolve_concrete_device(device, gpu_id),
    )
    return SessionScheduler(
        GestureHooks(gesture, max_open_sessions),
        max_open_sessions=max_open_sessions,
        max_concurrency=1,
    )


__all__ = ["GestureHooks", "create_gesture_executor"]
