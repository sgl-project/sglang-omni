# SPDX-License-Identifier: Apache-2.0
"""Advance the D2 session by one trained interaction interval per append."""

from dataclasses import replace

from sglang_omni.models.qwen3_omni_d2.config import (
    INPUT_SAMPLE_RATE_HZ,
    MAX_SESSIONS,
    NATIVE_FRAME_MS,
)
from sglang_omni.models.qwen3_omni_d2.runtime import D2Runtime
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import AUDIO, PCM16, PCM16_BYTES_PER_SAMPLE
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.utils.hf import qwen_d2_interval_ms


class D2SessionHooks(SessionHooks):
    def __init__(self, runtime: D2Runtime, interval_ms: int) -> None:
        self.runtime = runtime
        self.frames_per_interval = interval_ms // NATIVE_FRAME_MS
        self.input_bytes_per_interval = (
            interval_ms * INPUT_SAMPLE_RATE_HZ * PCM16_BYTES_PER_SAMPLE // 1000
        )

    def warm_up_serving_thread(self) -> None:
        """Compile the fixed system prompt and first interval before admitting a caller."""
        try:
            self.runtime.open_session()
            try:
                self.runtime.step(bytes(self.input_bytes_per_interval))
            finally:
                self.runtime.close_session()
        except BaseException:
            self.runtime.shutdown()
            raise

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.runtime.open_session()

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if (
            chunk.modality != AUDIO
            or chunk.format != PCM16
            or not isinstance(chunk.payload, bytes)
        ):
            raise ValueError("Qwen3-Omni D2 expects mono PCM16 audio")
        elif not chunk.payload and chunk.eos:
            context.emit(replace(chunk, payload=None, duration_ms=0))
        elif len(chunk.payload) != self.input_bytes_per_interval:
            raise ValueError(
                f"Qwen3-Omni D2 expects {self.input_bytes_per_interval} input bytes per interval"
            )
        else:
            frames = self.runtime.step(chunk.payload)
            assert len(frames) == self.frames_per_interval
            for index, frame in enumerate(frames):
                context.emit(
                    TimedChunk(
                        modality=AUDIO,
                        t_start_ms=chunk.t_start_ms + index * NATIVE_FRAME_MS,
                        duration_ms=NATIVE_FRAME_MS,
                        seq=frame.sequence,
                        payload=frame.model_dump(),
                        format="d2_frame",
                        eos=chunk.eos and index == len(frames) - 1,
                        turn=chunk.turn,
                    )
                )
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.runtime.close_session()


class D2SessionScheduler(SessionScheduler):
    def __init__(self, hooks: D2SessionHooks) -> None:
        super().__init__(hooks, max_open_sessions=MAX_SESSIONS, max_concurrency=1)
        self.runtime = hooks.runtime

    def start(self) -> None:
        try:
            super().start()
        finally:
            self.runtime.shutdown()


def create_scheduler(
    model_path: str,
    *,
    runtime_python: str,
    kv_budget: int,
    seed: int,
    device: str | None = None,
    gpu_id: int | None = None,
) -> D2SessionScheduler:
    release_directory = resolve_checkpoint(model_path)
    interval_ms = qwen_d2_interval_ms(release_directory)
    if interval_ms is None:
        raise ValueError("Expected a Qwen3-Omni D2 release with a d2.json manifest")
    else:
        runtime = D2Runtime(
            model_path=release_directory,
            device=str(resolve_concrete_device(device, gpu_id)),
            runtime_python=runtime_python,
            kv_budget=kv_budget,
            seed=seed,
        )
        return D2SessionScheduler(D2SessionHooks(runtime, interval_ms))
