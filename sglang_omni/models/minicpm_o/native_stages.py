# SPDX-License-Identifier: Apache-2.0
"""Model computations executed by the shared session stage scheduler."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import JsonValue
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    MiniCPMOPerceptionState,
    ProcessorFactory,
)
from sglang_omni.models.minicpm_o.components.tts_runtime import MiniCPMOVocoderRuntime
from sglang_omni.models.minicpm_o.engine_builder import MiniCPMOThinkerEngineBuilder
from sglang_omni.models.minicpm_o.stages import (
    create_sglang_talker_executor_from_config,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.utils.device import resolve_concrete_device


class PerceptionHooks(SessionHooks):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        processor_factory: ProcessorFactory,
        audio_encoder: MiniCPMOAudioEncoder,
        reference_audio: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor_factory = processor_factory
        self.audio_encoder = audio_encoder
        self.reference_audio = reference_audio
        self.states: dict[SessionIdentity, MiniCPMOPerceptionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = MiniCPMOPerceptionState.open(
            tokenizer=self.tokenizer,
            processor=self.processor_factory(),
            audio_encoder=self.audio_encoder,
            prompt=request.params.get("instructions", ""),
            reference_audio=self.reference_audio,
        )

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.eos and chunk.duration_ms == 0:
            payload.data = None
        else:
            state = self.states[context.session_identity]
            pcm = np.frombuffer(chunk.payload, dtype="<i2").astype(np.float32) / 32768.0
            payload.data = state.build_step_plan(state.encode_audio(pcm))
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity).close()

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.states[session_identity].held()


@dataclass
class SpeechState:
    session_id: str
    clock_ms: float = 0


class SpeechHooks(SessionHooks):
    def __init__(self, runtime: MiniCPMOVocoderRuntime, prompt_wav: str) -> None:
        self.runtime, self.prompt_wav = runtime, prompt_wav
        self.states: dict[SessionIdentity, SpeechState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.runtime.open_session(session_identity.id, prompt_wav=self.prompt_wav)
        self.states[session_identity] = SpeechState(session_identity.id)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        data = payload.data
        pcm = b""
        duration_ms = 0
        pairs = data["pairs"]
        end = data["end_of_turn"] or chunk.eos
        if end or (not data["is_listen"] and pairs):
            audio = self.runtime.synthesize(
                state.session_id,
                data["codec_tokens"],
                turn_start=data["speech_turn_start"],
                end_of_turn=end,
            )
            if audio is not None:
                samples = np.asarray(audio, dtype=np.float32).reshape(-1)
                pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()
                duration_ms = len(samples) / 24
            else:
                pass
        else:
            pass
        context.emit(
            TimedChunk(
                "voice",
                state.clock_ms,
                duration_ms,
                chunk.seq,
                dict(
                    text=data["text"],
                    pcm=pcm,
                    end_of_turn=end,
                    is_listen=(
                        None
                        if chunk.eos and chunk.duration_ms == 0
                        else data["is_listen"]
                    ),
                    model_end_of_turn=data["end_of_turn"],
                    prefill_schema=data.get("prefill_schema", []),
                ),
                eos=chunk.eos,
            )
        )
        state.clock_ms += duration_ms
        payload.data = None
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        state = self.states.pop(session_identity)
        self.runtime.close_session(state.session_id)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.runtime.held(self.states[session_identity].session_id)


def create_perception_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    reference_audio: str | None = None,
    **kwargs: JsonValue,
) -> SessionScheduler:
    """Build perception; extra factory options follow the stage loader contract."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoder = MiniCPMOAudioEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
    )
    hooks = PerceptionHooks(
        tokenizer,
        lambda: AutoProcessor.from_pretrained(model_path, trust_remote_code=True),
        encoder,
        reference_audio=reference_audio
        or str(Path(resolve_model_path(model_path)) / "assets" / "HT_ref_audio.wav"),
    )
    return SessionScheduler(hooks, max_open_sessions=2, max_concurrency=1)


def create_thinker_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, JsonValue] | None = None,
) -> OmniScheduler:
    return MiniCPMOThinkerEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_speech_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    reference_audio: str | None = None,
    **kwargs: JsonValue,
) -> SessionScheduler:
    """Build speech; dtype and extra options follow the stage loader contract."""
    device = str(resolve_concrete_device(device, gpu_id))
    codec = MiniCPMOCode2Wav(model_path, device=device, prompt_wav=reference_audio)
    runtime = MiniCPMOVocoderRuntime(codec.token2wav)
    return SessionScheduler(
        SpeechHooks(runtime, codec.default_prompt_wav),
        max_open_sessions=2,
        max_concurrency=1,
        # note (Junnan Li): Each speaker retains both reference and working flow caches.
        max_state_bytes=4 << 30,
    )


def create_talker_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: dict[str, JsonValue] | None = None,
) -> OmniScheduler:
    return create_sglang_talker_executor_from_config(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
        session_mode=True,
    )
