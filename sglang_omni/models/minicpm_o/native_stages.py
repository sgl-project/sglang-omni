# SPDX-License-Identifier: Apache-2.0
"""Model computations executed by the shared session stage scheduler."""

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import JsonValue
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    AudioBatchKey,
    MiniCPMOAudioEncoder,
    StreamingAudioChunk,
)
from sglang_omni.models.minicpm_o.components.code2wav import (
    OUTPUT_SAMPLE_RATE,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    SAMPLE_RATE,
    UNIT_SAMPLES,
    LogMelFilterBank,
    MiniCPMOPerceptionState,
    StreamingAudioProcessor,
    audio_feature_batch,
)
from sglang_omni.models.minicpm_o.components.tts_runtime import MiniCPMOVocoderRuntime
from sglang_omni.models.minicpm_o.config import CODE2WAV_DECODE_STREAM_PRIORITY
from sglang_omni.models.minicpm_o.engine_builder import MiniCPMOThinkerEngineBuilder
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.cache_key import hash_bytes
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.session import (
    BatchedSessionHooks,
    SessionAppend,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

# note (Junnan Li): A session's first unit and every later unit differ in mel window and encoder history, so warm-up runs one of each.
WARM_UP_UNITS = 2


@dataclass(frozen=True, kw_only=True)
class PendingMelUnit:
    payload: StagePayload
    state: MiniCPMOPerceptionState
    mel_window: np.ndarray
    image_embeds: tuple[torch.Tensor, ...]


@dataclass(frozen=True, kw_only=True)
class PendingAudioUnit:
    payload: StagePayload
    state: MiniCPMOPerceptionState
    audio: StreamingAudioChunk
    image_embeds: tuple[torch.Tensor, ...]


class PerceptionHooks(BatchedSessionHooks):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        processor: StreamingAudioProcessor,
        audio_encoder: MiniCPMOAudioEncoder,
        reference_audio: bytes,
        image_encoder: MiniCPMOImageEncoder,
        mel_filter_bank: LogMelFilterBank,
        reference_cache_capacity: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.audio_encoder = audio_encoder
        self.reference_audio = reference_audio
        self.image_encoder = image_encoder
        self.mel_filter_bank = mel_filter_bank
        self.reference_cache_capacity = reference_cache_capacity
        self.reference_embeds_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.states: dict[SessionIdentity, MiniCPMOPerceptionState] = {}

    def reference_embeds(self, reference_audio: bytes) -> torch.Tensor:
        """The thinker embeddings of one reference audio, encoded once per distinct reference."""
        reference_key = hash_bytes(reference_audio)
        if reference_key in self.reference_embeds_cache:
            self.reference_embeds_cache.move_to_end(reference_key)
        else:
            waveform, _ = AudioMediaIO(target_sr=SAMPLE_RATE).load_bytes(
                reference_audio
            )
            batch = audio_feature_batch(
                self.processor.process_audio(
                    np.asarray(waveform, dtype=np.float32).reshape(-1),
                    sampling_rate=SAMPLE_RATE,
                )
            )
            self.reference_embeds_cache[reference_key] = self.audio_encoder(
                audio_features=batch.audio_features,
                audio_feature_lens=batch.audio_feature_lens,
            )["audio_embeds"]
            # note (Junnan Li): Open sessions hold their own reference to the embeddings, so eviction does not affect them.
            if len(self.reference_embeds_cache) > self.reference_cache_capacity:
                self.reference_embeds_cache.popitem(last=False)
            else:
                pass
        return self.reference_embeds_cache[reference_key]

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = MiniCPMOPerceptionState.open(
            tokenizer=self.tokenizer,
            processor=self.processor,
            audio_encoder=self.audio_encoder,
            prompt=request.params["instructions"],
            reference_embeds=self.reference_embeds(
                request.params.get("reference_audio") or self.reference_audio
            ),
            image_encoder=self.image_encoder,
            max_slice_nums=request.params["max_slice_nums"],
            mel_filter_bank=self.mel_filter_bank,
        )

    def append_batch(self, appends: list[SessionAppend]) -> list[StagePayload]:
        """Prepare each session's unit, then compute the mel and encode the audio of equal-shaped chunks together."""
        pending_mel_units: list[PendingMelUnit] = []
        for append in appends:
            chunk, payload = append.chunk, append.payload
            if chunk.eos and chunk.duration_ms == 0:
                payload.data = None
            else:
                state = self.states[append.context.session_identity]
                if isinstance(chunk.payload, dict):
                    pcm, encoded_images = chunk.payload["pcm"], chunk.payload["images"]
                else:
                    pcm, encoded_images = chunk.payload, ()
                # note (Junnan Li): Frames are acked before decoding, so a bad frame is dropped, not fatal.
                image_embeds = []
                for encoded_image in encoded_images:
                    try:
                        image_embeds.append(state.encode_image(encoded_image))
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        logger.warning(
                            f"Dropping undecodable frame of unit {chunk.seq}: {exc}"
                        )
                waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                pending_mel_units.append(
                    PendingMelUnit(
                        payload=payload,
                        state=state,
                        mel_window=state.prepare_audio(waveform),
                        image_embeds=tuple(image_embeds),
                    )
                )
        windows: dict[int, list[PendingMelUnit]] = defaultdict(list)
        for unit in pending_mel_units:
            windows[unit.mel_window.size].append(unit)
        pending_units: list[PendingAudioUnit] = []
        for window_units in windows.values():
            log_mel = self.mel_filter_bank.log_mel(
                torch.from_numpy(np.stack([unit.mel_window for unit in window_units]))
            )
            for unit, window_log_mel in zip(window_units, log_mel, strict=True):
                pending_units.append(
                    PendingAudioUnit(
                        payload=unit.payload,
                        state=unit.state,
                        audio=unit.state.mel_chunk(window_log_mel),
                        image_embeds=unit.image_embeds,
                    )
                )
        groups: dict[AudioBatchKey, list[PendingAudioUnit]] = defaultdict(list)
        for unit in pending_units:
            groups[unit.audio.batch_key()].append(unit)
        for group in groups.values():
            encoded = self.audio_encoder.forward_streaming_batch(
                [unit.audio for unit in group]
            )
            for unit, (audio_embeds, audio_encoder_state) in zip(
                group, encoded, strict=True
            ):
                unit.state.finish_audio(audio_encoder_state)
                unit.payload.data = unit.state.build_step_plan(
                    audio_embeds, unit.image_embeds
                )
        return [append.payload for append in appends]

    def warm_up(self, max_batch_size: int) -> None:
        """Run the unit path at every batch size up to max_batch_size, so first-call work happens before serving.

        Covers a first unit, a later unit, and a later unit after the position reset; keeps no session state.
        """
        self.reference_embeds(self.reference_audio)
        silence = np.zeros(UNIT_SAMPLES, dtype=np.float32)
        for batch_size in range(1, max_batch_size + 1):
            states = [
                MiniCPMOPerceptionState(
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    audio_encoder=self.audio_encoder,
                    image_encoder=self.image_encoder,
                    max_slice_nums=1,
                    mel_filter_bank=self.mel_filter_bank,
                )
                for _ in range(batch_size)
            ]
            for _ in range(WARM_UP_UNITS):
                log_mel = self.mel_filter_bank.log_mel(
                    torch.from_numpy(
                        np.stack([state.prepare_audio(silence) for state in states])
                    )
                )
                chunks = [
                    state.mel_chunk(window_log_mel)
                    for state, window_log_mel in zip(states, log_mel, strict=True)
                ]
                for state, (_, audio_encoder_state) in zip(
                    states,
                    self.audio_encoder.forward_streaming_batch(chunks),
                    strict=True,
                ):
                    state.finish_audio(audio_encoder_state)
            # note (Junnan Li): The forward took the histories, so the same chunks now run without one.
            self.audio_encoder.forward_streaming_batch(chunks)

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity).close()

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.states[session_identity].held()


@dataclass
class SpeechState:
    session_id: str
    clock_ms: float = 0


class SpeechHooks(SessionHooks):
    def __init__(self, runtime: MiniCPMOVocoderRuntime, reference_audio: bytes) -> None:
        self.runtime, self.reference_audio = runtime, reference_audio
        self.states: dict[SessionIdentity, SpeechState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.runtime.open_session(
            session_identity.id,
            reference_audio=request.params.get("tts_reference_audio")
            or request.params.get("reference_audio")
            or self.reference_audio,
        )
        self.states[session_identity] = SpeechState(session_identity.id)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        talker_result = payload.data
        pcm = b""
        duration_ms = 0
        is_turn_end = talker_result["end_of_turn"] or chunk.eos
        if is_turn_end or (
            not talker_result["is_listen"] and talker_result["talker_conditions"]
        ):
            waveform = self.runtime.synthesize(
                state.session_id,
                talker_result["codec_tokens"],
                is_turn_start=talker_result["speech_turn_start"],
                end_of_turn=is_turn_end,
            )
            if waveform is not None:
                samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
                pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()
                duration_ms = len(samples) * 1000 / OUTPUT_SAMPLE_RATE
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
                    text=talker_result["text"],
                    pcm=pcm,
                    end_of_turn=is_turn_end,
                    is_listen=(
                        None
                        if chunk.eos and chunk.duration_ms == 0
                        else talker_result["is_listen"]
                    ),
                    model_end_of_turn=talker_result["end_of_turn"],
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
    max_open_sessions: int = DEFAULT_MAX_SESSIONS,
) -> SessionScheduler:
    device = str(resolve_concrete_device(device, gpu_id))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoder = MiniCPMOAudioEncoder(model_path, device=device, dtype=dtype)
    image_encoder = MiniCPMOImageEncoder(model_path, device=device, dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    hooks = PerceptionHooks(
        tokenizer,
        processor,
        encoder,
        reference_audio=Path(
            reference_audio
            or Path(resolve_model_path(model_path)) / "assets" / "HT_ref_audio.wav"
        ).read_bytes(),
        image_encoder=image_encoder,
        mel_filter_bank=LogMelFilterBank.from_feature_extractor(
            processor.audio_processor
        ),
        reference_cache_capacity=max_open_sessions,
    )
    # note (Junnan Li): The scheduler hands one call every ready session, so batches go up to the session limit.
    hooks.warm_up(max_open_sessions)
    return SessionScheduler(
        hooks,
        max_open_sessions=max_open_sessions,
    )


def create_thinker_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, JsonValue] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    return MiniCPMOThinkerEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )


def create_speech_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    reference_audio: str | None = None,
    max_open_sessions: int = DEFAULT_MAX_SESSIONS,
    max_state_bytes_per_session: int = DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
) -> SessionScheduler:
    device = str(resolve_concrete_device(device, gpu_id))
    # note (Junnan Li): Sessions stream one reference each, so the batched-offline options stay off.
    codec = MiniCPMOCode2Wav(
        model_path,
        device=device,
        prompt_wav=reference_audio,
        enable_flow_variable_length=False,
        reference_workers=1,
        prompt_cache_capacity=max_open_sessions,
        decode_stream_priority=CODE2WAV_DECODE_STREAM_PRIORITY,
        enable_flow_block_compile=False,
    )
    runtime = MiniCPMOVocoderRuntime(codec)
    return SessionScheduler(
        SpeechHooks(runtime, Path(codec.default_prompt_wav).read_bytes()),
        max_open_sessions=max_open_sessions,
        max_concurrency=1,
        max_state_bytes_per_session=max_state_bytes_per_session,
    )
