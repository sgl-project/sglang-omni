# SPDX-License-Identifier: Apache-2.0
"""Full-payload RedAE vocoder and shared audio result metadata."""

from __future__ import annotations

import contextlib
import os
import threading
from _thread import LockType
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict

import torch

from sglang_omni.models.firered_tts3.payload_types import (
    LEFT_CONTEXT_METADATA_KEY,
    PROMPT_CHUNK_METADATA_KEY,
    FireRedTTS3State,
    load_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    RedAEDecoderState,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

SAMPLES_PER_LATENT_FRAME = 960

DEBUG_DUMP_DIR_ENV = "FIREREDTTS3_DEBUG_DUMP_DIR"


class VocoderResultMetadata(TypedDict, total=False):
    modality: Literal["audio"]
    sample_rate: int
    finish_reason: str | None
    text: str
    usage: dict[str, int | float]


def vocoder_result_metadata(
    state: FireRedTTS3State, sample_rate: int
) -> VocoderResultMetadata:
    metadata: VocoderResultMetadata = {
        "modality": "audio",
        "sample_rate": sample_rate,
        "finish_reason": state.finish_reason,
    }
    if state.gen_text:
        metadata["text"] = state.gen_text
    else:
        pass
    usage = build_usage(state)
    if usage is not None:
        metadata["usage"] = usage
    else:
        pass
    return metadata


class FireRedTTS3BatchVocoder(BatchVocoderBase[FireRedTTS3State, torch.Tensor]):
    """Full-payload decoding through the shared vocoder scheduler."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        redae_scale: float,
        sample_rate: int,
        decode_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.decoder: RedAEAudioDecoder = decoder
        self.redae_scale: float = float(redae_scale)
        self.sample_rate: int = int(sample_rate)
        self.decode_stream: torch.cuda.Stream | None = decode_stream
        self.lock: LockType = threading.Lock()
        self.debug_dump_count: int = 0

    def decode_stream_context(
        self,
    ) -> contextlib.AbstractContextManager[None]:
        if self.decode_stream is None:
            return contextlib.nullcontext()
        else:
            pass
        return torch.cuda.stream(self.decode_stream)

    def maybe_dump_latents(self, latents: torch.Tensor) -> None:
        dump_dir = os.environ.get(DEBUG_DUMP_DIR_ENV)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"latents_{self.debug_dump_count}.pt")
            torch.save(latents.detach().cpu(), path)
            self.debug_dump_count += 1
        else:
            pass

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FireRedTTS3State, torch.Tensor]:
        state = load_firered_tts3_state(payload)
        if state.generated_latents is None or state.generated_latents.numel() == 0:
            raise RuntimeError("FireRedTTS3 vocoder received no generated latents")
        else:
            pass
        latent_parts: list[torch.Tensor] = []
        if state.prompt_latents is not None and state.prompt_latents.numel():
            latent_parts.append(state.prompt_latents)
        else:
            pass
        latent_parts.append(state.generated_latents)
        return state, torch.cat(latent_parts, dim=1)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a full latent sequence and read back on its decode stream."""
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "FireRedTTS3 latents must have shape [1, frames, redae_dim], "
                f"got {tuple(latents.shape)}"
            )
        else:
            pass
        if latents.shape[2] != self.decoder.in_proj.in_features:
            raise ValueError(
                f"FireRedTTS3 latent width must be {self.decoder.in_proj.in_features}, "
                f"got {latents.shape[2]}"
            )
        else:
            pass
        self.maybe_dump_latents(latents)
        with self.lock, torch.no_grad(), self.decode_stream_context():
            probe = ProfilingProbe.create()
            with probe.segment("redae_decode"):
                waveform, _ = self.decoder.forward_stream(
                    (latents / self.redae_scale).to(
                        device=next(self.decoder.parameters()).device,
                        dtype=torch.float32,
                    ),
                    None,
                    last_chunk=True,
                )
            probe.log("batch-vocoder")
            return waveform.cpu()

    async def decode_batch(
        self, items: list[tuple[FireRedTTS3State, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int]] = []
        for state, latents in items:
            outputs.append((self.decode_item(state, latents), self.sample_rate))
        return outputs

    def decode_item(
        self, state: FireRedTTS3State, latents: torch.Tensor
    ) -> torch.Tensor:
        waveform = self.decode_latents(latents)
        prompt_samples = int(state.left_context_size) * SAMPLES_PER_LATENT_FRAME
        if prompt_samples > 0:
            return waveform[:, prompt_samples:]
        else:
            return waveform

    def store_result(
        self,
        payload: StagePayload,
        state: FireRedTTS3State,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="fireredtts3",
        )
        payload.data.update(vocoder_result_metadata(state, sample_rate))
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass(kw_only=True)
class FireRedStreamState:
    """Per-request decoder state and buffered latent chunks."""

    decoder_state: RedAEDecoderState | None = None
    pending_latents: list[torch.Tensor] = field(default_factory=list)
    prompt_latents: torch.Tensor | None = None
    has_prompt: bool = False
    stream_start_sample: int = 0
    emitted_samples: int = 0
    codes_ready_event: torch.cuda.Event | None = None
    pending_codes_ready_event: torch.cuda.Event | None = None
    is_prompt_chunk: bool = False

    def latch_contract(
        self,
        request_id: str,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            return
        else:
            pass
        assert isinstance(source, Mapping)
        metadata = source
        ready_event = metadata.get("codes_ready_event")
        if ready_event is None or isinstance(ready_event, torch.cuda.Event):
            self.pending_codes_ready_event = ready_event
        else:
            raise TypeError("FireRedTTS3 codes_ready_event must be a CUDA event")
        if not metadata.get(PROMPT_CHUNK_METADATA_KEY):
            return
        else:
            pass
        if self.has_prompt:
            raise RuntimeError(
                f"FireRedTTS3 stream for {request_id!r} carried a second "
                "prompt chunk"
            )
        else:
            pass
        self.has_prompt = True
        left_context_value = metadata.get(LEFT_CONTEXT_METADATA_KEY)
        if left_context_value is None:
            left_context_frames = 0
        elif isinstance(left_context_value, (int, float, str)):
            left_context_frames = int(left_context_value or 0)
        else:
            raise TypeError("FireRedTTS3 left_context_size must be a frame count")
        if left_context_frames <= 0:
            raise ValueError(
                f"FireRedTTS3 prompt chunk for {request_id!r} must carry a "
                "positive left_context_size"
            )
        else:
            pass
        self.stream_start_sample = left_context_frames * SAMPLES_PER_LATENT_FRAME
        self.is_prompt_chunk = True

    def wait_codes_ready(self, device: torch.device) -> None:
        """Order this thread's stream after the talker's newest chunk."""
        if self.codes_ready_event is None:
            return
        else:
            pass
        torch.cuda.current_stream(device).wait_event(self.codes_ready_event)
        self.codes_ready_event = None

    def emit_cpu_window(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """Trim the waveform and read back on its producing stream."""
        start_sample = self.stream_start_sample - self.emitted_samples
        self.emitted_samples += int(waveform.shape[-1])
        if start_sample >= int(waveform.shape[-1]) and start_sample > 0:
            return None
        else:
            return waveform[:, max(start_sample, 0) :].cpu()

    def latent_parts(self) -> list[torch.Tensor]:
        """Buffered prompt frames followed by generated latent chunks."""
        latent_parts: list[torch.Tensor] = []
        if self.prompt_latents is not None:
            latent_parts.append(self.prompt_latents)
        else:
            pass
        latent_parts.extend(self.pending_latents)
        return latent_parts

    def step_group_key(self) -> tuple[int, bool, int]:
        """Group rows by latent frame count, cache presence, and decoded progress."""
        frame_count = sum(int(chunk.shape[1]) for chunk in self.pending_latents)
        if self.prompt_latents is not None:
            frame_count += int(self.prompt_latents.shape[1])
        else:
            pass
        decoder_state = self.decoder_state
        if decoder_state is None:
            return frame_count, False, 0
        else:
            pass
        return frame_count, True, int(decoder_state.frames_seen)


class StreamBatchVocoder(Protocol):
    batch: FireRedTTS3BatchVocoder

    @property
    def device(self) -> torch.device: ...

    def wait_codes_ready(self, state: FireRedStreamState) -> None: ...

    def complete_chunk(
        self, state: FireRedStreamState, waveform: torch.Tensor
    ) -> torch.Tensor | None: ...

    def decode_chunk(
        self,
        latents: torch.Tensor,
        decoder_state: RedAEDecoderState | None,
        *,
        last_chunk: bool,
        lead_samples: int = 0,
    ) -> tuple[torch.Tensor, RedAEDecoderState]: ...


def decode_stream_batch(
    vocoder: StreamBatchVocoder,
    participants: list[tuple[str, FireRedStreamState]],
    latent_parts: tuple[tuple[torch.Tensor, ...], ...],
) -> dict[str, torch.Tensor]:
    batch = vocoder.batch
    states = [state for _request_id, state in participants]
    probes = [ProfilingProbe.create() for _ in participants]
    with batch.lock, torch.no_grad(), batch.decode_stream_context():
        for state in states:
            vocoder.wait_codes_ready(state)
        # note (kermit): Concatenation reads producer tensors, so it follows the readiness waits.
        latents = [torch.cat(row_parts, dim=1) for row_parts in latent_parts]
        if len(participants) == 1:
            state = states[0]
            with probes[0].segment("stream_decode"):
                waveform, decoder_state = vocoder.decode_chunk(
                    latents[0] / batch.redae_scale,
                    state.decoder_state,
                    last_chunk=False,
                    lead_samples=state.stream_start_sample,
                )
            state.decoder_state = decoder_state
            waveform_rows = [waveform]
        else:
            merged = None
            if states[0].decoder_state is not None:
                merge_inputs: list[RedAEDecoderState] = []
                for state in states:
                    assert state.decoder_state is not None
                    merge_inputs.append(state.decoder_state)
                merged = batch.decoder.merge_states(merge_inputs)
            else:
                pass
            with ProfilingProbe.shared_segment("stream_decode", probes):
                waveform, merged_state = vocoder.decode_chunk(
                    torch.cat(latents, dim=0) / batch.redae_scale,
                    merged,
                    last_chunk=False,
                    lead_samples=min(
                        stream_state.stream_start_sample for stream_state in states
                    ),
                )
            split_states = batch.decoder.split_state(merged_state, len(participants))
            for state, split_row in zip(states, split_states):
                state.decoder_state = split_row
            waveform_rows = [waveform[row : row + 1] for row in range(len(states))]
        for (request_id, _state), probe in zip(participants, probes):
            probe.log(f"stream-decode {request_id}")
        decoded: dict[str, torch.Tensor] = {}
        for (request_id, state), waveform in zip(participants, waveform_rows):
            waveform = vocoder.complete_chunk(state, waveform)
            if waveform is not None:
                decoded[request_id] = waveform
            else:
                pass
    return decoded
