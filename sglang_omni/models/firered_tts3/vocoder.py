# SPDX-License-Identifier: Apache-2.0
"""Full-payload and streaming RedAE vocoder adapters."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from _thread import LockType
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

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
from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
    DEFAULT_COLD_BATCH_SIZES as GRAPH_DEFAULT_COLD_BATCH_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
    DEFAULT_COLD_LATENT_SIZES as GRAPH_DEFAULT_COLD_LATENT_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
    RedAEVocoderColdGraphRunner,
)
from sglang_omni.models.firered_tts3.vocoder_graph import (
    DEFAULT_BATCH_SIZES as GRAPH_DEFAULT_BATCH_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_graph import (
    DEFAULT_LATENT_SIZES as GRAPH_DEFAULT_LATENT_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_graph import RedAEVocoderGraphRunner
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

SAMPLES_PER_LATENT_FRAME = 960

DEBUG_DUMP_DIR_ENV = "FIREREDTTS3_DEBUG_DUMP_DIR"

WARMUP_FRAMES = 16


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


@dataclass(frozen=True, kw_only=True)
class FireRedCoalescedStep:
    """Unconcatenated latent chunks, read after their readiness waits."""

    latent_parts: tuple[tuple[torch.Tensor, ...], ...]


class FireRedTTS3StreamingVocoder(
    StreamingVocoderBase[FireRedStreamState, FireRedCoalescedStep]
):
    """Incremental decode; unused hook parameters follow StreamingVocoderBase."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        redae_scale: float,
        sample_rate: int,
        max_batch_size: int,
        max_batch_wait_ms: float,
        stream_slots: int,
        prompt_predecode: bool,
        stream_batching: bool,
        separate_decode_stream: bool,
        incremental_cuda_graph: bool,
        cuda_graph_latent_sizes: list[int] | None,
        cuda_graph_batch_sizes: list[int] | None,
        cold_cuda_graph: bool,
        cold_cuda_graph_latent_sizes: list[int] | None,
        cold_cuda_graph_batch_sizes: list[int] | None,
    ) -> None:
        device = next(decoder.parameters()).device
        decode_stream = (
            torch.cuda.Stream(device=device)
            if separate_decode_stream and device.type == "cuda"
            else None
        )
        self.batch: FireRedTTS3BatchVocoder = FireRedTTS3BatchVocoder(
            decoder,
            redae_scale=redae_scale,
            sample_rate=sample_rate,
            decode_stream=decode_stream,
        )
        self.stream_slots: int = int(stream_slots)
        self.prompt_predecode: bool = bool(prompt_predecode)
        self.graph_runner: RedAEVocoderGraphRunner | None = (
            RedAEVocoderGraphRunner(
                decoder,
                latent_sizes=(
                    tuple(cuda_graph_latent_sizes)
                    if cuda_graph_latent_sizes
                    else GRAPH_DEFAULT_LATENT_SIZES
                ),
                batch_sizes=(
                    tuple(cuda_graph_batch_sizes)
                    if cuda_graph_batch_sizes
                    else GRAPH_DEFAULT_BATCH_SIZES
                ),
            )
            if incremental_cuda_graph and device.type == "cuda"
            else None
        )
        self.can_batch_stream_chunks: bool = bool(stream_batching)
        self.cold_graph_runner: RedAEVocoderColdGraphRunner | None = (
            RedAEVocoderColdGraphRunner(
                decoder,
                latent_sizes=(
                    tuple(cold_cuda_graph_latent_sizes)
                    if cold_cuda_graph_latent_sizes
                    else GRAPH_DEFAULT_COLD_LATENT_SIZES
                ),
                batch_sizes=(
                    tuple(cold_cuda_graph_batch_sizes)
                    if cold_cuda_graph_batch_sizes
                    else GRAPH_DEFAULT_COLD_BATCH_SIZES
                ),
            )
            if cold_cuda_graph and device.type == "cuda"
            else None
        )
        if stream_batching:
            self.stream_chunk_batch_distinct_requests: bool = True
            self.stream_chunk_batch_max: int = self.stream_slots
        else:
            pass
        super().__init__(
            self.batch.decode_payload,
            batch_compute_fn=self.batch.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            sample_rate=sample_rate,
            stream_source_hint="fireredtts3",
            stream_input_modality="audio_latents",
        )

    @property
    def device(self) -> torch.device:
        return next(self.batch.decoder.parameters()).device

    def create_stream_state(self, request_id: str) -> FireRedStreamState:
        return FireRedStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: FireRedStreamState,
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
            state.pending_codes_ready_event = ready_event
        else:
            raise TypeError("FireRedTTS3 codes_ready_event must be a CUDA event")
        if not metadata.get(PROMPT_CHUNK_METADATA_KEY):
            return
        else:
            pass
        if state.has_prompt:
            raise RuntimeError(
                f"FireRedTTS3 stream for {request_id!r} carried a second "
                "prompt chunk"
            )
        else:
            pass
        state.has_prompt = True
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
        state.stream_start_sample = left_context_frames * SAMPLES_PER_LATENT_FRAME
        state.is_prompt_chunk = True

    def validate_chunk(
        self, request_id: str, state: FireRedStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] == 0:
            raise ValueError(
                "FireRedTTS3 latent chunks must have shape [1, frames, redae_dim] "
                f"with frames >= 1, got {tuple(codes.shape)}"
            )
        else:
            pass
        if codes.shape[-1] != self.batch.decoder.in_proj.in_features:
            raise ValueError(
                f"FireRedTTS3 latent width must be "
                f"{self.batch.decoder.in_proj.in_features}, got {codes.shape[-1]}"
            )
        else:
            pass
        if codes.device == self.device and codes.dtype == torch.float32:
            converted_latents = codes
        else:
            with self.batch.decode_stream_context():
                event = state.pending_codes_ready_event or state.codes_ready_event
                if event is not None and codes.is_cuda:
                    torch.cuda.current_stream(self.device).wait_event(event)
                else:
                    pass
                converted_latents = codes.to(device=self.device, dtype=torch.float32)
        decode_stream = self.batch.decode_stream
        if decode_stream is not None and converted_latents.is_cuda:
            # note (kermit): Producer allocations must survive the decode stream's reads.
            converted_latents.record_stream(decode_stream)
        else:
            pass
        return converted_latents

    def ingest(
        self, request_id: str, state: FireRedStreamState, codes: torch.Tensor
    ) -> None:
        if state.pending_codes_ready_event is not None:
            state.codes_ready_event = state.pending_codes_ready_event
            state.pending_codes_ready_event = None
        else:
            pass
        if state.is_prompt_chunk:
            state.is_prompt_chunk = False
            if self.prompt_predecode:
                self.decode_prompt(state, codes)
            else:
                state.prompt_latents = codes
            return
        else:
            pass
        state.pending_latents.append(codes)

    def decode_prompt(self, state: FireRedStreamState, prompt: torch.Tensor) -> None:
        """Warm decoder state without emitting the prompt waveform."""
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            self.wait_codes_ready(state)
            probe = ProfilingProbe.create()
            with probe.segment("prompt_predecode"):
                waveform, decoder_state = self.batch.decoder.forward_stream(
                    prompt / self.batch.redae_scale, None, last_chunk=False
                )
            probe.log("prompt-predecode")
        state.decoder_state = decoder_state
        state.emitted_samples += int(waveform.shape[-1])

    def should_decode(self, state: FireRedStreamState, *, is_final: bool) -> bool:
        return bool(state.pending_latents)

    def wait_codes_ready(self, state: FireRedStreamState) -> None:
        """Order this thread's stream after the talker's newest chunk."""
        if state.codes_ready_event is None:
            return
        else:
            pass
        torch.cuda.current_stream(self.device).wait_event(state.codes_ready_event)
        state.codes_ready_event = None

    def emit_cpu_window(
        self, state: FireRedStreamState, waveform: torch.Tensor
    ) -> torch.Tensor | None:
        """Trim the waveform and read back on its producing stream."""
        start_sample = state.stream_start_sample - state.emitted_samples
        state.emitted_samples += int(waveform.shape[-1])
        if start_sample >= int(waveform.shape[-1]) and start_sample > 0:
            return None
        else:
            return waveform[:, max(start_sample, 0) :].cpu()

    def complete_chunk(
        self, state: FireRedStreamState, waveform: torch.Tensor
    ) -> torch.Tensor | None:
        state.prompt_latents = None
        state.pending_latents.clear()
        return self.emit_cpu_window(state, waveform)

    def decode_chunk(
        self,
        latents: torch.Tensor,
        decoder_state: RedAEDecoderState | None,
        *,
        last_chunk: bool,
        lead_samples: int = 0,
    ) -> tuple[torch.Tensor, RedAEDecoderState]:
        """Decode one chunk with an eligible CUDA graph or eager execution."""
        if not last_chunk:
            runner = self.graph_runner
            if runner is not None and decoder_state is not None:
                result = runner.run(latents, decoder_state)
                if result is not None:
                    return result
                else:
                    pass
            else:
                pass
            cold = self.cold_graph_runner
            if cold is not None and decoder_state is None:
                result = cold.run(latents, lead_samples=lead_samples)
                if result is not None:
                    return result
                else:
                    pass
            else:
                pass
        else:
            pass
        return self.batch.decoder.forward_stream(
            latents, decoder_state, last_chunk=last_chunk
        )

    def decode_delta(
        self, request_id: str, state: FireRedStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            self.wait_codes_ready(state)
            latent_parts = self.latent_parts(state)
            if not latent_parts:
                if (
                    not is_final
                    or state.decoder_state is None
                    or state.decoder_state.istft_cache is None
                ):
                    return None
                else:
                    pass
                tail = self.batch.decoder.flush_stream(state.decoder_state)
                state.decoder_state = None
                return self.emit_cpu_window(state, tail)
            else:
                pass
            latents = torch.cat(latent_parts, dim=1)
            probe = ProfilingProbe.create()
            with probe.segment("stream_decode"):
                waveform, decoder_state = self.decode_chunk(
                    latents / self.batch.redae_scale,
                    state.decoder_state,
                    last_chunk=is_final,
                    lead_samples=state.stream_start_sample,
                )
            probe.log("stream-decode")
            state.decoder_state = None if is_final else decoder_state
            return self.complete_chunk(state, waveform)

    def latent_parts(self, state: FireRedStreamState) -> list[torch.Tensor]:
        """Buffered prompt frames followed by generated latent chunks."""
        latent_parts: list[torch.Tensor] = []
        if state.prompt_latents is not None:
            latent_parts.append(state.prompt_latents)
        else:
            pass
        latent_parts.extend(state.pending_latents)
        return latent_parts

    def step_group_key(self, state: FireRedStreamState) -> tuple[int, bool, int]:
        """Group rows by latent frame count, cache presence, and decoded progress."""
        frame_count = sum(int(chunk.shape[1]) for chunk in state.pending_latents)
        if state.prompt_latents is not None:
            frame_count += int(state.prompt_latents.shape[1])
        else:
            pass
        decoder_state = state.decoder_state
        if decoder_state is None:
            return frame_count, False, 0
        else:
            pass
        return frame_count, True, int(decoder_state.frames_seen)

    def select_step_participants(
        self,
    ) -> list[tuple[str, FireRedStreamState]]:
        groups: dict[tuple[int, bool, int], list[tuple[str, FireRedStreamState]]] = {}
        for request_id, state in self.stream_state_items():
            if state.pending_latents:
                groups.setdefault(self.step_group_key(state), []).append(
                    (request_id, state)
                )
            else:
                pass
        if not groups:
            return []
        else:
            pass
        return max(groups.values(), key=len)[: self.stream_chunk_batch_max]

    def build_step_plan(
        self, participants: list[tuple[str, FireRedStreamState]]
    ) -> FireRedCoalescedStep:
        latent_parts: list[tuple[torch.Tensor, ...]] = []
        for _request_id, state in participants:
            row_parts = self.latent_parts(state)
            assert row_parts
            latent_parts.append(tuple(row_parts))
        first_key = self.step_group_key(participants[0][1])
        for _request_id, state in participants[1:]:
            assert self.step_group_key(state) == first_key
        return FireRedCoalescedStep(latent_parts=tuple(latent_parts))

    def run_step(
        self,
        participants: list[tuple[str, FireRedStreamState]],
        plan: FireRedCoalescedStep,
    ) -> dict[str, torch.Tensor]:
        states = [state for _request_id, state in participants]
        probes = [ProfilingProbe.create() for _ in participants]
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            for state in states:
                self.wait_codes_ready(state)
            # note (kermit): Concatenation reads producer tensors, so it follows the readiness waits.
            latents = [torch.cat(row_parts, dim=1) for row_parts in plan.latent_parts]
            if len(participants) == 1:
                state = states[0]
                with probes[0].segment("stream_decode"):
                    waveform, decoder_state = self.decode_chunk(
                        latents[0] / self.batch.redae_scale,
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
                    merged = self.batch.decoder.merge_states(merge_inputs)
                else:
                    pass
                with ProfilingProbe.shared_segment("stream_decode", probes):
                    waveform, merged_state = self.decode_chunk(
                        torch.cat(latents, dim=0) / self.batch.redae_scale,
                        merged,
                        last_chunk=False,
                        lead_samples=min(
                            stream_state.stream_start_sample for stream_state in states
                        ),
                    )
                split_states = self.batch.decoder.split_state(
                    merged_state, len(participants)
                )
                for state, split_row in zip(states, split_states):
                    state.decoder_state = split_row
                waveform_rows = [waveform[row : row + 1] for row in range(len(states))]
            for (request_id, _state), probe in zip(participants, probes):
                probe.log(f"stream-decode {request_id}")
            decoded: dict[str, torch.Tensor] = {}
            for (request_id, state), waveform in zip(participants, waveform_rows):
                waveform = self.complete_chunk(state, waveform)
                if waveform is not None:
                    decoded[request_id] = waveform
                else:
                    pass
        return decoded

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: FireRedStreamState
    ) -> VocoderResultMetadata:
        tts_state = load_firered_tts3_state(payload)
        self.maybe_dump_stream_latents(request_id, tts_state)
        return vocoder_result_metadata(tts_state, self.sample_rate)

    @staticmethod
    def maybe_dump_stream_latents(request_id: str, state: FireRedTTS3State) -> None:
        """Save the complete working-domain latent sequence for diagnostics."""
        dump_dir = os.environ.get(DEBUG_DUMP_DIR_ENV)
        if not dump_dir:
            return
        else:
            pass
        latent_parts = [
            tensor
            for tensor in (state.prompt_latents, state.generated_latents)
            if tensor is not None and tensor.numel()
        ]
        if not latent_parts:
            return
        else:
            pass
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"latents_stream_{request_id}.pt")
        torch.save(torch.cat(latent_parts, dim=1).detach().cpu(), path)

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: FireRedStreamState
    ) -> torch.Tensor | None:
        tts_state, latents = self.batch.prepare_item(payload)
        return self.batch.decode_item(tts_state, latents)

    def release_stream_resources(
        self, request_id: str, state: FireRedStreamState
    ) -> None:
        state.decoder_state = None
        state.pending_latents.clear()
        state.prompt_latents = None

    def warmup_now(self) -> None:
        """Warm fresh-cache, cached, terminal, and graph decode paths."""
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            dummy = torch.zeros(
                1,
                WARMUP_FRAMES,
                self.batch.decoder.in_proj.in_features,
                device=self.device,
                dtype=torch.float32,
            )
            _, cache = self.batch.decoder.forward_stream(dummy, None, last_chunk=False)
            self.batch.decoder.forward_stream(dummy, cache, last_chunk=True)
            self.batch.decoder.flush_stream(cache)
            if self.graph_runner is not None:
                self.graph_runner.capture_all()
            else:
                pass
            if self.cold_graph_runner is not None:
                self.cold_graph_runner.capture_all()
            else:
                pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        logger.info("FireRedTTS3 streaming vocoder warmup complete")


__all__ = [
    "FireRedCoalescedStep",
    "FireRedTTS3BatchVocoder",
    "FireRedTTS3StreamingVocoder",
]
