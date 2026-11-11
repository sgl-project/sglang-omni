# SPDX-License-Identifier: Apache-2.0
"""Streaming RedAE vocoder adapter."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    load_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    RedAEDecoderState,
)
from sglang_omni.models.firered_tts3.vocoder_batch import (
    DEBUG_DUMP_DIR_ENV,
    FireRedStreamState,
    FireRedTTS3BatchVocoder,
    VocoderResultMetadata,
    decode_stream_batch,
    vocoder_result_metadata,
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
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase

logger = logging.getLogger(__name__)

WARMUP_FRAMES = 16


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
        state.latch_contract(request_id, source, origin=origin)

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
        state.wait_codes_ready(self.device)

    def emit_cpu_window(
        self, state: FireRedStreamState, waveform: torch.Tensor
    ) -> torch.Tensor | None:
        return state.emit_cpu_window(waveform)

    def complete_chunk(
        self, state: FireRedStreamState, waveform: torch.Tensor
    ) -> torch.Tensor | None:
        state.prompt_latents = None
        state.pending_latents.clear()
        return self.emit_cpu_window(state, waveform)

    def latent_parts(self, state: FireRedStreamState) -> list[torch.Tensor]:
        return state.latent_parts()

    def step_group_key(self, state: FireRedStreamState) -> tuple[int, bool, int]:
        return state.step_group_key()

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
        return decode_stream_batch(self, participants, plan.latent_parts)

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
