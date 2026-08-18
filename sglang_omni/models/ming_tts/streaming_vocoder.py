# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduling for Ming-Omni-TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    MingAudioDecoderState,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.payload_types import load_ming_tts_state
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase


@dataclass
class _StreamState:
    decoder_state: MingAudioDecoderState = field(default_factory=MingAudioDecoderState)
    expected_chunk_id: int = 0
    pending_patches: list[torch.Tensor] = field(default_factory=list)
    terminal_received: bool = False
    initial_group_consumed: bool = False
    emitted_samples: int = 0


class MingTTSStreamingVocoderScheduler(StreamingVocoderBase[_StreamState, None]):
    """Decode Ming acoustic latents with request-local AudioVAE state."""

    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        patch_size: int,
        latent_dim: int,
        initial_chunk_patches: int,
        steady_chunk_patches: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        keep_latents: bool = False,
    ) -> None:
        self._decoder = decoder
        self._patch_size = int(patch_size)
        self._latent_dim = int(latent_dim)
        self._initial_chunk_patches = int(initial_chunk_patches)
        self._steady_chunk_patches = int(steady_chunk_patches)
        super().__init__(
            partial(
                decode_ming_tts_audio_payload,
                decoder=decoder,
                keep_latents=bool(keep_latents),
            ),
            sample_rate=decoder.sample_rate,
            stream_source_hint="Ming-Omni-TTS",
            stream_input_modality="audio_latents",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> _StreamState:
        del request_id
        return _StreamState()

    def _ingest_stream_item(
        self,
        request_id: str,
        item: StreamItem,
    ) -> _StreamState | None:
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return None
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} must include "
                "metadata"
            )
        if item.chunk_id != state.expected_chunk_id:
            raise ValueError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} has "
                f"chunk_id={item.chunk_id}, expected {state.expected_chunk_id}"
            )
        if state.terminal_received:
            raise ValueError(
                f"Ming-Omni-TTS stream chunk arrived after the terminal patch "
                f"for {request_id!r}"
            )
        is_last = metadata.get("is_last")
        if not isinstance(is_last, bool):
            raise TypeError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} must include "
                "boolean metadata['is_last']"
            )
        super()._ingest_stream_item(request_id, item)
        state.expected_chunk_id += 1
        if is_last:
            state.terminal_received = True
        return state

    def validate_chunk(
        self,
        request_id: str,
        state: _StreamState,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if latents.device.type != "cpu":
            raise ValueError(
                "Ming-Omni-TTS stream latent must be on CPU, "
                f"got device {latents.device}"
            )
        if latents.dtype != torch.float32:
            raise TypeError(
                "Ming-Omni-TTS stream latent dtype must be torch.float32, "
                f"got {latents.dtype}"
            )
        expected_shape = (self._patch_size, self._latent_dim)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Ming-Omni-TTS stream latent shape must be {expected_shape}, "
                f"got {tuple(latents.shape)}"
            )
        return latents.contiguous()

    def ingest(
        self,
        request_id: str,
        state: _StreamState,
        latents: torch.Tensor,
    ) -> None:
        del request_id
        state.pending_patches.append(latents)

    def should_decode(self, state: _StreamState, *, is_final: bool) -> bool:
        del is_final
        if state.terminal_received:
            return True
        return len(state.pending_patches) >= self._next_chunk_patches(state)

    def _next_chunk_patches(self, state: _StreamState) -> int:
        if state.initial_group_consumed:
            return self._steady_chunk_patches
        return self._initial_chunk_patches

    def decode_delta(
        self,
        request_id: str,
        state: _StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if is_final:
            if not state.terminal_received:
                raise RuntimeError(
                    f"Ming-Omni-TTS stream {request_id!r} ended without a decoded "
                    "terminal latent patch"
                )
            if state.emitted_samples <= 0:
                raise RuntimeError(
                    f"Ming-Omni-TTS stream {request_id!r} completed without audio"
                )
            return None

        terminal = state.terminal_received
        if terminal:
            patch_count = len(state.pending_patches)
        else:
            patch_count = self._next_chunk_patches(state)
        latent_sequence = torch.cat(
            state.pending_patches[:patch_count],
            dim=0,
        )
        waveform = self._decoder.decode_streaming_step(
            latent_sequence,
            state=state.decoder_state,
            is_last=terminal,
        )

        del state.pending_patches[:patch_count]
        if not state.initial_group_consumed and not terminal:
            state.initial_group_consumed = True
        if waveform.numel() == 0:
            return None
        state.emitted_samples += int(waveform.numel())
        return waveform

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _StreamState,
    ) -> dict[str, Any]:
        del request_id
        final_state = load_ming_tts_state(payload)
        final_state.sample_rate = int(self._decoder.sample_rate)
        final_state.duration_s = float(
            state.emitted_samples / int(self._decoder.sample_rate)
        )
        data = final_state.to_dict()
        data["modality"] = "audio"
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data


__all__ = ["MingTTSStreamingVocoderScheduler"]
