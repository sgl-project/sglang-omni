# SPDX-License-Identifier: Apache-2.0
"""dots.tts AudioVAE adapters for Omni's shared vocoder schedulers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.models.dots_tts.payload_types import DotsTTSState, load_dots_tts_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

_LENGTH_BUCKET_FRAMES = 32
logger = logging.getLogger(__name__)


class DotsTTSBatchVocoder(BatchVocoderBase):
    def __init__(self, codec: DotsAudioCodec) -> None:
        self.codec = codec
        self._logged_batch = False

    def prepare_item(self, payload: StagePayload) -> tuple[DotsTTSState, torch.Tensor]:
        state = load_dots_tts_state(payload)
        if state.generated_latents is None:
            raise RuntimeError("dots.tts vocoder received no generated latents")
        return state, state.generated_latents

    async def decode_batch(
        self, items: list[tuple[DotsTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int] | None] = [None] * len(items)
        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, (_state, latents) in enumerate(items):
            self._validate_latents(latents)
            frames = int(latents.shape[1])
            bucket = (frames + _LENGTH_BUCKET_FRAMES - 1) // _LENGTH_BUCKET_FRAMES
            buckets.setdefault(bucket, []).append((index, latents))

        bucket_sizes = [len(bucket_items) for bucket_items in buckets.values()]
        if not self._logged_batch and max(bucket_sizes, default=0) > 1:
            logger.info(
                "dots.tts AudioVAE batched decode is active: requests=%d "
                "bucket_sizes=%s",
                len(items),
                bucket_sizes,
            )
            self._logged_batch = True

        with self.codec.lock:
            for bucket_items in buckets.values():
                frame_counts = [int(latents.shape[1]) for _, latents in bucket_items]
                max_frames = max(frame_counts)
                if len(bucket_items) == 1:
                    padded = bucket_items[0][1].to(self.codec.device)
                elif min(frame_counts) == max_frames:
                    padded = torch.cat(
                        [latents for _, latents in bucket_items], dim=0
                    ).to(self.codec.device)
                else:
                    padded = bucket_items[0][1].new_zeros(
                        len(bucket_items),
                        max_frames,
                        self.codec.latent_dim,
                        device=self.codec.device,
                    )
                    for row, ((_index, latents), frames) in enumerate(
                        zip(bucket_items, frame_counts, strict=True)
                    ):
                        padded[row, :frames].copy_(latents[0].to(self.codec.device))

                waveform_batch = self.codec.inference.decode_latents(padded)
                if waveform_batch.shape[0] != len(bucket_items):
                    raise RuntimeError(
                        "dots.tts AudioVAE returned an unexpected batch size: "
                        f"expected {len(bucket_items)}, got {waveform_batch.shape[0]}"
                    )
                for row, (index, latents) in enumerate(bucket_items):
                    valid_samples = int(latents.shape[1]) * self.codec.hop_size
                    outputs[index] = (
                        waveform_batch[row : row + 1, ..., :valid_samples],
                        self.codec.sample_rate,
                    )

        if any(output is None for output in outputs):
            raise RuntimeError("dots.tts AudioVAE did not produce every batch result")
        return [output for output in outputs if output is not None]

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "dots.tts latents must have shape [1, frames, latent_dim], "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[1] == 0:
            raise ValueError("dots.tts latents must contain at least one frame")
        if latents.shape[2] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {latents.shape[2]}"
            )

    def store_result(
        self,
        payload: StagePayload,
        state: DotsTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="dots.tts",
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass
class _DotsStreamState:
    codec_state: Any
    pending: list[torch.Tensor] = field(default_factory=list)
    received_patches: int = 0


class DotsTTSStreamingVocoder(StreamingVocoderBase[_DotsStreamState, None]):
    def __init__(
        self,
        codec: DotsAudioCodec,
        *,
        optimize: bool,
        merge_steps: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if merge_steps < 1:
            raise ValueError("dots.tts vocoder merge_steps must be positive")
        if max_batch_size < 1:
            raise ValueError("dots.tts vocoder max_batch_size must be positive")
        if max_batch_wait_ms < 0:
            raise ValueError("dots.tts vocoder max_batch_wait_ms must be non-negative")
        self.codec = codec
        self.optimize = bool(optimize)
        self.merge_steps = int(merge_steps) if optimize else 1
        self._batch_vocoder = DotsTTSBatchVocoder(codec)
        super().__init__(
            self._batch_vocoder.decode_payload,
            batch_compute_fn=self._batch_vocoder.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            sample_rate=codec.sample_rate,
            stream_source_hint="dots.tts",
            stream_input_modality="audio_latents",
        )

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        _state, latents = self._batch_vocoder.prepare_item(payload)
        self._batch_vocoder._validate_latents(latents)

    def create_stream_state(self, request_id: str) -> _DotsStreamState:
        del request_id
        with self.codec.lock:
            codec_state = self.codec.inference.init_stream_state(
                batch_size=1,
                chunk_size=self.codec.patch_size * self.merge_steps,
            )
        return _DotsStreamState(codec_state=codec_state)

    def validate_chunk(
        self,
        request_id: str,
        state: _DotsStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(
                "dots.tts latent chunks must have shape [1, frames, latent_dim]"
            )
        if codes.shape[-1] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {codes.shape[-1]}"
            )
        return codes.to(self.codec.device)

    def ingest(
        self, request_id: str, state: _DotsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        state.received_patches += 1

    def should_decode(self, state: _DotsStreamState, *, is_final: bool) -> bool:
        if is_final:
            return bool(state.pending)
        if state.received_patches <= 2:
            return bool(state.pending)
        return len(state.pending) >= self.merge_steps

    def decode_delta(
        self, request_id: str, state: _DotsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        chunks: list[torch.Tensor] = []
        if state.pending:
            take = (
                len(state.pending)
                if is_final
                else (1 if state.received_patches <= 2 else self.merge_steps)
            )
            patches, state.pending = state.pending[:take], state.pending[take:]
            latent_chunk = (
                patches[0] if len(patches) == 1 else torch.cat(patches, dim=1)
            )
            # note (db-ol): compiled stream step cudagraph trees corrupt the
            # backbone decode graph replay in this process, see issue 1392.
            with self.codec.lock:
                chunk = self.codec.inference.stream_step(
                    latent_chunk,
                    stream_state=state.codec_state,
                    optimize=self.optimize,
                    use_compiled=False,
                )
            if chunk.numel():
                chunks.append(chunk)
        if is_final:
            with self.codec.lock:
                tail = self.codec.inference.flush(state.codec_state)
            if tail.numel():
                chunks.append(tail)
        if not chunks:
            return None
        return torch.cat(chunks, dim=-1)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _DotsStreamState
    ) -> dict[str, Any]:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        result: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.codec.sample_rate,
        }
        usage = build_usage(tts_state)
        if usage is not None:
            result["usage"] = usage
        return result

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _DotsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        if tts_state.generated_latents is None:
            return None
        with self.codec.lock:
            return self.codec.inference.decode_latents(
                tts_state.generated_latents.to(self.codec.device)
            )


__all__ = ["DotsTTSBatchVocoder", "DotsTTSStreamingVocoder"]
