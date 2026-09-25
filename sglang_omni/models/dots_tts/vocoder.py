# SPDX-License-Identifier: Apache-2.0
"""dots.tts AudioVAE adapters for Omni's shared vocoder schedulers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.models.dots_tts.payload_types import DotsTTSState, load_dots_tts_state
from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

_LENGTH_BUCKET_FRAMES = 32
logger = logging.getLogger(__name__)


class DotsTTSBatchVocoder(BatchVocoderBase[DotsTTSState, torch.Tensor, torch.Tensor]):
    def __init__(self, codec: DotsAudioCodec) -> None:
        self.codec = codec
        self.logged_batch = False

    def prepare_item(self, payload: StagePayload) -> tuple[DotsTTSState, torch.Tensor]:
        state = load_dots_tts_state(payload)
        if state.generated_latents is None:
            raise RuntimeError("dots.tts vocoder received no generated latents")
        else:
            pass
        return state, state.generated_latents

    async def decode_batch(
        self, items: list[tuple[DotsTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int] | None] = [None] * len(items)
        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, (_state, latents) in enumerate(items):
            self.validate_latents(latents)
            frames = int(latents.shape[1])
            bucket = (frames + _LENGTH_BUCKET_FRAMES - 1) // _LENGTH_BUCKET_FRAMES
            buckets.setdefault(bucket, []).append((index, latents))

        bucket_sizes = [len(bucket_items) for bucket_items in buckets.values()]
        if not self.logged_batch and max(bucket_sizes, default=0) > 1:
            logger.info(
                "dots.tts AudioVAE batched decode is active: requests=%d "
                "bucket_sizes=%s",
                len(items),
                bucket_sizes,
            )
            self.logged_batch = True
        else:
            pass

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
                else:
                    pass
                for row, (index, latents) in enumerate(bucket_items):
                    valid_samples = int(latents.shape[1]) * self.codec.hop_size
                    outputs[index] = (
                        waveform_batch[row : row + 1, ..., :valid_samples],
                        self.codec.sample_rate,
                    )

        if any(output is None for output in outputs):
            raise RuntimeError("dots.tts AudioVAE did not produce every batch result")
        else:
            pass
        return [output for output in outputs if output is not None]

    def validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "dots.tts latents must have shape [1, frames, latent_dim], "
                f"got {tuple(latents.shape)}"
            )
        else:
            pass
        if latents.shape[1] == 0:
            raise ValueError("dots.tts latents must contain at least one frame")
        else:
            pass
        if latents.shape[2] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {latents.shape[2]}"
            )
        else:
            pass

    def store_result(
        self,
        payload: StagePayload,
        state: DotsTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        data: dict[str, bytes | list[int] | str | int | dict[str, int | float]] = dict(
            audio_waveform_payload(
                wav,
                sample_rate=sample_rate,
                modality="audio",
                source_hint="dots.tts",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        payload.data = data
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass
class DotsStreamState:
    slot: int | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    received_patches: int = 0


@dataclass(frozen=True)
class DotsCoalescedStepPlan:
    take_patches: int
    slot_latents: dict[int, torch.Tensor]


class DotsTTSStreamingVocoder(
    StreamingVocoderBase[DotsStreamState, DotsCoalescedStepPlan]
):
    """Streaming AudioVAE with slot-pooled equal-T batched eager decode."""

    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        codec: DotsAudioCodec,
        *,
        optimize: bool,
        merge_steps: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
        stream_slots: int = 16,
        slot_pool: DotsVocoderSlotPool | None = None,
    ) -> None:
        if merge_steps < 1:
            raise ValueError("dots.tts vocoder merge_steps must be positive")
        else:
            pass
        if max_batch_size < 1:
            raise ValueError("dots.tts vocoder max_batch_size must be positive")
        else:
            pass
        if max_batch_wait_ms < 0:
            raise ValueError("dots.tts vocoder max_batch_wait_ms must be non-negative")
        else:
            pass
        if stream_slots < 1:
            raise ValueError("dots.tts vocoder stream_slots must be positive")
        else:
            pass
        self.codec = codec
        self.optimize = bool(optimize)
        self.merge_steps = int(merge_steps) if optimize else 1
        self.stream_slots = int(stream_slots)
        self.batch_vocoder = DotsTTSBatchVocoder(codec)
        self.slot_pool = slot_pool
        # note (guozhihao-224): coalesce width follows max_batch_size only;
        # stream_slots is admission capacity and must not redefine the batch cap.
        self.stream_chunk_batch_max = int(max_batch_size)
        super().__init__(
            self.batch_vocoder.decode_payload,
            batch_compute_fn=self.batch_vocoder.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            sample_rate=codec.sample_rate,
            stream_source_hint="dots.tts",
            stream_input_modality="audio_latents",
        )

    def ensure_slot_pool(self) -> DotsVocoderSlotPool:
        """Allocate the slot pool at executor setup (not the first live chunk)."""
        with self.codec.lock:
            return self.get_or_create_pool_locked()

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        _state, latents = self.batch_vocoder.prepare_item(payload)
        self.batch_vocoder.validate_latents(latents)

    def create_stream_state(self, request_id: str) -> DotsStreamState:
        del request_id
        return DotsStreamState()

    def validate_chunk(
        self,
        request_id: str,
        state: DotsStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(
                "dots.tts latent chunks must have shape [1, frames, latent_dim]"
            )
        else:
            pass
        if codes.shape[-1] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {codes.shape[-1]}"
            )
        else:
            pass
        return codes.to(self.codec.device)

    def ingest(
        self, request_id: str, state: DotsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        state.received_patches += 1
        self.ensure_slot(state)

    def should_decode(self, state: DotsStreamState, *, is_final: bool) -> bool:
        # note (guozhihao-224): coalesced serving uses select_step_participants;
        # this gate remains for direct decode_delta / stream-done callers.
        if is_final:
            return bool(state.pending)
        else:
            pass
        return self.pending_ready(state)

    def decode_delta(
        self, request_id: str, state: DotsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done drain: pending -> slot step, flush, release.

        Steady chunks are emitted by the coalesced run_step pump.
        """
        del request_id
        if not is_final:
            return None
        else:
            pass
        chunks: list[torch.Tensor] = []
        with self.codec.lock:
            pool = self.get_or_create_pool_locked()
            if state.slot is None and state.pending:
                state.slot = pool.acquire()
            else:
                pass
            if state.slot is not None and state.pending:
                take = len(state.pending)
                patches, state.pending = state.pending[:take], state.pending[take:]
                # note (db-ol): compiled stream_step cudagraph trees corrupt the
                # backbone decode graph replay in this process, see issue 1392;
                # the slot pool stays on the eager kernels (#1395).
                chunk = pool.step({state.slot: torch.cat(patches, dim=1)})[state.slot]
                if chunk.numel():
                    chunks.append(chunk)
                else:
                    pass
            else:
                pass
            if state.slot is not None:
                tail = pool.flush(state.slot)
                pool.release(state.slot)
                state.slot = None
                if tail.numel():
                    chunks.append(tail)
                else:
                    pass
            else:
                pass
        if not chunks:
            return None
        else:
            pass
        return torch.cat(chunks, dim=-1)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: DotsStreamState
    ) -> dict[str, str | int | dict[str, int | float]]:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        result: dict[str, str | int | dict[str, int | float]] = {
            "modality": "audio",
            "sample_rate": self.codec.sample_rate,
        }
        usage = build_usage(tts_state)
        if usage is not None:
            result["usage"] = usage
        else:
            pass
        return result

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: DotsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        if tts_state.generated_latents is None:
            return None
        else:
            pass
        with self.codec.lock:
            return self.codec.inference.decode_latents(
                tts_state.generated_latents.to(self.codec.device)
            )

    def release_stream_resources(self, request_id: str, state: DotsStreamState) -> None:
        del request_id
        if state.slot is None:
            return
        else:
            pass
        with self.codec.lock:
            self.get_or_create_pool_locked().release(state.slot)
        state.slot = None

    def select_step_participants(self) -> list[tuple[str, DotsStreamState]]:
        slotted = [
            (request_id, state)
            for request_id, state in self.stream_state_items()
            if state.slot is not None and self.pending_ready(state)
        ]
        if not slotted:
            return []
        else:
            pass
        # note (guozhihao-224): exact-T groups only; padding would change AudioVAE
        # convolution boundaries. Cap with max_batch_size so one pool.step does
        # not grow to stream_slots under high concurrency.
        by_frames: dict[int, list[tuple[str, DotsStreamState]]] = {}
        for entry in slotted:
            frames = self.step_frames(entry[1])
            by_frames.setdefault(frames, []).append(entry)
        return max(by_frames.values(), key=len)[: self.stream_chunk_batch_max]

    def build_step_plan(
        self, participants: list[tuple[str, DotsStreamState]]
    ) -> DotsCoalescedStepPlan:
        take_patches = self.take_patches(participants[0][1])
        slot_latents: dict[int, torch.Tensor] = {}
        for _, state in participants:
            if state.slot is None:
                raise RuntimeError("dots.tts coalesced step is missing a vocoder slot")
            else:
                pass
            if self.take_patches(state) != take_patches:
                raise RuntimeError("dots.tts coalesced step mixed unequal patch counts")
            else:
                pass
            patches = state.pending[:take_patches]
            latents = torch.cat(patches, dim=1)
            if int(latents.shape[1]) != take_patches * self.codec.patch_size:
                raise RuntimeError(
                    "dots.tts coalesced step latent frames "
                    f"{int(latents.shape[1])} != "
                    f"{take_patches * self.codec.patch_size}"
                )
            else:
                pass
            slot_latents[state.slot] = latents
        return DotsCoalescedStepPlan(
            take_patches=take_patches,
            slot_latents=slot_latents,
        )

    def run_step(
        self,
        participants: list[tuple[str, DotsStreamState]],
        plan: DotsCoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        with self.codec.lock:
            decoded = self.get_or_create_pool_locked().step(plan.slot_latents)
        out: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            del state.pending[: plan.take_patches]
            if state.slot is None:
                continue
            else:
                pass
            waveform = decoded[state.slot]
            if waveform.numel():
                out[request_id] = waveform
            else:
                pass
        return out

    def get_or_create_pool_locked(self) -> DotsVocoderSlotPool:
        # Caller must hold self.codec.lock (RLock).
        if self.slot_pool is None:
            self.slot_pool = DotsVocoderSlotPool(
                self.codec.inference,
                num_slots=self.stream_slots,
                chunk_size=self.codec.patch_size * self.merge_steps,
            )
            logger.info(
                "dots.tts streaming vocoder slot pool ready: "
                "slots=%d merge_steps=%d chunk_size=%d",
                self.stream_slots,
                self.merge_steps,
                self.codec.patch_size * self.merge_steps,
            )
        else:
            pass
        return self.slot_pool

    def ensure_slot(self, state: DotsStreamState) -> None:
        if state.slot is not None:
            return
        else:
            pass
        with self.codec.lock:
            if state.slot is None:
                state.slot = self.get_or_create_pool_locked().acquire()
            else:
                pass

    def pending_ready(self, state: DotsStreamState) -> bool:
        if not state.pending:
            return False
        else:
            pass
        if state.received_patches <= 2:
            return True
        else:
            pass
        return len(state.pending) >= self.merge_steps

    def take_patches(self, state: DotsStreamState) -> int:
        if state.received_patches <= 2:
            return 1
        else:
            pass
        return min(self.merge_steps, len(state.pending))

    def step_frames(self, state: DotsStreamState) -> int:
        return self.take_patches(state) * self.codec.patch_size


__all__ = ["DotsTTSBatchVocoder", "DotsTTSStreamingVocoder"]
