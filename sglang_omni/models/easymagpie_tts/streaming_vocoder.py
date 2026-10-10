# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie codec stage: whole-utterance decode and coalesced streaming."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.easymagpie_tts.codec import EasyMagpieCodec
from sglang_omni.models.easymagpie_tts.codec_graphs import StreamingCodecRunner
from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase

DEFAULT_STARTUP_CHUNK_FRAMES = (2, 6)
DEFAULT_STEADY_CHUNK_FRAMES = 8


@dataclass
class EasyMagpieStreamState:
    pending: list[torch.Tensor] = field(default_factory=list)
    pending_frames: int = 0
    decoded_chunks: int = 0
    codec_slot: int = -1


class EasyMagpieStreamingVocoder(StreamingVocoderBase[EasyMagpieStreamState, int]):
    """Decode streamed [frames, stacked codebooks] rows chunk by chunk.

    Small first chunks cut time to first audio; later chunks are larger so
    each codec launch covers more audio. Streams waiting on the same chunk
    size decode together in one batched codec call, replayed from a CUDA
    graph when ``cuda_graph`` is set and the device is CUDA.
    """

    can_batch_stream_chunks = True
    accepts_stream_chunk_batch = True

    def __init__(
        self,
        codec: EasyMagpieCodec,
        *,
        startup_chunk_frames: Sequence[int] = DEFAULT_STARTUP_CHUNK_FRAMES,
        steady_chunk_frames: int = DEFAULT_STEADY_CHUNK_FRAMES,
        max_batch_size: int = 64,
        max_batch_wait_ms: float = 5,
        max_streams: int = 256,
        cuda_graph: bool = True,
    ) -> None:
        if steady_chunk_frames < 1 or any(f < 1 for f in startup_chunk_frames):
            raise ValueError("EasyMagpie vocoder chunk sizes must be positive")
        else:
            pass
        if max_batch_size < 1:
            raise ValueError("EasyMagpie vocoder max_batch_size must be positive")
        else:
            pass
        self.codec = codec
        self.codec_lock = threading.Lock()
        self.startup_chunk_frames = tuple(int(f) for f in startup_chunk_frames)
        self.steady_chunk_frames = int(steady_chunk_frames)
        self.stream_chunk_batch_max = int(max_batch_size)
        self.runner = StreamingCodecRunner(codec, max_streams=max_streams)
        self.cuda_graph = cuda_graph
        super().__init__(
            self.decode_payload,
            batch_compute_fn=self.decode_payloads,
            sample_rate=codec.config.output_sample_rate,
            stream_source_hint="EasyMagpie",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def decode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        states = [EasyMagpieTTSState.from_dict(p.data) for p in payloads]
        codes = []
        for state in states:
            if state.audio_codes is None or state.audio_codes.numel() == 0:
                raise ValueError("EasyMagpie generated no audio frames")
            else:
                codes.append(torch.as_tensor(state.audio_codes, dtype=torch.long))
        with self.codec_lock:
            audio = self.codec.decode_batch(codes)
        return [
            StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={**self.stream_payload(payload.request_id, wav), **usage(state)},
            )
            for payload, state, wav in zip(payloads, states, audio)
        ]

    def decode_payload(self, payload: StagePayload) -> StagePayload:
        return self.decode_payloads([payload])[0]

    def target_frames(self, state: EasyMagpieStreamState) -> int:
        if state.decoded_chunks < len(self.startup_chunk_frames):
            return self.startup_chunk_frames[state.decoded_chunks]
        else:
            return self.steady_chunk_frames

    def warmup_now(self) -> None:
        if self.cuda_graph:
            frames = {*self.startup_chunk_frames, self.steady_chunk_frames}
            self.runner.capture(sorted(frames), self.stream_chunk_batch_max)
        else:
            pass

    def create_stream_state(self, request_id: str) -> EasyMagpieStreamState:
        del request_id
        return EasyMagpieStreamState(codec_slot=self.runner.acquire())

    def release_stream_resources(
        self, request_id: str, state: EasyMagpieStreamState
    ) -> None:
        del request_id
        if state.codec_slot >= 0:
            self.runner.release(state.codec_slot)
        else:
            pass

    def validate_chunk(
        self, request_id: str, state: EasyMagpieStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        width = self.codec.config.num_stacked_codebooks
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        else:
            pass
        if codes.ndim != 2 or codes.shape[1] != width:
            raise ValueError(
                f"EasyMagpie stream chunks must be [frames, {width}] codes, "
                f"got {tuple(codes.shape)}"
            )
        else:
            return codes.to(torch.long)

    def ingest(
        self, request_id: str, state: EasyMagpieStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        state.pending_frames += int(codes.shape[0])

    def should_decode(self, state: EasyMagpieStreamState, *, is_final: bool) -> bool:
        return state.pending_frames > 0 if is_final else self.is_ready(state)

    def is_ready(self, state: EasyMagpieStreamState) -> bool:
        return state.pending_frames >= self.target_frames(state)

    def decode_delta(
        self, request_id: str, state: EasyMagpieStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done flush of the leftover frames; full chunks go through
        the coalesced ``run_step`` pump."""
        del request_id
        if not is_final or state.pending_frames == 0:
            return None
        else:
            return self.decode_streams([state], state.pending_frames)[0]

    def select_step_participants(self) -> list[tuple[str, EasyMagpieStreamState]]:
        by_frames: dict[int, list[tuple[str, EasyMagpieStreamState]]] = {}
        for entry in self.stream_state_items():
            if self.is_ready(entry[1]):
                by_frames.setdefault(self.target_frames(entry[1]), []).append(entry)
            else:
                pass
        if not by_frames:
            return []
        else:
            pass
        # First chunks set time to first audio, so their group goes first.
        group = max(
            by_frames.values(),
            key=lambda entries: (
                any(state.decoded_chunks == 0 for _, state in entries),
                len(entries),
            ),
        )
        return group[: self.stream_chunk_batch_max]

    def build_step_plan(
        self, participants: list[tuple[str, EasyMagpieStreamState]]
    ) -> int:
        return self.target_frames(participants[0][1])

    def run_step(
        self, participants: list[tuple[str, EasyMagpieStreamState]], plan: int
    ) -> dict[str, torch.Tensor]:
        audio = self.decode_streams([state for _, state in participants], plan)
        return {
            request_id: wav
            for (request_id, _), wav in zip(participants, audio, strict=True)
        }

    def decode_streams(
        self, states: list[EasyMagpieStreamState], frames: int
    ) -> list[torch.Tensor]:
        codes = torch.stack([take_frames(state, frames) for state in states])
        with self.codec_lock:
            audio = self.runner.decode(
                codes,
                [state.codec_slot for state in states],
                [state.decoded_chunks > 0 for state in states],
            )
        for state in states:
            state.decoded_chunks += 1
        return list(audio)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: EasyMagpieStreamState
    ) -> dict[str, Any]:
        del request_id, state
        return {
            "modality": "audio",
            "sample_rate": self.sample_rate,
            **usage(EasyMagpieTTSState.from_dict(payload.data)),
        }


def take_frames(state: EasyMagpieStreamState, frames: int) -> torch.Tensor:
    """Pop the first ``frames`` buffered rows of ``state``."""
    buffered = torch.cat(state.pending) if len(state.pending) > 1 else state.pending[0]
    rest = buffered[frames:]
    state.pending = [rest] if rest.shape[0] else []
    state.pending_frames = int(rest.shape[0])
    return buffered[:frames]


def usage(state: EasyMagpieTTSState) -> dict[str, Any]:
    stats = build_usage(state)
    return {} if stats is None else {"usage": stats}


__all__ = [
    "DEFAULT_STARTUP_CHUNK_FRAMES",
    "DEFAULT_STEADY_CHUNK_FRAMES",
    "EasyMagpieStreamState",
    "EasyMagpieStreamingVocoder",
]
