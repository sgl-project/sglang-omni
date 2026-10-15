# SPDX-License-Identifier: Apache-2.0
"""Streams the agent's Mimi codes into audio as they are generated.

Mimi's decoder is causal and its attention ring holds the widest ramp chunk, so
chunked decodes land on the samples a whole-sequence decode would; a chunk ramp
emits the first chunk on arrival and aggregates later frames into wider chunks,
amortizing per-decode launch and PCM transfer overhead. Streams whose ready
chunks share a width decode together, one row each, in a single batched step.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import torch

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CODEBOOKS_PER_STREAM,
    MIMI,
    SAMPLE_RATE,
)
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as emit_event
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

SOURCE_HINT = "PersonaPlex"


def trim_to_caller(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Cut the reply back to the caller recording's own length.

    Frames are whole 80 ms, so a recording that is not a multiple of one is
    padded before encoding; the reference trims that padding off the reply.
    """
    if num_samples and waveform.shape[-1] > num_samples:
        return waveform[..., :num_samples]
    else:
        pass
    return waveform


class AudioPayloadDecoder(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


@dataclass(kw_only=True)
class StreamState:
    slot_index: int | None = None
    pending_codes: deque[torch.Tensor] = field(default_factory=deque)
    pending_frames: int = 0
    decodes_done: int = 0
    decoded_frames: int = 0
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    emitted_samples: int = 0
    num_samples: int = 0
    stream_done: bool = False
    ready_order: int = 0
    client_streams: bool = False


class PersonaPlexCode2WavScheduler(StreamingVocoderBase[StreamState, int]):
    def __init__(
        self,
        decode_slots: MimiDecodeSlots,
        *,
        compute_fn: AudioPayloadDecoder,
        chunk_ramp_frames: tuple[int, ...] | list[int],
    ) -> None:
        super().__init__(
            compute_fn,
            sample_rate=SAMPLE_RATE,
            stream_source_hint=SOURCE_HINT,
            max_batch_size=decode_slots.max_batch_size,
        )
        self.decode_slots = decode_slots
        self.chunk_ramp_frames = tuple(chunk_ramp_frames)
        assert max(self.chunk_ramp_frames) <= decode_slots.max_chunk_frames
        self.next_ready_order = 0
        self.ingested_frames_to_decode = 0
        self.can_batch_stream_chunks = True
        # note (nightdecode): One decode batch may need several transport frames per row.
        self.stream_chunk_batch_max = self.max_batch_size * max(self.chunk_ramp_frames)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # Note (wilsonzheng0327): The LM streams every frame it produces, so a reply
        # with frames has chunks on the way whichever channel lands first; only an
        # empty reply is rendered whole.
        codes = PersonaPlexState.from_dict(payload.data).codes
        return codes is not None and codes.shape[0] > 0

    def create_stream_state(self, request_id: str) -> StreamState:
        return StreamState()

    def chunk_target_frames(self, state: StreamState) -> int:
        """Frames the next decode should cover; the last ramp entry is the steady width."""
        index = min(state.decodes_done, len(self.chunk_ramp_frames) - 1)
        return self.chunk_ramp_frames[index]

    def ready_frames(self, state: StreamState) -> int:
        target_frames = self.chunk_target_frames(state)
        if state.stream_done:
            return min(target_frames, state.pending_frames)
        elif state.pending_frames >= target_frames:
            return target_frames
        else:
            return 0

    def needs_decode_slot(self, state: StreamState) -> bool:
        return state.slot_index is None and self.ready_frames(state) > 0

    def bind_waiting_streams(self) -> None:
        waiting = sorted(
            (
                state
                for state in self.stream_states.values()
                if self.needs_decode_slot(state)
            ),
            key=lambda state: state.ready_order,
        )
        for state in waiting:
            state.slot_index = self.decode_slots.acquire()
            if state.slot_index is None:
                break
            else:
                pass

    def release_decode_slot(self, state: StreamState) -> None:
        if state.slot_index is not None:
            self.decode_slots.release(state.slot_index)
            state.slot_index = None
            self.bind_waiting_streams()
        else:
            pass

    def release_stream_resources(self, request_id: str, state: StreamState) -> None:
        self.release_decode_slot(state)

    def ingest_stream_item(
        self, request_id: str, item: StreamItem
    ) -> StreamState | None:
        # note (nightdecode): The base class requires a bool stream flag; a chunk
        # without one keeps its audio, as a non-streaming client's would.
        return super().ingest_stream_item(
            request_id,
            replace(item, metadata={"stream": False, **(item.metadata or {})}),
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: StreamState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        if isinstance(source, StagePayload):
            num_samples = PersonaPlexState.from_dict(source.data).num_samples
        else:
            # Note (wilsonzheng0327): The terminal payload only arrives after the LM finishes,
            # so the caller length and the client's stream flag travel with each chunk.
            num_samples = int(source.get("num_samples") or 0)
            state.client_streams = bool(source.get("stream"))
        if num_samples:
            if state.num_samples not in (0, num_samples):
                raise ValueError(
                    f"PersonaPlex num_samples changed for request {request_id!r}"
                )
            else:
                state.num_samples = num_samples
        else:
            pass

    def validate_chunk(
        self, request_id: str, state: StreamState, chunk: torch.Tensor
    ) -> torch.Tensor:
        if (
            chunk.ndim != 2
            or chunk.shape[0] == 0
            or chunk.shape[1] != AUDIO_CODEBOOKS_PER_STREAM
        ):
            raise ValueError(
                f"PersonaPlex codes for {request_id!r} must have shape [frames, 8] with frames > 0"
            )
        elif chunk.min() < 0 or chunk.max() >= MIMI.codebook_size:
            # Note (nightdecode): Reject the request before its codes join a batched step;
            # an out-of-range code there fails every row, and on CUDA the whole process.
            raise ValueError(
                f"PersonaPlex codes for {request_id!r} must lie in [0, {MIMI.codebook_size})"
            )
        else:
            return chunk.to(dtype=torch.long)

    def ingest(self, request_id: str, state: StreamState, chunk: torch.Tensor) -> None:
        was_ready = self.ready_frames(state) > 0
        state.pending_codes.append(chunk)
        state.pending_frames += chunk.shape[0]
        self.ingested_frames_to_decode += chunk.shape[0]
        if not was_ready and self.ready_frames(state):
            self.stamp_ready_order(state)
        else:
            pass
        self.bind_waiting_streams()

    def stamp_ready_order(self, state: StreamState) -> None:
        state.ready_order = self.next_ready_order
        self.next_ready_order += 1

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return bool(self.select_step_participants())

    def select_step_participants(self) -> list[tuple[str, StreamState]]:
        width_groups: dict[int, list[tuple[str, StreamState]]] = {}
        for request_id, state in self.stream_states.items():
            chunk_frames = self.ready_frames(state)
            if state.slot_index is not None and chunk_frames:
                width_groups.setdefault(chunk_frames, []).append((request_id, state))
            else:
                pass
        if not width_groups:
            return []
        else:
            pass
        for group in width_groups.values():
            group.sort(key=lambda entry: entry[1].ready_order)
        return min(width_groups.values(), key=lambda group: group[0][1].ready_order)[
            : self.max_batch_size
        ]

    def build_step_plan(self, participants: list[tuple[str, StreamState]]) -> int:
        return self.ready_frames(participants[0][1])

    def pump_streams(self) -> list[str]:
        # note (nightdecode): Bound message handling by received frames, not transport chunk count.
        try:
            while self.ingested_frames_to_decode > 0:
                failed = self.pump_one_step()
                if failed is None:
                    break
                elif failed:
                    return failed
                else:
                    pass
            return []
        finally:
            self.ingested_frames_to_decode = 0

    def pump_one_step(self) -> list[str] | None:
        failed = super().pump_one_step()
        if failed is not None and not failed:
            # note (nightdecode): Emit the last audio before the result, and free slots before payload arrival.
            for request_id, state in self.stream_state_items():
                if state.stream_done and state.pending_frames == 0:
                    self.release_decode_slot(state)
                    if request_id in self.stream_payloads:
                        self.complete_stream_request(
                            request_id, self.finish_stream(request_id)
                        )
                    else:
                        pass
                else:
                    pass
        else:
            pass
        return failed

    @torch.inference_mode()
    def run_step(
        self, participants: list[tuple[str, StreamState]], plan: int
    ) -> dict[str, torch.Tensor]:
        chunk_frames = plan
        slot_indices = [state.slot_index for _, state in participants]
        assert all(slot_index is not None for slot_index in slot_indices)
        decoded_frames = [state.decoded_frames for _, state in participants]
        code_rows: list[torch.Tensor] = []
        for _, state in participants:
            remaining_frames = chunk_frames
            parts: list[torch.Tensor] = []
            while remaining_frames:
                codes = state.pending_codes.popleft()
                take_frames = min(remaining_frames, codes.shape[0])
                parts.append(codes[:take_frames])
                if take_frames < codes.shape[0]:
                    state.pending_codes.appendleft(codes[take_frames:])
                else:
                    pass
                remaining_frames -= take_frames
            code_rows.append(torch.cat(parts).T)
        batched_codes = torch.stack(code_rows).to(self.decode_slots.codec.device)
        host_waveforms = (
            self.decode_slots.decode_step(batched_codes, slot_indices)[:, 0]
            .float()
            .cpu()
        )
        self.ingested_frames_to_decode = max(
            0, self.ingested_frames_to_decode - len(participants) * chunk_frames
        )
        decoded: dict[str, torch.Tensor] = {}
        for row_index, (request_id, state) in enumerate(participants):
            state.pending_frames -= chunk_frames
            state.decoded_frames += chunk_frames
            state.decodes_done += 1
            if self.ready_frames(state):
                self.stamp_ready_order(state)
            else:
                pass
            waveform = host_waveforms[row_index]
            if state.num_samples:
                waveform = waveform[: max(state.num_samples - state.emitted_samples, 0)]
            else:
                pass
            # note (nightdecode): A saved row must not retain the whole batch's CPU storage.
            waveform = waveform.clone()
            state.emitted_samples += waveform.shape[-1]
            if not state.client_streams:
                state.audio_parts.append(waveform)
            else:
                pass
            decoded[request_id] = waveform
        if get_recorder().is_active():
            emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_chunk_batch",
                metadata={
                    "batch_size": len(participants),
                    "chunk_frames": chunk_frames,
                    "graph_replayed": (len(participants), chunk_frames)
                    in self.decode_slots.graphs,
                    "decoded_frames": decoded_frames,
                    "slot_indices": slot_indices,
                    "participant_request_ids": [
                        request_id for request_id, _ in participants
                    ],
                    "active_slots": self.decode_slots.active_slot_count,
                    "waiting_streams": sum(
                        self.needs_decode_slot(state)
                        for state in self.stream_states.values()
                    ),
                    "inbox_depth": self.inbox.qsize(),
                },
            )
        else:
            pass
        return decoded

    def finish_stream_decode(self, state: StreamState) -> None:
        was_ready = self.ready_frames(state) > 0
        state.stream_done = True
        if not was_ready and self.ready_frames(state):
            self.stamp_ready_order(state)
            self.bind_waiting_streams()
        elif state.pending_frames == 0:
            self.release_decode_slot(state)
        else:
            pass

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is not None:
            self.finish_stream_decode(state)
        else:
            pass
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            self.finish_stream_decode(state)
        if state.pending_frames:
            return None
        else:
            return self.finish_stream(request_id)

    def decode_delta(
        self, request_id: str, state: StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        # note (nightdecode): All audio uses the coalesced pump, including one-row and terminal steps.
        assert is_final and state.pending_frames == 0
        return None

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: StreamState
    ) -> dict[str, bytes | list[int] | str | int]:
        if payload.request.params.get("stream"):
            # Note (wilsonzheng0327): A streaming client already holds every chunk; only a
            # non-streaming client reads the reply from the result.
            return {"modality": "audio", "sample_rate": SAMPLE_RATE}
        else:
            waveform = (
                torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
            )
            return audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint=SOURCE_HINT,
            )


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
