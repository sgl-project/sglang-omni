# SPDX-License-Identifier: Apache-2.0
"""Stream generated Mimi codes through a capacity-bounded decode arena."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.components.mimi import (
    MimiCodec,
    MimiDecodeStateArena,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as emit_event
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
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
class Code2WavStreamState:
    pending_chunks: deque[torch.Tensor] = field(default_factory=deque)
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    emitted_samples: int = 0
    num_samples: int = 0
    slot_index: int | None = None
    stream_done: bool = False
    last_decode_step_index: int = 0


class PersonaPlexCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        codec: MimiCodec,
        *,
        compute_fn: AudioPayloadDecoder,
        max_batch_size: int,
        stream_slots: int,
    ) -> None:
        super().__init__(compute_fn, max_batch_size=max_batch_size)
        self.codec = codec
        self.stream_states: dict[str, Code2WavStreamState] = {}
        self.decode_step_index = 0
        stream_slot_count = max(int(stream_slots), 1)
        self.decode_arena = MimiDecodeStateArena(codec, stream_slot_count)
        self.free_slot_indices = list(reversed(range(stream_slot_count)))
        if self.max_batch_size > 1:
            self.can_batch_stream_chunks = True
            self.stream_chunk_batch_max = self.max_batch_size
        else:
            pass

    @property
    def active_slot_count(self) -> int:
        return self.decode_arena.slot_count - len(self.free_slot_indices)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # Note (wilsonzheng0327): The LM streams every frame it produces, so a reply
        # with frames has chunks on the way whichever channel lands first; only an
        # empty reply is rendered whole.
        codes = PersonaPlexState.from_dict(payload.data).codes
        return codes is not None and codes.shape[0] > 0

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.get_or_create_stream_state(request_id)

    def get_or_create_stream_state(self, request_id: str) -> Code2WavStreamState:
        state = self.stream_states.get(request_id)
        if state is not None:
            return state
        else:
            pass
        state = Code2WavStreamState()
        self.assign_decode_slot(state)
        self.stream_states[request_id] = state
        return state

    def assign_decode_slot(self, state: Code2WavStreamState) -> None:
        if state.slot_index is not None:
            pass
        elif state.stream_done and not state.pending_chunks:
            pass
        elif not self.free_slot_indices:
            raise QueueFullError()
        else:
            slot_index = self.free_slot_indices.pop()
            self.decode_arena.reset_slot(slot_index)
            state.slot_index = slot_index

    def release_decode_slot(self, state: Code2WavStreamState) -> None:
        slot_index = state.slot_index
        assert slot_index is not None
        state.slot_index = None
        self.free_slot_indices.append(slot_index)

    def clear_stream_state(self, request_id: str) -> None:
        state = self.stream_states.pop(request_id, None)
        if state is not None and state.slot_index is not None:
            self.release_decode_slot(state)
        else:
            pass

    def ingest_chunk(
        self, request_id: str, stream_item: StreamItem
    ) -> Code2WavStreamState:
        state = self.get_or_create_stream_state(request_id)
        metadata = stream_item.metadata or {}
        num_samples = int(metadata.get("num_samples") or 0)
        if num_samples:
            if state.num_samples not in (0, num_samples):
                raise ValueError(
                    f"PersonaPlex num_samples changed from {state.num_samples} "
                    f"to {num_samples} for request {request_id!r}"
                )
            else:
                pass
            state.num_samples = num_samples
        else:
            pass
        state.pending_chunks.append(
            torch.as_tensor(
                stream_item.data, dtype=torch.long, device=self.codec.device
            )
        )
        return state

    @torch.inference_mode()
    def on_stream_chunk(
        self, request_id: str, stream_item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self.ingest_chunk(request_id, stream_item)
        assert state.slot_index is not None
        return self.decode_slot_batch([(request_id, state)])

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Ingest chunks, then run one decode step so sustained message
        arrival cannot postpone decode work."""
        failed_request_ids: list[str] = []
        with self.state_lock:
            for request_id, stream_item in items:
                if self.is_aborted(request_id):
                    continue
                else:
                    pass
                try:
                    self.ingest_chunk(request_id, stream_item)
                except Exception as error:
                    self.emit_error(request_id, error)
                    self.abort_state(request_id)
                    failed_request_ids.append(request_id)
        for request_id in failed_request_ids:
            self.cleanup_aborted_request(request_id)
        self.run_ready_step()

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return bool(self.select_step_participants())

    def select_step_participants(
        self,
    ) -> list[tuple[str, Code2WavStreamState]]:
        compatible_groups: dict[
            tuple[int, bool], list[tuple[str, Code2WavStreamState]]
        ] = {}
        for request_id, state in self.stream_states.items():
            if state.slot_index is not None and state.pending_chunks:
                batch_key = self.chunk_batch_key(state)
                compatible_groups.setdefault(batch_key, []).append((request_id, state))
            else:
                pass
        compatible_state_groups = list(compatible_groups.values())
        for compatible_states in compatible_state_groups:
            compatible_states.sort(key=lambda entry: entry[1].last_decode_step_index)
        if not compatible_state_groups:
            return []
        else:
            pass
        selected_group = min(
            compatible_state_groups,
            key=lambda group: group[0][1].last_decode_step_index,
        )
        return selected_group[: self.max_batch_size]

    def run_ready_step(self) -> None:
        failed_request_ids: list[str] = []
        with self.state_lock:
            participants = self.select_step_participants()
            if not participants:
                return
            else:
                pass
            try:
                stream_messages = self.decode_slot_batch(participants)
            except Exception as error:
                for request_id, _ in participants:
                    self.emit_error(request_id, error)
                    self.abort_state(request_id)
                    failed_request_ids.append(request_id)
            else:
                for stream_message in stream_messages:
                    if not self.is_aborted(stream_message.request_id):
                        self.outbox.put(stream_message)
                    else:
                        pass
                for request_id, state in participants:
                    if state.stream_done and not state.pending_chunks:
                        if state.slot_index is not None:
                            self.release_decode_slot(state)
                        else:
                            pass
                        if request_id in self.stream_payloads:
                            self.complete_stream_request(
                                request_id,
                                [self.build_result_message(request_id, state)],
                            )
                        else:
                            pass
                    else:
                        pass
        for request_id in failed_request_ids:
            self.cleanup_aborted_request(request_id)

    def chunk_batch_key(self, state: Code2WavStreamState) -> tuple[int, bool]:
        slot_index = state.slot_index
        assert slot_index is not None and state.pending_chunks
        return (
            state.pending_chunks[0].shape[0],
            self.decode_arena.decoded_frame_count(slot_index) == 0,
        )

    @torch.inference_mode()
    def decode_slot_batch(
        self, participants: list[tuple[str, Code2WavStreamState]]
    ) -> list[OutgoingMessage]:
        assert participants
        batch_key = self.chunk_batch_key(participants[0][1])
        assert all(
            self.chunk_batch_key(state) == batch_key for _, state in participants
        )
        slot_indices = [state.slot_index for _, state in participants]
        assert all(slot_index is not None for slot_index in slot_indices)
        concrete_slot_indices = [int(slot_index) for slot_index in slot_indices]
        decoded_frame_counts = [
            self.decode_arena.decoded_frame_count(slot_index)
            for slot_index in concrete_slot_indices
        ]
        pending_chunks = [state.pending_chunks[0] for _, state in participants]
        batched_codes = torch.stack([frame_codes.T for frame_codes in pending_chunks])
        waveform_batch = self.decode_arena.decode_step(
            batched_codes, slot_indices=concrete_slot_indices
        )
        self.decode_step_index += 1
        for _, state in participants:
            state.pending_chunks.popleft()
            state.last_decode_step_index = self.decode_step_index
        if get_recorder().is_active():
            emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_chunk_batch",
                metadata={
                    "batch_size": len(participants),
                    "chunk_frames": batch_key[0],
                    "decoded_frames": decoded_frame_counts,
                    "slot_indices": concrete_slot_indices,
                    "participant_request_ids": [
                        request_id for request_id, _ in participants
                    ],
                    "active_slots": self.active_slot_count,
                    "inbox_depth": self.inbox.qsize(),
                },
            )
        else:
            pass
        host_waveforms = waveform_batch[:, 0].float().cpu()
        return [
            self.build_stream_message(request_id, state, host_waveforms[row_index])
            for row_index, (request_id, state) in enumerate(participants)
        ]

    def build_stream_message(
        self,
        request_id: str,
        state: Code2WavStreamState,
        waveform: torch.Tensor,
    ) -> OutgoingMessage:
        if state.num_samples:
            waveform = waveform[
                ..., : max(state.num_samples - state.emitted_samples, 0)
            ]
        else:
            pass
        state.emitted_samples += waveform.shape[-1]
        state.audio_parts.append(waveform)
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint=SOURCE_HINT,
            ),
            metadata={"modality": "audio"},
        )

    def finish_stream_decode(
        self, request_id: str, state: Code2WavStreamState
    ) -> list[OutgoingMessage]:
        state.stream_done = True
        if not state.pending_chunks and state.slot_index is not None:
            self.release_decode_slot(state)
        else:
            pass
        return []

    def build_result_message(
        self, request_id: str, state: Code2WavStreamState
    ) -> OutgoingMessage:
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return OutgoingMessage(
            request_id=request_id,
            type="result",
            data=StagePayload(
                request_id=request_id,
                request=self.stream_payloads[request_id].request,
                data=audio_waveform_payload(
                    waveform,
                    sample_rate=SAMPLE_RATE,
                    modality="audio",
                    source_hint=SOURCE_HINT,
                ),
            ),
        )

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            pass
        return self.finish_stream_decode(request_id, state)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            pass
        stream_messages = self.finish_stream_decode(request_id, state)
        if state.pending_chunks:
            return None
        else:
            pass
        stream_messages.append(self.build_result_message(request_id, state))
        return stream_messages


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
