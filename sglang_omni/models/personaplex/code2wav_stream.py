# SPDX-License-Identifier: Apache-2.0
"""Streams the agent's Mimi codes into audio as they are generated.

Mimi's decoder is causal and its attention ring holds the widest ramp chunk, so
chunked decodes land on the samples a whole-sequence decode would; a chunk ramp
emits the first chunk on arrival and aggregates later frames into wider chunks,
amortizing per-decode launch and PCM transfer overhead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.mimi_decode_graph import DecodeSlot, MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
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
class StreamState:
    decode_slot: DecodeSlot
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    emitted_samples: int = 0
    pending_codes: deque[torch.Tensor] = field(default_factory=deque)
    pending_frames: int = 0
    decodes_done: int = 0
    num_samples: int = 0


class PersonaPlexCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        decode_slots: MimiDecodeSlots,
        *,
        compute_fn: AudioPayloadDecoder,
        chunk_ramp_frames: tuple[int, ...] | list[int],
    ) -> None:
        super().__init__(compute_fn)
        self.decode_slots = decode_slots
        self.stream_states: dict[str, StreamState] = {}
        self.chunk_ramp_frames: tuple[int, ...] = tuple(chunk_ramp_frames)
        assert max(self.chunk_ramp_frames) <= decode_slots.max_chunk_frames

    def chunk_target_frames(self, state: StreamState) -> int:
        """Frames the next decode should cover; the last ramp entry is the steady width."""
        index = min(state.decodes_done, len(self.chunk_ramp_frames) - 1)
        return self.chunk_ramp_frames[index]

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # Note (wilsonzheng0327): The LM streams every frame it produces, so a reply
        # with frames has chunks on the way whichever channel lands first; only an
        # empty reply is rendered whole.
        codes = PersonaPlexState.from_dict(payload.data).codes
        return codes is not None and codes.shape[0] > 0

    def stream_state(self, request_id: str) -> StreamState:
        state = self.stream_states.get(request_id)
        if state is None:
            state = StreamState(decode_slot=self.decode_slots.acquire())
            self.stream_states[request_id] = state
        else:
            pass
        return state

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.stream_state(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        state = self.stream_states.pop(request_id, None)
        if state is not None:
            self.decode_slots.release(state.decode_slot)
        else:
            pass

    @torch.inference_mode()
    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self.stream_state(request_id)
        codes_FK = torch.as_tensor(item.data, dtype=torch.long)
        state.pending_codes.append(codes_FK)
        state.pending_frames += codes_FK.shape[0]
        # Note (wilsonzheng0327): The terminal payload only arrives after the LM finishes,
        # so the caller length travels with each chunk.
        num_samples = int((item.metadata or {}).get("num_samples") or 0)
        if num_samples:
            state.num_samples = num_samples
        else:
            pass
        messages: list[OutgoingMessage] = []
        while state.pending_frames >= self.chunk_target_frames(state):
            messages.append(
                self.decode_chunk(request_id, state, self.chunk_target_frames(state))
            )
        return messages

    @torch.inference_mode()
    def decode_chunk(
        self, request_id: str, state: StreamState, frame_count: int
    ) -> OutgoingMessage:
        """Decode the oldest frame_count frames of pending codes and emit the chunk."""
        pending_codes = torch.cat(list(state.pending_codes))
        codes_FK = pending_codes[:frame_count]
        remaining_codes = pending_codes[frame_count:]
        state.pending_codes = (
            deque([remaining_codes]) if remaining_codes.shape[0] else deque()
        )
        state.pending_frames -= frame_count
        state.decodes_done += 1
        codes_BKF = codes_FK.T[None].to(self.decode_slots.codec.device)
        waveform = self.decode_slots.decode_step(codes_BKF, state.decode_slot)[0, 0]
        waveform = waveform.float().cpu()
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

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is not None and state.pending_frames > 0:
            return [self.decode_chunk(request_id, state, state.pending_frames)]
        else:
            return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            pass
        messages = self.on_stream_done_before_payload(request_id)
        request = self.stream_payloads[request_id].request
        if request.params.get("stream"):
            # Note (wilsonzheng0327): A streaming client already holds every chunk; only a
            # non-streaming client reads the reply from the result.
            result_data = {"modality": "audio", "sample_rate": SAMPLE_RATE}
        else:
            waveform = (
                torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
            )
            result_data = audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint=SOURCE_HINT,
            )
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=request_id, request=request, data=result_data
                ),
            )
        )
        return messages


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
