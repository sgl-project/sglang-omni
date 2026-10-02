# SPDX-License-Identifier: Apache-2.0
"""Streams the agent's Mimi codes into audio as they are generated.

Mimi's decoder is causal, so frames decode in chunks of any width and land
on the same samples as a whole-sequence decode; a chunk ramp emits the first
chunk on arrival and aggregates later frames into wider chunks, amortizing
per-decode launch and PCM transfer overhead.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiDecodeState
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
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


class StreamState:
    def __init__(self, codec: MimiCodec) -> None:
        self.decode_state: MimiDecodeState = codec.init_decode_state()
        self.audio_parts: list[torch.Tensor] = []
        self.emitted_samples = 0
        self.pending_codes: deque[torch.Tensor] = deque()
        self.pending_frames = 0
        self.decodes_done = 0
        self.num_samples = 0


class PersonaPlexCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        codec: MimiCodec,
        *,
        compute_fn: AudioPayloadDecoder,
        chunk_ramp_frames: tuple[int, ...] | list[int],
    ) -> None:
        super().__init__(compute_fn)
        self.codec = codec
        self.stream_states: dict[str, StreamState] = {}
        self.chunk_ramp_frames = self.parse_chunk_ramp_frames(chunk_ramp_frames)

    @staticmethod
    def parse_chunk_ramp_frames(chunk_ramp_frames: object) -> tuple[int, ...]:
        """Normalize the chunk_ramp_frames option: a YAML list of positive ints."""
        if isinstance(chunk_ramp_frames, (list, tuple)):
            try:
                ramp_frames = tuple(int(width) for width in chunk_ramp_frames)
            except (TypeError, ValueError):
                ramp_frames = ()
        else:
            ramp_frames = ()
        if not ramp_frames or any(width <= 0 for width in ramp_frames):
            raise ValueError(
                f"code2wav chunk_ramp_frames {chunk_ramp_frames!r} is not a "
                "non-empty list of positive integers (e.g. [1, 4])"
            )
        else:
            pass
        return ramp_frames

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

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.stream_states.setdefault(request_id, StreamState(self.codec))

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_states.pop(request_id, None)

    @torch.inference_mode()
    def on_stream_chunk(
        self, request_id: str, item: IncomingMessage
    ) -> list[OutgoingMessage]:
        state = self.stream_states.setdefault(request_id, StreamState(self.codec))
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

    def decode_chunk(
        self, request_id: str, state: StreamState, frame_count: int
    ) -> OutgoingMessage:
        """Decode the oldest frame_count frames of pending codes and emit the chunk."""
        picked_codes: list[torch.Tensor] = []
        frames_taken = 0
        while frames_taken < frame_count:
            head = state.pending_codes[0]
            frames_needed = frame_count - frames_taken
            if head.shape[0] > frames_needed:
                picked_codes.append(head[:frames_needed])
                state.pending_codes[0] = head[frames_needed:]
                break
            else:
                picked_codes.append(state.pending_codes.popleft())
                frames_taken += head.shape[0]
        codes_FK = torch.cat(picked_codes) if len(picked_codes) > 1 else picked_codes[0]
        state.pending_frames -= frame_count
        state.decodes_done += 1
        codes_BKF = codes_FK.T[None].to(self.codec.device)
        waveform = self.codec.decode_step(codes_BKF, state.decode_state)[0, 0]
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

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            pass
        messages: list[OutgoingMessage] = []
        if state.pending_frames > 0:
            messages.append(self.decode_chunk(request_id, state, state.pending_frames))
        else:
            pass
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        messages.append(
            OutgoingMessage(
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
        )
        return messages


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
