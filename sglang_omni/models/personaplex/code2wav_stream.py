# SPDX-License-Identifier: Apache-2.0
"""Streams the agent's Mimi codes into audio as they are generated.

Mimi's decoder is causal and its chunked path lands on exactly the samples a
whole-sequence decode would, so each frame goes out as soon as its codes
arrive: 1920 samples per 80 ms frame, no holdback.
"""

from __future__ import annotations

import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiDecodeState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

SOURCE_HINT = "PersonaPlex"


def _audio_message(request_id: str, waveform: torch.Tensor) -> OutgoingMessage:
    # Note (wilsonzheng0327): Chunks to the coordinator are msgpack'd, so the waveform
    # travels as a payload, not a tensor.
    return OutgoingMessage(
        request_id=request_id,
        type="stream",
        data=audio_waveform_payload(
            waveform, sample_rate=SAMPLE_RATE, modality="audio", source_hint=SOURCE_HINT
        ),
        metadata={"modality": "audio"},
    )


def trim_to_caller(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Cut the reply back to the caller recording's own length.

    Frames are whole 80 ms, so a recording that is not a multiple of one is
    padded before encoding; the reference trims that padding off the reply.
    """
    if num_samples and waveform.shape[-1] > num_samples:
        return waveform[..., :num_samples]
    return waveform


class _StreamState:
    def __init__(self, codec: MimiCodec) -> None:
        self.decode_state: MimiDecodeState = codec.init_decode_state()
        self.audio_parts: list[torch.Tensor] = []
        self.emitted = 0


class PersonaPlexCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(self, codec: MimiCodec, *, compute_fn) -> None:
        super().__init__(compute_fn)
        self._codec = codec
        self._states: dict[str, _StreamState] = {}

    def is_streaming_payload(self, payload) -> bool:
        return payload.request_id in self._states

    def on_streaming_new_request(self, request_id: str, payload) -> None:
        self._states.setdefault(request_id, _StreamState(self._codec))

    def clear_stream_state(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    @torch.inference_mode()
    def on_stream_chunk(self, request_id: str, item) -> list[OutgoingMessage]:
        state = self._states.setdefault(request_id, _StreamState(self._codec))
        codes_FK = torch.as_tensor(
            item.data, dtype=torch.long, device=self._codec.device
        )
        waveform = self._codec.decode_step(codes_FK.T[None], state.decode_state)[0, 0]
        waveform = waveform.float().cpu()
        # Note (wilsonzheng0327): The terminal payload only arrives after the LM finishes,
        # so the caller length travels with each chunk.
        num_samples = int((item.metadata or {}).get("num_samples") or 0)
        if num_samples:
            waveform = trim_to_caller(waveform, max(num_samples - state.emitted, 0))
        state.emitted += waveform.shape[-1]
        state.audio_parts.append(waveform)
        return [_audio_message(request_id, waveform)]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self._states.get(request_id)
        if state is None:
            return []
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=request_id,
                    request=self._stream_payloads[request_id].request,
                    data=audio_waveform_payload(
                        waveform,
                        sample_rate=SAMPLE_RATE,
                        modality="audio",
                        source_hint=SOURCE_HINT,
                    ),
                ),
            )
        ]


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
