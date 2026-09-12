from __future__ import annotations

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

OUTPUT_SAMPLE_RATE = 22_050
DECODE_WINDOW_FRAMES = 16
TAIL_HOLDBACK_SAMPLES = 256


class StreamingCodec:
    def __init__(self, decoder, device) -> None:
        self.decoder = decoder
        self.device = device
        self.codes_rows: list[torch.Tensor] = []
        self.emitted_samples = 0

    @torch.inference_mode()
    def push(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        for row in codes_TQ.to(self.device):
            self.codes_rows.append(row)
        return self._advance(final=False)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if not self.codes_rows:
            return torch.zeros(0)
        return self._advance(final=True)

    def _advance(self, *, final: bool) -> torch.Tensor:
        frames = len(self.codes_rows)
        samples_per_frame = self.decoder.samples_per_frame
        first = max(0, frames - DECODE_WINDOW_FRAMES)
        audio = self.decoder(torch.stack(self.codes_rows[first:]))
        available = frames * samples_per_frame - (0 if final else TAIL_HOLDBACK_SAMPLES)
        start = self.emitted_samples - first * samples_per_frame
        fresh = audio[start : available - first * samples_per_frame].float().cpu()
        self.emitted_samples = available
        return fresh


class _StreamState:
    def __init__(self, decoder, device) -> None:
        self.codec = StreamingCodec(decoder, device)
        self.audio_parts: list[torch.Tensor] = []


class NemotronCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(self, decoder, device, *, compute_fn) -> None:
        super().__init__(compute_fn)
        self._decoder = decoder
        self._device = device
        self._states: dict[str, _StreamState] = {}

    def _new_state(self) -> _StreamState:
        return _StreamState(self._decoder, self._device)

    def is_streaming_payload(self, payload) -> bool:
        return payload.request_id in self._states

    def on_streaming_new_request(self, request_id: str, payload) -> None:
        self._states.setdefault(request_id, self._new_state())

    def clear_stream_state(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    @torch.inference_mode()
    def on_stream_chunk(self, request_id: str, item) -> list[OutgoingMessage]:
        state = self._states.setdefault(request_id, self._new_state())
        tail = state.codec.push(item.data)
        state.audio_parts.append(tail)
        # Chunks to the coordinator are msgpack'd, so the waveform travels in
        # the shared payload format rather than as a tensor.
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=audio_waveform_payload(
                    tail,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    modality="audio",
                    source_hint="NemotronVoiceChat",
                ),
                metadata={"modality": "audio"},
            )
        ]

    @torch.inference_mode()
    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self._states.get(request_id)
        if state is None:
            return []
        messages: list[OutgoingMessage] = []
        if state.codec.codes_rows:
            tail = state.codec.flush()
            if tail.numel():
                state.audio_parts.append(tail)
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=audio_waveform_payload(
                            tail,
                            sample_rate=OUTPUT_SAMPLE_RATE,
                            modality="audio",
                            source_hint="NemotronVoiceChat",
                        ),
                        metadata={"modality": "audio"},
                    )
                )
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return messages + [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=request_id,
                    request=self._stream_payloads[request_id].request,
                    data=audio_waveform_payload(
                        waveform,
                        sample_rate=OUTPUT_SAMPLE_RATE,
                        modality="audio",
                        source_hint="NemotronVoiceChat",
                    ),
                ),
            )
        ]
