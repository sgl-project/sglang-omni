from __future__ import annotations

from collections.abc import Callable, Coroutine

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

OUTPUT_SAMPLE_RATE = 22_050
DECODE_WINDOW_FRAMES = 16
TAIL_HOLDBACK_SAMPLES = 256


class StreamingCodec:
    def __init__(self, decoder, device) -> None:
        self.decoder = decoder
        self.device = device
        self.window_codes_TQ: torch.Tensor = torch.empty(
            0, decoder.num_quantizers, dtype=torch.long, device=device
        )
        self.pushed_frames: int = 0
        self.emitted_samples = 0

    @torch.inference_mode()
    def push(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        window_codes_TQ = torch.cat((self.window_codes_TQ, codes_TQ.to(self.device)))
        self.window_codes_TQ = window_codes_TQ[-DECODE_WINDOW_FRAMES:]
        self.pushed_frames += codes_TQ.shape[0]
        return self.advance(final=False)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if not self.pushed_frames:
            return torch.zeros(0)
        else:
            pass
        return self.advance(final=True)

    def advance(self, *, final: bool) -> torch.Tensor:
        frames = self.pushed_frames
        samples_per_frame = self.decoder.samples_per_frame
        first = frames - self.window_codes_TQ.shape[0]
        audio = self.decoder(self.window_codes_TQ)
        available = frames * samples_per_frame - (0 if final else TAIL_HOLDBACK_SAMPLES)
        start = self.emitted_samples - first * samples_per_frame
        fresh = audio[start : available - first * samples_per_frame].float().cpu()
        self.emitted_samples = available
        return fresh


class StreamState:
    def __init__(self, decoder, device) -> None:
        self.codec = StreamingCodec(decoder, device)
        self.audio_parts: list[torch.Tensor] = []


class NemotronCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        decoder,
        device,
        *,
        compute_fn: (
            Callable[
                [StagePayload],
                StagePayload | Coroutine[None, None, StagePayload],
            ]
            | None
        ),
    ) -> None:
        super().__init__(compute_fn)
        self.decoder = decoder
        self.device = device
        self.states: dict[str, StreamState] = {}

    def new_state(self) -> StreamState:
        return StreamState(self.decoder, self.device)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return payload.request_id in self.states

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.states.setdefault(request_id, self.new_state())

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    @torch.inference_mode()
    def on_stream_chunk(self, request_id: str, item) -> list[OutgoingMessage]:
        state = self.states.setdefault(request_id, self.new_state())
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
        state = self.states.get(request_id)
        if state is None:
            return []
        else:
            pass
        messages: list[OutgoingMessage] = []
        if state.codec.pushed_frames:
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
            else:
                pass
        else:
            pass
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return messages + [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=request_id,
                    request=self.stream_payloads[request_id].request,
                    data=audio_waveform_payload(
                        waveform,
                        sample_rate=OUTPUT_SAMPLE_RATE,
                        modality="audio",
                        source_hint="NemotronVoiceChat",
                    ),
                ),
            )
        ]
