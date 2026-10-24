from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

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
        self.append(codes_TQ)
        return self.take(self.decoder(self.window()), final=False).float().cpu()

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if not self.codes_rows:
            return torch.zeros(0)
        else:
            pass
        return self.take(self.decoder(self.window()), final=True).float().cpu()

    def append(self, codes_TQ: torch.Tensor) -> None:
        for row in codes_TQ.to(self.device):
            self.codes_rows.append(row)

    def window(self) -> torch.Tensor:
        return torch.stack(self.codes_rows[-DECODE_WINDOW_FRAMES:])

    def take(self, audio_S: torch.Tensor, *, final: bool) -> torch.Tensor:
        frames = len(self.codes_rows)
        samples_per_frame = self.decoder.samples_per_frame
        first = max(0, frames - DECODE_WINDOW_FRAMES)
        available = frames * samples_per_frame - (0 if final else TAIL_HOLDBACK_SAMPLES)
        start = self.emitted_samples - first * samples_per_frame
        fresh = audio_S[start : available - first * samples_per_frame]
        self.emitted_samples = available
        return fresh


class StreamState:
    def __init__(self, decoder, device) -> None:
        self.codec = StreamingCodec(decoder, device)
        self.audio_parts: list[torch.Tensor] = []


class NemotronCode2WavScheduler(StreamingSimpleScheduler):
    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

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
        max_batch_size: int,
    ) -> None:
        super().__init__(compute_fn, max_batch_size=max_batch_size)
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

    def audio_message(self, request_id: str, audio_S: torch.Tensor) -> OutgoingMessage:
        # Chunks to the coordinator are msgpack'd, so the waveform travels in
        # the shared payload format rather than as a tensor.
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=audio_waveform_payload(
                audio_S,
                sample_rate=OUTPUT_SAMPLE_RATE,
                modality="audio",
                source_hint="NemotronVoiceChat",
            ),
            metadata={"modality": "audio"},
        )

    @torch.inference_mode()
    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        request_ids = [request_id for request_id, _ in items]
        with self.state_lock:
            try:
                request_ids_by_window_frames: dict[int, list[str]] = {}
                for request_id, item in items:
                    codec = self.states.setdefault(request_id, self.new_state()).codec
                    codec.append(item.data)
                    request_ids_by_window_frames.setdefault(
                        min(len(codec.codes_rows), DECODE_WINDOW_FRAMES), []
                    ).append(request_id)
                fresh_by_request_id: dict[str, torch.Tensor] = {}
                for group_request_ids in request_ids_by_window_frames.values():
                    codecs = [
                        self.states[request_id].codec
                        for request_id in group_request_ids
                    ]
                    audio_BS = self.decoder.decode_batch(
                        torch.stack([codec.window() for codec in codecs])
                    )
                    for request_id, codec, audio_S in zip(
                        group_request_ids, codecs, audio_BS, strict=True
                    ):
                        fresh_by_request_id[request_id] = codec.take(
                            audio_S, final=False
                        )
                fresh_audio = [
                    fresh_by_request_id[request_id] for request_id in request_ids
                ]
                # note (Xinhao Tan): every host copy waits for the GPU, so a step copies once.
                host_audio = (
                    torch.cat(fresh_audio)
                    .float()
                    .cpu()
                    .split([audio_S.numel() for audio_S in fresh_audio])
                )
            except Exception as exc:
                logger.exception(
                    f"code2wav decode failed for {len(request_ids)} requests"
                )
                for request_id in request_ids:
                    self.emit_error(request_id, exc)
                    self.abort_state(request_id)
                failed_request_ids = request_ids
            else:
                for request_id, audio_S in zip(request_ids, host_audio, strict=True):
                    # note (Xinhao Tan): a split view would pin the whole step's audio.
                    self.states[request_id].audio_parts.append(audio_S.clone())
                    if not self.is_aborted(request_id):
                        self.outbox.put(self.audio_message(request_id, audio_S))
                    else:
                        pass
                failed_request_ids = []
        for request_id in failed_request_ids:
            self.cleanup_aborted_request(request_id)

    @torch.inference_mode()
    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.states.get(request_id)
        if state is None:
            return []
        else:
            pass
        messages: list[OutgoingMessage] = []
        if state.codec.codes_rows:
            tail = state.codec.flush()
            if tail.numel():
                state.audio_parts.append(tail)
                messages.append(self.audio_message(request_id, tail))
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
