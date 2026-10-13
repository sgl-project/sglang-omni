# SPDX-License-Identifier: Apache-2.0
"""Shared scheduling for streaming windows and padded codec batches."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.audio_stream import (
    AudioCodec,
    FinalAudioData,
    StreamVocoderState,
    WindowDecoder,
    build_stream_vocoder_chunk,
    flush_stream_vocoder_chunk,
    resolve_stream_overlap_tokens,
)
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


class StreamingVocoderScheduler(StreamingSimpleScheduler, ABC):
    """Schedule codec windows while model subclasses provide final frame payloads."""

    def __init__(
        self,
        codec: AudioCodec,
        *,
        device: str,
        stream_stride: int,
        stream_followup_stride: int,
        stream_overlap_tokens: int | None,
        stream_crossfade_samples: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        decode_window: WindowDecoder | None,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0 or max_batch_size <= 0:
            raise ValueError(
                "stream_stride, stream_followup_stride, and max_batch_size must be > 0"
            )
        else:
            pass
        if stream_crossfade_samples < 0 or max_batch_wait_ms < 0:
            raise ValueError(
                "stream_crossfade_samples and max_batch_wait_ms must be >= 0"
            )
        else:
            pass

        self.codec: AudioCodec = codec
        self.device: torch.device = torch.device(device)
        self.stream_stride: int = stream_stride
        self.stream_followup_stride: int = stream_followup_stride
        self.stream_overlap_tokens: int = resolve_stream_overlap_tokens(
            codec, stream_overlap_tokens
        )
        self.stream_crossfade_samples: int = stream_crossfade_samples
        self.decode_window: WindowDecoder | None = decode_window
        self.stream_states: dict[str, StreamVocoderState] = {}

        super().__init__(
            self.vocode_payload,
            batch_compute_fn=self.vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )
        self.payloads: dict[str, StagePayload] = self.stream_payloads

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return bool(payload.request.params.get("stream"))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        """The scheduler supplies payload; codec state needs only the request id."""
        self.stream_states.setdefault(request_id, StreamVocoderState())

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        stream_state = self.stream_states.setdefault(request_id, StreamVocoderState())
        frame_codes = chunk.data
        if not isinstance(frame_codes, torch.Tensor):
            raise TypeError(
                f"Codec stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(frame_codes).__name__}"
            )
        else:
            pass
        audio_chunk = build_stream_vocoder_chunk(
            stream_state,
            frame_codes,
            codec=self.codec,
            device=self.device,
            stream_stride=self.stream_stride,
            stream_followup_stride=self.stream_followup_stride,
            stream_overlap_tokens=self.stream_overlap_tokens,
            stream_crossfade_samples=self.stream_crossfade_samples,
            decode_window=self.decode_window,
        )
        if audio_chunk is None:
            return []
        else:
            pass
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=audio_chunk,
                metadata={"modality": "audio"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        stream_state = self.stream_states.get(request_id)
        if stream_state is None:
            return []
        else:
            pass

        had_streamed_audio = stream_state.last_vocode_tokens > 0
        audio_chunk = flush_stream_vocoder_chunk(
            stream_state,
            codec=self.codec,
            device=self.device,
            stream_overlap_tokens=self.stream_overlap_tokens,
            stream_crossfade_samples=self.stream_crossfade_samples,
            decode_window=self.decode_window,
        )
        messages: list[OutgoingMessage] = []
        if audio_chunk is not None:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=audio_chunk,
                    metadata={"modality": "audio"},
                )
            )
        else:
            pass

        payload = self.payloads[request_id]
        if audio_chunk is None and not had_streamed_audio:
            result = self.vocode_payload(payload)
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        field_name: field_value
                        for field_name, field_value in result.data.items()
                        if field_name in {"audio_data", "sample_rate", "modality"}
                    },
                    metadata={"modality": "audio"},
                )
            )
        else:
            pass

        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=self.final_result_data(payload),
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_states.pop(request_id, None)

    def vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self.vocode_payloads([payload])[0]

    def decode_codebook_codes(
        self, codebook_codes: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        code_batches = [frame_codes.to(self.device) for frame_codes in codebook_codes]
        frame_counts = [int(frame_codes.shape[-1]) for frame_codes in code_batches]
        max_frame_count = max(frame_counts)
        padded_codebook_codes = [
            torch.nn.functional.pad(
                frame_codes, (0, max_frame_count - frame_count), value=0
            )
            for frame_codes, frame_count in zip(code_batches, frame_counts)
        ]
        batched_codebook_codes = torch.stack(padded_codebook_codes, dim=0)

        with torch.no_grad():
            batched_audio = (
                self.codec.from_indices
                if self.decode_window is None
                else self.decode_window
            )(batched_codebook_codes)

        samples_per_frame = int(self.codec.frame_length)
        return [
            batched_audio[request_index, 0, : frame_count * samples_per_frame]
            .float()
            .cpu()
            for request_index, frame_count in enumerate(frame_counts)
        ]

    @abstractmethod
    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        """Validate the model's completed frame payload."""

    @abstractmethod
    def final_result_data(self, payload: StagePayload) -> FinalAudioData:
        """Build the model's final usage and finish metadata."""

    @abstractmethod
    def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        """Decode completed frame payloads in their original request order."""
