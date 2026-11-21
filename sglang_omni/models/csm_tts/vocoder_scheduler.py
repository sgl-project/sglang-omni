# SPDX-License-Identifier: Apache-2.0
"""Streaming Mimi vocoder for CSM.

No delay pattern. Overlap-trim hides Mimi's stateless edge frames.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from numpy.typing import ArrayLike

from sglang_omni.models.csm_tts.audio_codec import CsmAudioCodec, sanitize_for_mimi
from sglang_omni.models.csm_tts.payload_types import CsmTtsState, build_csm_usage
from sglang_omni.models.csm_tts.utils import (
    NUM_CODEBOOKS,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    trim_at_first_all_zero_frame,
    trimmed_output_codes,
)
from sglang_omni.models.csm_tts.vocoder_stream_state import (
    CsmVocoderStreamState,
    latch_initial_codec_chunk_frames,
    latch_stream_contract,
    latch_stream_metadata,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

__all__ = [
    "CsmStreamingVocoderScheduler",
    "DEFAULT_CSM_STREAM_STRIDE",
    "DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE",
]

DEFAULT_CSM_STREAM_STRIDE = 13
DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE = 6


class CsmStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Streaming incremental decode; batched final decode otherwise."""

    def __init__(
        self,
        codec: CsmAudioCodec,
        *,
        stream_stride: int = DEFAULT_CSM_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE,
        stream_overlap_frames: int = 4,
        stream_holdback_frames: int = 2,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        else:
            pass
        if stream_overlap_frames < 0:
            raise ValueError("stream_overlap_frames must be >= 0")
        else:
            pass
        if stream_holdback_frames < 0:
            raise ValueError("stream_holdback_frames must be >= 0")
        else:
            pass

        self.codec = codec
        self.stream_stride = int(stream_stride)
        self.stream_followup_stride = int(stream_followup_stride)
        self.stream_overlap_frames = int(stream_overlap_frames)
        self.stream_holdback_frames = int(stream_holdback_frames)
        self.sample_rate = SAMPLE_RATE
        self.samples_per_frame = SAMPLES_PER_FRAME
        self.stream_states: dict[str, CsmVocoderStreamState] = {}

        super().__init__(
            self.vocode_payload,
            batch_compute_fn=self.vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        """True when request.params.stream is set."""
        params = payload.request.params
        if not isinstance(params, Mapping):
            raise TypeError(
                f"CSM request params must be a dict, got {type(params).__name__}"
            )
        else:
            pass
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        """Latch codebook contract and initial_codec_chunk_frames."""
        if self.is_aborted(request_id):
            return
        else:
            pass
        state = self.stream_states.setdefault(request_id, CsmVocoderStreamState())
        payload_data = payload.data if isinstance(payload.data, Mapping) else {}
        latch_stream_contract(request_id, state, payload_data, source="payload")
        params = payload.request.params
        latch_initial_codec_chunk_frames(
            state,
            params if isinstance(params, Mapping) else None,
            steady_chunk_frames=self.stream_stride,
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        """Append one code row and emit any newly available audio."""
        if self.is_aborted(request_id):
            return []
        else:
            pass
        state = self.stream_states.setdefault(request_id, CsmVocoderStreamState())
        latch_stream_metadata(
            request_id,
            state,
            item.metadata,
            steady_chunk_frames=self.stream_stride,
        )

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"CSM stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        else:
            pass
        row = row.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        num_codebooks = state.num_codebooks or NUM_CODEBOOKS
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"CSM stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        else:
            pass
        state.code_rows.append(row)
        return self.decode_delta(request_id, state)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        """Flush held-back frames, trim at the first all-zero frame, emit usage."""
        if self.is_aborted(request_id):
            return []
        else:
            pass
        payload = self.stream_payloads[request_id]
        state = self.stream_states.setdefault(request_id, CsmVocoderStreamState())
        messages = self.decode_delta(request_id, state, final=True)
        if not messages and state.frames_emitted == 0:
            # frame-0 EOS falls back to the engine payload output_codes.
            output = self.audio_payload_from_stage_payload(request_id, payload)
            if output is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=output,
                        metadata={"modality": "audio"},
                    )
                )
            else:
                pass
        else:
            pass

        final_data: dict[str, object] = {
            "modality": "audio",
            "sample_rate": self.sample_rate,
        }
        usage = build_csm_usage(CsmTtsState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        else:
            pass
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_states.pop(request_id, None)

    def decode_delta(
        self, request_id: str, state: CsmVocoderStreamState, *, final: bool = False
    ) -> list[OutgoingMessage]:
        """Overlap-trim incremental decode once enough frames have arrived."""
        available = len(state.code_rows)
        if available == 0:
            return []
        else:
            pass

        emit_until = self.emit_boundary(state, available, final=final)
        if emit_until <= state.frames_emitted:
            return []
        else:
            pass

        window_start = max(0, state.frames_emitted - self.stream_overlap_frames)
        codes = torch.stack(state.code_rows[window_start:emit_until], dim=0)
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            logger.warning(
                f"CSM vocoder stream {request_id} clamped {num_clamped} OOB codes "
                "(should not fire at temperature=0)"
            )
        else:
            pass
        audio = self.codec.decode(codes).reshape(-1)

        trim_samples = (state.frames_emitted - window_start) * self.samples_per_frame
        if final:
            delta = audio[trim_samples:]
        else:
            new_frames = emit_until - state.frames_emitted
            delta = audio[
                trim_samples : trim_samples + new_frames * self.samples_per_frame
            ]
        delta = delta.contiguous()
        if delta.numel() == 0:
            return []
        else:
            pass

        state.frames_emitted = emit_until
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=self.audio_payload(delta, source_hint="CSM TTS streaming"),
                metadata={"modality": "audio"},
            )
        ]

    def emit_boundary(
        self, state: CsmVocoderStreamState, available: int, *, final: bool
    ) -> int:
        """Frame index to decode up to, or frames_emitted while still holding back."""
        if final:
            rows_F32 = torch.stack(state.code_rows, dim=0)
            return int(trim_at_first_all_zero_frame(rows_F32).shape[0])
        else:
            pass

        initial = state.initial_codec_chunk_frames or 0
        if state.frames_emitted > 0:
            needed = (
                state.frames_emitted
                + self.stream_followup_stride
                + self.stream_holdback_frames
            )
            use_initial = False
        elif 0 < initial < self.stream_stride:
            needed = initial
            use_initial = True
        else:
            needed = self.stream_stride
            use_initial = False
        if available < needed:
            return state.frames_emitted
        elif use_initial:
            return initial
        else:
            return available - self.stream_holdback_frames

    def audio_payload_from_stage_payload(
        self, request_id: str, payload: StagePayload
    ) -> dict[str, object] | None:
        """Full-payload decode fallback for streams that never emitted."""
        if not isinstance(payload.data, Mapping):
            return None
        else:
            pass
        codes = trimmed_output_codes(CsmTtsState.from_dict(payload.data).output_codes)
        if codes is None:
            return None
        else:
            pass
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            logger.warning(
                f"CSM vocoder fallback decode for {request_id} clamped "
                f"{num_clamped} OOB codes"
            )
        else:
            pass
        return self.audio_payload(
            self.codec.decode(codes), source_hint="CSM TTS streaming"
        )

    def vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self.vocode_payloads([payload])[0]

    def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        """Non-streaming batch decode of output_codes."""
        items: list[tuple[CsmTtsState, torch.Tensor | None]] = []
        for payload in payloads:
            state = CsmTtsState.from_dict(payload.data)
            codes = trimmed_output_codes(state.output_codes)
            if codes is not None:
                codes, num_clamped = sanitize_for_mimi(codes)
                if num_clamped:
                    logger.warning(
                        f"CSM vocoder batch decode for {payload.request_id} "
                        f"clamped {num_clamped} OOB codes"
                    )
                else:
                    pass
            else:
                pass
            items.append((state, codes))

        valid = [(i, codes) for i, (_, codes) in enumerate(items) if codes is not None]
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        if valid:
            indices, codes_list = zip(*valid)
            decoded = self.codec.decode_batch(list(codes_list))
            if len(decoded) != len(valid):
                raise RuntimeError(
                    f"CSM vocoder decode_batch returned {len(decoded)} audios "
                    f"for {len(valid)} requests"
                )
            else:
                pass
            for index, waveform in zip(indices, decoded):
                waveforms[index] = waveform
        else:
            pass

        results: list[StagePayload] = []
        for payload, (state, _), waveform in zip(payloads, items, waveforms):
            data = self.audio_payload(
                waveform if waveform is not None else [],
                source_hint="CSM TTS vocoder",
            )
            usage = build_csm_usage(state)
            if usage is not None:
                data["usage"] = usage
            else:
                pass
            payload.data = data
            results.append(payload)
        return results

    def audio_payload(
        self, audio: ArrayLike | torch.Tensor, *, source_hint: str
    ) -> dict[str, object]:
        return audio_waveform_payload(
            audio,
            sample_rate=self.sample_rate,
            modality="audio",
            source_hint=source_hint,
        )
