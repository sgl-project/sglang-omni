# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Delay.

Delayed rows are reversed into raw codec frames per request. Due streams are
coalesced into one batched decode step on the stateful codec session shared
with MOSS-TTS Local, so each frame is decoded once with its full causal context.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerVocoder,
    MossAudioTokenizerVocoderDecoder,
)
from sglang_omni.models.moss_tts.codec_session import (
    CoalescedStepPlan,
    CodecStreamSession,
)
from sglang_omni.models.moss_tts.payload_types import (
    load_moss_tts_state,
    resolve_moss_audio_pad_code,
)
from sglang_omni.models.moss_tts.vocoder import MossTTSVocoder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)

logger = logging.getLogger(__name__)


def streaming_codec_view(codec: MossAudioTokenizerVocoder) -> MossAudioTokenizerVocoder:
    """Weight-sharing codec whose decoder owns the streaming state.

    The native decoder also serves the non-streaming batched and standalone
    decodes, which must never see live streaming offsets or KV.
    """
    view = MossAudioTokenizerVocoder(
        vars(codec.config),
        parameter_device="meta",
        decoder_dtype=codec.decoder_dtype,
        compute_dtype=codec.compute_dtype,
        attention_backend=codec.attention_backend,
    )
    view.uses_moss_audio_tokenizer_v1_weights = (
        codec.uses_moss_audio_tokenizer_v1_weights
    )
    view.quantizer = codec.quantizer
    view.decoder = MossAudioTokenizerVocoderDecoder(
        source_decoder=codec.decoder,
        attention_backend=codec.attention_backend,
    )
    return view.eval()


def stream_contract_integer(
    request_id: str,
    field_name: Literal["n_vq", "audio_pad_code", "sample_rate"],
    value: object,
    previous: int | None,
) -> int:
    """Read a positive wire integer without changing the stream contract."""
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError(
            f"MOSS-TTS stream {field_name} for {request_id!r} must be > 0, got {value}"
        )
    elif previous is not None and previous != resolved_value:
        raise ValueError(
            f"MOSS-TTS stream {field_name} changed for {request_id!r}: "
            f"{previous} -> {value}"
        )
    else:
        return resolved_value


@dataclass
class MossStreamState:
    delay_window: deque[torch.Tensor] = field(default_factory=deque)
    pending: list[torch.Tensor] = field(default_factory=list)
    ready_audio: list[torch.Tensor] = field(default_factory=list)
    slot: int | None = None
    segment_open: bool = False
    segment_started: bool = False
    threshold: int = 0
    n_vq: int | None = None
    audio_pad_code: int | None = None
    sample_rate: int = 24000


class MossStreamingVocoderScheduler(
    StreamingVocoderBase[MossStreamState, CoalescedStepPlan]
):
    """Reverse MOSS delay rows and decode them in coalesced session steps."""

    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        vocoder: MossTTSVocoder,
        *,
        stream_chunk_frames: int = 16,
        initial_chunk_frames: int = 4,
        coalesce_floor_frames: int = 4,
        stream_slots: int = 32,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        vocoder_cuda_graph: bool = True,
        vocoder_cuda_graph_min_free_gb: float = 3.0,
    ) -> None:
        if stream_chunk_frames <= 0 or stream_slots <= 0:
            raise ValueError("stream_chunk_frames and stream_slots must be > 0")
        else:
            pass
        self.vocoder = vocoder
        self.audio_vocoder = vocoder.audio_vocoder
        model_config = getattr(vocoder.processor, "model_config", None)
        self.default_n_vq = int(getattr(model_config, "n_vq", 0) or 0)
        self.default_audio_pad_code = resolve_moss_audio_pad_code(model_config)
        self.stream_chunk_frames = int(stream_chunk_frames)
        self.default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), self.stream_chunk_frames)
        )
        self.coalesce_floor_frames = max(
            1, min(int(coalesce_floor_frames), self.stream_chunk_frames)
        )
        self.stream_slots = int(stream_slots)
        self.stream_chunk_batch_max = self.stream_slots
        self.vocoder_cuda_graph = bool(vocoder_cuda_graph)
        self.vocoder_cuda_graph_min_free_gb = float(vocoder_cuda_graph_min_free_gb)
        self.session: CodecStreamSession | None = None
        super().__init__(
            self.vocode_payload,
            batch_compute_fn=self.vocode_payloads,
            sample_rate=int(self.audio_vocoder.sample_rate),
            stream_source_hint="MOSS-TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> MossStreamState:
        del request_id
        return MossStreamState(
            n_vq=self.default_n_vq or None,
            audio_pad_code=self.default_audio_pad_code,
            sample_rate=self.sample_rate,
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: MossStreamState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            params = source.request.params
            self.latch_threshold(
                request_id, state, params if isinstance(params, dict) else None
            )
            return
        else:
            pass
        n_vq = source.get("n_vq")
        if n_vq is not None:
            state.n_vq = stream_contract_integer(request_id, "n_vq", n_vq, state.n_vq)
        else:
            pass
        audio_pad_code = source.get("audio_pad_code")
        if audio_pad_code is not None:
            state.audio_pad_code = stream_contract_integer(
                request_id, "audio_pad_code", audio_pad_code, state.audio_pad_code
            )
        else:
            pass
        sample_rate = source.get("sample_rate")
        if sample_rate is not None:
            state.sample_rate = stream_contract_integer(
                request_id, "sample_rate", sample_rate, state.sample_rate
            )
        else:
            pass
        if state.threshold == 0:
            self.latch_threshold(request_id, state, source)
        else:
            pass

    def latch_threshold(
        self,
        request_id: str,
        state: MossStreamState,
        values: Mapping[str, object] | None,
    ) -> None:
        initial_frames = resolve_initial_codec_chunk_frames(
            values,
            steady_chunk_frames=self.stream_chunk_frames,
            default_frames=self.default_initial_chunk_frames,
        )
        if initial_frames > 0 and not self.stream_has_emitted(request_id):
            state.threshold = initial_frames
        else:
            state.threshold = self.stream_chunk_frames

    def validate_chunk(
        self, request_id: str, state: MossStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        rows = codes.to(dtype=torch.long)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        elif rows.ndim != 2:
            raise ValueError(
                f"MOSS-TTS stream chunk for {request_id!r} must be [N] or "
                f"[T, N], got {tuple(rows.shape)}"
            )
        else:
            pass
        if state.n_vq is None or state.audio_pad_code is None:
            raise RuntimeError(
                f"MOSS-TTS stream contract for {request_id!r} is incomplete"
            )
        else:
            pass
        if int(rows.shape[1]) != state.n_vq:
            raise ValueError(
                f"MOSS-TTS stream chunk has {int(rows.shape[1])} codebooks, "
                f"expected {state.n_vq}"
            )
        else:
            pass
        return rows

    def ingest(
        self, request_id: str, state: MossStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        n_vq = int(state.n_vq)
        for row in codes.detach().to(device="cpu", dtype=torch.long).unbind(0):
            state.delay_window.append(row)
            if len(state.delay_window) < n_vq:
                continue
            else:
                pass
            frame = torch.stack(
                [state.delay_window[channel][channel] for channel in range(n_vq)]
            )
            state.delay_window.popleft()
            self.ingest_raw_frame(state, frame)

    def ingest_raw_frame(self, state: MossStreamState, frame: torch.Tensor) -> None:
        if not bool(torch.all((frame >= 0) & (frame < int(state.audio_pad_code)))):
            self.close_segment(state)
            return
        else:
            pass
        if not state.segment_open and state.segment_started:
            # Note (Jiaxin Deng): each pad-separated segment decodes from a fresh
            # codec state, like the non-streaming per-segment decode.
            if state.slot is not None:
                self.ensure_session().reset_slots([state.slot])
            else:
                pass
        else:
            pass
        state.segment_open = True
        state.segment_started = True
        state.pending.append(frame)
        if state.slot is None:
            state.slot = self.ensure_session().acquire()
        else:
            pass

    def close_segment(self, state: MossStreamState) -> None:
        if not state.segment_open:
            return
        else:
            pass
        state.segment_open = False
        self.drain_pending(state)

    def drain_pending(self, state: MossStreamState) -> None:
        """Decode every buffered frame of the open segment for one request."""
        if not state.pending:
            return
        else:
            pass
        codes = torch.stack(state.pending, dim=0)
        state.pending = []
        if state.slot is None:
            state.ready_audio.append(
                self.vocoder.decode_segments([codes])[0].reshape(-1)
            )
            return
        else:
            pass
        audio = self.ensure_session().step({state.slot: codes.transpose(0, 1)})
        state.ready_audio.append(audio[state.slot].reshape(-1))

    def take_ready_audio(self, state: MossStreamState) -> torch.Tensor | None:
        parts = state.ready_audio
        state.ready_audio = []
        if not parts:
            return None
        else:
            pass
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def decode_delta(
        self, request_id: str, state: MossStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done drain; steady chunks go through the coalesced steps."""
        del request_id, is_final
        self.drain_pending(state)
        if state.slot is not None and self.session is not None:
            self.session.release(state.slot)
            state.slot = None
        else:
            pass
        return self.take_ready_audio(state)

    def pump_one_step(self) -> list[str] | None:
        # Note (Jiaxin Deng): a pad-closed segment drains into ready_audio outside
        # any step; emit it now instead of holding it until the next step or done.
        for request_id, state in self.stream_state_items():
            if not state.ready_audio or self.is_aborted(request_id):
                continue
            else:
                pass
            self.mark_stream_emitted(request_id)
            self.outbox.put(
                self.stream_chunk_message(request_id, self.take_ready_audio(state))
            )
        return super().pump_one_step()

    def select_step_participants(self) -> list[tuple[str, MossStreamState]]:
        """Streams whose buffer crossed their threshold are due; already-emitting
        peers with at least the join floor ride along in the same step."""
        slotted = [
            (request_id, state)
            for request_id, state in self.stream_state_items()
            if state.slot is not None and state.threshold > 0
        ]
        due = [
            entry for entry in slotted if len(entry[1].pending) >= entry[1].threshold
        ]
        if not due:
            return []
        else:
            pass
        floor = min(
            min(len(state.pending) for _, state in due), self.coalesce_floor_frames
        )
        return [
            (request_id, state)
            for request_id, state in slotted
            if len(state.pending) >= state.threshold
            or (self.stream_has_emitted(request_id) and len(state.pending) >= floor)
        ]

    def build_step_plan(
        self, participants: list[tuple[str, MossStreamState]]
    ) -> CoalescedStepPlan:
        step_t = min(
            min(len(state.pending) for _, state in participants),
            self.stream_chunk_frames,
        )
        for request_id, state in participants:
            if not self.stream_has_emitted(request_id):
                step_t = min(step_t, state.threshold)
            else:
                pass
        return CoalescedStepPlan(
            step_t=step_t,
            slot_codes={
                state.slot: torch.stack(state.pending[:step_t], dim=1)
                for _, state in participants
            },
        )

    def run_step(
        self,
        participants: list[tuple[str, MossStreamState]],
        plan: CoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        decoded = self.ensure_session().step(plan.slot_codes)
        out: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            del state.pending[: plan.step_t]
            state.threshold = self.stream_chunk_frames
            state.ready_audio.append(decoded[state.slot].reshape(-1))
            out[request_id] = self.take_ready_audio(state)
        return out

    def release_stream_resources(self, request_id: str, state: MossStreamState) -> None:
        del request_id
        if state.slot is not None and self.session is not None:
            self.session.release(state.slot)
            state.slot = None
        else:
            pass

    def ensure_session(self) -> CodecStreamSession:
        if self.session is None:
            self.session = CodecStreamSession(
                streaming_codec_view(self.audio_vocoder.model),
                stream_slots=self.stream_slots,
                n_vq=self.default_n_vq,
            )
        else:
            pass
        return self.session

    def warmup_now(self) -> None:
        """Capture the session's decode graphs before the stage is marked ready;
        best-effort, serving falls back to eager decode."""
        on_cuda = next(self.audio_vocoder.model.parameters()).device.type == "cuda"
        if not self.vocoder_cuda_graph or not on_cuda:
            return
        else:
            pass
        session = self.ensure_session()
        try:
            captured = session.warmup_cuda_graph(
                list(range(1, self.stream_chunk_frames + 1)),
                min_free_gb=self.vocoder_cuda_graph_min_free_gb,
            )
        except Exception:
            logger.exception(
                "MOSS-TTS streaming vocoder CUDA-graph capture failed; serving eager"
            )
            return
        logger.info("MOSS-TTS streaming vocoder CUDA graphs captured: T=%s", captured)

    def on_serving_stop(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        else:
            pass

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: MossStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        final_state, delayed_codes = self.vocoder.prepare_item(payload)
        waveform, _ = self.vocoder.decode_audio(final_state, delayed_codes)
        return waveform

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: MossStreamState
    ) -> dict[str, object]:
        del request_id
        final_state = load_moss_tts_state(payload)
        final_state.delayed_audio_codes = None
        final_state.sample_rate = int(state.sample_rate or self.sample_rate)
        data = final_state.to_dict()
        data["modality"] = "audio"
        data["sample_rate"] = final_state.sample_rate
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        return data

    async def vocode_payload(self, payload: StagePayload) -> StagePayload:
        return (await self.vocode_payloads([payload]))[0]

    async def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self.vocoder.prepare_item(payload) for payload in payloads]
        results = await self.vocoder.decode_batch(items)
        if len(results) != len(items):
            raise RuntimeError(
                f"MOSS-TTS vocoder returned {len(results)} results for "
                f"{len(items)} requests"
            )
        else:
            pass
        return [
            self.vocoder.store_result(payload, item[0], waveform, sample_rate)
            for payload, item, (waveform, sample_rate) in zip(payloads, items, results)
        ]


__all__ = ["MossStreamingVocoderScheduler"]
