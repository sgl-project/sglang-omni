# SPDX-License-Identifier: Apache-2.0
"""Streaming Zonos3 codec prefixes with stable iSTFT and resampling boundaries."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping

import torch

from sglang_omni.models.zonos3.batched_decode import Zonos3BatchedDecode
from sglang_omni.models.zonos3.components.codec import (
    CODEBOOK_COUNT,
    EXTERNAL_SAMPLES_PER_FRAME,
    SAMPLE_RATE_HZ,
    Zonos3CodecDecoder,
)
from sglang_omni.models.zonos3.components.codec_stream import (
    Zonos3DecoderSlots,
    Zonos3DecoderStream,
)
from sglang_omni.models.zonos3.request_builders import (
    Zonos3GenerationResult,
    result_metadata,
)
from sglang_omni.models.zonos3.stream_steps import (
    DecodeUrgency,
    Zonos3StreamState,
    extend_step,
    stable_sample_count,
    step_urgency,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

# note (zhaochenyang20): The model can sample end-of-audio at the first spoken frame, which leaves no frame (4 of about 560,000 SeedTTS EN requests on one H100; a seed that ends there does so streaming or not); the reference runtime keeps only the frames before the end, so such a request answers with one frame of silence, the shortest audio a stream can carry, instead of an error.
NO_SPEECH_SAMPLE_COUNT = EXTERNAL_SAMPLES_PER_FRAME
# Queued engine chunks ingested before the next decode step is ranked.
MAX_COALESCED_STREAM_CHUNKS = 256


class Zonos3StreamingVocoderScheduler(StreamingVocoderBase[Zonos3StreamState, None]):
    """Decode causal prefixes while retaining the boundary affected by future frames."""

    can_batch_stream_chunks: bool = True
    stream_chunk_batch_max: int = MAX_COALESCED_STREAM_CHUNKS
    # The serving loop drains queued chunks first; has_ready_work then ranks one decode step.
    pump_on_chunk_batch: bool = False

    def __init__(
        self,
        decoder: Zonos3CodecDecoder,
        *,
        slots: Zonos3DecoderSlots | None,
        headroom_gain: float,
        initial_chunk_frames: int,
        steady_chunk_frames: int,
        max_step_streams: int,
    ) -> None:
        self.batched_decode = Zonos3BatchedDecode(
            decoder, headroom_gain=headroom_gain, slots=slots
        )
        self.initial_chunk_frames = initial_chunk_frames
        self.steady_chunk_frames = steady_chunk_frames
        self.max_step_streams = max_step_streams
        super().__init__(
            self.decode_full,
            sample_rate=SAMPLE_RATE_HZ,
            stream_source_hint="Zonos3",
        )

    def decode_full(self, payload: StagePayload) -> StagePayload:
        """Non-streaming requests decode their whole result at once."""
        result = payload.data
        assert isinstance(result, Zonos3GenerationResult)
        if result.codes.shape[1] == 0:
            waveform = torch.zeros(NO_SPEECH_SAMPLE_COUNT)
        else:
            [waveform] = self.batched_decode.decode_waveforms(
                [result.codes],
                [(0, result.codes.shape[1] * EXTERNAL_SAMPLES_PER_FRAME)],
                exact_prefix_count=1,
                decoder_streams=[None],
            )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                **audio_waveform_payload(waveform, sample_rate=SAMPLE_RATE_HZ),
                **result_metadata(result),
            },
        )

    def create_stream_state(self, request_id: str) -> Zonos3StreamState:
        """The base keys the state by request_id."""
        return Zonos3StreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: Zonos3StreamState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        """The first payload or chunk fixes the request's initial chunk size."""
        if state.initial_chunk_frames is not None:
            return
        elif isinstance(source, StagePayload):
            request_params = source.request.params
        else:
            request_params = source
        state.initial_chunk_frames = (
            resolve_initial_codec_chunk_frames(
                request_params,
                steady_chunk_frames=self.steady_chunk_frames,
                default_frames=self.initial_chunk_frames,
            )
            or self.steady_chunk_frames
        )

    def validate_chunk(
        self, request_id: str, state: Zonos3StreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        """The engine's stream output builder emits clamped [codebooks, frames] codes."""
        assert codes.dim() == 2 and codes.shape[0] == CODEBOOK_COUNT, codes.shape
        return codes

    def ingest(
        self, request_id: str, state: Zonos3StreamState, codes: torch.Tensor
    ) -> None:
        state.code_chunks.append(codes)
        state.frame_count += codes.shape[1]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        """The final decode runs on the serving loop, which first ingests queued chunks so they can join it."""
        state = self.get_or_create_stream_state(request_id)
        if state is None:
            return []
        else:
            state.is_finished = True
            return None

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return any(
                state.is_finished for state in self.stream_states.values()
            ) or bool(self.select_step_participants())

    def run_non_streaming_batch(
        self, batch: list[IncomingMessage], loop: asyncio.AbstractEventLoop
    ) -> None:
        """A whole-result decode is followed by one ready step when a stream waits for its first audio or runs dry within PLAYBACK_DECODE_HORIZON_SECONDS."""
        super().run_non_streaming_batch(batch, loop)
        with self.state_lock:
            now_seconds = time.monotonic()
            # note (zhaochenyang20): The serving loop drains its inbox before a ready step, and queued non-streaming requests kept it full: at 48 non-streaming plus 16 streaming SeedTTS EN clients on one H100 no stream finished for 115 s; with this step streams finished at 11.4 requests per second at the same total throughput. A finished stream with audio buffered still waits for an empty inbox.
            is_stream_waiting = any(
                step_urgency(state, now_seconds)[0] != DecodeUrgency.BUFFERED
                for state in self.stream_states.values()
                if state.is_finished
            ) or bool(self.select_step_participants())
        if is_stream_waiting:
            self.run_ready_step()
        else:
            pass

    def run_ready_step(self) -> None:
        """One step led by the most urgent stream in step_urgency order; a finished lead runs its final decode, which other streams join."""
        with self.state_lock:
            now_seconds = time.monotonic()
            finished_streams = [
                (request_id, state)
                for request_id, state in self.stream_states.items()
                if state.is_finished
            ]
            final_lead = min(
                finished_streams,
                key=lambda finished_stream: step_urgency(
                    finished_stream[1], now_seconds
                ),
                default=None,
            )
            streaming_participants = self.select_step_participants()
            # note (zhaochenyang20): A first chunk that rode along a long final decode waited 93 ms on average in the vocoder; ranking finals that are neither due nor unplayed behind ready streams brought SeedTTS EN first audio from 0.294 s to 0.277 s and throughput from 22.6 to 23.5 requests per second at concurrency 16 on one H100 (4 interleaved runs each).
            if final_lead is not None and (
                not streaming_participants
                or step_urgency(final_lead[1], now_seconds)
                <= step_urgency(streaming_participants[0][1], now_seconds)
            ):
                request_id, state = final_lead
                try:
                    self.complete_stream_request(
                        request_id, self.finish_stream(request_id)
                    )
                except Exception as exc:
                    failed = self.on_step_failure([(request_id, state)], exc)
                else:
                    failed = []
            else:
                failed = self.pump_one_step() or []
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def select_step_participants(self) -> list[tuple[str, Zonos3StreamState]]:
        """A streaming step decodes only unfinished streams; a finished one waits for its final decode."""
        return extend_step(
            [],
            self.stream_states,
            self.session_appends,
            include_finished=False,
            max_step_streams=self.max_step_streams,
            steady_chunk_frames=self.steady_chunk_frames,
        )

    def stream_payload(
        self, request_id: str, waveform: torch.Tensor
    ) -> dict[str, bytes | list[int] | str | int]:
        """Each emitted chunk extends its stream's playback deadline by its duration."""
        payload = super().stream_payload(request_id, waveform)
        state = self.stream_states[request_id]
        now_seconds = time.monotonic()
        state.playback_deadline_seconds = (
            max(state.playback_deadline_seconds or now_seconds, now_seconds)
            + waveform.numel() / SAMPLE_RATE_HZ
        )
        return payload

    def build_step_plan(
        self, participants: list[tuple[str, Zonos3StreamState]]
    ) -> None:
        """A step decodes every participant's whole prefix, so it needs no plan."""
        return None

    def run_step(
        self, participants: list[tuple[str, Zonos3StreamState]], plan: None
    ) -> dict[str, torch.Tensor]:
        prefixes = [self.compact_prefix(state) for _, state in participants]
        sample_ranges = [
            (state.emitted_sample_count, stable_sample_count(state))
            for _, state in participants
        ]
        waveforms = self.batched_decode.decode_waveforms(
            prefixes,
            sample_ranges,
            exact_prefix_count=0,
            decoder_streams=[self.decoder_stream(state) for _, state in participants],
        )
        for (_, state), (_, stop_sample) in zip(
            participants, sample_ranges, strict=True
        ):
            state.emitted_sample_count = stop_sample
        return {
            request_id: waveform
            for (request_id, _), waveform in zip(participants, waveforms, strict=True)
        }

    def decoder_stream(self, state: Zonos3StreamState) -> Zonos3DecoderStream | None:
        """A stream takes a free slot at its first decode, unless that decode is its last; it keeps the slot to its end."""
        slots = self.batched_decode.slots
        if (
            slots is not None
            and state.decoder_stream is None
            and state.emitted_sample_count == 0
            and not state.is_finished
        ):
            with (
                self.batched_decode.decode_lock,
                self.batched_decode.stream_context(),
            ):
                state.decoder_stream = slots.acquire()
        else:
            pass
        return state.decoder_stream

    def release_stream_resources(
        self, request_id: str, state: Zonos3StreamState
    ) -> None:
        """A completed or aborted stream frees its slot; its last decode has finished on the device."""
        if state.decoder_stream is None:
            pass
        else:
            assert self.batched_decode.slots is not None
            self.batched_decode.slots.release(state.decoder_stream)
            state.decoder_stream = None

    def compact_prefix(self, state: Zonos3StreamState) -> torch.Tensor:
        """The stream's codes [codebooks, frames] so far, kept as one chunk for the next decode."""
        prefix = torch.cat(state.code_chunks, dim=1)
        state.code_chunks = [prefix]
        return prefix

    def decode_delta(
        self, request_id: str, state: Zonos3StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """The final decode flushes the held-back boundary from the finished result's codes; ready and finished streams join it, and earlier decodes run in run_step."""
        assert is_final
        result = self.stream_payloads[request_id].data
        assert isinstance(result, Zonos3GenerationResult)
        final_sample_count = result.codes.shape[1] * EXTERNAL_SAMPLES_PER_FRAME
        assert final_sample_count >= state.emitted_sample_count
        if final_sample_count == 0:
            return torch.zeros(NO_SPEECH_SAMPLE_COUNT)
        elif final_sample_count == state.emitted_sample_count:
            return None
        else:
            pass
        companions = extend_step(
            [(request_id, state)],
            self.stream_states,
            self.session_appends,
            include_finished=True,
            max_step_streams=self.max_step_streams,
            steady_chunk_frames=self.steady_chunk_frames,
        )[1:]
        finished_companions = [
            (companion_request_id, companion_state)
            for companion_request_id, companion_state in companions
            if companion_state.is_finished
        ]
        streaming_companions = [
            (companion_request_id, companion_state)
            for companion_request_id, companion_state in companions
            if not companion_state.is_finished
        ]
        finished_codes: list[torch.Tensor] = []
        for companion_request_id, _ in finished_companions:
            companion_result = self.stream_payloads[companion_request_id].data
            assert isinstance(companion_result, Zonos3GenerationResult)
            finished_codes.append(companion_result.codes)
        companion_ranges = [
            (
                companion_state.emitted_sample_count,
                codes.shape[1] * EXTERNAL_SAMPLES_PER_FRAME,
            )
            for (_, companion_state), codes in zip(
                finished_companions, finished_codes, strict=True
            )
        ] + [
            (
                companion_state.emitted_sample_count,
                stable_sample_count(companion_state),
            )
            for _, companion_state in streaming_companions
        ]
        waveform, *companion_waveforms = self.batched_decode.decode_waveforms(
            [result.codes]
            + finished_codes
            + [
                self.compact_prefix(companion_state)
                for _, companion_state in streaming_companions
            ],
            [(state.emitted_sample_count, final_sample_count)] + companion_ranges,
            exact_prefix_count=1 + len(finished_companions),
            decoder_streams=[
                self.decoder_stream(stream_state)
                for _, stream_state in [(request_id, state)]
                + finished_companions
                + streaming_companions
            ],
        )
        state.emitted_sample_count = final_sample_count
        for (
            (companion_request_id, companion_state),
            (_, companion_stop_sample),
            companion_waveform,
        ) in zip(
            finished_companions + streaming_companions,
            companion_ranges,
            companion_waveforms,
            strict=True,
        ):
            companion_state.emitted_sample_count = companion_stop_sample
            self.mark_stream_emitted(companion_request_id)
            if companion_state.is_finished:
                self.complete_stream_request(
                    companion_request_id,
                    [
                        self.stream_chunk_message(
                            companion_request_id, companion_waveform
                        ),
                        *self.terminal_messages(
                            companion_request_id,
                            self.stream_payloads[companion_request_id],
                            companion_state,
                        ),
                    ],
                )
            else:
                self.outbox.put(
                    self.stream_chunk_message(companion_request_id, companion_waveform)
                )
        return waveform

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: Zonos3StreamState
    ) -> dict[str, str | int | dict[str, int]]:
        result = payload.data
        assert isinstance(result, Zonos3GenerationResult)
        return result_metadata(result)
