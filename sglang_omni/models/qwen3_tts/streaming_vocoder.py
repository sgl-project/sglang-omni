# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Qwen3-TTS."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Mapping

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.cuda_staging import GrowablePinnedBuffer, PinnedTransferSlot

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_TTS_STREAM_STRIDE = 16
DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE = 8
DEFAULT_QWEN3_TTS_STREAM_INITIAL_FOLLOWUP_STRIDE = 8
DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES = 8
DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES = 16
_QWEN3_TTS_CODEBOOK_SIZE = 2048


@dataclass
class _Qwen3TTSStreamState:
    code_chunks: list[torch.Tensor] = field(default_factory=list)
    total_frames: int = 0
    pruned_frames: int = 0
    ref_frames: int = 0
    emitted_generated_frames: int = 0
    next_decode_generated_frames: int = 0
    decoded_chunks: int = 0
    num_quantizers: int | None = None
    pending_ref_frames: int = 0
    initial_chunk_frames: int = DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES
    initial_pending: bool = False
    followup_pending: bool = False
    final_pending: bool = False
    playback_deadline_s: float = 0.0
    incremental_codec_state: Qwen3TTSIncrementalCodecState | None = None
    incremental_codec_fallback: bool = False


class _Qwen3TTSInvalidCodeRows(ValueError):
    """Carries which rows of a decode batch held out-of-range codec ids."""

    def __init__(self, indices: list[int], message: str) -> None:
        super().__init__(message)
        self.indices = tuple(indices)


@dataclass(frozen=True)
class _Qwen3TTSDecodePlan:
    decoder_input: torch.Tensor
    absolute_emitted_frames: int
    generated_frames: int
    window_start: int
    emitted_generated_frames: int


def _bad_row_message(indices: list[int] | tuple[int, ...]) -> str:
    return (
        "Qwen3-TTS decoder input contains codec ids outside "
        f"[0, {_QWEN3_TTS_CODEBOOK_SIZE}) in rows {list(indices)}"
    )


def _raise_for_bad_rows(bad_rows: Any, count: int) -> None:
    indices = bad_rows[:count].nonzero().flatten().tolist()
    if not indices:
        return
    raise _Qwen3TTSInvalidCodeRows(indices, _bad_row_message(indices))


@dataclass(eq=False)
class _DecodeSlot:
    """Per-thread pinned transfer resources for one in-flight decode group.

    Data flow:
        CPU codes [B, Q, T] -> input_codes (pinned) -> CUDA decoder
        CUDA audio deltas [S] -> output_transfer (pinned + completion event)
            -> independent CPU tensors (not pinned)

    ``busy`` is set from acquisition until the handle that owns the slot
    releases it. ``broken`` is sticky: the slot is never acquired, grown, or
    reused again. A slot that is both busy and broken belongs to a decode
    whose CUDA completion could not be proven; it is retained for the rest
    of the process.
    """

    input_codes: GrowablePinnedBuffer
    output_transfer: PinnedTransferSlot
    busy: bool = False
    broken: bool = False


@dataclass(eq=False)
class _RetainedDecodeResources:
    """Strong references kept alive when CUDA completion could not be proven."""

    owner: Any
    stream: Any
    slot: _DecodeSlot | None
    decoder_input: torch.Tensor | None
    keepalives: list[Any]


# Note (jiannan-17): when neither the completion event nor the exact decode
# stream can be synchronized, nothing proves the GPU is done with the pinned
# buffers, the decoder input, or the decoder output of that decode. They are
# kept here for the rest of the process so the allocator cannot hand their
# memory to later work; recovery needs a process restart, not slot reuse.
_CONTEXT_FATAL_RETAINED: list[_RetainedDecodeResources] = []


@dataclass(eq=False)
class _Qwen3TTSDecodeHandle:
    """Result of one launched decode group.

    A pending handle owns its thread's ``_DecodeSlot`` until ``resolve()``
    returns or raises. ``resolve()`` is terminal: it waits for the completion
    event, materializes independent CPU deltas, releases the slot, and raises
    ``_Qwen3TTSInvalidCodeRows`` for rows that held out-of-range codec ids, so
    no caller can emit audio decoded from clamped codes. Later calls return
    the cached deltas or raise again without touching the event or slot.
    """

    deltas: list[torch.Tensor]
    bad_rows: torch.Tensor | None
    slot: _DecodeSlot | None = None
    owner: Any = None
    stream: Any = None
    decoder_input_keepalive: torch.Tensor | None = None
    keepalives: list[Any] = field(default_factory=list)
    _done: bool = field(default=False, init=False, repr=False)
    _failure: str | None = field(default=None, init=False, repr=False)
    _bad_row_indices: tuple[int, ...] | None = field(
        default=None, init=False, repr=False
    )

    def resolve(self) -> list[torch.Tensor]:
        """Wait for completion, release the slot, and return owned CPU deltas."""
        if self._done:
            if self._bad_row_indices is not None:
                raise _Qwen3TTSInvalidCodeRows(
                    list(self._bad_row_indices),
                    _bad_row_message(self._bad_row_indices),
                )
            if self._failure is not None:
                raise RuntimeError(
                    "Qwen3-TTS decode handle resolution previously failed: "
                    f"{self._failure}"
                )
            return self.deltas
        self._done = True
        try:
            if self.slot is not None:
                self._wait_and_release()
            if self.bad_rows is not None:
                bad_rows = self.bad_rows
                self.bad_rows = None
                indices = bad_rows[: len(self.deltas)].nonzero().flatten().tolist()
                if indices:
                    self._bad_row_indices = tuple(indices)
                    self.deltas = []
                    raise _Qwen3TTSInvalidCodeRows(indices, _bad_row_message(indices))
            return self.deltas
        except BaseException as exc:
            if self._bad_row_indices is None:
                self._failure = f"{type(exc).__name__}: {exc}"
            raise

    def _wait_and_release(self) -> None:
        slot = self.slot
        assert slot is not None
        try:
            slot.output_transfer.synchronize()
        except BaseException as event_exc:
            try:
                self.stream.synchronize()
            except BaseException:
                # Note (jiannan-17): neither the event nor the stream could be
                # synchronized, so GPU completion is unknown. Keep everything
                # this decode touched alive for the rest of the process and
                # stop issuing CUDA decodes from this vocoder.
                slot.broken = True
                if self.owner is not None:
                    self.owner._cuda_decode_failed = True
                _CONTEXT_FATAL_RETAINED.append(
                    _RetainedDecodeResources(
                        owner=self.owner,
                        stream=self.stream,
                        slot=slot,
                        decoder_input=self.decoder_input_keepalive,
                        keepalives=[*self.keepalives, *self.deltas],
                    )
                )
                logger.error(
                    "Qwen3-TTS decode event and stream synchronization both "
                    "failed; disabling CUDA decode and retaining the in-flight "
                    "buffers",
                    exc_info=True,
                )
                raise event_exc
            # Note (jiannan-17): the stream drained, so the buffers are safe
            # to release, but the event that failed is never trusted again.
            self._drop_views()
            slot.broken = True
            slot.busy = False
            self.slot = None
            logger.warning(
                "Qwen3-TTS decode event synchronization failed; the staging "
                "slot will not be reused",
                exc_info=True,
            )
            raise
        try:
            self.deltas = [delta.clone() for delta in self.deltas]
        except BaseException:
            self.deltas = []
            raise
        finally:
            self.keepalives.clear()
            self.decoder_input_keepalive = None
            slot.busy = False
            self.slot = None

    def _drop_views(self) -> None:
        self.deltas = []
        self.keepalives.clear()
        self.decoder_input_keepalive = None


_ASYNC_STOP = None


class _Qwen3TTSInitialDecodeGraphs:
    """CUDA graphs for fixed shape streaming decodes, one holder per stream."""

    def __init__(
        self,
        decoder: Any,
        *,
        device: torch.device,
        num_quantizers: int,
        input_frames: int | tuple[int, ...],
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        enabled: bool = True,
    ) -> None:
        self._decoder = decoder
        self._device = device
        self._num_quantizers = int(num_quantizers)
        frames = (
            input_frames if isinstance(input_frames, (tuple, list)) else (input_frames,)
        )
        self._input_frames = tuple(sorted(set(int(f) for f in frames if int(f) > 0)))
        self._batch_sizes = tuple(sorted(set(int(size) for size in batch_sizes)))
        self._enabled = bool(enabled and device.type == "cuda")
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._inputs: dict[tuple[int, int], torch.Tensor] = {}
        self._outputs: dict[tuple[int, int], torch.Tensor] = {}

    def capture(self) -> None:
        if not self._enabled or self._graphs:
            return
        capture_stream = torch.cuda.Stream(device=self._device)
        graph_pool = torch.cuda.graph_pool_handle()
        for input_frames, batch_size in (
            (f, b) for f in self._input_frames for b in self._batch_sizes
        ):
            try:
                static_input = torch.zeros(
                    (batch_size, self._num_quantizers, input_frames),
                    dtype=torch.long,
                    device=self._device,
                )
                capture_stream.wait_stream(torch.cuda.current_stream(self._device))
                with torch.inference_mode(), torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        self._decoder(static_input)
                capture_stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(
                        graph,
                        pool=graph_pool,
                        stream=capture_stream,
                    ),
                ):
                    static_output = self._decoder(static_input)
            except Exception:
                logger.warning(
                    "Qwen3-TTS decoder graph capture failed for frames=%d batch=%d",
                    input_frames,
                    batch_size,
                    exc_info=True,
                )
                continue
            self._graphs[(input_frames, batch_size)] = graph
            self._inputs[(input_frames, batch_size)] = static_input
            self._outputs[(input_frames, batch_size)] = static_output
        if self._graphs:
            logger.info(
                "Qwen3-TTS decoder graphs captured for (frames, batch) %s",
                sorted(self._graphs),
            )

    def decode(self, codes: torch.Tensor) -> torch.Tensor | None:
        if (
            not self._graphs
            or codes.ndim != 3
            or int(codes.shape[1]) != self._num_quantizers
            or int(codes.shape[2]) not in self._input_frames
        ):
            return None
        batch_size = int(codes.shape[0])
        bucket = next((size for size in self._batch_sizes if size >= batch_size), None)
        key = (int(codes.shape[2]), bucket) if bucket is not None else None
        if key is None or key not in self._graphs:
            return None
        static_input = self._inputs[key]
        static_input.zero_()
        static_input[:batch_size].copy_(codes)
        self._graphs[key].replay()
        return self._outputs[key][:batch_size].clone()


class Qwen3TTSStreamingVocoderScheduler(
    StreamingVocoderBase[_Qwen3TTSStreamState, None]
):
    """Decode Qwen3-TTS codec frames on a priority CUDA stream."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        device: str,
        stream_stride: int = DEFAULT_QWEN3_TTS_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE,
        stream_initial_followup_stride: int | None = None,
        initial_chunk_frames: int | None = None,
        stream_chunk_ramp: tuple[int, ...] | list[int] | None = None,
        stream_left_context_frames: int = DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        async_decode: bool | None = None,
        initial_max_batch_size: int = 32,
        initial_batch_wait_ms: int = 2,
        followup_max_batch_size: int = 8,
        followup_batch_wait_ms: int = 1,
        initial_cuda_graph: bool = True,
        enable_deterministic_inference: bool = False,
        followup_cuda_graph: bool = True,
        fused_snake_activation: bool = False,
        enable_stateful_codec_decoder: bool = False,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream strides must be > 0")
        if (
            stream_initial_followup_stride is not None
            and stream_initial_followup_stride <= 0
        ):
            raise ValueError("stream_initial_followup_stride must be > 0")
        if initial_chunk_frames is not None and initial_chunk_frames < 0:
            raise ValueError("initial_chunk_frames must be >= 0")
        if stream_chunk_ramp is not None:
            # note (Junnan Li): the ramp is the generalized form of the two
            # legacy knobs; a mixed configuration has no single source of
            # truth, so refuse it.
            if (
                stream_initial_followup_stride is not None
                or initial_chunk_frames is not None
            ):
                raise ValueError(
                    "stream_chunk_ramp replaces initial_chunk_frames and "
                    "stream_initial_followup_stride; set only one form"
                )
            if not isinstance(stream_chunk_ramp, (tuple, list)):
                raise TypeError("stream_chunk_ramp must be a tuple or list of ints")
            if not stream_chunk_ramp:
                raise ValueError("stream_chunk_ramp must contain at least one entry")
            if any(
                isinstance(frames, bool) or not isinstance(frames, int)
                for frames in stream_chunk_ramp
            ):
                raise TypeError("stream_chunk_ramp entries must be ints")
            chunk_ramp = tuple(int(frames) for frames in stream_chunk_ramp)
            if any(frames <= 0 for frames in chunk_ramp):
                raise ValueError("stream_chunk_ramp entries must be > 0")
            # note (Junnan Li): the request-time resolver clamps the first
            # chunk to the steady stride, so a larger configured value would
            # silently run a different schedule with unusable graph shapes.
            if chunk_ramp[0] > stream_stride:
                raise ValueError("stream_chunk_ramp[0] must be <= stream_stride")
            initial_chunk_frames = chunk_ramp[0]
            followup_stride_ramp = chunk_ramp[1:]
        else:
            if initial_chunk_frames is None:
                initial_chunk_frames = DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES
            followup_stride_ramp = (
                (
                    min(
                        DEFAULT_QWEN3_TTS_STREAM_INITIAL_FOLLOWUP_STRIDE,
                        stream_followup_stride,
                    )
                    if stream_initial_followup_stride is None
                    else stream_initial_followup_stride
                ),
            )
        if stream_left_context_frames < 0:
            raise ValueError("stream_left_context_frames must be >= 0")
        if initial_max_batch_size <= 0 or followup_max_batch_size <= 0:
            raise ValueError("async batch sizes must be > 0")
        if initial_batch_wait_ms < 0 or followup_batch_wait_ms < 0:
            raise ValueError("async batch waits must be >= 0")
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._decoder = tokenizer.model.decoder
        if fused_snake_activation:
            from sglang_omni.models.qwen3_tts.vocoder_kernels import (
                fuse_vocoder_decoder,
            )

            logger.info(
                "Qwen3-TTS vocoder fused SnakeBeta modules: %d",
                fuse_vocoder_decoder(self._decoder),
            )
        tokenizer_config = getattr(tokenizer.model, "config", None)
        decoder_config = getattr(tokenizer_config, "decoder_config", tokenizer_config)
        num_quantizers = int(getattr(decoder_config, "num_quantizers", 0) or 0)
        self._deterministic_inference = bool(enable_deterministic_inference)
        self._enable_stateful_codec_decoder = bool(enable_stateful_codec_decoder)
        self._incremental_decoder = (
            Qwen3TTSIncrementalDecoder(self._decoder)
            if self._enable_stateful_codec_decoder
            else None
        )
        self._initial_decode_graphs = _Qwen3TTSInitialDecodeGraphs(
            self._decoder,
            device=self._device,
            num_quantizers=num_quantizers,
            input_frames=int(stream_left_context_frames) + int(initial_chunk_frames),
            batch_sizes=(1,) if self._deterministic_inference else (1, 2, 4, 8),
            enabled=bool(
                initial_cuda_graph
                and num_quantizers > 0
                and not self._enable_stateful_codec_decoder
            ),
        )
        # note (Junnan Li): windows truncated below the full left context
        # (short or absent reference codes early in a stream) fall back to
        # eager decode, as the legacy first follow-up already does.
        followup_frames = tuple(
            int(stream_left_context_frames) + int(stride)
            for stride in (*followup_stride_ramp, stream_followup_stride)
        )
        self._followup_decode_graphs = _Qwen3TTSInitialDecodeGraphs(
            self._decoder,
            device=self._device,
            num_quantizers=num_quantizers,
            input_frames=followup_frames,
            batch_sizes=(1,) if self._deterministic_inference else (1, 2, 4, 8),
            enabled=bool(
                followup_cuda_graph
                and num_quantizers > 0
                and not self._enable_stateful_codec_decoder
            ),
        )
        self._samples_per_frame = int(self._decoder.total_upsample)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        # note (Junnan Li): ``_followup_stride_ramp[i]`` sizes decode chunk
        # ``i + 2``; past the ramp the steady stride takes over.
        self._followup_stride_ramp = tuple(
            int(stride) for stride in followup_stride_ramp
        )
        self._chunk_ramp_configured = stream_chunk_ramp is not None
        self._initial_max_batch_size = int(initial_max_batch_size)
        self._initial_batch_wait_s = float(initial_batch_wait_ms) / 1000.0
        self._followup_max_batch_size = int(followup_max_batch_size)
        self._followup_batch_wait_s = float(followup_batch_wait_ms) / 1000.0
        self._default_initial_chunk_frames = int(initial_chunk_frames)
        self._stream_left_context_frames = int(stream_left_context_frames)
        self._async_decode = (
            False
            if self._enable_stateful_codec_decoder
            else (
                self._device.type == "cuda"
                if async_decode is None
                else bool(async_decode)
            )
        )
        self._decode_staging = threading.local()
        self._pinned_staging_disabled = self._device.type != "cuda"
        self._cuda_decode_failed = False
        if self._device.type == "cuda":
            least_priority, greatest_priority = torch.cuda.Stream.priority_range()
            followup_priority = min(least_priority, greatest_priority + 1)
            self._decode_stream = torch.cuda.Stream(
                device=self._device,
                priority=followup_priority,
            )
            self._followup_decode_stream = (
                torch.cuda.Stream(
                    device=self._device,
                    priority=followup_priority,
                )
                if self._async_decode
                else None
            )
        else:
            self._decode_stream = None
            self._followup_decode_stream = None
        self._initial_queue: queue.Queue[tuple[str, _Qwen3TTSStreamState] | None] = (
            queue.Queue()
        )
        self._followup_queue: queue.PriorityQueue[
            tuple[float, int, str, _Qwen3TTSStreamState | None]
        ] = queue.PriorityQueue()
        self._followup_sequence = count()
        self._async_stop = threading.Event()
        self._initial_worker: threading.Thread | None = None
        self._followup_worker: threading.Thread | None = None
        sample_rate = int(tokenizer.get_output_sample_rate())

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=sample_rate,
            stream_source_hint="Qwen3-TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def start(self) -> None:
        try:
            super().start()
        finally:
            self._join_async_workers()

    def stop(self) -> None:
        self._signal_async_stop()
        super().stop()
        self._join_async_workers()

    def warmup_now(self) -> None:
        if not self._async_decode:
            return
        self._initial_decode_graphs.capture()
        self._followup_decode_graphs.capture()

    def on_serving_start(self) -> None:
        if not self._async_decode:
            return
        self._initial_queue = queue.Queue()
        self._followup_queue = queue.PriorityQueue()
        self._followup_sequence = count()
        self._async_stop.clear()
        self._initial_worker = threading.Thread(
            target=self._run_initial_worker,
            name="qwen3-tts-vocoder-initial",
            daemon=True,
        )
        self._followup_worker = threading.Thread(
            target=self._run_followup_worker,
            name="qwen3-tts-vocoder-followup",
            daemon=True,
        )
        self._initial_worker.start()
        self._followup_worker.start()

    def on_serving_stop(self) -> None:
        self._signal_async_stop()

    def _signal_async_stop(self) -> None:
        if self._async_stop.is_set():
            return
        self._async_stop.set()
        if self._initial_worker is not None:
            self._initial_queue.put(_ASYNC_STOP)
        if self._followup_worker is not None:
            self._followup_queue.put(
                (float("inf"), next(self._followup_sequence), "", _ASYNC_STOP)
            )

    def _join_async_workers(self) -> None:
        for worker in (self._initial_worker, self._followup_worker):
            if worker is not None and worker is not threading.current_thread():
                worker.join()
        self._initial_worker = None
        self._followup_worker = None

    def create_stream_state(self, request_id: str) -> _Qwen3TTSStreamState:
        del request_id
        return _Qwen3TTSStreamState(
            initial_chunk_frames=self._default_initial_chunk_frames
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            params = source.request.params
            if isinstance(params, Mapping):
                state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                    params,
                    steady_chunk_frames=self._stream_stride,
                    default_frames=self._default_initial_chunk_frames,
                )
            return

        metadata: Mapping[str, Any] = source
        if "num_quantizers" not in metadata and state.num_quantizers is None:
            raise RuntimeError(
                f"Qwen3-TTS stream chunk for {request_id!r} is missing num_quantizers"
            )
        if "num_quantizers" in metadata:
            num_quantizers = int(metadata["num_quantizers"])
            if num_quantizers <= 0:
                raise ValueError("Qwen3-TTS num_quantizers must be > 0")
            if (
                state.num_quantizers is not None
                and state.num_quantizers != num_quantizers
            ):
                raise ValueError(
                    f"Qwen3-TTS num_quantizers changed for {request_id!r}: "
                    f"{state.num_quantizers} -> {num_quantizers}"
                )
            state.num_quantizers = num_quantizers
        if "ref_code_len" in metadata:
            ref_frames = int(metadata["ref_code_len"])
            if ref_frames < 0:
                raise ValueError("Qwen3-TTS ref_code_len must be >= 0")
            if state.total_frames or state.ref_frames:
                raise ValueError(
                    f"Qwen3-TTS reference codes arrived after stream start for "
                    f"{request_id!r}"
                )
            state.pending_ref_frames = ref_frames
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                metadata,
                steady_chunk_frames=self._stream_stride,
                default_frames=self._default_initial_chunk_frames,
            )

    def validate_chunk(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        chunk = codes.detach().to(dtype=torch.long)
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        elif chunk.ndim != 2:
            raise ValueError(
                f"Qwen3-TTS stream chunk must be [Q] or [T, Q], "
                f"got {tuple(chunk.shape)}"
            )
        if chunk.shape[0] == 0:
            raise ValueError("Qwen3-TTS stream chunk must not be empty")
        if state.num_quantizers is None:
            raise RuntimeError(
                f"Qwen3-TTS stream contract for {request_id!r} is missing "
                "num_quantizers"
            )
        if int(chunk.shape[1]) != state.num_quantizers:
            raise ValueError(
                f"Qwen3-TTS stream chunk has {int(chunk.shape[1])} quantizers, "
                f"expected {state.num_quantizers}"
            )
        if not chunk.is_cuda and (
            bool((chunk < 0).any()) or bool((chunk >= _QWEN3_TTS_CODEBOOK_SIZE).any())
        ):
            raise ValueError(
                f"Qwen3-TTS stream chunk for {request_id!r} contains codec ids "
                f"outside [0, {_QWEN3_TTS_CODEBOOK_SIZE})"
            )
        return chunk

    def ingest(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        if state.pending_ref_frames:
            if state.pending_ref_frames >= int(codes.shape[0]):
                raise ValueError(
                    "Qwen3-TTS first stream chunk must include at least one "
                    "generated codec frame after the reference"
                )
            state.ref_frames = state.pending_ref_frames
            state.pending_ref_frames = 0
        state.code_chunks.append(codes)
        state.total_frames += int(codes.shape[0])

    def should_decode(self, state: _Qwen3TTSStreamState, *, is_final: bool) -> bool:
        if is_final:
            return True
        generated_frames = state.total_frames - state.ref_frames
        next_frames = self._next_decode_threshold(state)
        return generated_frames >= next_frames

    def _next_decode_threshold(self, state: _Qwen3TTSStreamState) -> int:
        if state.next_decode_generated_frames:
            return state.next_decode_generated_frames
        return state.initial_chunk_frames or self._stream_stride

    def decode_delta(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        force_legacy_decode = False
        if self._enable_stateful_codec_decoder and not state.incremental_codec_fallback:
            try:
                incremental = self._decode_incremental_eager(state)
            except Exception:
                state.incremental_codec_fallback = True
                force_legacy_decode = True
                logger.warning(
                    "Qwen3-TTS stateful codec decode failed for %r; using the "
                    "legacy left-context decoder for the rest of the request",
                    request_id,
                    exc_info=True,
                )
            else:
                if incremental is None:
                    return None
                plan, candidate_state, delta = incremental
                delta = self._commit_decode_plan(state, plan, delta)
                state.incremental_codec_state = candidate_state
                self._prune_incremental_codes(state)
                return delta

        plan = self._build_decode_plan(state, is_final=is_final or force_legacy_decode)
        if plan is None:
            return None
        handle = self._launch_decode_plans([plan], stream=self._decode_stream)
        deltas = handle.resolve()
        return self._commit_decode_plan(state, plan, deltas[0])

    def _decode_incremental_eager(
        self,
        state: _Qwen3TTSStreamState,
    ) -> (
        tuple[
            _Qwen3TTSDecodePlan,
            Qwen3TTSIncrementalCodecState,
            torch.Tensor,
        ]
        | None
    ):
        available_generated_frames = state.total_frames - state.ref_frames
        if available_generated_frames <= state.emitted_generated_frames:
            return None

        committed_state = state.incremental_codec_state
        if committed_state is None:
            if state.emitted_generated_frames:
                raise RuntimeError(
                    "Qwen3-TTS incremental codec state is missing after emitted frames"
                )
            candidate_state = Qwen3TTSIncrementalCodecState()
        else:
            candidate_state = committed_state.clone()
        consumed_frames = candidate_state.frame_position
        expected_consumed_frames = state.ref_frames + state.emitted_generated_frames
        if committed_state is not None and consumed_frames != expected_consumed_frames:
            raise RuntimeError(
                "Qwen3-TTS incremental codec position does not match emitted frames"
            )
        end_frame = state.ref_frames + available_generated_frames
        if consumed_frames < state.pruned_frames:
            raise RuntimeError(
                "Qwen3-TTS incremental codec codes were pruned too early"
            )
        codes = torch.cat(state.code_chunks, dim=0)
        decoder_input = (
            codes[
                consumed_frames - state.pruned_frames : end_frame - state.pruned_frames
            ]
            .transpose(0, 1)
            .unsqueeze(0)
        )
        bad_rows = self._screen_out_of_range_codes(decoder_input)
        _raise_for_bad_rows(bad_rows, 1)
        incremental_decoder = self._incremental_decoder
        if incremental_decoder is None:
            raise RuntimeError("Qwen3-TTS incremental codec decoder is unavailable")
        with torch.inference_mode():
            waveform = incremental_decoder.decode(
                decoder_input.to(self._device), candidate_state
            )
        if candidate_state.frame_position != end_frame:
            raise RuntimeError(
                "Qwen3-TTS incremental codec position did not advance to the decode end"
            )
        waveform = self._split_batch_waveform(waveform, 1)[0]
        reference_frames = max(0, state.ref_frames - consumed_frames)
        trim_samples = reference_frames * self._samples_per_frame
        emit_frames = available_generated_frames - state.emitted_generated_frames
        emit_samples = emit_frames * self._samples_per_frame
        delta = (
            waveform[trim_samples : trim_samples + emit_samples]
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
        )
        if int(delta.numel()) != emit_samples:
            raise RuntimeError(
                "Qwen3-TTS incremental codec decoder returned the wrong delta length"
            )
        plan = _Qwen3TTSDecodePlan(
            decoder_input=decoder_input,
            absolute_emitted_frames=expected_consumed_frames,
            generated_frames=available_generated_frames,
            window_start=consumed_frames,
            emitted_generated_frames=state.emitted_generated_frames,
        )
        return plan, candidate_state, delta

    def _prune_incremental_codes(self, state: _Qwen3TTSStreamState) -> None:
        committed_state = state.incremental_codec_state
        assert committed_state is not None
        retention_start = max(
            0,
            committed_state.frame_position - self._stream_left_context_frames,
        )
        while (
            state.code_chunks
            and state.pruned_frames + int(state.code_chunks[0].shape[0])
            <= retention_start
        ):
            state.pruned_frames += int(state.code_chunks.pop(0).shape[0])

    def _build_decode_plan(
        self,
        state: _Qwen3TTSStreamState,
        *,
        is_final: bool,
        max_generated_frames: int | None = None,
    ) -> _Qwen3TTSDecodePlan | None:
        available_generated_frames = state.total_frames - state.ref_frames
        if available_generated_frames <= state.emitted_generated_frames:
            return None
        next_frames = self._next_decode_threshold(state)
        if not is_final and available_generated_frames < next_frames:
            state.next_decode_generated_frames = next_frames
            return None

        generated_frames = available_generated_frames
        if max_generated_frames is not None:
            generated_frames = min(generated_frames, max_generated_frames)

        absolute_emitted = state.ref_frames + state.emitted_generated_frames
        window_start = max(0, absolute_emitted - self._stream_left_context_frames)
        window_end = state.ref_frames + generated_frames
        # Note (Jiaxin Deng): window_start only moves forward, so frames behind
        # it are dead; prune whole chunks to keep this cat O(window), not
        # O(stream) per decode. Slices below translate by the pruned offset.
        while (
            state.code_chunks
            and state.pruned_frames + int(state.code_chunks[0].shape[0]) <= window_start
        ):
            state.pruned_frames += int(state.code_chunks.pop(0).shape[0])
        codes = torch.cat(state.code_chunks, dim=0)
        decoder_input = (
            codes[window_start - state.pruned_frames : window_end - state.pruned_frames]
            .transpose(0, 1)
            .unsqueeze(0)
        )
        return _Qwen3TTSDecodePlan(
            decoder_input=decoder_input,
            absolute_emitted_frames=absolute_emitted,
            generated_frames=generated_frames,
            window_start=window_start,
            emitted_generated_frames=state.emitted_generated_frames,
        )

    def _screen_out_of_range_codes(self, decoder_input: torch.Tensor) -> Any:
        # Note (Jiaxin Deng): an out-of-range id makes the codec embedding lookup
        # raise a device-side assert, which poisons the CUDA context and kills
        # every in-flight stream in this process; validate_chunk cannot catch it
        # because it skips device tensors. Clamp into range so the lookup is
        # always safe, and return the per-row verdict: the CPU and deterministic
        # multi-plan paths check it before decoding, the async CUDA path reads it
        # back inside ``resolve()`` once the completion event has fired, so no
        # added synchronization buys the same protection. Rows that needed
        # clamping are failed, never emitted.
        bad_rows = (
            ((decoder_input < 0) | (decoder_input >= _QWEN3_TTS_CODEBOOK_SIZE))
            .flatten(start_dim=1)
            .any(dim=1)
        )
        decoder_input.clamp_(0, _QWEN3_TTS_CODEBOOK_SIZE - 1)
        return bad_rows

    def _launch_decode_plans(
        self,
        plans: list[_Qwen3TTSDecodePlan],
        *,
        stream: torch.cuda.Stream | None,
    ) -> _Qwen3TTSDecodeHandle:
        """Launch one decode batch and return its handle.

        Asynchronous CUDA path:
            stage CPU input -> run decoder -> extract audio deltas
            -> copy deltas into the thread's pinned slot -> record its event

        Return behavior:
            asynchronous CUDA path -> return a pending handle that owns the slot
            CPU or pageable CUDA fallback -> return a complete handle
            deterministic multi-plan mode -> resolve each plan before returning

        ``resolve()`` raises ``_Qwen3TTSInvalidCodeRows`` for rows that held
        out-of-range codec ids. The CPU and deterministic multi-plan paths
        raise it here instead, before anything is decoded.
        """
        if stream is not None and self._cuda_decode_failed:
            raise RuntimeError(
                "Qwen3-TTS CUDA decode is disabled after an unrecoverable "
                "stream failure"
            )
        if self._deterministic_inference and len(plans) > 1:
            # Note (jiannan-17): in deterministic mode, decode each plan at
            # B=1 so its output does not depend on the other requests in the
            # batch (#1475). Validate the combined batch before the per-plan
            # loop so bad-row indices still refer to the original group.
            decoder_input = torch.cat([plan.decoder_input for plan in plans], dim=0)
            bad_rows = self._screen_out_of_range_codes(decoder_input)
            _raise_for_bad_rows(bad_rows, len(plans))
            deltas: list[torch.Tensor] = []
            for plan in plans:
                single = self._launch_decode_plans([plan], stream=stream)
                deltas.extend(single.resolve())
            return _Qwen3TTSDecodeHandle(deltas, bad_rows=None)

        decoder_input = torch.cat([plan.decoder_input for plan in plans], dim=0)
        bad_rows = self._screen_out_of_range_codes(decoder_input)
        with torch.inference_mode():
            if stream is None:
                _raise_for_bad_rows(bad_rows, len(plans))
                waveforms = self._split_batch_waveform(
                    self._decoder.chunked_decode(decoder_input), len(plans)
                )
                return _Qwen3TTSDecodeHandle(
                    [
                        self._extract_delta(plan, waveform)
                        .detach()
                        .to(torch.float32)
                        .contiguous()
                        for plan, waveform in zip(plans, waveforms)
                    ],
                    bad_rows=None,
                )
            return self._launch_async(plans, decoder_input, bad_rows, stream)

    def _launch_async(
        self,
        plans: list[_Qwen3TTSDecodePlan],
        decoder_input: torch.Tensor,
        bad_rows: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> _Qwen3TTSDecodeHandle:
        slot = self._thread_decode_slot()
        pinned = self._reserve_slot(
            slot,
            input_numel=(
                0
                if decoder_input.device.type == self._device.type
                else int(decoder_input.numel())
            ),
            output_numel=(
                sum(
                    max(0, plan.generated_frames - plan.emitted_generated_frames)
                    for plan in plans
                )
                * self._samples_per_frame
            ),
        )
        gpu_input: torch.Tensor | None = None
        keepalives: list[Any] = []
        try:
            stream.wait_stream(torch.cuda.current_stream(self._device))
            with torch.cuda.stream(stream):
                gpu_input = self._stage_decoder_input(
                    decoder_input, slot if pinned else None
                )
                graphs = (
                    self._initial_decode_graphs
                    if stream is self._decode_stream
                    else (
                        self._followup_decode_graphs
                        if stream is self._followup_decode_stream
                        else None
                    )
                )
                waveform = graphs.decode(gpu_input) if graphs is not None else None
                if waveform is None:
                    waveform = self._decoder.chunked_decode(gpu_input)
                keepalives.append(waveform)
                waveforms = self._split_batch_waveform(waveform, len(plans))
                deltas = [
                    self._extract_delta(plan, waveform).detach().to(torch.float32)
                    for plan, waveform in zip(plans, waveforms)
                ]
                keepalives.extend(deltas)
                if not pinned:
                    host = [delta.contiguous().cpu() for delta in deltas]
                    # Note (jiannan-17): a zero-element .cpu() call enqueues
                    # no D2H copy, so it does not wait for earlier decode
                    # work. Synchronize before returning.
                    stream.synchronize()
                    return _Qwen3TTSDecodeHandle(
                        host, bad_rows, owner=self, stream=stream
                    )
                staged = self._stage_deltas(deltas, slot)
                # Note (jiannan-17): recorded even when every delta is empty;
                # the event also fences the H2D copy and the decoder work.
                slot.output_transfer.record(stream)
            return _Qwen3TTSDecodeHandle(
                staged,
                bad_rows,
                slot=slot,
                owner=self,
                stream=stream,
                decoder_input_keepalive=gpu_input,
                keepalives=keepalives,
            )
        except BaseException as launch_exc:
            # Note (jiannan-17): CUDA work may already be using the decoder
            # input, the slot buffers, or the decoder output. Synchronizing
            # the exact stream proves it finished. If even that fails, nothing
            # may be reused or freed: keep every reference alive for the rest
            # of the process and stop issuing CUDA decodes. Either way a slot
            # whose launch failed is never trusted again.
            try:
                stream.synchronize()
            except BaseException:
                if pinned:
                    slot.broken = True
                self._cuda_decode_failed = True
                _CONTEXT_FATAL_RETAINED.append(
                    _RetainedDecodeResources(
                        owner=self,
                        stream=stream,
                        slot=slot if pinned else None,
                        decoder_input=gpu_input,
                        keepalives=[*keepalives, decoder_input],
                    )
                )
                logger.error(
                    "Qwen3-TTS decode launch failed and the decode stream could "
                    "not be synchronized; disabling CUDA decode and retaining "
                    "the in-flight buffers",
                    exc_info=True,
                )
                raise launch_exc
            if pinned:
                slot.broken = True
                slot.busy = False
            raise

    def _thread_decode_slot(self) -> _DecodeSlot:
        slot = getattr(self._decode_staging, "value", None)
        if slot is None:
            # Note (jiannan-17): the slot records on the decode streams, so it
            # takes their exact (indexed) device instead of re-resolving a
            # bare "cuda" on this thread.
            slot_device = (
                self._decode_stream.device
                if self._decode_stream is not None
                else self._device
            )
            slot = _DecodeSlot(
                input_codes=GrowablePinnedBuffer(torch.long),
                output_transfer=PinnedTransferSlot(slot_device, torch.float32),
            )
            self._decode_staging.value = slot
        return slot

    def _reserve_slot(
        self, slot: _DecodeSlot, *, input_numel: int, output_numel: int
    ) -> bool:
        """Grow and acquire the thread's slot before any async work is enqueued.

        Return ``False`` when the slot cannot be used; the launch then falls
        back to pageable transfers and a synchronous stream wait.
        """
        if slot.broken or self._pinned_staging_disabled:
            return False
        if slot.busy:
            raise RuntimeError(
                "Qwen3-TTS decode slot is still owned by a pending handle"
            )
        try:
            slot.input_codes.ensure_capacity(input_numel)
            slot.output_transfer.ensure_capacity(output_numel)
        except RuntimeError:
            self._pinned_staging_disabled = True
            logger.warning(
                "Qwen3-TTS streaming vocoder pinned staging allocation failed; "
                "falling back to pageable transfers",
                exc_info=True,
            )
            return False
        slot.busy = True
        return True

    def _stage_decoder_input(
        self, decoder_input: torch.Tensor, slot: _DecodeSlot | None
    ) -> torch.Tensor:
        """Move decoder input to the configured device.

        CPU codes go through the slot's pinned buffer when one is reserved so
        the copy can run asynchronously. CUDA input skips staging.
        """
        if decoder_input.device.type == self._device.type or slot is None:
            return decoder_input.to(self._device)
        pinned = slot.input_codes.view(int(decoder_input.numel()))
        pinned = pinned.view(decoder_input.shape)
        pinned.copy_(decoder_input)
        return pinned.to(self._device, non_blocking=True)

    def _stage_deltas(
        self, deltas: list[torch.Tensor], slot: _DecodeSlot
    ) -> list[torch.Tensor]:
        """Copy GPU deltas into the slot's pinned buffer, one view per delta.

        Example:
            delta lengths [3, 2] -> flat[0:3], flat[3:5]

        The caller records the slot's event afterwards and ``resolve()``
        clones the views before the slot is reused.
        """
        total = sum(int(delta.numel()) for delta in deltas)
        flat = slot.output_transfer.view(total)
        staged: list[torch.Tensor] = []
        offset = 0
        for delta in deltas:
            numel = int(delta.numel())
            segment = flat[offset : offset + numel]
            segment.copy_(delta, non_blocking=True)
            staged.append(segment)
            offset += numel
        return staged

    def _split_batch_waveform(
        self, waveform: torch.Tensor, batch_size: int
    ) -> list[torch.Tensor]:
        """Return one 1-D waveform per request.

        Examples:
            [B, 1, S] -> B tensors shaped [S]
            [1, S]    -> one tensor shaped [S], valid only for batch_size == 1
        """
        if waveform.ndim == 3:
            if waveform.shape[0] != batch_size:
                raise RuntimeError(
                    "Qwen3-TTS streaming decoder returned the wrong batch size"
                )
            return [waveform[index, 0] for index in range(batch_size)]
        if waveform.ndim == 2:
            if batch_size != 1:
                raise RuntimeError(
                    "Qwen3-TTS streaming decoder dropped the batch dimension"
                )
            return [waveform[0]]
        raise ValueError(
            "Qwen3-TTS decoder returned unexpected waveform shape "
            f"{tuple(waveform.shape)}"
        )

    def _extract_delta(
        self, plan: _Qwen3TTSDecodePlan, waveform: torch.Tensor
    ) -> torch.Tensor:
        trim_frames = plan.absolute_emitted_frames - plan.window_start
        trim_samples = min(
            trim_frames * self._samples_per_frame,
            int(waveform.shape[-1]),
        )
        new_frames = plan.generated_frames - plan.emitted_generated_frames
        emit_samples = new_frames * self._samples_per_frame
        return waveform[trim_samples : trim_samples + emit_samples]

    def _commit_decode_plan(
        self,
        state: _Qwen3TTSStreamState,
        plan: _Qwen3TTSDecodePlan,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if state.emitted_generated_frames != plan.emitted_generated_frames:
            raise RuntimeError("Qwen3-TTS streaming decode plan committed out of order")
        if delta.numel() == 0:
            raise RuntimeError("Qwen3-TTS streaming decoder returned an empty delta")

        state.emitted_generated_frames = plan.generated_frames
        state.decoded_chunks += 1
        state.next_decode_generated_frames = (
            plan.generated_frames + self._next_followup_stride(state)
        )
        now = time.monotonic()
        duration_s = float(delta.numel()) / float(self._sample_rate)
        state.playback_deadline_s = max(state.playback_deadline_s, now) + duration_s
        return delta

    def _next_followup_stride(self, state: _Qwen3TTSStreamState) -> int:
        """Stride of the next decode chunk after a commit.

        A ramp is cursored by emitted frames, so a backlog that overshot the
        ramp resumes at the steady stride; the legacy schedule keeps its
        decode-count selection."""
        if not self._chunk_ramp_configured:
            return (
                self._followup_stride_ramp[0]
                if state.decoded_chunks == 1
                else self._stream_followup_stride
            )
        cumulative = state.initial_chunk_frames or self._stream_stride
        for stride in self._followup_stride_ramp:
            if state.emitted_generated_frames < cumulative + stride:
                return stride
            cumulative += stride
        return self._stream_followup_stride

    def _decode_and_emit(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> list[OutgoingMessage]:
        if not self.should_decode(state, is_final=False):
            return []
        if self._async_decode:
            if state.decoded_chunks:
                self._schedule_followup(request_id, state)
            else:
                self._schedule_initial(request_id, state)
            return []
        delta = self.decode_delta(request_id, state, is_final=False)
        if delta is None:
            return []

        self._mark_stream_emitted(request_id)
        if state.decoded_chunks == 1 and state.initial_chunk_frames > 0:
            # note (Junnan Li): keep client-visible chunk sizes ramp-shaped
            # when the first decode flushed a backlog; without a ramp only the
            # legacy initial-boundary split applies.
            split_frames = (state.initial_chunk_frames,)
            if self._chunk_ramp_configured:
                split_frames += self._followup_stride_ramp
            slices: list[torch.Tensor] = []
            total_samples = int(delta.shape[-1])
            start = 0
            for frames in split_frames:
                end = min(start + frames * self._samples_per_frame, total_samples)
                if end <= start:
                    break
                slices.append(delta[start:end])
                start = end
            if start < total_samples:
                slices.append(delta[start:])
            if len(slices) > 1:
                return [
                    self._stream_chunk_message(request_id, piece) for piece in slices
                ]
        return [self._stream_chunk_message(request_id, delta)]

    def _schedule_initial(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        if state.initial_pending:
            return
        if self._initial_worker is None:
            raise RuntimeError("Qwen3-TTS initial decoder is not running")
        state.initial_pending = True
        self._initial_queue.put((request_id, state))

    def _schedule_followup(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        if state.followup_pending:
            return
        if self._followup_worker is None:
            raise RuntimeError("Qwen3-TTS follow-up decoder is not running")
        state.followup_pending = True
        self._enqueue_followup(request_id, state)

    def _enqueue_followup(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        self._followup_queue.put(
            (
                state.playback_deadline_s,
                next(self._followup_sequence),
                request_id,
                state,
            )
        )

    def _collect_async_batch(
        self,
        work_queue: queue.Queue[tuple[str, _Qwen3TTSStreamState] | None],
        *,
        max_batch_size: int,
        batch_wait_s: float,
    ) -> list[tuple[str, _Qwen3TTSStreamState]] | None:
        queued = work_queue.get()
        if queued is None or self._async_stop.is_set():
            return None
        batch = [queued]
        deadline = time.monotonic() + batch_wait_s
        while len(batch) < max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_queued = work_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_queued is None:
                return None
            batch.append(next_queued)
        return batch

    def _run_initial_worker(self) -> None:
        while True:
            batch = self._collect_async_batch(
                self._initial_queue,
                max_batch_size=self._initial_max_batch_size,
                batch_wait_s=self._initial_batch_wait_s,
            )
            if batch is None:
                return
            self._run_initial_batch(batch)

    def _run_initial_batch(
        self,
        batch: list[tuple[str, _Qwen3TTSStreamState]],
    ) -> None:
        planned: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]] = []
        with self._state_lock:
            for request_id, state in batch:
                if (
                    self._stream_states.get(request_id) is not state
                    or state.decoded_chunks
                ):
                    continue
                plan = self._build_decode_plan(
                    state,
                    is_final=state.final_pending,
                    max_generated_frames=(
                        state.initial_chunk_frames or self._stream_stride
                    ),
                )
                if plan is None:
                    state.initial_pending = False
                    continue
                planned.append((request_id, state, plan))

        for group in self._group_decode_plans(planned):
            decoded = self._decode_group(group, stream=self._decode_stream)
            if decoded is None:
                continue
            for entry, delta in zip(*decoded):
                request_id, state, plan = entry
                self._commit_initial(request_id, state, plan, delta)

    def _decode_group(
        self,
        group: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]],
        *,
        stream: torch.cuda.Stream | None,
    ) -> (
        tuple[list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]], list] | None
    ):
        """Decode a group, failing only the rows that carried invalid codes."""
        while group:
            try:
                handle = self._launch_decode_plans(
                    [entry[2] for entry in group], stream=stream
                )
                deltas = handle.resolve()
            except _Qwen3TTSInvalidCodeRows as exc:
                bad = set(exc.indices)
                for index, (request_id, state, _) in enumerate(group):
                    if index in bad:
                        self._fail_async_stream(request_id, state, exc)
                group = [e for i, e in enumerate(group) if i not in bad]
                continue
            except Exception as exc:
                for request_id, state, _ in group:
                    self._fail_async_stream(request_id, state, exc)
                return None
            return group, deltas
        return None

    @staticmethod
    def _group_decode_plans(
        planned: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]],
    ) -> list[list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]]]:
        groups: dict[
            tuple[int, ...],
            list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]],
        ] = {}
        for entry in planned:
            groups.setdefault(tuple(entry[2].decoder_input.shape), []).append(entry)
        return list(groups.values())

    def _commit_initial(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        plan: _Qwen3TTSDecodePlan,
        delta: torch.Tensor,
    ) -> None:
        cleanup_abort = False
        with self._state_lock:
            if self._stream_states.get(request_id) is not state:
                return
            try:
                delta = self._commit_decode_plan(state, plan, delta)
            except Exception as exc:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                cleanup_abort = True
            else:
                state.initial_pending = False
                if not self._is_aborted(request_id):
                    self._mark_stream_emitted(request_id)
                    self.outbox.put(self._stream_chunk_message(request_id, delta))
                has_remainder = (
                    state.total_frames - state.ref_frames
                    > state.emitted_generated_frames
                )
                if state.final_pending and not has_remainder:
                    self._finish_async_stream(request_id, state)
                elif state.final_pending or self.should_decode(state, is_final=False):
                    self._schedule_followup(request_id, state)
        if cleanup_abort:
            self._cleanup_aborted_request(request_id)

    def _run_followup_worker(self) -> None:
        while True:
            batch = self._collect_followup_batch()
            if batch is None:
                return
            self._run_followup_batch(batch)

    def _collect_followup_batch(
        self,
    ) -> list[tuple[str, _Qwen3TTSStreamState]] | None:
        _, _, request_id, state = self._followup_queue.get()
        if state is None or self._async_stop.is_set():
            return None
        batch = [(request_id, state)]
        deadline = time.monotonic() + self._followup_batch_wait_s
        while len(batch) < self._followup_max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                _, _, request_id, state = self._followup_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if state is None:
                return None
            batch.append((request_id, state))
        return batch

    def _run_followup_batch(
        self,
        batch: list[tuple[str, _Qwen3TTSStreamState]],
    ) -> None:
        planned: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]] = []
        with self._state_lock:
            for request_id, state in batch:
                if self._stream_states.get(request_id) is not state:
                    continue
                plan = self._build_decode_plan(
                    state,
                    is_final=state.final_pending,
                    max_generated_frames=self._next_decode_threshold(state),
                )
                if plan is None:
                    state.followup_pending = False
                    if state.final_pending:
                        self._finish_async_stream(request_id, state)
                    continue
                planned.append((request_id, state, plan))

        for group in self._group_decode_plans(planned):
            decoded = self._decode_group(group, stream=self._followup_decode_stream)
            if decoded is None:
                continue
            for entry, delta in zip(*decoded):
                request_id, state, plan = entry
                self._commit_followup(request_id, state, plan, delta)

    def _commit_followup(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        plan: _Qwen3TTSDecodePlan,
        delta: torch.Tensor,
    ) -> None:
        cleanup_abort = False
        with self._state_lock:
            if self._stream_states.get(request_id) is not state:
                return
            try:
                delta = self._commit_decode_plan(state, plan, delta)
            except Exception as exc:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                cleanup_abort = True
            else:
                if not self._is_aborted(request_id):
                    self._mark_stream_emitted(request_id)
                    self.outbox.put(self._stream_chunk_message(request_id, delta))
                has_remainder = (
                    state.total_frames - state.ref_frames
                    > state.emitted_generated_frames
                )
                if state.final_pending and not has_remainder:
                    state.followup_pending = False
                    self._finish_async_stream(request_id, state)
                elif state.final_pending or self.should_decode(state, is_final=False):
                    self._enqueue_followup(request_id, state)
                else:
                    state.followup_pending = False
        if cleanup_abort:
            self._cleanup_aborted_request(request_id)

    def _fail_async_stream(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        exc: BaseException,
    ) -> None:
        cleanup_abort = False
        with self._state_lock:
            if self._stream_states.get(request_id) is state:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                cleanup_abort = True
        if cleanup_abort:
            self._cleanup_aborted_request(request_id)

    def _handle_stream_done(self, request_id: str) -> None:
        with self._state_lock:
            if request_id not in self._stream_payloads:
                if request_id in self._completed_non_streaming_request_ids:
                    return
                self._pending_done.add(request_id)
                return
            state = self._get_or_create_stream_state(request_id)
            if (
                self._async_decode
                and state is not None
                and (state.initial_pending or state.decoded_chunks)
            ):
                state.final_pending = True
                if not state.initial_pending:
                    self._schedule_followup(request_id, state)
                return
        # Note (jiannan-17): requests that finish before the initial decode
        # threshold flush synchronously below, so that decode and its resolve
        # run under _state_lock. Kept as-is here; moving short finals onto the
        # initial worker is a separate change.
        super()._handle_stream_done(request_id)

    def _finish_async_stream(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        payload = self._stream_payloads.get(request_id)
        if payload is None or self._is_aborted(request_id):
            return
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=self.final_result_data(request_id, payload, state),
                ),
            )
        )
        self._record_completed_stream_request_id(request_id)
        self._clear_request_state(request_id)

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Qwen3TTSStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_state_audio(Qwen3TTSState.from_dict(payload.data))

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Qwen3TTSStreamState,
    ) -> dict[str, Any]:
        del request_id, state
        final_state = Qwen3TTSState.from_dict(payload.data)
        data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return (await self._vocode_payloads([payload]))[0]

    async def _vocode_payloads(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        states = [Qwen3TTSState.from_dict(payload.data) for payload in payloads]
        codes = []
        for state in states:
            if state.audio_codes is None:
                raise RuntimeError(
                    "Qwen3-TTS vocoder requires audio_codes from tts_engine"
                )
            codes.append(torch.as_tensor(state.audio_codes, dtype=torch.long))

        if self._deterministic_inference:
            wavs = []
            for item in codes:
                decoded, sample_rate = self._tokenizer.decode([{"audio_codes": item}])
                (wav,) = decoded
                wavs.append(wav)
        else:
            wavs, sample_rate = self._tokenizer.decode(
                [{"audio_codes": item} for item in codes]
            )
        if len(wavs) != len(payloads):
            raise RuntimeError(
                f"Qwen3-TTS speech tokenizer returned {len(wavs)} audios for "
                f"{len(payloads)} requests"
            )
        return [
            self._store_vocoder_result(payload, state, wav, sample_rate)
            for payload, state, wav in zip(payloads, states, wavs)
        ]

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: Qwen3TTSState,
        waveform: Any,
        sample_rate: int,
    ) -> StagePayload:
        if waveform is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")
        if state.ref_code_len:
            total_frames = len(state.audio_codes)
            cut = int(state.ref_code_len / max(total_frames, 1) * waveform.shape[0])
            waveform = waveform[cut:]

        data = audio_waveform_payload(
            waveform,
            sample_rate=int(sample_rate),
            modality="audio",
            source_hint="Qwen3-TTS",
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    def _decode_state_audio(self, state: Qwen3TTSState) -> torch.Tensor | None:
        if state.audio_codes is None:
            return None
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        wavs, _ = self._tokenizer.decode([{"audio_codes": codes}])
        if not wavs:
            return None
        waveform = torch.as_tensor(wavs[0], dtype=torch.float32)
        if state.ref_code_len:
            total_frames = len(codes)
            cut = int(state.ref_code_len / max(total_frames, 1) * waveform.shape[0])
            waveform = waveform[cut:]
        return waveform.contiguous()


__all__ = ["Qwen3TTSStreamingVocoderScheduler"]
