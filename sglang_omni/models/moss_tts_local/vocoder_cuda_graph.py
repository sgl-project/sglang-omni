# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph runner for the MOSS streaming codec decode: one graph per T (B fixed at slot width), captured once at warmup. Adapted from the Higgs vocoder graph; bit-identity gated by the test."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from types import MethodType
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_ATTN_ORIGINAL_UPDATE_CACHE_ATTR = "_sglang_omni_original_update_streaming_cache"

MOSSL_FRAME_GRAPH_ENV = "SGLANG_OMNI_MOSSL_FRAME_GRAPH"
_NONSTREAM_MAX_CAPTURE_FAILURES = 8
# Note: (Jiaxin Deng) one shared load threshold for the AR emit fast path and
# the nonstream vocoder graphs: they must engage together. The fast path lets
# the launch queue run deep, which the EAGER vocoder's ~13 per-utterance host
# syncs each drain (measured stage-latency explosion); the graphed vocoder
# syncs once. Below the threshold both stay off = the env-off code path.
MOSSL_FRAME_GRAPH_MIN_AR_BATCH = 12


def mossl_frame_graph_enabled() -> bool:
    value = os.environ.get(MOSSL_FRAME_GRAPH_ENV, "1").strip().lower()
    return value not in ("0", "false", "off")


class _ArDecodeLoadBeacon:
    """Latest AR decode batch size, published by the model runner every step.

    The colocated vocoder thread reads it as the in-flight load signal for the
    nonstream-graph gate. Note: (Jiaxin Deng) a split-process vocoder reads 0,
    so its nonstream decode stays eager (safe, just ungraphed).
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


_ar_decode_load = _ArDecodeLoadBeacon()


def publish_ar_decode_batch(batch_size: int) -> None:
    _ar_decode_load.value = int(batch_size)


def last_ar_decode_batch() -> int:
    return _ar_decode_load.value


class _CapturedVocoderGraph(NamedTuple):
    """One captured per-T graph and its static replay buffers (named to avoid positional unpack)."""

    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor


def _decoder_attention_modules(codec) -> list:
    """Decoder attention modules whose streaming KV cache must be made graph-stable."""
    modules_by_id: dict[int, object] = {}
    decoder = getattr(codec, "decoder", ())
    for decoder_module in decoder:
        modules = decoder_module.modules() if hasattr(decoder_module, "modules") else ()
        for module in modules:
            if hasattr(module, "attention_implementation"):
                modules_by_id.setdefault(id(module), module)
    return list(modules_by_id.values())


def _cuda_graph_update_streaming_cache(
    self, state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k
) -> None:
    context = getattr(self, "context", None)
    original = getattr(self, _ATTN_ORIGINAL_UPDATE_CACHE_ATTR, None)
    if context is None:
        if callable(original):
            return original(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
        raise RuntimeError("CUDA graph codec attention requires finite context")
    state_cached_keys = getattr(state, "cached_keys", None)
    state_cached_values = getattr(state, "cached_values", None)
    state_cached_positions = getattr(state, "cached_positions", None)
    if (
        state_cached_keys is None
        or state_cached_values is None
        or state_cached_positions is None
    ):
        if callable(original):
            return original(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
        raise RuntimeError("CUDA graph codec attention cache is not initialized")
    exec_mask = state.exec_mask.view(-1, 1, 1, 1)
    exec_mask_pos = state.exec_mask.view(-1, 1)
    new_cached_k = k_all[:, :, -int(context) :, :].contiguous()
    new_cached_v = v_all[:, :, -int(context) :, :].contiguous()
    new_cached_pos = pos_k[:, -int(context) :].contiguous()
    state_cached_keys.copy_(torch.where(exec_mask, new_cached_k, cached_k))
    state_cached_values.copy_(torch.where(exec_mask, new_cached_v, cached_v))
    state_cached_positions.copy_(torch.where(exec_mask_pos, new_cached_pos, cached_pos))


def patch_codec_attention_cache_for_cuda_graph(codec) -> None:
    """Rebind the decoder streaming attention cache update to an in-place write (stable address,
    value-identical to eager) so a CUDA graph can capture it."""
    for module in _decoder_attention_modules(codec):
        update_cache = getattr(module, "_update_streaming_cache", None)
        if not callable(update_cache):
            continue
        if hasattr(module, _ATTN_ORIGINAL_UPDATE_CACHE_ATTR):
            continue
        setattr(module, _ATTN_ORIGINAL_UPDATE_CACHE_ATTR, update_cache)
        module._update_streaming_cache = MethodType(
            _cuda_graph_update_streaming_cache, module
        )


class MossVocoderCudaGraphRunner:
    """Warmup-captured, sealed replay of exact-T CUDA graphs for the MOSS codec decode (B fixed)."""

    def __init__(
        self,
        codec,
        *,
        batch_size: int,
        n_vq: int,
        max_frames: int = 128,
        max_graphs: int = 160,
        warmup_iters: int = 3,
        min_free_gb: float = 3.0,
    ) -> None:
        self._codec = codec
        self._batch_size = int(batch_size)
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._max_frames = int(max_frames)
        self._max_graphs = int(max_graphs)
        self._warmup_iters = int(warmup_iters)
        # Min free VRAM to attempt a capture (each graph is multi-GB); below it we skip -> eager,
        # so a VRAM-tight box degrades gracefully instead of OOM-ing.
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._graphs: dict[int, _CapturedVocoderGraph] = {}
        self._pool = None
        self._sealed = False
        # Reused all-active mask for the warmup-only state reset (avoid re-allocating it per captured T).
        self._reset_mask = torch.ones(
            self._batch_size, dtype=torch.bool, device=self._device
        )

    def _is_supported_frame_count(self, frame_count: int) -> bool:
        return 1 <= frame_count <= self._max_frames

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    @torch.no_grad()
    def _reset_state(self) -> None:
        """Reset every streaming module's offset/positions to 0 in-place (warmup-only, between
        captures; the full state.reset is a one-time startup cost, not per-step)."""

        def _r(module) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is not None:
                state.reset(self._reset_mask.to(state.device))

        self._codec.apply(_r)

    @torch.no_grad()
    def _capture_frame_count(self, frame_count: int) -> None:
        b, n = self._batch_size, self._n_vq
        device = self._device
        static_codes = torch.zeros(n, b, frame_count, dtype=torch.long, device=device)
        # Capture all-active; the live exec_mask at replay decides which slots advance.
        static_lengths = torch.full((b,), frame_count, dtype=torch.long, device=device)
        exec_mask = torch.ones(b, dtype=torch.bool, device=device)
        self._codec._set_streaming_exec_mask(exec_mask)
        # Note: (Jiaxin Deng) side-stream warmup forces lazy allocs (conv algo / workspaces) out of the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._codec._decode_frame(static_codes, static_lengths)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Note: (Jiaxin Deng) reset to offset 0 AFTER warmup, BEFORE capture -- capturing at the
        # warmup-advanced offset bakes a wrong start state (~0.4 PCM error). reset re-activates all slots.
        self._reset_state()
        self._codec._set_streaming_exec_mask(exec_mask)
        # Shared mempool across the T graphs to bound memory (large B=16 intermediates); capture order in warmup.
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            result = self._codec._decode_frame(static_codes, static_lengths)
            static_audio = result.audio
            static_audio_lengths = result.audio_lengths
        self._graphs[frame_count] = _CapturedVocoderGraph(
            graph=graph,
            static_codes=static_codes,
            static_lengths=static_lengths,
            static_audio=static_audio,
            static_audio_lengths=static_audio_lengths,
        )
        logger.info(
            "Captured MOSS vocoder CUDA graph T=%d (B=%d) -> audio %s (%d cached)",
            frame_count,
            b,
            tuple(static_audio.shape),
            len(self._graphs),
        )

    @torch.no_grad()
    def warmup(self, frames: Iterable[int]) -> None:
        """Capture one graph per T, once, then seal (startup, GPU quiescent). Caller MUST reset all
        slots after this returns (warmup advances per-slot state)."""
        if self._sealed:
            logger.warning(
                "MossVocoderCudaGraphRunner.warmup called after seal; ignoring"
            )
            return
        # Bind capture to the codec's device: the stream/pool/graph use the current device, and
        # factory-time capture can precede the stage device switch (split puts the codec off cuda:0).
        with torch.cuda.device(self._device):
            # Note: (Jiaxin Deng) capture LARGEST T first -- the graphs share one mempool; capturing a larger
            # graph after a smaller one grows the pool and invalidates earlier graphs' addresses (replay segfaults).
            for t in sorted(dict.fromkeys(int(x) for x in frames), reverse=True):
                if t in self._graphs:
                    continue
                if not self._is_supported_frame_count(t):
                    logger.warning(
                        "skip MOSS vocoder CG T=%d: outside [1, %d]",
                        t,
                        self._max_frames,
                    )
                    continue
                if len(self._graphs) >= self._max_graphs:
                    logger.warning(
                        "MOSS vocoder CG cap %d reached; skipping rest",
                        self._max_graphs,
                    )
                    break
                # Note: (Jiaxin Deng) VRAM headroom guard -- skip capture (-> eager) rather than risk OOM on
                # a tight box. Checked per-T because each capture allocates; free only drops through the loop.
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "MOSS vocoder CG: free VRAM %.1fGB < %.1fGB headroom; skipping T=%d+ (eager)",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                        t,
                    )
                    break
                # best-effort: an uncaptured T falls back to eager
                try:
                    self._capture_frame_count(t)
                except Exception as exc:
                    self._graphs.pop(t, None)
                    logger.warning(
                        "MOSS vocoder CG capture failed for T=%d: %s; will use eager",
                        t,
                        exc,
                    )
        self._sealed = True
        logger.info(
            "MOSS vocoder CUDA graphs sealed: %d T captured %s",
            len(self._graphs),
            sorted(self._graphs.keys()),
        )

    def captured_frames(self) -> list[int]:
        return sorted(self._graphs.keys())

    @torch.no_grad()
    def decode_step(
        self,
        codes_step: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay the captured graph for ``[n_vq, B_full, T]`` codes (set live exec_mask, copy codes, replay), else None. Returns the static buffers directly; caller consumes before the next replay.

        Per-slot lengths are not needed: every captured graph decodes the full T for all slots and
        ``exec_mask`` gates which outputs are valid (the eager fallback still uses lengths).
        """
        if not codes_step.is_cuda:
            return None
        n, b, t = codes_step.shape
        if b != self._batch_size or n != self._n_vq:
            return None
        entry = self._graphs.get(int(t))
        if entry is None:
            return None
        # Replicate eager inputs exactly (codes + live exec_mask) so replay is bit-for-bit identical.
        self._codec._set_streaming_exec_mask(exec_mask)
        entry.static_codes.copy_(codes_step)
        entry.graph.replay()
        return entry.static_audio, entry.static_audio_lengths


class _CapturedNonstreamGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_audio: torch.Tensor
    samples_per_frame: int


class MossNonstreamVocoderGraphRunner:
    """(B, T)-bucketed CUDA graphs over the packed non-streaming codec decode.

    Captured at warmup with every sequence pinned to the bucket's dense length
    (``assume_full_lengths``). Replay is bit-identical to the same-geometry
    eager decode (gated in tests). Note: (Jiaxin Deng) the eager ragged decode
    is itself batch-composition-dependent at the bit level (varlen flash
    geometry; measured max|delta| ~2e-2 between batch layouts of the same
    utterance), so bucket padding stays within the pre-existing numerical
    family rather than adding a new error mode. Replay copies padded codes
    into the static buffer and slices per-utterance audio by ``frames *
    samples_per_frame`` (the length map is exactly linear; validated at
    capture).
    """

    def __init__(
        self,
        codec,
        nonstream_decoder,
        *,
        n_vq: int,
        batch_buckets: Iterable[int],
        frame_buckets: Iterable[int],
        warmup_iters: int = 2,
        min_free_gb: float = 3.0,
    ) -> None:
        self._codec = codec
        self._nonstream_decoder = nonstream_decoder
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._batch_buckets = sorted({int(b) for b in batch_buckets if int(b) >= 1})
        self._frame_buckets = sorted({int(t) for t in frame_buckets if int(t) >= 1})
        self._warmup_iters = int(warmup_iters)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._graphs: dict[tuple[int, int], _CapturedNonstreamGraph] = {}
        # Note: (Jiaxin Deng) one shared mempool per batch bucket, captured
        # largest-T first: growing a shared pool after a capture invalidates
        # earlier graphs' addresses (streaming-runner lesson).
        self._pools: dict[int, object] = {}
        self._capture_failures = 0
        self._disabled = False
        self._sealed = False
        self._graph_steps = 0
        self._eager_misses = 0

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    @torch.no_grad()
    def _capture(self, batch_bucket: int, frame_bucket: int) -> None:
        device = self._device
        static_codes = torch.zeros(
            self._n_vq, batch_bucket, frame_bucket, dtype=torch.long, device=device
        )
        static_lengths = torch.full(
            (batch_bucket,), frame_bucket, dtype=torch.long, device=device
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        warmup_result = None
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                warmup_result = self._codec._decode_frame(static_codes, static_lengths)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # Validate the linear frame->sample map on the executed warmup result
        # (capture only records, so the static outputs hold no values yet).
        assert warmup_result is not None
        warmup_lengths = warmup_result.audio_lengths
        audio_len_max = int(warmup_lengths.max().item())
        audio_len_min = int(warmup_lengths.min().item())
        if (
            audio_len_max != audio_len_min
            or audio_len_max <= 0
            or audio_len_max % frame_bucket != 0
        ):
            raise RuntimeError(
                "MOSS nonstream vocoder length map is not uniformly linear: "
                f"T={frame_bucket} -> audio_lengths in "
                f"[{audio_len_min}, {audio_len_max}]"
            )
        samples_per_frame = audio_len_max // frame_bucket

        pool = self._pools.get(batch_bucket)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            self._pools[batch_bucket] = pool
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                result = self._codec._decode_frame(static_codes, static_lengths)
                static_audio = result.audio
        except Exception:
            try:
                graph.reset()
            except Exception:
                pass
            raise
        self._graphs[(batch_bucket, frame_bucket)] = _CapturedNonstreamGraph(
            graph=graph,
            static_codes=static_codes,
            static_audio=static_audio,
            samples_per_frame=samples_per_frame,
        )
        logger.info(
            "Captured MOSS nonstream vocoder CUDA graph B=%d T=%d -> audio %s "
            "(%d cached)",
            batch_bucket,
            frame_bucket,
            tuple(static_audio.shape),
            len(self._graphs),
        )

    @torch.no_grad()
    def warmup(self) -> None:
        """Capture all (B, T) buckets once, then seal; failed/skipped keys fall
        back to eager. Must run while the GPU is otherwise quiescent."""
        if self._sealed:
            logger.warning(
                "MossNonstreamVocoderGraphRunner.warmup called after seal; ignoring"
            )
            return
        original_decoder = self._codec.decoder
        self._codec.decoder = self._nonstream_decoder
        try:
            with torch.cuda.device(self._device):
                with self._nonstream_decoder.assume_full_lengths():
                    # B ascending: a VRAM-budget stop keeps the small-B keys,
                    # which carry ~all production hits. T stays largest-first
                    # within each per-B shared pool (pool-growth rule).
                    for batch_bucket in sorted(self._batch_buckets):
                        for frame_bucket in sorted(self._frame_buckets, reverse=True):
                            if self._disabled:
                                break
                            enough, free = self._enough_free_vram()
                            if not enough:
                                logger.warning(
                                    "MOSS nonstream vocoder CG: free VRAM %.1fGB < "
                                    "%.1fGB headroom; skipping remaining captures",
                                    free / 1024**3,
                                    self._min_free_bytes / 1024**3,
                                )
                                self._sealed = True
                                return
                            try:
                                self._capture(batch_bucket, frame_bucket)
                            except Exception:
                                self._capture_failures += 1
                                self._log_capture_failure(batch_bucket, frame_bucket)
                                if (
                                    self._capture_failures
                                    >= _NONSTREAM_MAX_CAPTURE_FAILURES
                                ):
                                    self._disabled = True
                                    logger.warning(
                                        "Disabling MOSS nonstream vocoder CUDA "
                                        "graphs after %d capture failures",
                                        self._capture_failures,
                                    )
                                continue
                            # Re-check AFTER the capture: one near the boundary
                            # can eat into the promised eager headroom.
                            enough_after, free_after = self._enough_free_vram()
                            if not enough_after:
                                entry = self._graphs.pop(
                                    (batch_bucket, frame_bucket), None
                                )
                                if entry is not None:
                                    try:
                                        entry.graph.reset()
                                    except Exception:
                                        pass
                                logger.warning(
                                    "MOSS nonstream vocoder CG: capture B=%d T=%d "
                                    "left free VRAM %.1fGB below the %.1fGB "
                                    "reserve; rolled back, stopping captures",
                                    batch_bucket,
                                    frame_bucket,
                                    free_after / 1024**3,
                                    self._min_free_bytes / 1024**3,
                                )
                                return
                        if self._disabled:
                            break
        finally:
            self._codec.decoder = original_decoder
            self._sealed = True
        logger.info(
            "MOSS nonstream vocoder CUDA graphs sealed: %d keys %s",
            len(self._graphs),
            sorted(self._graphs.keys()),
        )

    def _log_capture_failure(self, batch_bucket: int, frame_bucket: int) -> None:
        logger.warning(
            "MOSS nonstream vocoder CG capture failed for "
            "B=%d T=%d (failure %d/%d); eager for this key",
            batch_bucket,
            frame_bucket,
            self._capture_failures,
            _NONSTREAM_MAX_CAPTURE_FAILURES,
            exc_info=True,
        )

    def captured_keys(self) -> list[tuple[int, int]]:
        return sorted(self._graphs.keys())

    def bucket_for(self, live_batch: int, live_frames: int) -> tuple[int, int] | None:
        """Smallest captured (batch, frames) bucket covering the live shape."""
        for batch_bucket in self._batch_buckets:
            if batch_bucket < live_batch:
                continue
            for frame_bucket in self._frame_buckets:
                if frame_bucket < live_frames:
                    continue
                if (batch_bucket, frame_bucket) in self._graphs:
                    return (batch_bucket, frame_bucket)
        return None

    def _find_entry(
        self, live_batch: int, live_frames: int
    ) -> _CapturedNonstreamGraph | None:
        key = self.bucket_for(live_batch, live_frames)
        return self._graphs[key] if key is not None else None

    @torch.no_grad()
    def decode_padded(
        self,
        padded_codes: torch.Tensor,
        codes_lengths: list[int],
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Replay a bucketed graph for ``[n_vq, B, T]`` zero-padded codes with
        host-side per-utterance frame counts; ``None`` -> caller decodes eager.
        Returns (static audio buffer sliced to live rows, per-utterance sample
        counts); the caller must consume the audio before the next replay.
        """
        if self._disabled or not self._graphs:
            return None
        if not padded_codes.is_cuda:
            return None
        n_vq, live_batch, live_frames = padded_codes.shape
        if n_vq != self._n_vq or live_batch == 0 or live_frames == 0:
            return None
        if len(codes_lengths) != live_batch:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        entry = self._find_entry(live_batch, live_frames)
        if entry is None:
            self._eager_misses += 1
            if self._eager_misses <= 5:
                logger.info(
                    "MOSS nonstream vocoder CG miss: B=%d T=%d",
                    live_batch,
                    live_frames,
                )
            return None
        with torch.cuda.device(self._device):
            entry.static_codes.zero_()
            entry.static_codes[:, :live_batch, :live_frames].copy_(padded_codes)
            entry.graph.replay()
        self._graph_steps += 1
        if self._graph_steps % 25 == 0:
            logger.info(
                "MOSS nonstream vocoder CG: %d graphed decodes, %d eager misses",
                self._graph_steps,
                self._eager_misses,
            )
        audio_lengths = [
            int(frames) * entry.samples_per_frame for frames in codes_lengths
        ]
        return entry.static_audio[:live_batch], audio_lengths
