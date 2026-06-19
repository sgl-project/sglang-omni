# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph runner for the MOSS streaming codec decode: one graph per T (B fixed at slot width), captured once at warmup. Adapted from the Higgs vocoder graph; bit-identity gated by the test."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from types import MethodType
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_ATTN_ORIGINAL_UPDATE_CACHE_ATTR = "_sglang_omni_original_update_streaming_cache"


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
