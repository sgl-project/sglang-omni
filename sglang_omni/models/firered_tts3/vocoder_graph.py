# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for steady-state RedAE decoder chunks."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
from transformers import DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask

from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    RedAEDecoderState,
    build_initialized_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_LATENT_SIZES = (4, 8, 16)
# note (kermit): Wider batch buckets become compute-bound on L20.
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)

WARMUP_ITERATIONS = 3


@dataclass(kw_only=True)
class BucketBuffers:
    """Fixed-address decoder state shared by one batch bucket."""

    input_key_values: torch.Tensor
    output_key_values: torch.Tensor
    input_istft_frames: torch.Tensor
    output_istft_frames: torch.Tensor


@dataclass(kw_only=True)
class VocoderGraphEntry:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    positions: torch.Tensor
    causal_mask: torch.Tensor
    waveform: torch.Tensor
    buffers: BucketBuffers
    # note (kermit): Captured kernels retain raw addresses, so every cache and mask owner stays alive.
    graph_cache: DynamicCache
    attention_mask: dict[str, torch.Tensor | None]


class RedAEVocoderGraphRunner:
    """Replay steady-state RedAE chunks at captured widths and batch buckets."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        latent_sizes: tuple[int, ...] = DEFAULT_LATENT_SIZES,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    ) -> None:
        self.decoder: RedAEAudioDecoder = decoder
        self.device: torch.device = next(decoder.parameters()).device
        config = decoder.qwen3_config
        self.layer_count: int = int(config.num_hidden_layers)
        attention = decoder.qwen3.layers[0].self_attn
        self.key_value_head_count: int = int(
            attention.k_proj.out_features // attention.head_dim
        )
        self.attention_head_size: int = int(attention.head_dim)
        self.stored_frames: int = int(config.sliding_window) - 1
        self.latent_frame_buckets: tuple[int, ...] = tuple(
            sorted({int(size) for size in latent_sizes if int(size) > 0})
        )
        self.batch_buckets: tuple[int, ...] = tuple(
            sorted({int(size) for size in batch_sizes if int(size) > 0})
        )
        self.enabled: bool = self.device.type == "cuda"
        self.graphs: dict[tuple[int, int], VocoderGraphEntry] = {}
        self.buckets: dict[int, BucketBuffers] = {}
        self.failed_keys: set[tuple[int, int]] = set()
        self.graph_pool: tuple[int, int] | None = None
        # note (kermit): Dedicated capture streams keep cuBLAS workspaces separate across graph families.
        self.capture_stream: torch.cuda.Stream | None = None
        self.counters: dict[str, int] = {
            "calls": 0,
            "replays": 0,
            "eager": 0,
            "captures": 0,
        }

    def stats(self) -> dict[str, int | list[tuple[int, int]]]:
        return {
            **self.counters,
            "graphs": sorted(self.graphs),
            "failed_keys": sorted(self.failed_keys),
        }

    def capture_all(self) -> None:
        """Capture the configured shapes before serving."""
        if not self.enabled:
            return
        else:
            pass
        for batch_size in sorted(self.batch_buckets, reverse=True):
            for latent_frame_count in sorted(self.latent_frame_buckets, reverse=True):
                key = (latent_frame_count, batch_size)
                if key in self.graphs or key in self.failed_keys:
                    continue
                else:
                    pass
                entry = self.capture(latent_frame_count, batch_size)
                if entry is None:
                    self.failed_keys.add(key)
                    continue
                else:
                    pass
                self.graphs[key] = entry
                self.counters["captures"] += 1
        if not self.graphs:
            self.enabled = False
            logger.warning(
                "FireRedTTS3 vocoder CUDA graphs: no key captured; eager only"
            )
            return
        else:
            pass
        logger.info(
            f"FireRedTTS3 vocoder CUDA graphs captured: {sorted(self.graphs)}"
            + (f" (skipped: {sorted(self.failed_keys)})" if self.failed_keys else "")
        )

    def bucket_buffers(self, batch_size: int) -> BucketBuffers:
        buffers = self.buckets.get(batch_size)
        if buffers is None:
            shape = (
                2 * self.layer_count,
                batch_size,
                self.key_value_head_count,
                self.stored_frames,
                self.attention_head_size,
            )
            istft = self.decoder.istft_head.istft
            overlap_frame_count = istft.win_length // istft.hop_length - 1
            buffers = BucketBuffers(
                input_key_values=torch.zeros(
                    shape, device=self.device, dtype=torch.float32
                ),
                output_key_values=torch.zeros(
                    shape, device=self.device, dtype=torch.float32
                ),
                input_istft_frames=torch.zeros(
                    batch_size,
                    istft.n_fft,
                    overlap_frame_count,
                    device=self.device,
                    dtype=torch.float32,
                ),
                output_istft_frames=torch.zeros(
                    batch_size,
                    istft.n_fft,
                    overlap_frame_count,
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            self.buckets[batch_size] = buffers
        else:
            pass
        return buffers

    def capture(
        self, latent_frame_count: int, batch_size: int
    ) -> VocoderGraphEntry | None:
        """Capture one latent-frame width and batch bucket."""
        decoder = self.decoder
        config = decoder.qwen3_config
        hidden_size = int(config.hidden_size)
        frame_count = latent_frame_count * int(decoder.upsample_rate)
        buffers = self.bucket_buffers(batch_size)
        latents = torch.zeros(
            batch_size,
            latent_frame_count,
            decoder.in_proj.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.zeros(1, frame_count, dtype=torch.long, device=self.device)
        waveform = torch.zeros(
            batch_size,
            frame_count * decoder.istft_head.hop_length,
            device=self.device,
            dtype=torch.float32,
        )
        graph_cache = build_initialized_cache(config, device=self.device)
        for index, layer in enumerate(graph_cache.layers):
            layer.keys = buffers.input_key_values[2 * index]
            layer.values = buffers.input_key_values[2 * index + 1]
            # note (kermit): Saturated windows have the same mask geometry at every absolute position.
            layer.cumulative_length = self.stored_frames + 1
        causal_mask = create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=torch.zeros(
                batch_size,
                frame_count,
                hidden_size,
                device=self.device,
                dtype=torch.float32,
            ),
            attention_mask=None,
            past_key_values=graph_cache,
            position_ids=positions,
        )
        attention_masks: dict[str, torch.Tensor | None] = {
            "full_attention": None,
            "sliding_attention": causal_mask,
        }

        def body() -> None:
            hidden_states = decoder.in_proj(latents)
            hidden_states = hidden_states.reshape(
                hidden_states.shape[0], -1, hidden_size
            )
            model_output = decoder.qwen3.forward(
                inputs_embeds=hidden_states,
                use_cache=True,
                past_key_values=graph_cache,
                position_ids=positions,
                attention_mask=attention_masks,
            )
            chunk, new_istft_cache = decoder.istft_head.forward_stream(
                model_output.last_hidden_state, buffers.input_istft_frames, False
            )
            waveform.copy_(chunk)
            buffers.output_istft_frames.copy_(new_istft_cache)
            for index, layer in enumerate(graph_cache.layers):
                buffers.output_key_values[2 * index].copy_(layer.keys)
                buffers.output_key_values[2 * index + 1].copy_(layer.values)

        try:
            with torch.no_grad():
                for _ in range(WARMUP_ITERATIONS):
                    body()
                torch.cuda.synchronize()
                for index, layer in enumerate(graph_cache.layers):
                    layer.keys = buffers.input_key_values[2 * index]
                    layer.values = buffers.input_key_values[2 * index + 1]
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                else:
                    pass
                if self.capture_stream is None:
                    self.capture_stream = torch.cuda.Stream(device=self.device)
                else:
                    pass
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=self.graph_pool,
                    stream=self.capture_stream,
                    capture_error_mode="thread_local",
                ):
                    body()
        except Exception:
            logger.warning(
                f"FireRedTTS3 vocoder CUDA graph capture failed for "
                f"latents={latent_frame_count} batch={batch_size}; eager fallback",
                exc_info=True,
            )
            return None
        return VocoderGraphEntry(
            graph=graph,
            latents=latents,
            positions=positions,
            causal_mask=causal_mask,
            waveform=waveform,
            buffers=buffers,
            graph_cache=graph_cache,
            attention_mask=attention_masks,
        )

    def find_entry(
        self, latents: torch.Tensor, state: RedAEDecoderState
    ) -> VocoderGraphEntry | None:
        """Find an eligible captured decoder shape."""
        if (
            not self.enabled
            or latents.device.type != "cuda"
            or latents.dtype != torch.float32
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        else:
            pass
        batch_size, latent_frame_count = int(latents.shape[0]), int(latents.shape[1])
        batch_bucket_size = next(
            (size for size in self.batch_buckets if size >= batch_size), None
        )
        if batch_bucket_size is None:
            return None
        else:
            pass
        entry = self.graphs.get((latent_frame_count, batch_bucket_size))
        if entry is None or state.qwen3_cache is None or state.istft_cache is None:
            return None
        else:
            pass
        keys = state.qwen3_cache.layers[0].keys
        if (
            keys is None
            or keys.ndim != 4
            or int(keys.shape[0]) != batch_size
            or int(keys.shape[-2]) != self.stored_frames
        ):
            return None
        else:
            pass
        return entry

    def run(
        self, latents: torch.Tensor, state: RedAEDecoderState
    ) -> tuple[torch.Tensor, RedAEDecoderState] | None:
        """Replay a decoder graph; return None when eager execution is required."""
        self.counters["calls"] += 1
        entry = self.find_entry(latents, state)
        if entry is None:
            self.counters["eager"] += 1
            return None
        else:
            pass
        batch_size = int(latents.shape[0])
        batch_bucket_size = int(entry.latents.shape[0])
        buffers = entry.buffers
        frame_count = int(entry.positions.shape[1])
        try:
            with torch.no_grad():
                entry.latents[:batch_size].copy_(latents)
                cache_layers = state.qwen3_cache.layers
                key_value_tensors: list[torch.Tensor] = []
                for layer in cache_layers:
                    key_value_tensors.append(layer.keys)
                    key_value_tensors.append(layer.values)
                if batch_size == batch_bucket_size:
                    torch.stack(key_value_tensors, dim=0, out=buffers.input_key_values)
                else:
                    buffers.input_key_values[:, :batch_size].copy_(
                        torch.stack(key_value_tensors, dim=0)
                    )
                buffers.input_istft_frames[:batch_size].copy_(state.istft_cache)
                torch.arange(
                    state.frames_seen,
                    state.frames_seen + frame_count,
                    dtype=torch.long,
                    device=self.device,
                    out=entry.positions[0],
                )
                entry.graph.replay()
            self.counters["replays"] += 1
        except Exception:
            self.enabled = False
            self.counters["eager"] += 1
            logger.warning(
                "FireRedTTS3 vocoder CUDA graph replay failed; disabling graphs",
                exc_info=True,
            )
            return None
        # note (kermit): Request state must survive the next replay's writes to static buffers.
        kv_snapshot = buffers.output_key_values[:, :batch_size].clone()
        istft_snapshot = buffers.output_istft_frames[:batch_size].clone()
        new_frames_seen = state.frames_seen + frame_count
        merged_cache = DynamicCache(config=self.decoder.qwen3_config)
        for index, layer in enumerate(merged_cache.layers):
            source = copy.copy(cache_layers[index])
            source.keys = kv_snapshot[2 * index]
            source.values = kv_snapshot[2 * index + 1]
            source.cumulative_length = new_frames_seen
            merged_cache.layers[index] = source
        new_state = RedAEDecoderState(merged_cache)
        new_state.istft_cache = istft_snapshot
        new_state.frames_seen = new_frames_seen
        return entry.waveform[:batch_size], new_state


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_LATENT_SIZES",
    "RedAEVocoderGraphRunner",
]
