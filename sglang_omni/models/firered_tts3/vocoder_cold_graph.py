# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for left-padded fresh-cache RedAE decoding."""

from __future__ import annotations

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
from sglang_omni.models.firered_tts3.vocoder_graph import WARMUP_ITERATIONS

logger = logging.getLogger(__name__)

DEFAULT_COLD_LATENT_SIZES = (32, 48, 64, 96, 128)
# note (kermit): Padding wider groups loses the launch savings on L20.
DEFAULT_COLD_BATCH_SIZES = (1, 2)


@dataclass(kw_only=True)
class ColdGraphEntry:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    positions: torch.Tensor
    causal_mask: torch.Tensor
    base_attention_mask: torch.Tensor
    row_indices: torch.Tensor
    column_indices: torch.Tensor
    waveform: torch.Tensor
    output_key_values: torch.Tensor
    output_istft_frames: torch.Tensor
    # note (kermit): Captured kernels retain raw addresses, so every cache and mask owner stays alive.
    graph_cache: DynamicCache
    attention_mask: dict[str, torch.Tensor | None]


class RedAEVocoderColdGraphRunner:
    """Replay fresh-cache RedAE chunks with left padding."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        latent_sizes: tuple[int, ...] = DEFAULT_COLD_LATENT_SIZES,
        batch_sizes: tuple[int, ...] = DEFAULT_COLD_BATCH_SIZES,
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
        self.minimum_latent_frames: int = -(-self.stored_frames // 2)
        istft = decoder.istft_head.istft
        self.minimum_lead_samples: int = int(istft.win_length) - int(istft.hop_length)
        self.latent_frame_buckets: tuple[int, ...] = tuple(
            sorted(
                {
                    int(size)
                    for size in latent_sizes
                    if int(size) >= self.minimum_latent_frames
                }
            )
        )
        self.batch_buckets: tuple[int, ...] = tuple(
            sorted({int(size) for size in batch_sizes if int(size) > 0})
        )
        self.enabled: bool = (
            self.device.type == "cuda"
            and bool(self.latent_frame_buckets)
            and bool(self.batch_buckets)
        )
        self.graphs: dict[tuple[int, int], ColdGraphEntry] = {}
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
                "FireRedTTS3 vocoder cold CUDA graphs: no key captured; eager only"
            )
            return
        else:
            pass
        logger.info(
            f"FireRedTTS3 vocoder cold CUDA graphs captured: {sorted(self.graphs)}"
            + (f" (skipped: {sorted(self.failed_keys)})" if self.failed_keys else "")
        )

    def capture(
        self, latent_frame_count: int, batch_size: int
    ) -> ColdGraphEntry | None:
        """Capture one latent-frame width and batch bucket."""
        decoder = self.decoder
        config = decoder.qwen3_config
        hidden_size = int(config.hidden_size)
        frame_count = latent_frame_count * int(decoder.upsample_rate)
        istft = decoder.istft_head.istft
        latents = torch.zeros(
            batch_size,
            latent_frame_count,
            decoder.in_proj.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.arange(
            frame_count, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        head_padding_samples = (istft.win_length - istft.hop_length) // 2
        emitted_sample_count = frame_count * istft.hop_length - head_padding_samples
        waveform = torch.zeros(
            batch_size, emitted_sample_count, device=self.device, dtype=torch.float32
        )
        output_key_values = torch.zeros(
            2 * self.layer_count,
            batch_size,
            self.key_value_head_count,
            self.stored_frames,
            self.attention_head_size,
            device=self.device,
            dtype=torch.float32,
        )
        overlap_frame_count = istft.win_length // istft.hop_length - 1
        output_istft_frames = torch.zeros(
            batch_size,
            istft.n_fft,
            overlap_frame_count,
            device=self.device,
            dtype=torch.float32,
        )
        graph_cache = build_initialized_cache(config, device=self.device)
        causal_mask = create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=torch.zeros(1, frame_count, hidden_size, device=self.device),
            attention_mask=None,
            past_key_values=graph_cache,
            position_ids=None,
        )
        assert causal_mask is not None and causal_mask.dtype == torch.bool
        causal_mask = causal_mask.clone()
        base_attention_mask = causal_mask[0, 0].clone()
        row_indices = torch.arange(frame_count, device=self.device).unsqueeze(1)
        column_indices = torch.arange(frame_count, device=self.device).unsqueeze(0)
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
                model_output.last_hidden_state, None, False
            )
            waveform.copy_(chunk)
            output_istft_frames.copy_(new_istft_cache)
            for index, layer in enumerate(graph_cache.layers):
                output_key_values[2 * index].copy_(layer.keys)
                output_key_values[2 * index + 1].copy_(layer.values)

        try:
            with torch.no_grad():
                empty = torch.zeros(0, device=self.device, dtype=torch.float32)
                for _ in range(WARMUP_ITERATIONS + 1):
                    # note (kermit): Each cold warmup restarts from an empty cache.
                    for layer in graph_cache.layers:
                        layer.keys = empty
                        layer.values = empty
                        layer.cumulative_length = 0
                    body()
                torch.cuda.synchronize()
                for layer in graph_cache.layers:
                    layer.keys = empty
                    layer.values = empty
                    layer.cumulative_length = 0
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
                f"FireRedTTS3 vocoder cold CUDA graph capture failed for "
                f"latents={latent_frame_count} batch={batch_size}; eager fallback",
                exc_info=True,
            )
            return None
        return ColdGraphEntry(
            graph=graph,
            latents=latents,
            positions=positions,
            causal_mask=causal_mask,
            base_attention_mask=base_attention_mask,
            row_indices=row_indices,
            column_indices=column_indices,
            waveform=waveform,
            output_key_values=output_key_values,
            output_istft_frames=output_istft_frames,
            graph_cache=graph_cache,
            attention_mask=attention_masks,
        )

    def find_entry(
        self, latents: torch.Tensor, lead_samples: int
    ) -> ColdGraphEntry | None:
        """Find an eligible captured decoder shape."""
        if (
            not self.enabled
            or latents.device.type != "cuda"
            or latents.dtype != torch.float32
            or torch.cuda.is_current_stream_capturing()
            or lead_samples < self.minimum_lead_samples
        ):
            return None
        else:
            pass
        batch_size, latent_frame_count = int(latents.shape[0]), int(latents.shape[1])
        if latent_frame_count < self.minimum_latent_frames:
            return None
        else:
            pass
        batch_bucket_size = next(
            (size for size in self.batch_buckets if size >= batch_size), None
        )
        if batch_bucket_size is None:
            return None
        else:
            pass
        latent_bucket_frames = next(
            (size for size in self.latent_frame_buckets if size >= latent_frame_count),
            None,
        )
        if latent_bucket_frames is None:
            return None
        else:
            pass
        return self.graphs.get((latent_bucket_frames, batch_bucket_size))

    def run(
        self, latents: torch.Tensor, *, lead_samples: int
    ) -> tuple[torch.Tensor, RedAEDecoderState] | None:
        """Replay a decoder graph; return None when eager execution is required."""
        self.counters["calls"] += 1
        entry = self.find_entry(latents, lead_samples)
        if entry is None:
            self.counters["eager"] += 1
            return None
        else:
            pass
        batch_size, latent_frame_count = int(latents.shape[0]), int(latents.shape[1])
        latent_bucket_frames = int(entry.latents.shape[1])
        upsample_rate = int(self.decoder.upsample_rate)
        real_frames = latent_frame_count * upsample_rate
        pad_frames = latent_bucket_frames * upsample_rate - real_frames
        try:
            with torch.no_grad():
                entry.latents.zero_()
                entry.latents[
                    :batch_size, latent_bucket_frames - latent_frame_count :, :
                ].copy_(latents)
                if pad_frames:
                    entry.positions[0, :pad_frames].zero_()
                else:
                    pass
                torch.arange(
                    real_frames,
                    dtype=torch.long,
                    device=self.device,
                    out=entry.positions[0, pad_frames:],
                )
                pad_block = (entry.column_indices >= pad_frames) | (
                    entry.row_indices < pad_frames
                )
                torch.logical_and(
                    entry.base_attention_mask, pad_block, out=entry.causal_mask[0, 0]
                )
                entry.graph.replay()
            self.counters["replays"] += 1
        except Exception:
            self.enabled = False
            self.counters["eager"] += 1
            logger.warning(
                "FireRedTTS3 vocoder cold CUDA graph replay failed; disabling graphs",
                exc_info=True,
            )
            return None
        kv_snapshot = entry.output_key_values[:, :batch_size].clone()
        istft_snapshot = entry.output_istft_frames[:batch_size].clone()
        cache = build_initialized_cache(
            self.decoder.qwen3_config, device=self.device, dtype=torch.float32
        )
        for index, layer in enumerate(cache.layers):
            layer.keys = kv_snapshot[2 * index]
            layer.values = kv_snapshot[2 * index + 1]
            layer.cumulative_length = real_frames
        new_state = RedAEDecoderState(cache)
        new_state.istft_cache = istft_snapshot
        new_state.frames_seen = real_frames
        hop_samples = int(self.decoder.istft_head.istft.hop_length)
        # note (kermit): The lead-sample gate keeps padded overlap artifacts inside the trimmed prompt.
        start_sample = pad_frames * hop_samples
        emitted_sample_count = (
            real_frames * hop_samples - self.minimum_lead_samples // 2
        )
        return (
            entry.waveform[
                :batch_size, start_sample : start_sample + emitted_sample_count
            ],
            new_state,
        )


__all__ = [
    "DEFAULT_COLD_BATCH_SIZES",
    "DEFAULT_COLD_LATENT_SIZES",
    "RedAEVocoderColdGraphRunner",
]
