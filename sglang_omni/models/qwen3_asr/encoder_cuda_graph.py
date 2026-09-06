# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Qwen3-ASR audio encoder layer stack.

The chunk/conv front end reads each clip's length, so its shapes and control
flow change from request to request — we leave it on the eager path. What
we capture is the 24-layer transformer stack and the output projection that
follow. By then the batch is packed as [total_tokens, hidden], so capture
buckets only need to track total token count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.layers.attention.vision import VisionAttentionMetadata

from sglang_omni.platforms import current_platform

if TYPE_CHECKING:
    from sglang_omni.platforms.device_graph import DeviceGraphBackend

logger = logging.getLogger(__name__)


def build_buckets(max_batch: int, max_tokens_per_clip: int) -> tuple[int, ...]:
    """Return token-count bucket sizes for encoder CUDA-graph capture.

    Each captured graph pads the packed [total_tokens, hidden] encoder
    input up to one of these sizes. The caller supplies two deployment
    limits:
    1. max_batch: pre_lm_max_batch_size on the per-LM encoder service
      (maximum clips in one encode batch).
    2. max_tokens_per_clip: encoder output tokens for the longest clip the
      pipeline admits; derived from AudioChunkingConfig.max_audio_clip_s
      via qwen3_asr_num_audio_tokens.

    Buckets are power-of-two sizes from 128 up to max_batch * max_tokens_per_clip.
    """
    if max_batch < 1 or max_tokens_per_clip < 1:
        raise ValueError(
            f"build_buckets needs positive limits, got max_batch={max_batch} "
            f"max_tokens_per_clip={max_tokens_per_clip}"
        )
    ceiling = int(max_batch) * int(max_tokens_per_clip)
    buckets: list[int] = []
    step = 128
    while step < ceiling:
        buckets.append(step)
        step *= 2
    buckets.append(ceiling)
    return tuple(buckets)


@dataclass
class _CapturedGraph:
    graph: Any  # the accelerator's graph type, named per backend
    hidden_states: torch.Tensor  # [bucket, hidden] static input
    cu_seqlens: torch.Tensor  # [max_windows + 1] static window boundaries
    attention_metadata: VisionAttentionMetadata | None
    output: torch.Tensor  # [bucket, output_dim] static result


class Qwen3ASREncoderLayerStackGraphRunner:
    """Captures the audio tower's transformer stack (post-conv) per bucket.

    The chunk/conv front end stays eager; callers hand over the packed
    hidden_states it produced plus the window boundaries, and get back the
    projected embeddings. max_seqlen is the architectural cap on tokens per
    attention window; materializing it once on the host removes the per-layer
    sync inside VisionAttention.
    """

    def __init__(
        self,
        audio_tower: Any,
        *,
        buckets: tuple[int, ...],
        max_batch_size: int,
        graph_backend: DeviceGraphBackend,
    ) -> None:
        self._tower = audio_tower
        self._graph_backend = graph_backend
        param = next(audio_tower.parameters())
        self._device = param.device
        self._dtype = param.dtype
        self._device_module = torch.get_device_module(self._device)
        cfg = audio_tower.config

        chunk_tokens = _get_feat_extract_output_lengths_int(cfg.n_window * 2)
        self._max_seqlen = chunk_tokens * (cfg.n_window_infer // (cfg.n_window * 2))
        self._max_windows_for = (
            lambda bucket_size: max_batch_size + bucket_size // self._max_seqlen + 1
        )
        top = buckets[-1]
        self._buckets = buckets[:-1] + (top + self._max_windows_for(top),)
        self._graphs: dict[int, _CapturedGraph] = {}  # bucket size -> recorded graph
        self._failed: set[int] = set()
        self._capture_attention_metadata: VisionAttentionMetadata | None = None

    @property
    def tokens_per_window(self) -> int:
        return self._max_seqlen

    def capture_all(self) -> None:
        """Capture every bucket up front. A failed bucket stays eager."""
        for bucket_size in self._buckets:
            if bucket_size in self._graphs or bucket_size in self._failed:
                continue
            try:
                self._graphs[bucket_size] = self._capture(bucket_size)
            except Exception as exc:
                logger.warning(
                    "[qwen3-asr] encoder graph capture failed for bucket=%d: %s; "
                    "bucket stays eager",
                    bucket_size,
                    exc,
                )
                self._failed.add(bucket_size)

    def _layer_stack(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """The computation we capture: 24 layers + ln_post + proj chain."""
        tower = self._tower
        h = hidden_states
        for layer in tower.layers:
            residual = h
            h = layer.self_attn_layer_norm(h)
            attention_kwargs = dict(
                max_seqlen=self._max_seqlen,
                forward_metadata=self._capture_attention_metadata,
            )
            h = layer.self_attn(x=h, cu_seqlens=cu_seqlens, **attention_kwargs)
            h = residual + h
            residual = h
            h = layer.final_layer_norm(h)
            h, _ = layer.fc1(h)
            h = layer.activation_fn(h)
            h, _ = layer.fc2(h)
            h = residual + h
        h = tower.ln_post(h)
        h = tower.proj1(h)[0]
        h = tower.act(h)
        return tower.proj2(h)[0]

    def _capture(self, bucket_size: int) -> _CapturedGraph:
        """Record one graph for a bucket-sized packed input."""
        device, dtype = self._device, self._dtype
        d_model = self._tower.ln_post.normalized_shape[0]
        static_hs = torch.zeros(bucket_size, d_model, device=device, dtype=dtype)

        max_windows = self._max_windows_for(bucket_size)
        base, rem = divmod(bucket_size, max_windows)
        sizes = [base + 1] * rem + [base] * (max_windows - rem)
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        static_cu = torch.tensor(bounds, dtype=torch.int32, device=device)
        attention_metadata = None
        if current_platform.is_rocm():
            # VisionAiterAttention otherwise recomputes max_seqlen with
            # seq_lens.max().item() inside the captured region. The device-to-host
            # sync is illegal during HIP graph capture. Keep the mutable tensor
            # metadata static and supply the architectural maximum as a host scalar.
            attention_metadata = VisionAttentionMetadata(
                cu_seqlens=static_cu,
                seq_lens=static_cu[1:] - static_cu[:-1],
                max_seqlen=self._max_seqlen,
            )
        self._capture_attention_metadata = attention_metadata

        def run_once() -> torch.Tensor:
            with torch.no_grad():
                return self._layer_stack(static_hs, static_cu)

        device_module = self._device_module
        side = device_module.Stream(device)
        side.wait_stream(device_module.current_stream(device))
        with device_module.stream(side):
            for _ in range(3):
                run_once()
        device_module.current_stream(device).wait_stream(side)
        device_module.synchronize(device)

        # Note (siju): each bucket keeps its own pool. Sharing one is only safe
        # for graphs replayed in capture order, and a request picks its bucket
        # from the clip length, so any order is possible.
        with self._graph_backend.capture(thread_local_errors=True) as graph:
            static_out = run_once()
        logger.info(
            "[qwen3-asr] captured encoder layer-stack graph bucket=%d windows=%d out=%s",
            bucket_size,
            max_windows,
            tuple(static_out.shape),
        )
        return _CapturedGraph(
            graph=graph,
            hidden_states=static_hs,
            cu_seqlens=static_cu,
            attention_metadata=attention_metadata,
            output=static_out,
        )

    def run(
        self, hidden_states: torch.Tensor, window_lens: list[int]
    ) -> torch.Tensor | None:
        """Replay the recorded graph for a batch of hidden states."""
        total = int(hidden_states.shape[0])
        if not window_lens or sum(window_lens) != total:
            return None
        if max(window_lens) > self._max_seqlen:
            return None

        plan = self._plan(total, len(window_lens))
        if plan is None:
            return None
        bucket_size, dummy_sizes = plan
        if bucket_size in self._failed:
            return None

        entry = self._graphs.get(bucket_size)
        if entry is None:
            try:
                entry = self._capture(bucket_size)
            except Exception as exc:
                logger.warning(
                    "[qwen3-asr] encoder graph capture failed for bucket=%d: %s; "
                    "bucket stays eager",
                    bucket_size,
                    exc,
                )
                self._failed.add(bucket_size)
                return None
            self._graphs[bucket_size] = entry

        bounds = [0]
        for size in window_lens + dummy_sizes:
            bounds.append(bounds[-1] + size)
        cu = torch.tensor(bounds, dtype=torch.int32)

        entry.hidden_states[:total].copy_(hidden_states)
        entry.cu_seqlens.copy_(cu, non_blocking=True)
        if entry.attention_metadata is not None:
            entry.attention_metadata.seq_lens.copy_(cu[1:] - cu[:-1], non_blocking=True)
        entry.graph.replay()
        out = entry.output
        if out.dim() == 3:  # attention backends emit [1, tokens, dim]
            out = out.squeeze(0)
        return out[:total].clone()

    def _plan(self, total: int, real_windows: int) -> tuple[int, list[int]] | None:
        """Pick a bucket and the dummy-window sizes that absorb its padding."""

        for bucket_size in self._buckets:
            if bucket_size < total:
                continue
            slots = self._max_windows_for(bucket_size) - real_windows
            pad = bucket_size - total
            if slots < 0:
                continue
            if slots == 0:
                if pad == 0:
                    return bucket_size, []
                continue
            if not (slots <= pad <= slots * self._max_seqlen):
                continue
            base, rem = divmod(pad, slots)
            sizes = [base + 1] * rem + [base] * (slots - rem)
            return bucket_size, sizes
        return None


def _get_feat_extract_output_lengths_int(frames: int) -> int:
    """Compute conv output length for a mel-frame count."""
    from .audio_lengths import qwen3_asr_num_audio_tokens

    return int(qwen3_asr_num_audio_tokens(frames))


def eager_preamble(
    tower: Any, input_features: torch.Tensor, feature_lens: torch.Tensor
) -> torch.Tensor:

    import torch.nn.functional as F

    chunk_width = tower.n_window * 2
    chunk_num = torch.ceil(feature_lens / chunk_width).long()
    chunk_lengths = torch.tensor(
        [chunk_width] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_width
    chunk_lengths[chunk_lengths == 0] = chunk_width

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(
        chunk_list, batch_first=True
    ).transpose(1, 2)

    feature_lens_after_cnn = _get_feat_extract_output_lengths_tensor(chunk_lengths)
    max_len_after_cnn = (
        int(feature_lens_after_cnn.max().item())
        if feature_lens_after_cnn.numel()
        else 0
    )
    idx = torch.arange(max_len_after_cnn, device=padded_feature.device)
    padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

    padded_feature = padded_feature.unsqueeze(1)
    if padded_feature.size(0) <= tower.conv_chunksize:
        padded_embed = F.gelu(tower.conv2d1(padded_feature))
        padded_embed = F.gelu(tower.conv2d2(padded_embed))
        padded_embed = F.gelu(tower.conv2d3(padded_embed))
    else:
        padded_embeds = []
        for chunk in padded_feature.split(tower.conv_chunksize, dim=0):
            x = F.gelu(tower.conv2d1(chunk))
            x = F.gelu(tower.conv2d2(x))
            x = F.gelu(tower.conv2d3(x))
            padded_embeds.append(x)
        padded_embed = torch.cat(padded_embeds, dim=0)

    b, c, f, t = padded_embed.size()
    padded_embed = tower.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )[0]
    positional_embedding = (
        tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    return padded_embed[padded_mask_after_cnn]


def _get_feat_extract_output_lengths_tensor(
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    leave = input_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def window_lens_from_token_counts(
    token_counts: list[int], *, tokens_per_window: int
) -> list[int]:
    out: list[int] = []
    for count in token_counts:
        full, rem = divmod(int(count), tokens_per_window)
        out.extend([tokens_per_window] * full)
        if rem:
            out.append(rem)
    return out
