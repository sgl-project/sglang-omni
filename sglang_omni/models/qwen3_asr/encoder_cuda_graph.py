# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Qwen3-ASR audio encoder layer stack.

The chunk/conv front end reads each clip's length, so its shapes and control
flow change from request to request — we leave it on the eager path. What
we capture is the 24-layer transformer stack and the output projection that
follow. By then the batch is packed as [total_tokens, hidden]. Graphs are keyed
by token bucket. Ascend captures its host-side window boundaries as updatable
graph inputs so different layouts in one bucket can replay the same graph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.layers.attention.vision import VisionAttentionMetadata

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import get_npu_graph_update_stream

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
    npu_update_tasks: tuple[_NpuGraphUpdateTask, ...] = ()


@dataclass
class _NpuGraphUpdateTask:
    """One captured FIA task whose host-side sequence boundaries can change."""

    operation: Any
    kwargs: dict[str, Any]
    handle: Any
    event: Any

    def apply(
        self,
        device_module: Any,
        update_stream: Any,
        cumulative_window_lens: list[int],
    ) -> None:
        device_module.graph_task_update_begin(update_stream, self.handle)
        self.operation(
            **self.kwargs,
            actual_seq_lengths=cumulative_window_lens,
            actual_seq_lengths_kv=cumulative_window_lens,
        )
        device_module.graph_task_update_end(update_stream)
        self.event.record(update_stream)


@dataclass
class _NpuGraphCaptureState:
    tasks: list[_NpuGraphUpdateTask]
    workspace: torch.Tensor | None = None


class _NpuUpdatableAttentionBackend(torch.nn.Module):
    """Capture Ascend FIA as an explicitly updatable graph task group."""

    _FIA_BLOCK_SIZE = 128
    _INT_MAX = 2_147_483_647

    def __init__(
        self, capture_state: _NpuGraphCaptureState, device_module: Any
    ) -> None:
        super().__init__()
        self._capture_state = capture_state
        self._device_module = device_module

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        forward_metadata: VisionAttentionMetadata | None = None,
        attention_mask: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if forward_metadata is None or attention_mask is not None:
            raise RuntimeError(
                "NPU encoder graph capture requires unmasked Ascend attention "
                "with precomputed sequence metadata"
            )

        import torch_npu

        cumulative_window_lens = (
            forward_metadata.cu_seqlens[1:].to(torch.int32).tolist()
        )
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        scale = softmax_scale if softmax_scale is not None else q.shape[2] ** -0.5
        output = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
        fia_kwargs = {
            "query": q,
            "key": k,
            "value": v,
            "atten_mask": None,
            "block_table": None,
            "input_layout": "TND",
            "block_size": self._FIA_BLOCK_SIZE,
            "num_key_value_heads": num_kv_heads,
            "num_heads": num_heads,
            "scale": scale,
            "sparse_mode": 0,
            "pre_tokens": self._INT_MAX,
            "next_tokens": self._INT_MAX,
        }
        state = self._capture_state
        if state.workspace is None:
            state.workspace = (
                torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    **fia_kwargs,
                    actual_seq_lengths=cumulative_window_lens,
                    actual_seq_lengths_kv=cumulative_window_lens,
                )
            )

        operation = torch_npu.npu_fused_infer_attention_score.out
        operation_kwargs = {
            **fia_kwargs,
            "workspace": state.workspace,
            "out": [output, softmax_lse],
        }
        device_module = self._device_module
        stream = device_module.current_stream()
        event = device_module.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        device_module.graph_task_group_begin(stream)
        operation(
            **operation_kwargs,
            actual_seq_lengths=cumulative_window_lens,
            actual_seq_lengths_kv=cumulative_window_lens,
        )
        handle = device_module.graph_task_group_end(stream)
        state.tasks.append(
            _NpuGraphUpdateTask(
                operation=operation,
                kwargs=operation_kwargs,
                handle=handle,
                event=event,
            )
        )
        return output


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
        self._is_npu = current_platform.is_npu()
        cfg = audio_tower.config

        chunk_tokens = _get_feat_extract_output_lengths_int(cfg.n_window * 2)
        self._max_seqlen = chunk_tokens * (cfg.n_window_infer // (cfg.n_window * 2))
        self._max_windows_for = lambda bucket_size: (
            max_batch_size + bucket_size // self._max_seqlen + 1
        )
        top = buckets[-1]
        self._buckets = buckets[:-1] + (top + self._max_windows_for(top),)
        self._graphs: dict[int, _CapturedGraph] = {}
        self._failed: set[int] = set()
        self._graph_pool: Any | None = None
        self._capture_failed = False
        self._npu_update_stream = (
            get_npu_graph_update_stream() if self._is_npu else None
        )

    @property
    def tokens_per_window(self) -> int:
        return self._max_seqlen

    def capture_all(self) -> None:
        """Capture every bucket up front, except on NPU where capture is lazy."""
        if self._is_npu:
            return
        for bucket_size in self._buckets:
            if bucket_size in self._graphs or bucket_size in self._failed:
                continue
            try:
                self._graphs[bucket_size] = self._capture(bucket_size)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[qwen3-asr] encoder graph capture failed for bucket=%d: %s; "
                    "bucket stays eager",
                    bucket_size,
                    exc,
                )
                self._failed.add(bucket_size)

    def _layer_stack(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_metadata: VisionAttentionMetadata | None,
    ) -> torch.Tensor:
        """The computation we capture: 24 layers + ln_post + proj chain."""
        tower = self._tower
        h = hidden_states
        for layer in tower.layers:
            residual = h
            h = layer.self_attn_layer_norm(h)
            h = layer.self_attn(
                x=h,
                cu_seqlens=cu_seqlens,
                max_seqlen=self._max_seqlen,
                forward_metadata=attention_metadata,
            )
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

    def _make_cu_seqlens(
        self, sizes: Sequence[int], *, device: torch.device | str
    ) -> torch.Tensor:
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        return torch.tensor(bounds, dtype=torch.int32, device=device)

    def _capture_pool(self) -> Any | None:
        if not self._is_npu:
            return None
        if self._graph_pool is None:
            self._graph_pool = self._device_module.graph_pool_handle()
        return self._graph_pool

    @contextmanager
    def _capture_npu_attention_tasks(
        self, state: _NpuGraphCaptureState
    ) -> Iterator[None]:
        """Temporarily route each Qwen3-ASR FIA call through task-group capture."""
        replacements: list[tuple[Any, torch.nn.Module]] = []
        for layer in self._tower.layers:
            attention = layer.self_attn
            if attention.qkv_backend_name != "ascend_attn":
                raise RuntimeError(
                    "NPU encoder graph capture requires the ascend_attn backend"
                )
            original = attention.qkv_backend
            replacements.append((attention, original))
            attention.qkv_backend = _NpuUpdatableAttentionBackend(
                state, self._device_module
            )
        try:
            yield
        finally:
            for attention, original in replacements:
                attention.qkv_backend = original

    def _update_npu_attention_tasks(
        self,
        entry: _CapturedGraph,
        cumulative_window_lens: list[int],
    ) -> None:
        update_stream = self._npu_update_stream
        if update_stream is None:
            raise RuntimeError("NPU encoder graph update stream is not initialized")
        with self._device_module.stream(update_stream):
            for task in entry.npu_update_tasks:
                task.apply(
                    self._device_module,
                    update_stream,
                    cumulative_window_lens,
                )

    def _capture(
        self,
        bucket_size: int,
        *,
        window_lens: tuple[int, ...] | None = None,
    ) -> _CapturedGraph:
        """Record one graph for a bucket-sized packed input."""
        device, dtype = self._device, self._dtype
        d_model = self._tower.ln_post.normalized_shape[0]
        static_hs = torch.zeros(bucket_size, d_model, device=device, dtype=dtype)

        if self._is_npu:
            if not window_lens or sum(window_lens) != bucket_size:
                raise ValueError(
                    "NPU encoder graph capture requires an exact window "
                    f"signature for bucket {bucket_size}"
                )
            sizes = list(window_lens)
            max_windows = len(sizes)
        else:
            max_windows = self._max_windows_for(bucket_size)
            base, rem = divmod(bucket_size, max_windows)
            sizes = [base + 1] * rem + [base] * (max_windows - rem)
        static_cu = self._make_cu_seqlens(
            sizes, device="cpu" if self._is_npu else self._device
        )
        attention_metadata = None
        if current_platform.is_rocm() or self._is_npu:
            # VisionAiterAttention otherwise recomputes max_seqlen with
            # seq_lens.max().item() inside the captured region. The device-to-host
            # sync is illegal during HIP graph capture. Ascend similarly turns
            # the boundaries into host-side operator parameters, so keep them
            # host-resident before capture.
            attention_metadata = VisionAttentionMetadata(
                cu_seqlens=static_cu,
                seq_lens=static_cu[1:] - static_cu[:-1],
                max_seqlen=self._max_seqlen,
            )

        def run_once() -> torch.Tensor:
            with torch.no_grad():
                return self._layer_stack(static_hs, static_cu, attention_metadata)

        device_module = self._device_module
        side = device_module.Stream(device)
        side.wait_stream(device_module.current_stream(device))
        with device_module.stream(side):
            for _ in range(3):
                run_once()
        device_module.current_stream(device).wait_stream(side)
        device_module.synchronize(device)

        # NPU captures share one pool; CUDA/ROCm keep their existing private-pool
        # behavior.
        capture_state = _NpuGraphCaptureState(tasks=[])
        attention_capture = (
            self._capture_npu_attention_tasks(capture_state)
            if self._is_npu
            else nullcontext()
        )
        with (
            attention_capture,
            self._graph_backend.capture(
                pool=self._capture_pool(), thread_local_errors=True
            ) as graph,
        ):
            static_out = run_once()
        if self._is_npu and len(capture_state.tasks) != len(self._tower.layers):
            raise RuntimeError(
                "NPU encoder graph did not capture one FIA task per encoder layer: "
                f"tasks={len(capture_state.tasks)} layers={len(self._tower.layers)}"
            )
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
            npu_update_tasks=tuple(capture_state.tasks),
        )

    def run(
        self, hidden_states: torch.Tensor, window_lens: list[int]
    ) -> torch.Tensor | None:
        """Replay the recorded graph for a batch of hidden states."""
        if self._is_npu and self._capture_failed:
            raise RuntimeError(
                "NPU encoder graph capture previously failed; restart the "
                "encoder process with encoder graphs disabled before retrying"
            )
        total = int(hidden_states.shape[0])
        if not window_lens or sum(window_lens) != total:
            return None
        if max(window_lens) > self._max_seqlen:
            return None

        plan = self._plan(total, len(window_lens))
        if plan is None:
            return None
        bucket_size, dummy_sizes = plan
        effective_window_lens = tuple(window_lens + dummy_sizes)
        graph_key = bucket_size
        if not self._is_npu and graph_key in self._failed:
            return None

        entry = self._graphs.get(graph_key)
        if entry is None:
            try:
                entry = self._capture(
                    bucket_size,
                    window_lens=(effective_window_lens if self._is_npu else None),
                )
            except Exception as exc:
                if not self._is_npu:
                    raise
                self._capture_failed = True
                raise RuntimeError(
                    "NPU encoder graph capture failed; restart the encoder "
                    "process with encoder graphs disabled before retrying"
                ) from exc
            self._graphs[graph_key] = entry

        entry.hidden_states[:total].copy_(hidden_states)
        cu = self._make_cu_seqlens(effective_window_lens, device="cpu")
        if self._is_npu:
            cumulative_window_lens = cu[1:].tolist()
        else:
            entry.cu_seqlens.copy_(cu, non_blocking=True)
            if entry.attention_metadata is not None:
                entry.attention_metadata.seq_lens.copy_(
                    cu[1:] - cu[:-1], non_blocking=True
                )
        if self._is_npu:
            from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_submission import (
                npu_graph_submission,
            )

            with npu_graph_submission(
                self._device_module, self._npu_update_stream, source="encoder"
            ):
                self._graph_backend.replay(entry.graph)
                self._update_npu_attention_tasks(entry, cumulative_window_lens)
        else:
            self._graph_backend.replay(entry.graph)
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
