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
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.layers.attention.vision import VisionAttentionMetadata

from sglang_omni.platforms import current_platform

if TYPE_CHECKING:
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass

logger = logging.getLogger(__name__)


class EncoderGraphUnrecoverableError(RuntimeError):
    """An NPU graph capture cannot be safely retried."""


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
    else:
        pass
    ceiling = int(max_batch) * int(max_tokens_per_clip)
    buckets: list[int] = []
    step = 128
    while step < ceiling:
        buckets.append(step)
        step *= 2
    buckets.append(ceiling)
    return tuple(buckets)


@dataclass
class CapturedGraph:
    graph: Any  # the accelerator's graph type, named per backend
    hidden_states: torch.Tensor  # [bucket, hidden] static input
    cu_seqlens: torch.Tensor  # [max_windows + 1] static window boundaries
    attention_metadata: VisionAttentionMetadata | None
    output: torch.Tensor  # [bucket, output_dim] static result
    npu_update_tasks: tuple[NpuGraphUpdateTask, ...] = ()


@dataclass
class NpuGraphUpdateTask:
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
class NpuGraphCaptureContext:
    tasks: list[NpuGraphUpdateTask]
    workspace: torch.Tensor | None = None


class NpuGraphCaptureAttention(torch.nn.Module):
    """Capture Ascend FIA as an explicitly updatable graph task group."""

    FIA_BLOCK_SIZE = 128
    INT32_MAX = torch.iinfo(torch.int32).max

    def __init__(
        self, capture_context: NpuGraphCaptureContext, device_module: Any
    ) -> None:
        super().__init__()
        self.capture_context = capture_context
        self.device_module = device_module

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
        else:
            pass

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
            "block_size": self.FIA_BLOCK_SIZE,
            "num_key_value_heads": num_kv_heads,
            "num_heads": num_heads,
            "scale": scale,
            "sparse_mode": 0,
            "pre_tokens": self.INT32_MAX,
            "next_tokens": self.INT32_MAX,
        }
        context = self.capture_context
        if context.workspace is None:
            context.workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                **fia_kwargs,
                actual_seq_lengths=cumulative_window_lens,
                actual_seq_lengths_kv=cumulative_window_lens,
            )
        else:
            pass

        operation = torch_npu.npu_fused_infer_attention_score.out
        operation_kwargs = {
            **fia_kwargs,
            "workspace": context.workspace,
            "out": [output, softmax_lse],
        }
        device_module = self.device_module
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
        context.tasks.append(
            NpuGraphUpdateTask(
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
        self.tower = audio_tower
        self.graph_backend = graph_backend
        param = next(audio_tower.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self.device_module = torch.get_device_module(self.device)
        cfg = audio_tower.config

        chunk_tokens = get_feat_extract_output_lengths_int(cfg.n_window * 2)
        self.max_seqlen = chunk_tokens * (cfg.n_window_infer // (cfg.n_window * 2))
        self.max_windows_for = (
            lambda bucket_size: max_batch_size + bucket_size // self.max_seqlen + 1
        )
        top = buckets[-1]
        self.buckets = buckets[:-1] + (top + self.max_windows_for(top),)
        self.graphs: dict[int, CapturedGraph] = {}  # bucket size -> recorded graph
        self.failed: set[int] = set()
        self.graph_pool: Any | None = None
        # Keep encoder task updates on a stream owned by this runner and device.
        self.npu_update_stream = (
            self.device_module.Stream(device=self.device)
            if self.graph_backend.supports_graph_task_update
            else None
        )

    @property
    def tokens_per_window(self) -> int:
        return self.max_seqlen

    def capture_all(self) -> None:
        """Capture every bucket up front, except on NPU where capture is lazy."""
        if self.graph_backend.supports_graph_task_update:
            return
        else:
            pass
        for bucket_size in self.buckets:
            if bucket_size in self.graphs or bucket_size in self.failed:
                continue
            else:
                pass
            try:
                self.graphs[bucket_size] = self.capture(bucket_size)
            except Exception as exc:
                logger.warning(
                    "[qwen3-asr] encoder graph capture failed for bucket=%d: %s; "
                    "bucket stays eager",
                    bucket_size,
                    exc,
                )
                self.failed.add(bucket_size)

    def layer_stack(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_metadata: VisionAttentionMetadata | None,
    ) -> torch.Tensor:
        """Run the captured stack with capture-local attention metadata.

        Passing metadata explicitly keeps one capture's host-side boundaries out
        of mutable runner state and makes the dependency visible to each layer.
        """
        tower = self.tower
        h = hidden_states
        for layer in tower.layers:
            residual = h
            h = layer.self_attn_layer_norm(h)
            h = layer.self_attn(
                x=h,
                cu_seqlens=cu_seqlens,
                max_seqlen=self.max_seqlen,
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

    def capture_pool(self) -> Any | None:
        if not self.graph_backend.supports_graph_task_update:
            return None
        else:
            pass
        if self.graph_pool is None:
            self.graph_pool = self.device_module.graph_pool_handle()
        else:
            pass
        return self.graph_pool

    @contextmanager
    def capture_npu_attention_tasks(
        self, context: NpuGraphCaptureContext
    ) -> Iterator[None]:
        """Temporarily route each Qwen3-ASR FIA call through task-group capture."""
        replacements: list[tuple[Any, torch.nn.Module]] = []
        try:
            for layer in self.tower.layers:
                attention = layer.self_attn
                if attention.qkv_backend_name != "ascend_attn":
                    raise RuntimeError(
                        "NPU encoder graph capture requires the ascend_attn backend"
                    )
                else:
                    pass
                original = attention.qkv_backend
                replacements.append((attention, original))
                attention.qkv_backend = NpuGraphCaptureAttention(
                    context, self.device_module
                )
            yield
        finally:
            for attention, original in replacements:
                attention.qkv_backend = original

    def update_npu_attention_tasks(
        self,
        entry: CapturedGraph,
        cumulative_window_lens: list[int],
    ) -> None:
        update_stream = self.npu_update_stream
        if update_stream is None:
            raise RuntimeError("NPU encoder graph update stream is not initialized")
        else:
            pass
        with self.device_module.stream(update_stream):
            for task in entry.npu_update_tasks:
                task.apply(
                    self.device_module,
                    update_stream,
                    cumulative_window_lens,
                )

    def capture(
        self,
        bucket_size: int,
        *,
        window_lens: tuple[int, ...] | None = None,
    ) -> CapturedGraph:
        """Record one graph for a bucket-sized packed input."""
        device, dtype = self.device, self.dtype
        d_model = self.tower.ln_post.normalized_shape[0]
        static_hs = torch.zeros(bucket_size, d_model, device=device, dtype=dtype)

        if self.graph_backend.supports_graph_task_update:
            if not window_lens or sum(window_lens) != bucket_size:
                raise ValueError(
                    "NPU encoder graph capture requires an exact window "
                    f"signature for bucket {bucket_size}"
                )
            else:
                pass
            sizes = list(window_lens)
            max_windows = len(sizes)
        else:
            max_windows = self.max_windows_for(bucket_size)
            base, rem = divmod(bucket_size, max_windows)
            sizes = [base + 1] * rem + [base] * (max_windows - rem)
        static_cu = torch.tensor(
            list(accumulate(sizes, initial=0)),
            dtype=torch.int32,
            device=(
                "cpu" if self.graph_backend.supports_graph_task_update else self.device
            ),
        )
        attention_metadata = None
        if current_platform.is_rocm() or self.graph_backend.supports_graph_task_update:
            # VisionAiterAttention otherwise recomputes max_seqlen with
            # seq_lens.max().item() inside the captured region. The device-to-host
            # sync is illegal during HIP graph capture. Ascend similarly turns
            # the boundaries into host-side operator parameters, so keep them
            # host-resident before capture.
            attention_metadata = VisionAttentionMetadata(
                cu_seqlens=static_cu,
                seq_lens=static_cu[1:] - static_cu[:-1],
                max_seqlen=self.max_seqlen,
            )
        else:
            pass

        def run_once() -> torch.Tensor:
            with torch.no_grad():
                return self.layer_stack(static_hs, static_cu, attention_metadata)

        device_module = self.device_module
        side = device_module.Stream(device)
        side.wait_stream(device_module.current_stream(device))
        with device_module.stream(side):
            for _ in range(3):
                run_once()
        device_module.current_stream(device).wait_stream(side)
        device_module.synchronize(device)

        # NPU captures share one pool; CUDA/ROCm keep their existing private-pool
        # behavior.
        capture_context = NpuGraphCaptureContext(tasks=[])
        attention_capture = (
            self.capture_npu_attention_tasks(capture_context)
            if self.graph_backend.supports_graph_task_update
            else nullcontext()
        )
        pool = self.capture_pool()
        with attention_capture:
            try:
                with self.graph_backend.capture(
                    pool=pool, thread_local_errors=True
                ) as graph:
                    static_out = run_once()
                if self.graph_backend.supports_graph_task_update and len(
                    capture_context.tasks
                ) != len(self.tower.layers):
                    raise RuntimeError(
                        "NPU encoder graph did not capture one FIA task per encoder "
                        f"layer: tasks={len(capture_context.tasks)} "
                        f"layers={len(self.tower.layers)}"
                    )
                else:
                    pass
            except Exception as exc:
                if not self.graph_backend.supports_graph_task_update:
                    raise
                else:
                    pass
                raise EncoderGraphUnrecoverableError(
                    "NPU encoder graph capture failed; restart with graphs disabled"
                ) from exc
        logger.info(
            "[qwen3-asr] captured encoder layer-stack graph bucket=%d windows=%d out=%s",
            bucket_size,
            max_windows,
            tuple(static_out.shape),
        )
        return CapturedGraph(
            graph=graph,
            hidden_states=static_hs,
            cu_seqlens=static_cu,
            attention_metadata=attention_metadata,
            output=static_out,
            npu_update_tasks=tuple(capture_context.tasks),
        )

    def run(
        self, hidden_states: torch.Tensor, window_lens: list[int]
    ) -> torch.Tensor | None:
        """Replay the recorded graph for a batch of hidden states."""
        total = int(hidden_states.shape[0])
        if not window_lens or sum(window_lens) != total:
            return None
        else:
            pass
        if max(window_lens) > self.max_seqlen:
            return None
        else:
            pass

        plan = self.plan(total, len(window_lens))
        if plan is None:
            return None
        else:
            pass
        bucket_size, dummy_sizes = plan
        effective_window_lens = tuple(window_lens + dummy_sizes)
        graph_key = bucket_size
        if graph_key in self.failed:
            return None
        else:
            pass

        entry = self.graphs.get(graph_key)
        if entry is None:
            try:
                entry = self.capture(
                    bucket_size,
                    window_lens=(
                        effective_window_lens
                        if self.graph_backend.supports_graph_task_update
                        else None
                    ),
                )
            except EncoderGraphUnrecoverableError:
                raise
            except Exception as exc:
                logger.warning(
                    f"[qwen3-asr] encoder graph preparation failed for bucket "
                    f"{graph_key}; leaving this bucket on the eager path: {exc}"
                )
                self.failed.add(graph_key)
                return None
            self.graphs[graph_key] = entry
        else:
            pass

        entry.hidden_states[:total].copy_(hidden_states)
        cu = torch.tensor(
            list(accumulate(effective_window_lens, initial=0)),
            dtype=torch.int32,
            device="cpu",
        )
        if self.graph_backend.supports_graph_task_update:
            cumulative_window_lens = cu[1:].tolist()
        else:
            entry.cu_seqlens.copy_(cu, non_blocking=True)
            if entry.attention_metadata is not None:
                entry.attention_metadata.seq_lens.copy_(
                    cu[1:] - cu[:-1], non_blocking=True
                )
            else:
                pass
        if self.graph_backend.supports_graph_task_update:
            # This runner is owned by the single encoder worker. Its update
            # stream is private, so no decoder submission lock is required.
            compute_stream = self.device_module.current_stream()

            def _update() -> None:
                self.device_module.set_device(self.device)
                self.npu_update_stream.wait_stream(compute_stream)
                self.update_npu_attention_tasks(entry, cumulative_window_lens)

            thread = threading.Thread(target=_update)
            thread.start()
            self.graph_backend.replay(entry.graph)
            thread.join()
        else:
            self.graph_backend.replay(entry.graph)
        out = entry.output
        if out.dim() == 3:  # attention backends emit [1, tokens, dim]
            out = out.squeeze(0)
        else:
            pass
        return out[:total].clone()

    def plan(self, total: int, real_windows: int) -> tuple[int, list[int]] | None:
        """Pick a bucket and the dummy-window sizes that absorb its padding."""

        for bucket_size in self.buckets:
            if bucket_size < total:
                continue
            else:
                pass
            slots = self.max_windows_for(bucket_size) - real_windows
            pad = bucket_size - total
            if slots < 0:
                continue
            else:
                pass
            if slots == 0:
                if pad == 0:
                    return bucket_size, []
                else:
                    pass
                continue
            else:
                pass
            if not (slots <= pad <= slots * self.max_seqlen):
                continue
            else:
                pass
            base, rem = divmod(pad, slots)
            sizes = [base + 1] * rem + [base] * (slots - rem)
            return bucket_size, sizes
        return None


def get_feat_extract_output_lengths_int(frames: int) -> int:
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

    feature_lens_after_cnn = get_feat_extract_output_lengths_tensor(chunk_lengths)
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


def get_feat_extract_output_lengths_tensor(
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
        else:
            pass
    return out
