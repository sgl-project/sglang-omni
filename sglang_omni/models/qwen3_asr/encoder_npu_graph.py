# SPDX-License-Identifier: Apache-2.0
"""Updatable Ascend graphs for the Qwen3-ASR encoder layer stack."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from sglang.srt.layers.attention.vision import VisionAttention, VisionAttentionMetadata
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    CapturedGraph,
    Qwen3ASREncoderLayerStackGraphRunner,
)
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)


@dataclass
class NpuGraphUpdateTask:
    """One captured FIA task whose host-side sequence boundaries can change."""

    kwargs: dict[str, Any]
    handle: Any
    event: Any

    def apply(
        self,
        device_module: ModuleType,
        update_stream: torch.Stream,
        cumulative_window_lens: list[int],
    ) -> None:
        device_module.graph_task_update_begin(update_stream, self.handle)
        torch.ops.npu.npu_fused_infer_attention_score.out(
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
        self, capture_context: NpuGraphCaptureContext, device_module: ModuleType
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
        forward_metadata: VisionAttentionMetadata,
        softmax_scale: float | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
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
            "pre_tokens": NpuGraphCaptureAttention.INT32_MAX,
            "next_tokens": NpuGraphCaptureAttention.INT32_MAX,
        }
        context = self.capture_context
        if context.workspace is None:
            get_workspace = (
                torch.ops.npu._npu_fused_infer_attention_score_get_max_workspace  # noqa: PGH004  # leading-underscore
            )
            context.workspace = get_workspace(
                **fia_kwargs,
                actual_seq_lengths=cumulative_window_lens,
                actual_seq_lengths_kv=cumulative_window_lens,
            )
        else:
            pass

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
        torch.ops.npu.npu_fused_infer_attention_score.out(
            **operation_kwargs,
            actual_seq_lengths=cumulative_window_lens,
            actual_seq_lengths_kv=cumulative_window_lens,
        )
        handle = device_module.graph_task_group_end(stream)
        context.tasks.append(
            NpuGraphUpdateTask(
                kwargs=operation_kwargs,
                handle=handle,
                event=event,
            )
        )
        return output


class Qwen3ASREncoderNpuGraphRunner(Qwen3ASREncoderLayerStackGraphRunner):
    """Capture FIA task groups lazily and update them alongside graph replay."""

    def __init__(
        self,
        audio_tower: Qwen3OmniMoeAudioEncoder,
        *,
        buckets: tuple[int, ...],
        max_batch_size: int,
        graph_backend: DeviceGraphBackend,
    ) -> None:
        super().__init__(
            audio_tower,
            buckets=buckets,
            max_batch_size=max_batch_size,
            graph_backend=graph_backend,
        )
        self.graph_pool: DeviceGraphPool | None = None
        self.update_tasks: dict[int, tuple[NpuGraphUpdateTask, ...]] = {}
        self.update_stream = self.device_module.Stream(device=self.device)
        # Note (wangyao-i): The runner and executor live for the model's lifetime.
        self.update_executor = ThreadPoolExecutor(
            max_workers=1,
            initializer=self.device_module.set_device,
            initargs=(self.device,),
        )

    def capture_all(self) -> None:
        """Ascend captures each bucket lazily on first use."""

    def capture_boundaries(
        self, bucket_size: int
    ) -> tuple[torch.Tensor, VisionAttentionMetadata]:
        cu = super().capture_boundaries(bucket_size)[0].cpu()
        return cu, VisionAttentionMetadata(
            cu_seqlens=cu,
            seq_lens=cu[1:] - cu[:-1],
            max_seqlen=self.max_seqlen,
        )

    def record_graph(
        self, bucket_size: int, run_once: Callable[[], torch.Tensor]
    ) -> tuple[ReplayableGraph, torch.Tensor]:
        context = NpuGraphCaptureContext(tasks=[])
        if self.graph_pool is None:
            self.graph_pool = self.graph_backend.graph_pool_handle()
        else:
            pass
        replacements: list[tuple[VisionAttention, torch.nn.Module]] = []
        try:
            for layer in self.tower.layers:
                attention = layer.self_attn
                if attention.qkv_backend_name != "ascend_attn":
                    raise RuntimeError(
                        "NPU encoder graph capture requires the ascend_attn backend"
                    )
                else:
                    pass
                replacements.append((attention, attention.qkv_backend))
                attention.qkv_backend = NpuGraphCaptureAttention(
                    context, self.device_module
                )
            with self.graph_backend.capture(
                pool=self.graph_pool, thread_local_errors=True
            ) as graph:
                output = run_once()
        finally:
            for attention, original in replacements:
                attention.qkv_backend = original
        if len(context.tasks) != len(self.tower.layers):
            raise RuntimeError("NPU graph must capture one FIA task per encoder layer")
        else:
            pass
        self.update_tasks[bucket_size] = tuple(context.tasks)
        return graph, output

    def replay(self, entry: CapturedGraph, cu: torch.Tensor) -> None:
        cumulative_window_lens = cu[1:].tolist()
        # Note (wangyao-i): Record the dependency before replay waits for FIA updates.
        self.update_stream.wait_stream(self.device_module.current_stream())

        def update_attention_tasks() -> None:
            with self.device_module.stream(self.update_stream):
                for task in self.update_tasks[entry.hidden_states.shape[0]]:
                    task.apply(
                        self.device_module, self.update_stream, cumulative_window_lens
                    )

        update = self.update_executor.submit(update_attention_tasks)
        entry.graph.replay()
        update.result()
