# SPDX-License-Identifier: Apache-2.0
"""Drama 3 model runner: stages each forward's frames and collects sampled frames.

Everything runs inside the model's forward, behind SGLang's own forward path,
so the deferred clear of reused linear-attention state slots always happens.
Decode steps go through the model's fixed buffers, which a decode CUDA graph reads.
"""

from __future__ import annotations

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.request_id_prefill_cuda_graph_runner import (
    RequestIdPrefillCudaGraphRunner,
)
from sglang_omni.models.fishaudio_drama3.decode_state import (
    Drama3DecodeStatePool,
    extend_frame_rows,
    ras_window,
    request_data_of,
    request_ids_of,
)
from sglang_omni.models.fishaudio_drama3.model_inputs import (
    Drama3DecodeBuffers,
    Drama3StepInputs,
)
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import (
    ForwardBatch,
    GenerationBatchResult,
    ScheduleBatch,
)

__all__ = [
    "Drama3ModelRunner",
    "Drama3DecodeStatePool",
    "extend_frame_rows",
    "ras_window",
    "request_data_of",
    "request_ids_of",
]


class Drama3ModelRunner(ModelRunner[Drama3RequestData]):
    """Hands each forward its frames, RAS windows and sampling knobs, then appends
    the frame the model sampled for every request."""

    model: Drama3ForCausalLM

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        sglang_runner = tp_worker.model_runner
        model = sglang_runner.model
        assert isinstance(model, Drama3ForCausalLM)
        text_config = model.config.text_config
        attention_backend = sglang_runner.attn_backend
        kv_pool = sglang_runner.token_to_kv_pool
        request_pool = sglang_runner.req_to_token_pool
        prefill_graph_runner = sglang_runner.prefill_cuda_graph_runner
        decode_graph_runner = sglang_runner.decode_cuda_graph_runner
        decode_buffers = model.decode_buffers
        if not isinstance(attention_backend, HybridLinearAttnBackend):
            raise RuntimeError(
                f"Drama 3 needs the hybrid linear-attention backend; got "
                f"{type(attention_backend).__name__}"
            )
        elif not isinstance(attention_backend.full_attn_backend, TritonAttnBackend):
            raise RuntimeError(
                f"Drama 3 needs triton for full attention; got "
                f"{type(attention_backend.full_attn_backend).__name__}"
            )
        elif not (
            decode_graph_runner is None
            or isinstance(decode_graph_runner, DecodeCudaGraphRunner)
        ) or not (
            prefill_graph_runner is None
            or isinstance(
                prefill_graph_runner, (EagerRunner, RequestIdPrefillCudaGraphRunner)
            )
        ):
            raise RuntimeError(
                "Drama 3 replays full decode CUDA graphs and runs its prefill eager "
                "or as a breakable CUDA graph"
            )
        elif decode_buffers is None or decode_buffers.frames.shape[0] < max(
            request_pool.size,
            0 if decode_graph_runner is None else decode_graph_runner.max_bs,
        ):
            raise RuntimeError(
                "the Drama 3 decode buffers need a row for every request slot and "
                "every captured decode batch size"
            )
        elif not (
            isinstance(kv_pool, HybridLinearKVPool)
            and set(kv_pool.full_attention_layer_id_mapping)
            == set(text_config.full_attention_layer_ids)
        ):
            raise RuntimeError(
                f"the KV pool does not hold exactly the backbone and tower layers "
                f"{text_config.full_attention_layer_ids}"
            )
        elif not (
            isinstance(request_pool, HybridReqToTokenPool)
            and set(request_pool.mamba_map) == set(text_config.linear_layer_ids)
        ):
            raise RuntimeError(
                f"the state pool does not hold exactly the linear-attention layers "
                f"{text_config.linear_layer_ids}"
            )
        else:
            pass
        self.decode_buffers: Drama3DecodeBuffers = decode_buffers
        self.request_slot_count = request_pool.size
        self.state_pool = Drama3DecodeStatePool.allocate(
            request_slot_count=request_pool.size,
            frame_width=decode_buffers.frames.shape[1],
            device=self.device,
        )
        self.row_launch_key: tuple[int, ...] | None = None
        self.row_upload_buffers: list[torch.Tensor] = []
        self.row_upload_slot = 0
        self.prefill_inputs: Drama3StepInputs | None = None

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        return True

    def state_rows(self, batch_request_data: list[Drama3RequestData]) -> list[int]:
        rows = []
        for request_data in batch_request_data:
            request = request_data.req
            assert request is not None
            row = request.kv.req_pool_idx
            if row is None or not 0 < row <= self.request_slot_count:
                raise RuntimeError(
                    f"{request.rid}: request slot {row} has no state row"
                )
            else:
                rows.append(row)
        return rows

    def gather_decode_rows(self, batch_request_data: list[Drama3RequestData]) -> None:
        state_pool = self.state_pool
        rows = self.state_rows(batch_request_data)
        row_count = len(rows)
        row_launch_key = tuple(rows)
        if row_launch_key != self.row_launch_key:
            host_rows = self.pinned_pingpong(
                "row_upload_buffers",
                "row_upload_slot",
                (row_count,),
                torch.int64,
                realloc_on_grow=True,
            )
            host_rows.numpy()[:row_count] = rows
            state_pool.row_indices[:row_count].copy_(
                host_rows[:row_count], non_blocking=True
            )
            self.row_launch_key = row_launch_key
        else:
            pass
        row_indices = state_pool.row_indices[:row_count]
        buffers = self.decode_buffers
        torch.index_select(
            state_pool.frames, 0, row_indices, out=buffers.frames[:row_count]
        )
        torch.index_select(
            state_pool.ras_windows, 0, row_indices, out=buffers.ras_windows[:row_count]
        )
        torch.index_select(
            state_pool.temperature, 0, row_indices, out=buffers.temperature[:row_count]
        )
        torch.index_select(
            state_pool.top_k, 0, row_indices, out=buffers.top_k[:row_count]
        )
        torch.index_select(
            state_pool.top_p, 0, row_indices, out=buffers.top_p[:row_count]
        )

    def publish_decode_frames(self, row_count: int) -> torch.Tensor:
        state_pool = self.state_pool
        row_indices = state_pool.row_indices[:row_count]
        frames = self.decode_buffers.sampled_frames[:row_count].clone()
        ras_windows = state_pool.ras_windows.index_select(0, row_indices)
        state_pool.frames.index_copy_(0, row_indices, frames)
        state_pool.ras_windows.index_copy_(
            0, row_indices, torch.cat((ras_windows[:, 1:], frames[:, :1]), dim=1)
        )
        return frames

    def initialize_state_rows(
        self,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
        first_frames: torch.Tensor,
    ) -> None:
        prefill_inputs = self.prefill_inputs
        request_ids = request_ids_of([request_data_of(request) for request in requests])
        if prefill_inputs is None or prefill_inputs.request_ids != request_ids:
            raise RuntimeError(
                f"the staged prefill inputs are for "
                f"{None if prefill_inputs is None else prefill_inputs.request_ids}, "
                f"not the batch's {request_ids}"
            )
        elif forward_batch is None:
            raise RuntimeError("a Drama 3 prefill has no forward batch")
        else:
            state_pool = self.state_pool
            rows = forward_batch.req_pool_indices.to(torch.int64)
            state_pool.frames.index_copy_(0, rows, first_frames)
            state_pool.ras_windows.index_copy_(
                0,
                rows,
                torch.cat(
                    (prefill_inputs.ras_windows[:, 1:], first_frames[:, :1]), dim=1
                ),
            )
            state_pool.temperature.index_copy_(0, rows, prefill_inputs.temperature)
            state_pool.top_k.index_copy_(0, rows, prefill_inputs.top_k)
            state_pool.top_p.index_copy_(0, rows, prefill_inputs.top_p)

    def prepare_and_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        is_prefill: bool,
        *,
        is_lookahead: bool = False,
    ) -> GenerationBatchResult:
        try:
            return super().prepare_and_forward(
                forward_batch,
                schedule_batch,
                requests,
                is_prefill,
                is_lookahead=is_lookahead,
            )
        except BaseException:
            self.prefill_inputs = None
            raise

    def stage_step(
        self, requests: list[SchedulerRequest], frame_rows: list[list[int]]
    ) -> Drama3StepInputs:
        request_data = [request_data_of(request) for request in requests]
        step_inputs = Drama3StepInputs(
            request_ids=request_ids_of(request_data),
            frames=torch.tensor(frame_rows, dtype=torch.int64, device=self.device),
            ras_windows=torch.tensor(
                [ras_window(data.frame_rows) for data in request_data],
                dtype=torch.int64,
                device=self.device,
            ),
            temperature=torch.tensor(
                [data.temperature for data in request_data],
                dtype=torch.float32,
                device=self.device,
            ),
            top_k=torch.tensor(
                [data.top_k for data in request_data],
                dtype=torch.int64,
                device=self.device,
            ),
            top_p=torch.tensor(
                [data.top_p for data in request_data],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.model.step_exchange.stage(step_inputs)
        return step_inputs

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Stage every request's whole frame history; the batch arguments are the
        interface contract."""
        self.prefill_inputs = None
        frame_rows: list[list[int]] = []
        for request in requests:
            frame_rows.extend(extend_frame_rows(request_data_of(request)))
        self.prefill_inputs = self.stage_step(requests, frame_rows)

    def cleanup_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Drop inputs a failed prefill left staged; the arguments are the contract."""
        self.model.step_exchange.clear()

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Stage every request's newest frame into the decode buffers, in batch
        order; the other arguments are the interface contract."""
        batch_request_data = [request_data_of(request) for request in requests]
        request_ids = request_ids_of(batch_request_data)
        if forward_batch is None or forward_batch.rids != request_ids:
            raise RuntimeError(
                f"decode requests {request_ids} are not the batch's "
                f"{None if forward_batch is None else forward_batch.rids}"
            )
        else:
            self.gather_decode_rows(batch_request_data)

    def collect_frames(
        self,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
        frames: torch.Tensor,
    ) -> None:
        result.next_token_ids = frames[:, 0]
        for request, frame_row in zip(requests, frames.tolist(), strict=True):
            request_data_of(request).frame_rows.append(frame_row)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Collect the first frames; the schedule batch is the interface contract."""
        try:
            first_frames = self.model.step_exchange.take_frames()
            self.initialize_state_rows(forward_batch, requests, first_frames)
            self.collect_frames(result, requests, first_frames)
        finally:
            self.prefill_inputs = None

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Collect the next frames; the batch arguments are the interface contract."""
        self.collect_frames(result, requests, self.publish_decode_frames(len(requests)))

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        if not requests:
            return None
        else:
            row_count = len(requests)
            frames = self.publish_decode_frames(row_count)
            host_frames = self.next_host_staging(tuple(frames.shape), frames.dtype)
            host_frames[:row_count].copy_(frames, non_blocking=True)
            result.next_token_ids = frames[:, 0]
            return host_frames

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        if launch_buf is None or not requests:
            return
        else:
            frames = launch_buf[: len(requests)]
            kept_rows = []
            for row, request in enumerate(requests):
                sglang_request = request_data_of(request).req
                assert sglang_request is not None
                if sglang_request.finished() or self.req_is_retracted(sglang_request):
                    pass
                else:
                    kept_rows.append(row)
            self.collect_frames(
                result, [requests[row] for row in kept_rows], frames[kept_rows]
            )
            result.next_token_ids = frames[:, 0]
