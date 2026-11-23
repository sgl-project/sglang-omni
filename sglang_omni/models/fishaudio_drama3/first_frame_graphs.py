# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import time
from typing import Protocol

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

WARMUP_RUN_COUNT = 2


class FirstFrameSampler(Protocol):
    def __call__(
        self,
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor: ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class FirstFrameInputs:
    semantic_logits: torch.Tensor
    hidden_states: torch.Tensor
    positions: torch.Tensor
    input_frames: torch.Tensor
    previous_main_tokens: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    request_pool_indices: torch.Tensor
    sequence_lengths: torch.Tensor
    cache_slots: torch.Tensor

    def rows(self, request_count: int) -> FirstFrameInputs:
        return FirstFrameInputs(
            **{
                field.name: getattr(self, field.name)[:request_count]
                for field in dataclasses.fields(self)
            }
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3FirstFrameGraphs:
    static_inputs: FirstFrameInputs
    forward_batches: dict[int, ForwardBatch]
    graphs: dict[int, ReplayableGraph]
    sampled_frames: dict[int, torch.Tensor]
    attention_backend: TritonAttnBackend
    capture_seconds: float
    reserved_bytes: int

    @property
    def max_request_count(self) -> int:
        return max(self.graphs)

    @property
    def static_input_bytes(self) -> int:
        return sum(
            getattr(self.static_inputs, field.name).nbytes
            for field in dataclasses.fields(self.static_inputs)
        )

    @classmethod
    def capture(
        cls,
        *,
        sample_first_frames: FirstFrameSampler,
        attention_backend: HybridLinearAttnBackend,
        max_request_count: int,
        vocab_size: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        frame_width: int,
        device: torch.device,
    ) -> Drama3FirstFrameGraphs:
        graph_backend = current_platform.get_device_graph_backend(device)
        full_attention_backend = attention_backend.full_attn_backend
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {device}; set "
                f"--tts_engine.factory.first_frame_cuda_graphs false to run the "
                f"first-frame step eagerly"
            )
        elif not isinstance(full_attention_backend, TritonAttnBackend):
            raise RuntimeError(
                f"Drama 3 first-frame CUDA graphs need triton for full attention; "
                f"got {type(full_attention_backend).__name__}"
            )
        else:
            pass
        device_module = torch.get_device_module(device)
        reserved_bytes_before = device_module.memory_reserved(device)
        capture_start = time.perf_counter()
        sequence_length_fill_value = (
            full_attention_backend.get_cuda_graph_seq_len_fill_value()
        )
        static_inputs = FirstFrameInputs(
            semantic_logits=torch.zeros(
                (max_request_count, vocab_size), dtype=torch.float32, device=device
            ),
            hidden_states=torch.zeros(
                (max_request_count, hidden_size), dtype=hidden_dtype, device=device
            ),
            positions=torch.zeros(max_request_count, dtype=torch.int64, device=device),
            input_frames=torch.zeros(
                (max_request_count, frame_width), dtype=torch.int64, device=device
            ),
            previous_main_tokens=torch.full(
                (max_request_count, RAS_WINDOW_FRAMES),
                EMPTY_RAS_SLOT,
                dtype=torch.int64,
                device=device,
            ),
            temperature=torch.ones(
                max_request_count, dtype=torch.float32, device=device
            ),
            top_k=torch.ones(max_request_count, dtype=torch.int64, device=device),
            top_p=torch.ones(max_request_count, dtype=torch.float32, device=device),
            request_pool_indices=torch.zeros(
                max_request_count, dtype=torch.int64, device=device
            ),
            sequence_lengths=torch.full(
                (max_request_count,),
                sequence_length_fill_value,
                dtype=torch.int64,
                device=device,
            ),
            cache_slots=torch.zeros(
                max_request_count, dtype=torch.int64, device=device
            ),
        )
        forward_batches: dict[int, ForwardBatch] = {}
        for request_count in range(1, max_request_count + 1):
            rows = static_inputs.rows(request_count)
            forward_batches[request_count] = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=request_count,
                input_ids=rows.input_frames[:, 0],
                req_pool_indices=rows.request_pool_indices,
                seq_lens=rows.sequence_lengths,
                out_cache_loc=rows.cache_slots,
                seq_lens_sum=request_count * sequence_length_fill_value,
                positions=rows.positions,
            )
        graphs: dict[int, ReplayableGraph] = {}
        sampled_frames: dict[int, torch.Tensor] = {}
        current_stream = device_module.current_stream(device)
        capture_stream = device_module.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        graph_pool = device_module.graph_pool_handle()
        with (
            torch.no_grad(),
            device_module.device(device),
            device_module.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=attention_backend)),
            model_capture_mode(),
        ):
            for request_count in reversed(forward_batches):
                rows = static_inputs.rows(request_count)
                forward_batch = forward_batches[request_count]
                sampler_arguments = (
                    rows.semantic_logits,
                    rows.hidden_states,
                    rows.positions,
                    rows.input_frames,
                    rows.previous_main_tokens,
                    rows.temperature,
                    rows.top_k,
                    rows.top_p,
                    forward_batch,
                )
                try:
                    full_attention_backend.init_forward_metadata_out_graph(
                        forward_batch, in_capture=True
                    )
                    for _ in range(WARMUP_RUN_COUNT):
                        sample_first_frames(*sampler_arguments)
                    capture_stream.synchronize()
                    with graph_backend.capture(
                        pool=graph_pool, stream=capture_stream
                    ) as graph:
                        sampled_frames[request_count] = sample_first_frames(
                            *sampler_arguments
                        )
                    graphs[request_count] = graph
                except RuntimeError as error:
                    raise RuntimeError(
                        f"capturing the Drama 3 first-frame CUDA graph for "
                        f"{request_count} requests failed: {error}; set "
                        f"--tts_engine.factory.first_frame_cuda_graphs false to run "
                        f"the first-frame step eagerly"
                    ) from error
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(device)
        return cls(
            static_inputs=static_inputs,
            forward_batches=forward_batches,
            graphs=graphs,
            sampled_frames=sampled_frames,
            attention_backend=full_attention_backend,
            capture_seconds=time.perf_counter() - capture_start,
            reserved_bytes=device_module.memory_reserved(device)
            - reserved_bytes_before,
        )

    def replay(self, inputs: FirstFrameInputs) -> torch.Tensor:
        request_count = inputs.semantic_logits.shape[0]
        static_rows = self.static_inputs.rows(request_count)
        for field in dataclasses.fields(inputs):
            getattr(static_rows, field.name).copy_(getattr(inputs, field.name))
        self.attention_backend.init_forward_metadata_out_graph(
            self.forward_batches[request_count]
        )
        self.graphs[request_count].replay()
        return self.sampled_frames[request_count].clone()
