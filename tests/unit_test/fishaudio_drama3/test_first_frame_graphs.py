# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import math
from unittest.mock import Mock, call

import pytest
import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    has_forward_context,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

from sglang_omni.models.fishaudio_drama3.first_frame_graphs import (
    WARMUP_RUN_COUNT,
    Drama3FirstFrameGraphs,
    FirstFrameInputs,
)
from sglang_omni.models.fishaudio_drama3.sampling import RAS_WINDOW_FRAMES
from tests.unit_test.fixtures.accelerator import require_cuda

VOCAB_SIZE = 6
HIDDEN_SIZE = 3
FRAME_WIDTH = 5
MAX_REQUEST_COUNT = 4
STATIC_ROW_START = -100


def numbered(
    shape: tuple[int, ...], dtype: torch.dtype, start: int, device: torch.device
) -> torch.Tensor:
    return (
        (torch.arange(math.prod(shape), device=device) + start).reshape(shape).to(dtype)
    )


def first_frame_inputs(
    request_count: int, start: int, device: torch.device
) -> FirstFrameInputs:
    return FirstFrameInputs(
        semantic_logits=numbered(
            (request_count, VOCAB_SIZE), torch.float32, start, device
        ),
        hidden_states=numbered(
            (request_count, HIDDEN_SIZE), torch.bfloat16, start, device
        ),
        positions=numbered((request_count,), torch.int64, start, device),
        input_frames=numbered((request_count, FRAME_WIDTH), torch.int64, start, device),
        previous_main_tokens=numbered(
            (request_count, RAS_WINDOW_FRAMES), torch.int64, start, device
        ),
        temperature=numbered((request_count,), torch.float32, start, device),
        top_k=numbered((request_count,), torch.int64, start, device),
        top_p=numbered((request_count,), torch.float32, start, device),
        request_pool_indices=numbered((request_count,), torch.int64, start, device),
        sequence_lengths=numbered((request_count,), torch.int64, start, device),
        cache_slots=numbered((request_count,), torch.int64, start, device),
    )


def summed_frames(frame_inputs: FirstFrameInputs) -> torch.Tensor:
    return (
        frame_inputs.input_frames
        + frame_inputs.positions[:, None]
        + frame_inputs.cache_slots[:, None]
    )


class SummingGraph:
    def __init__(
        self,
        static_inputs: FirstFrameInputs,
        sampled_frames: torch.Tensor,
        events: list[str],
    ) -> None:
        self.static_inputs = static_inputs
        self.sampled_frames = sampled_frames
        self.events = events

    def replay(self) -> None:
        request_count = self.sampled_frames.shape[0]
        self.events.append(f"replay {request_count}")
        self.sampled_frames.copy_(summed_frames(self.static_inputs.rows(request_count)))


def cpu_first_frame_graphs(events: list[str]) -> Drama3FirstFrameGraphs:
    device = torch.device("cpu")
    static_inputs = first_frame_inputs(MAX_REQUEST_COUNT, STATIC_ROW_START, device)
    sampled_frames = {
        request_count: torch.zeros((request_count, FRAME_WIDTH), dtype=torch.int64)
        for request_count in range(1, MAX_REQUEST_COUNT + 1)
    }
    attention_backend = Mock(spec=TritonAttnBackend)
    attention_backend.init_forward_metadata_out_graph.side_effect = (
        lambda forward_batch: events.append(f"refill {forward_batch.batch_size}")
    )
    return Drama3FirstFrameGraphs(
        static_inputs=static_inputs,
        forward_batches={
            request_count: Mock(spec=ForwardBatch, batch_size=request_count)
            for request_count in sampled_frames
        },
        graphs={
            request_count: SummingGraph(static_inputs, frames, events)
            for request_count, frames in sampled_frames.items()
        },
        sampled_frames=sampled_frames,
        attention_backend=attention_backend,
        capture_seconds=0.0,
        reserved_bytes=0,
    )


def test_replay_returns_current_frames_and_preserves_previous_results() -> None:
    events: list[str] = []
    graphs = cpu_first_frame_graphs(events)
    replay_results: list[tuple[torch.Tensor, torch.Tensor]] = []

    for request_count in (2, MAX_REQUEST_COUNT, 1, 3):
        live_inputs = first_frame_inputs(
            request_count, 10 * request_count, torch.device("cpu")
        )
        replay_results.append((graphs.replay(live_inputs), summed_frames(live_inputs)))

    for sampled_frames, expected_frames in replay_results:
        assert torch.equal(sampled_frames, expected_frames)
    assert events == [
        event
        for request_count in (2, MAX_REQUEST_COUNT, 1, 3)
        for event in (f"refill {request_count}", f"replay {request_count}")
    ]


def never_sampled(
    semantic_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    input_frames: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    raise AssertionError("a refused capture samples nothing")


def hybrid_attention_backend() -> tuple[Mock, Mock]:
    full_attention_backend = Mock(spec=TritonAttnBackend)
    full_attention_backend.get_cuda_graph_seq_len_fill_value.return_value = 1
    attention_backend = Mock(spec=HybridLinearAttnBackend)
    attention_backend.full_attn_backend = full_attention_backend
    return attention_backend, full_attention_backend


def test_capture_refuses_a_device_without_graphs() -> None:
    attention_backend, _ = hybrid_attention_backend()

    with pytest.raises(RuntimeError, match="first_frame_cuda_graphs false"):
        Drama3FirstFrameGraphs.capture(
            sample_first_frames=never_sampled,
            attention_backend=attention_backend,
            max_request_count=2,
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            hidden_dtype=torch.bfloat16,
            frame_width=FRAME_WIDTH,
            device=torch.device("cpu"),
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerCall:
    request_count: int
    is_capturing: bool
    is_capture_mode: bool
    attention_backend: AttentionBackend
    forward_mode: ForwardMode


@pytest.mark.accelerator
def test_capture_records_every_request_count_largest_first_in_capture_mode() -> None:
    require_cuda()
    device = torch.device("cuda")
    attention_backend, full_attention_backend = hybrid_attention_backend()
    calls: list[SamplerCall] = []

    def sample(
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        is_capturing = torch.cuda.is_current_stream_capturing()
        calls.append(
            SamplerCall(
                request_count=forward_batch.batch_size,
                is_capturing=is_capturing,
                is_capture_mode=get_is_capture_mode(),
                attention_backend=get_attn_backend(),
                forward_mode=forward_batch.forward_mode,
            )
        )
        return input_frames + positions[:, None] + forward_batch.out_cache_loc[:, None]

    had_forward_context = has_forward_context()
    graphs = Drama3FirstFrameGraphs.capture(
        sample_first_frames=sample,
        attention_backend=attention_backend,
        max_request_count=3,
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        hidden_dtype=torch.bfloat16,
        frame_width=FRAME_WIDTH,
        device=device,
    )

    assert [(sampled.request_count, sampled.is_capturing) for sampled in calls] == [
        (request_count, is_capturing)
        for request_count in (3, 2, 1)
        for is_capturing in [False] * WARMUP_RUN_COUNT + [True]
    ]
    assert all(
        sampled.is_capture_mode
        and sampled.attention_backend is attention_backend
        and sampled.forward_mode == ForwardMode.DECODE
        for sampled in calls
    )
    assert full_attention_backend.init_forward_metadata_out_graph.call_args_list == [
        call(graphs.forward_batches[request_count], in_capture=True)
        for request_count in (3, 2, 1)
    ]
    assert not get_is_capture_mode()
    assert has_forward_context() == had_forward_context
    assert sorted(graphs.graphs) == [1, 2, 3]
    for request_count in (2, 3, 1):
        live_inputs = first_frame_inputs(request_count, 10 * request_count, device)

        frames = graphs.replay(live_inputs)

        assert full_attention_backend.init_forward_metadata_out_graph.call_args == (
            call(graphs.forward_batches[request_count])
        )
        assert torch.equal(frames, summed_frames(live_inputs))
    assert len(calls) == 3 * (WARMUP_RUN_COUNT + 1)


@pytest.mark.accelerator
def test_a_failed_capture_names_its_request_count_and_the_switch() -> None:
    require_cuda()
    attention_backend, _ = hybrid_attention_backend()

    def sample(
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing() and forward_batch.batch_size == 2:
            raise RuntimeError("injected capture failure")
        else:
            return input_frames + 1

    had_forward_context = has_forward_context()
    with pytest.raises(
        RuntimeError,
        match=(
            r"for 2 requests failed: injected capture failure; set "
            r"--tts_engine\.factory\.first_frame_cuda_graphs false"
        ),
    ):
        Drama3FirstFrameGraphs.capture(
            sample_first_frames=sample,
            attention_backend=attention_backend,
            max_request_count=3,
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            hidden_dtype=torch.bfloat16,
            frame_width=FRAME_WIDTH,
            device=torch.device("cuda"),
        )
    assert not get_is_capture_mode()
    assert has_forward_context() == had_forward_context
