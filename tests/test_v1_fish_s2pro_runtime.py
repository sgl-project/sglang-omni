# SPDX-License-Identifier: Apache-2.0
"""Regression tests for V1 FishAudio S2-Pro request state transitions."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni_v1.models.fishaudio_s2_pro.fish_scheduler import (
    FishIterationController,
    FishScheduler,
)
from sglang_omni_v1.models.fishaudio_s2_pro.model_runner import (
    FishS2ProModelRunner,
    collect_s2pro_step_outputs,
)
from sglang_omni_v1.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
)
from sglang_omni_v1.scheduling.types import SchedulerRequest

IM_END_TOKEN_ID = 151645
SEMANTIC_TOKEN_ID = 151678


class _CountingTreeCache:
    def __init__(self) -> None:
        self.cached_requests = 0

    def cache_unfinished_req(self, req) -> None:
        del req
        self.cached_requests += 1


def _make_request(request_id: str, *, is_chunked: int = 0) -> SchedulerRequest:
    req = SimpleNamespace(
        is_chunked=is_chunked,
        output_ids=[],
        decode_batch_idx=0,
        finished=lambda: False,
    )
    data = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=req,
    )
    return SchedulerRequest(request_id=request_id, data=data)


def _collect_model_step(
    requests: list[SchedulerRequest],
    code_rows: list[list[int]],
) -> SimpleNamespace:
    result = SimpleNamespace(next_token_ids=None)
    output_codes = torch.tensor(code_rows, dtype=torch.long)
    output_semantic_ids = output_codes[:, 0].clone()
    collect_s2pro_step_outputs(
        result,
        requests,
        output_codes=output_codes,
        output_semantic_ids=output_semantic_ids,
        im_end_token_id=IM_END_TOKEN_ID,
        rep_history_len=4,
    )
    return result


def _update_request_from_result(
    controller: FishIterationController,
    request: SchedulerRequest,
    result: SimpleNamespace,
    row_idx: int = 0,
) -> int:
    token_id = int(result.next_token_ids[row_idx].item())
    controller.update_request(request, token_id)
    return token_id


def test_v1_s2pro_audio_timestep_updates_audio_and_stream_state() -> None:
    request = _make_request("req-audio")
    data = request.data

    result = _collect_model_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])

    assert int(result.next_token_ids[0].item()) == SEMANTIC_TOKEN_ID
    assert len(data.output_codes) == 1
    assert torch.equal(
        data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert torch.equal(data.latest_stream_code_chunk, data.output_codes[0])
    assert data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert data.semantic_history_count == 1
    assert torch.equal(
        data.semantic_history_tokens,
        torch.tensor([SEMANTIC_TOKEN_ID, 0, 0, 0], dtype=torch.long),
    )
    assert torch.equal(data.last_codebook_values, torch.tensor([11, 22]))


def test_v1_s2pro_prepare_decode_uses_gpu_history_buffer() -> None:
    request = _make_request("req-history")
    data = request.data
    data.previous_semantic_tokens = [9999]
    data.semantic_history_tokens = torch.tensor(
        [SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0],
        dtype=torch.long,
    )
    data.semantic_history_count = 2
    data.last_codebook_values = torch.tensor([11, 22], dtype=torch.long)

    runner = FishS2ProModelRunner.__new__(FishS2ProModelRunner)
    runner.model = SimpleNamespace(
        _semantic_begin_id=SEMANTIC_TOKEN_ID,
        _semantic_end_id=SEMANTIC_TOKEN_ID + 10,
        _rep_history_len=4,
        _vq_mask=torch.zeros(1, dtype=torch.bool),
        _sampling_temperature=torch.zeros(1),
        _sampling_top_p=torch.zeros(1),
        _sampling_top_k=torch.zeros(1, dtype=torch.long),
        _sampling_rep_penalty=torch.zeros(1),
        _ras_temperature=torch.zeros(1),
        _ras_top_p=torch.zeros(1),
        _prev_tokens=torch.zeros(1, 4, dtype=torch.long),
        _prev_token_count=torch.zeros(1, dtype=torch.long),
        _vq_codes=torch.zeros(1, 2, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([SEMANTIC_TOKEN_ID]))

    runner.prepare_decode(forward_batch, None, [request])

    assert torch.equal(
        runner.model._prev_tokens[0],
        torch.tensor([SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0]),
    )
    assert int(runner.model._prev_token_count[0].item()) == 2
    assert torch.equal(runner.model._vq_codes[0], torch.tensor([11, 22]))
    assert bool(runner.model._vq_mask[0])


def test_v1_s2pro_terminal_im_end_is_not_audio_codebook_frame() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_request("req-terminal")
    data = request.data
    req = data.req

    result = _collect_model_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])
    _update_request_from_result(controller, request, result)
    data.latest_stream_code_chunk = None

    result = _collect_model_step([request], [[IM_END_TOKEN_ID, 33, 44]])
    eos_token = _update_request_from_result(controller, request, result)

    assert controller.is_finished(request, eos_token)
    assert req.output_ids == [SEMANTIC_TOKEN_ID, IM_END_TOKEN_ID]
    assert len(data.output_codes) == 1
    assert torch.equal(
        data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert data.latest_stream_code_chunk is None
    assert data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert torch.equal(data.last_codebook_values, torch.tensor([11, 22]))
    assert tree_cache.cached_requests == 1


def test_v1_s2pro_immediate_im_end_leaves_no_audio_codebook_frames() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_request("req-immediate-terminal")
    data = request.data

    result = _collect_model_step([request], [[IM_END_TOKEN_ID, 33, 44]])
    eos_token = _update_request_from_result(controller, request, result)

    assert controller.is_finished(request, eos_token)
    assert data.req.output_ids == [IM_END_TOKEN_ID]
    assert data.output_codes == []
    assert data.latest_stream_code_chunk is None
    assert data.previous_semantic_tokens == []
    assert data.last_codebook_values is None
    assert tree_cache.cached_requests == 0


def test_v1_s2pro_emit_finished_errors_before_vocoder_for_empty_codes() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_request("req-immediate-terminal")

    result = _collect_model_step([request], [[IM_END_TOKEN_ID, 33, 44]])
    _update_request_from_result(controller, request, result)

    def result_adapter(_data):
        raise AssertionError(
            "empty S2-Pro output_codes must not route to result_adapter"
        )

    scheduler = FishScheduler(
        tree_cache=tree_cache,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        prefill_manager=None,
        decode_manager=None,
        server_args=SimpleNamespace(max_running_requests=1),
        model_runner=None,
        request_builder=None,
        result_adapter=result_adapter,
        im_end_token_id=IM_END_TOKEN_ID,
        max_new_tokens=2048,
    )
    scheduler._submit_times[request.request_id] = 1.0
    scheduler._requests[request.request_id] = request

    scheduler.emit_finished([request])
    output = scheduler.outbox.get_nowait()

    assert output.request_id == request.request_id
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert "S2-Pro generated no audio codec tokens" in str(output.data)
    assert scheduler._submit_times == {}
    assert request.request_id not in scheduler._requests


def test_v1_s2pro_mixed_batch_keeps_terminal_and_audio_state_separate() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    audio_request = _make_request("req-audio")
    terminal_request = _make_request("req-terminal")

    result = _collect_model_step(
        [audio_request, terminal_request],
        [
            [SEMANTIC_TOKEN_ID, 11, 22],
            [IM_END_TOKEN_ID, 33, 44],
        ],
    )
    audio_token = _update_request_from_result(controller, audio_request, result, 0)
    terminal_token = _update_request_from_result(
        controller, terminal_request, result, 1
    )

    assert not controller.is_finished(audio_request, audio_token)
    assert controller.is_finished(terminal_request, terminal_token)
    assert audio_request.data.req.output_ids == [SEMANTIC_TOKEN_ID]
    assert terminal_request.data.req.output_ids == [IM_END_TOKEN_ID]
    assert len(audio_request.data.output_codes) == 1
    assert terminal_request.data.output_codes == []
    assert torch.equal(
        audio_request.data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert torch.equal(
        audio_request.data.latest_stream_code_chunk,
        audio_request.data.output_codes[0],
    )
    assert terminal_request.data.latest_stream_code_chunk is None
    assert audio_request.data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert terminal_request.data.previous_semantic_tokens == []
    assert torch.equal(audio_request.data.last_codebook_values, torch.tensor([11, 22]))
    assert terminal_request.data.last_codebook_values is None
    assert tree_cache.cached_requests == 1


def test_v1_s2pro_chunked_step_does_not_mutate_decode_state() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_request("req-chunked", is_chunked=1)
    data = request.data

    result = _collect_model_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])
    semantic_token = _update_request_from_result(controller, request, result)

    assert not controller.is_finished(request, semantic_token)
    assert data.req.is_chunked == 0
    assert data.req.output_ids == []
    assert data.output_codes == []
    assert data.latest_stream_code_chunk is None
    assert data.previous_semantic_tokens == []
    assert data.last_codebook_values is None
    assert tree_cache.cached_requests == 0
