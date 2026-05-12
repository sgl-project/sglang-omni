# SPDX-License-Identifier: Apache-2.0

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
from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerRequest,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishModel,
    FakeFishReq,
    make_s2pro_payload,
)

IM_END_TOKEN_ID = 151645
SEMANTIC_TOKEN_ID = 151678


def test_fish_model_runner_vq_injection_and_code_collection_contracts() -> None:
    """Preserves VQ prompt embedding injection and semantic code collection."""
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner._semantic_begin_id = 200
    runner._semantic_end_id = 295
    runner._im_end_token_id = 99
    prefill_request = SchedulerRequest(
        request_id="prefill",
        data=SimpleNamespace(
            req=FakeFishReq(extend_input_len=3),
            vq_mask_tokens=torch.tensor([True, False, True]),
            vq_parts=[torch.tensor([[7, 8], [9, 10]], dtype=torch.long)],
        ),
    )
    embeds = runner._build_prefill_input_embeds(
        SimpleNamespace(input_ids=torch.tensor([10, 11, 12])),
        [prefill_request],
    )
    assert torch.equal(embeds[0], torch.tensor([1007.0, 1009.0]))
    assert torch.equal(embeds[1], torch.tensor([11.0, 11.0]))

    active = SchedulerRequest(
        request_id="active",
        data=SimpleNamespace(
            req=FakeFishReq(is_chunked=0),
            output_codes=[],
            previous_semantic_tokens=[],
            semantic_history_tokens=None,
            semantic_history_count=0,
            last_codebook_values=None,
            latest_stream_code_chunk=None,
        ),
    )
    runner._collect_step_outputs(SimpleNamespace(next_token_ids=None), [active])
    assert len(active.data.output_codes) == 1
    assert torch.equal(active.data.last_codebook_values, torch.tensor([1, 2]))
    assert torch.equal(
        active.data.latest_stream_code_chunk,
        active.data.output_codes[0],
    )
    assert active.data.previous_semantic_tokens == [201]
    assert active.data.semantic_history_count == 1
    assert torch.equal(
        active.data.semantic_history_tokens,
        torch.tensor([201, 0, 0, 0], dtype=torch.long),
    )


class _FakePlanner:
    def __init__(self) -> None:
        self.recorded = None

    def select_requests(self, waiting, running):
        del running
        return list(waiting)

    def build_batch(self, requests):
        return SimpleNamespace(request_ids=[request.request_id for request in requests])

    def record_last_batch(self, batch_data) -> None:
        self.recorded = batch_data


class _FakeResourceManager:
    def __init__(self) -> None:
        self.freed: list[str] = []

    def free(self, request) -> None:
        self.freed.append(request.request_id)


def make_fish_scheduler() -> FishScheduler:
    def request_builder(payload):
        return SimpleNamespace(
            req=FakeFishReq(rid=payload.request_id),
            stage_payload=payload,
            output_codes=[torch.tensor([[100], [1], [2]], dtype=torch.long)],
            previous_semantic_tokens=[],
            last_codebook_values=None,
            latest_stream_code_chunk=None,
            max_new_tokens=4,
            input_ids=[1, 2, 3],
        )

    def result_adapter(data):
        payload = make_s2pro_payload(request_id=data.req.rid)
        payload.data = {"output_ids": list(data.req.output_ids)}
        return payload

    scheduler = FishScheduler(
        tree_cache=SimpleNamespace(cache_unfinished_req=lambda req: None),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        prefill_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(),
        server_args=SimpleNamespace(),
        model_runner=SimpleNamespace(),
        request_builder=request_builder,
        result_adapter=result_adapter,
        im_end_token_id=99,
        max_new_tokens=4,
    )
    scheduler.batch_planner = _FakePlanner()
    scheduler.resource_manager = _FakeResourceManager()
    return scheduler


def test_fish_scheduler_lifecycle_abort_and_iteration_contracts() -> None:
    """Preserves chunked iteration state, finished emission, and abort cleanup."""
    request = SchedulerRequest(
        request_id="chunked",
        data=SimpleNamespace(
            req=FakeFishReq(is_chunked=2),
            output_codes=[],
            previous_semantic_tokens=[],
        ),
    )
    controller = FishIterationController(
        tree_cache=SimpleNamespace(cache_unfinished_req=lambda req: None),
        im_end_token_id=99,
        max_new_tokens=4,
    )
    controller.update_request(request, 10)
    assert request.data.req.is_chunked == 1
    assert request.data.req.output_ids == []

    scheduler = make_fish_scheduler()
    scheduler.process_input_requests([make_s2pro_payload(request_id="req-1")])
    batch = scheduler.schedule()
    finished = scheduler.update(
        batch,
        ModelRunnerOutput(outputs={"req-1": RequestOutput("req-1", data=99)}),
    )
    scheduler.emit_finished(finished)
    message = scheduler.outbox.get_nowait()
    assert batch.request_ids == ["req-1"]
    assert scheduler.resource_manager.freed == ["req-1"]
    assert message.type == "result"
    assert message.data.data["output_ids"] == [99]

    scheduler.process_input_requests([make_s2pro_payload(request_id="req-2")])
    scheduler.abort("req-2")
    scheduler.inbox.put(
        IncomingMessage("req-2", "new_request", make_s2pro_payload(request_id="req-2"))
    )
    assert scheduler.recv_requests() == []
    scheduler._cleanup_aborted_requests()
    assert "req-2" not in scheduler._requests


class _CountingTreeCache:
    def __init__(self) -> None:
        self.cached_requests = 0

    def cache_unfinished_req(self, req, *args, **kwargs) -> None:
        del req, args, kwargs
        self.cached_requests += 1


def _make_s2pro_request(request_id: str, *, is_chunked: int = 0) -> SchedulerRequest:
    req = FakeFishReq(is_chunked=is_chunked)
    req.finished = lambda: False
    return SchedulerRequest(
        request_id=request_id,
        data=SimpleNamespace(
            req=req,
            output_codes=[],
            previous_semantic_tokens=[],
            semantic_history_tokens=None,
            semantic_history_count=0,
            last_codebook_values=None,
            latest_stream_code_chunk=None,
            max_new_tokens=2048,
            temperature=0.8,
            top_p=0.8,
            top_k=30,
            repetition_penalty=1.1,
            ras_temperature=1.0,
            ras_top_p=0.9,
        ),
    )


def _collect_s2pro_step(
    requests: list[SchedulerRequest],
    code_rows: list[list[int]],
    *,
    rep_history_len: int | None = None,
) -> SimpleNamespace:
    result = SimpleNamespace(next_token_ids=None)
    output_codes = torch.tensor(code_rows, dtype=torch.long)
    collect_s2pro_step_outputs(
        result,
        requests,
        output_codes=output_codes,
        output_semantic_ids=output_codes[:, 0].clone(),
        im_end_token_id=IM_END_TOKEN_ID,
        rep_history_len=rep_history_len,
    )
    return result


def _update_request_from_step(
    controller: FishIterationController,
    request: SchedulerRequest,
    result: SimpleNamespace,
    row_idx: int = 0,
) -> int:
    token_id = int(result.next_token_ids[row_idx].item())
    controller.update_request(request, token_id)
    return token_id


def test_fish_s2pro_audio_timestep_updates_audio_and_stream_state() -> None:
    request = _make_s2pro_request("req-audio")
    data = request.data

    result = _collect_s2pro_step(
        [request],
        [[SEMANTIC_TOKEN_ID, 11, 22]],
        rep_history_len=4,
    )

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


def test_fish_s2pro_prepare_decode_uses_gpu_history_buffer() -> None:
    req = FakeFishReq()
    data = S2ProSGLangRequestData(input_ids=torch.tensor([], dtype=torch.long), req=req)
    data.previous_semantic_tokens = [9999]
    data.semantic_history_tokens = torch.tensor(
        [SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0],
        dtype=torch.long,
    )
    data.semantic_history_count = 2
    data.last_codebook_values = torch.tensor([11, 22], dtype=torch.long)
    request = SchedulerRequest(request_id="req-history", data=data)

    runner = object.__new__(FishS2ProModelRunner)
    runner._semantic_begin_id = SEMANTIC_TOKEN_ID
    runner._semantic_end_id = SEMANTIC_TOKEN_ID + 10
    runner.model = SimpleNamespace(
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


def test_fish_s2pro_terminal_im_end_is_not_audio_codebook_frame() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_s2pro_request("req-terminal")

    result = _collect_s2pro_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])
    _update_request_from_step(controller, request, result)
    request.data.latest_stream_code_chunk = None

    result = _collect_s2pro_step([request], [[IM_END_TOKEN_ID, 33, 44]])
    eos_token = _update_request_from_step(controller, request, result)

    assert controller.is_finished(request, eos_token)
    assert request.data.req.output_ids == [SEMANTIC_TOKEN_ID, IM_END_TOKEN_ID]
    assert len(request.data.output_codes) == 1
    assert torch.equal(
        request.data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert request.data.latest_stream_code_chunk is None
    assert request.data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert torch.equal(request.data.last_codebook_values, torch.tensor([11, 22]))
    assert tree_cache.cached_requests == 1


def test_fish_s2pro_immediate_im_end_leaves_no_audio_codebook_frames() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_s2pro_request("req-immediate-terminal")

    result = _collect_s2pro_step([request], [[IM_END_TOKEN_ID, 33, 44]])
    eos_token = _update_request_from_step(controller, request, result)

    assert controller.is_finished(request, eos_token)
    assert request.data.req.output_ids == [IM_END_TOKEN_ID]
    assert request.data.output_codes == []
    assert request.data.latest_stream_code_chunk is None
    assert request.data.previous_semantic_tokens == []
    assert request.data.last_codebook_values is None
    assert tree_cache.cached_requests == 0


def test_fish_s2pro_emit_finished_errors_before_vocoder_for_empty_codes() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_s2pro_request("req-immediate-terminal")

    result = _collect_s2pro_step([request], [[IM_END_TOKEN_ID, 33, 44]])
    _update_request_from_step(controller, request, result)

    def result_adapter(data):
        del data
        raise AssertionError("empty output_codes must not route to result_adapter")

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


def test_fish_s2pro_mixed_batch_keeps_terminal_and_audio_state_separate() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    audio_request = _make_s2pro_request("req-audio")
    terminal_request = _make_s2pro_request("req-terminal")

    result = _collect_s2pro_step(
        [audio_request, terminal_request],
        [
            [SEMANTIC_TOKEN_ID, 11, 22],
            [IM_END_TOKEN_ID, 33, 44],
        ],
    )
    audio_token = _update_request_from_step(controller, audio_request, result, 0)
    terminal_token = _update_request_from_step(
        controller,
        terminal_request,
        result,
        1,
    )

    assert not controller.is_finished(audio_request, audio_token)
    assert controller.is_finished(terminal_request, terminal_token)
    assert audio_request.data.req.output_ids == [SEMANTIC_TOKEN_ID]
    assert terminal_request.data.req.output_ids == [IM_END_TOKEN_ID]
    assert len(audio_request.data.output_codes) == 1
    assert terminal_request.data.output_codes == []
    assert torch.equal(
        audio_request.data.latest_stream_code_chunk,
        audio_request.data.output_codes[0],
    )
    assert terminal_request.data.latest_stream_code_chunk is None
    assert torch.equal(
        audio_request.data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert audio_request.data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert terminal_request.data.previous_semantic_tokens == []
    assert torch.equal(audio_request.data.last_codebook_values, torch.tensor([11, 22]))
    assert terminal_request.data.last_codebook_values is None
    assert tree_cache.cached_requests == 1


def test_fish_s2pro_chunked_step_does_not_mutate_decode_state() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_s2pro_request("req-chunked", is_chunked=1)

    result = _collect_s2pro_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])
    semantic_token = _update_request_from_step(controller, request, result)

    assert not controller.is_finished(request, semantic_token)
    assert request.data.req.is_chunked == 0
    assert request.data.req.output_ids == []
    assert request.data.output_codes == []
    assert request.data.latest_stream_code_chunk is None
    assert request.data.previous_semantic_tokens == []
    assert request.data.last_codebook_values is None
    assert tree_cache.cached_requests == 0
