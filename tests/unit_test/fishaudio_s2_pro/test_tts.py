# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.fish_scheduler import (
    FishIterationController,
    FishScheduler,
)
from sglang_omni.models.fishaudio_s2_pro.model_runner import (
    FishS2ProModelRunner,
    collect_s2pro_step_outputs,
)
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
    validate_s2pro_top_k,
)
from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
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


def _stream_payload(request_id: str, *, stream: bool = True):
    return make_s2pro_payload(request_id=request_id, params={"stream": stream})


def _code(value: int = 1) -> torch.Tensor:
    return torch.full((11, 1), value, dtype=torch.long)


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


def test_fish_s2pro_before_decode_uses_gpu_history_buffer() -> None:
    req = FakeFishReq()
    data = S2ProSGLangRequestData(input_ids=torch.tensor([], dtype=torch.long), req=req)
    data.previous_semantic_tokens = [9999]
    data.semantic_history_tokens = torch.tensor(
        [SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0],
        dtype=torch.long,
    )
    data.semantic_history_count = 2
    data.last_codebook_values = torch.tensor([11, 22], dtype=torch.long)
    data.temperature = 0.6
    data.top_p = 0.7
    data.top_k = 8
    data.repetition_penalty = 1.3
    data.ras_temperature = 0.4
    data.ras_top_p = 0.5
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
        _sampling_seeds=torch.full((1,), -1, dtype=torch.long),
        _step_count=torch.zeros(1, dtype=torch.long),
        _ras_temperature=torch.zeros(1),
        _ras_top_p=torch.zeros(1),
        _prev_tokens=torch.zeros(1, 4, dtype=torch.long),
        _prev_token_count=torch.zeros(1, dtype=torch.long),
        _vq_codes=torch.zeros(1, 2, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([SEMANTIC_TOKEN_ID]))

    runner.before_decode(forward_batch, None, [request])

    assert torch.equal(
        runner.model._prev_tokens[0],
        torch.tensor([SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0]),
    )
    assert int(runner.model._prev_token_count[0].item()) == 2
    assert torch.equal(runner.model._vq_codes[0], torch.tensor([11, 22]))
    assert bool(runner.model._vq_mask[0])
    assert torch.allclose(runner.model._sampling_temperature, torch.tensor([0.6]))
    assert torch.allclose(runner.model._sampling_top_p, torch.tensor([0.7]))
    assert int(runner.model._sampling_top_k[0].item()) == 8
    assert torch.allclose(runner.model._sampling_rep_penalty, torch.tensor([1.3]))
    assert torch.allclose(runner.model._ras_temperature, torch.tensor([0.4]))
    assert torch.allclose(runner.model._ras_top_p, torch.tensor([0.5]))


def test_fish_s2pro_before_prefill_syncs_decode_state() -> None:
    first = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=FakeFishReq(extend_input_len=1),
    )
    first.temperature = 0.55
    first.top_p = 0.65
    first.top_k = 6
    first.repetition_penalty = 1.25
    first.ras_temperature = 0.35
    first.ras_top_p = 0.45

    second = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=FakeFishReq(extend_input_len=1),
    )
    second.temperature = 0.75
    second.top_p = 0.85
    second.top_k = 12
    second.repetition_penalty = 1.05
    second.ras_temperature = 0.25
    second.ras_top_p = 0.95
    second.semantic_history_tokens = torch.tensor(
        [SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0],
        dtype=torch.long,
    )
    second.semantic_history_count = 2

    runner = object.__new__(FishS2ProModelRunner)

    def _embed(input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 2)

    runner.model = SimpleNamespace(
        get_embed_tokens=lambda: _embed,
        _audio_decoder=SimpleNamespace(
            embed_text_dim=lambda embeds, parts, mask: embeds
        ),
        _rep_history_len=4,
        _sampling_temperature=torch.zeros(2),
        _sampling_top_p=torch.zeros(2),
        _sampling_top_k=torch.zeros(2, dtype=torch.long),
        _sampling_rep_penalty=torch.zeros(2),
        _sampling_seeds=torch.full((2,), -1, dtype=torch.long),
        _step_count=torch.zeros(2, dtype=torch.long),
        _ras_temperature=torch.zeros(2),
        _ras_top_p=torch.zeros(2),
        _prev_tokens=torch.full((2, 4), 999, dtype=torch.long),
        _prev_token_count=torch.full((2,), 99, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([10, 11]))

    runner.before_prefill(
        forward_batch,
        None,
        [
            SchedulerRequest(request_id="req-first", data=first),
            SchedulerRequest(request_id="req-second", data=second),
        ],
    )

    assert hasattr(forward_batch, "input_embeds")
    assert torch.equal(runner.model._prev_tokens[0], torch.zeros(4, dtype=torch.long))
    assert int(runner.model._prev_token_count[0].item()) == 0
    assert torch.equal(
        runner.model._prev_tokens[1],
        torch.tensor([SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0]),
    )
    assert int(runner.model._prev_token_count[1].item()) == 2
    assert runner.model._sampling_top_k.tolist() == [6, 12]
    assert torch.allclose(
        runner.model._sampling_temperature,
        torch.tensor([0.55, 0.75]),
    )
    assert torch.allclose(runner.model._sampling_top_p, torch.tensor([0.65, 0.85]))
    assert torch.allclose(
        runner.model._sampling_rep_penalty,
        torch.tensor([1.25, 1.05]),
    )
    assert torch.allclose(runner.model._ras_temperature, torch.tensor([0.35, 0.25]))
    assert torch.allclose(runner.model._ras_top_p, torch.tensor([0.45, 0.95]))


def test_fish_s2pro_accepts_default_top_k_sentinel() -> None:
    validate_s2pro_top_k(-1)


def test_fish_s2pro_setup_vq_decode_allocates_sampling_state() -> None:
    model = SimpleNamespace(
        vocab_size=80,
        embed_tokens=SimpleNamespace(weight=torch.empty(1, device="cpu")),
    )
    audio_decoder = SimpleNamespace(
        codebook_embeddings=torch.nn.Embedding(16, 4),
        codebook_offsets=torch.tensor([0, 8], dtype=torch.long),
    )

    S2ProSGLangTextModel.setup_vq_decode(
        model,
        audio_decoder,
        num_codebooks=2,
        codebook_size=8,
        semantic_begin_id=10,
        semantic_end_id=20,
        im_end_token_id=30,
        max_batch_size=3,
        rep_history_len=5,
    )

    assert model._rep_history_len == 5
    assert model._prev_tokens.shape == (3, 5)
    assert model._prev_token_count.shape == (3,)
    assert model._sampling_temperature.shape == (3,)
    assert model._sampling_top_p.shape == (3,)
    assert model._sampling_top_k.tolist() == [30, 30, 30]
    assert model._sampling_rep_penalty.shape == (3,)
    assert model._ras_temperature.shape == (3,)
    assert model._ras_top_p.shape == (3,)
    assert model._rep_positions.tolist() == [0, 1, 2, 3, 4]
    assert model._top_k_positions.shape == (30,)
    assert model._vq_ready


def test_fish_s2pro_decode_codebooks_keeps_eos_out_of_audio_embedding(
    monkeypatch,
) -> None:
    # multinomial_with_seed is a GPU-only Triton kernel; this CPU test only
    # exercises EOS handling, and all rows are unseeded, so stub it out.
    def _fake_multinomial_with_seed(logprobs, seeds, pos):
        del seeds, pos
        assert torch.all(logprobs <= 0)
        assert torch.isneginf(logprobs).any()
        return logprobs.argmax(-1, keepdim=True)

    decode_codebooks = S2ProSGLangTextModel._decode_codebooks.__wrapped__
    monkeypatch.setitem(
        decode_codebooks.__globals__,
        "multinomial_with_seed",
        _fake_multinomial_with_seed,
    )

    class _AudioDecoder:
        def __init__(self) -> None:
            self.seen_embedding_ids: list[torch.Tensor] = []

        def reset_caches(self) -> None:
            pass

        def project_in(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

        def forward_kvcached(
            self,
            hidden_states: torch.Tensor,
            *,
            codebook_idx: int,
        ) -> torch.Tensor:
            del hidden_states, codebook_idx
            return torch.zeros(1, 1, 8)

        def embeddings(self, ids: torch.Tensor) -> torch.Tensor:
            assert int(ids.max().item()) < 8
            self.seen_embedding_ids.append(ids.detach().clone())
            return torch.zeros(ids.shape[0], 4)

    audio_decoder = _AudioDecoder()
    model = SimpleNamespace(
        _semantic_bias=torch.full((40,), -float("inf")),
        _prev_token_count=torch.zeros(1, dtype=torch.long),
        _ras_range=torch.arange(4, 0, -1),
        _prev_tokens=torch.zeros(1, 4, dtype=torch.long),
        _ras_temperature=torch.ones(1),
        _sampling_temperature=torch.ones(1),
        _ras_top_p=torch.ones(1),
        _sampling_top_p=torch.ones(1),
        _sampling_rep_penalty=torch.ones(1),
        _sampling_seeds=torch.full((1,), -1, dtype=torch.long),
        _step_count=torch.zeros(1, dtype=torch.long),
        _rep_positions=torch.arange(4),
        _graph_top_k=30,
        _sampling_top_k=torch.full((1,), 30, dtype=torch.long),
        _top_k_positions=torch.arange(30),
        _audio_decoder=audio_decoder,
        _semantic_begin_id=10,
        _im_end_token_id=30,
        _codebook_size=8,
        _num_codebooks=2,
        _output_codes=torch.zeros(1, 3, dtype=torch.long),
        _output_semantic_ids=torch.zeros(1, dtype=torch.long),
    )
    model._semantic_bias[10:18] = 0.0
    model._semantic_bias[30] = 0.0
    logits = torch.full((1, 40), -1_000_000.0)
    logits[0, 30] = 1_000_000.0

    S2ProSGLangTextModel._decode_codebooks(
        model,
        logits,
        torch.zeros(1, 4),
    )

    assert int(model._output_semantic_ids[0].item()) == 30
    assert int(model._output_codes[0, 0].item()) == 30
    assert int(audio_decoder.seen_embedding_ids[0][0].item()) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="multinomial_with_seed needs CUDA"
)
def test_fish_s2pro_seeded_sampler_preserves_probability_distribution() -> None:
    batch = 20_000
    device = torch.device("cuda")
    semantic_begin_id = 10
    im_end_token_id = 39
    vocab_size = 40

    class _AudioDecoder:
        def reset_caches(self) -> None:
            pass

        def project_in(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

        def forward_kvcached(
            self,
            hidden_states: torch.Tensor,
            *,
            codebook_idx: int,
        ) -> torch.Tensor:
            del codebook_idx
            return torch.zeros(
                hidden_states.shape[0], 1, 8, device=hidden_states.device
            )

        def embeddings(self, ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(ids.shape[0], 4, device=ids.device)

    semantic_bias = torch.full((vocab_size,), -float("inf"), device=device)
    semantic_bias[semantic_begin_id : semantic_begin_id + 2] = 0.0
    semantic_bias[im_end_token_id] = 0.0
    logits = torch.full((batch, vocab_size), -float("inf"), device=device)
    logits[:, semantic_begin_id] = torch.log(torch.tensor(0.9, device=device))
    logits[:, semantic_begin_id + 1] = torch.log(torch.tensor(0.1, device=device))

    model = SimpleNamespace(
        _semantic_bias=semantic_bias,
        _prev_token_count=torch.zeros(batch, dtype=torch.long, device=device),
        _ras_range=torch.arange(4, 0, -1, device=device),
        _prev_tokens=torch.zeros(batch, 4, dtype=torch.long, device=device),
        _ras_temperature=torch.ones(batch, device=device),
        _sampling_temperature=torch.ones(batch, device=device),
        _ras_top_p=torch.ones(batch, device=device),
        _sampling_top_p=torch.ones(batch, device=device),
        _sampling_rep_penalty=torch.ones(batch, device=device),
        _sampling_seeds=torch.arange(1, batch + 1, dtype=torch.long, device=device),
        _step_count=torch.zeros(batch, dtype=torch.long, device=device),
        _rep_positions=torch.arange(4, device=device),
        _graph_top_k=30,
        _sampling_top_k=torch.full((batch,), 2, dtype=torch.long, device=device),
        _top_k_positions=torch.arange(30, device=device),
        _audio_decoder=_AudioDecoder(),
        _semantic_begin_id=semantic_begin_id,
        _im_end_token_id=im_end_token_id,
        _codebook_size=8,
        _num_codebooks=1,
        _output_codes=torch.zeros(batch, 2, dtype=torch.long, device=device),
        _output_semantic_ids=torch.zeros(batch, dtype=torch.long, device=device),
    )

    S2ProSGLangTextModel._decode_codebooks(
        model,
        logits,
        torch.zeros(batch, 4, device=device),
    )

    token0_rate = (
        (model._output_semantic_ids == semantic_begin_id).float().mean().item()
    )
    assert 0.87 < token0_rate < 0.93


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


def test_fish_s2pro_max_tokens_sets_length_finish_reason() -> None:
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, IM_END_TOKEN_ID)
    request = _make_s2pro_request("req-length")
    request.data.max_new_tokens = 1

    result = _collect_s2pro_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])
    semantic_token = _update_request_from_step(controller, request, result)

    assert controller.is_finished(request, semantic_token)
    assert request.data.finish_reason == "length"


def test_fish_scheduler_emits_code_chunks_only_for_streaming_requests() -> None:
    class _IterationController:
        def update_request(self, request, output_token_id) -> None:
            pass

        def is_finished(self, request, output_token_id) -> bool:
            return False

    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = set()
    scheduler.iteration_controller = _IterationController()

    stream_codes = _code(7)
    stream_req = SchedulerRequest(
        request_id="stream",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(
            stage_payload=_stream_payload("stream", stream=True),
            latest_stream_code_chunk=stream_codes,
        ),
    )
    non_stream_req = SchedulerRequest(
        request_id="non-stream",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(
            stage_payload=_stream_payload("non-stream", stream=False),
            latest_stream_code_chunk=_code(8),
        ),
    )

    finished = scheduler.update(
        SchedulerOutput(
            requests=[stream_req, non_stream_req],
            batch_data=None,
        ),
        ModelRunnerOutput(
            outputs={
                "stream": RequestOutput("stream", data=1),
                "non-stream": RequestOutput("non-stream", data=1),
            }
        ),
    )

    assert finished == []
    out = scheduler.outbox.get_nowait()
    assert out.request_id == "stream"
    assert out.type == "stream"
    assert out.target == "vocoder"
    assert out.data is stream_codes
    assert stream_req.data.latest_stream_code_chunk is None
    assert scheduler.outbox.empty()


def test_fish_scheduler_abort_during_update_suppresses_stream_chunk() -> None:
    freed = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = ["req"]
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = set()
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )

    class _IterationController:
        def update_request(self, request, output_token_id) -> None:
            del request, output_token_id
            scheduler.abort("req")

        def is_finished(self, request, output_token_id) -> bool:
            del request, output_token_id
            return True

    scheduler.iteration_controller = _IterationController()
    request = SchedulerRequest(
        request_id="req",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(
            stage_payload=_stream_payload("req", stream=True),
            latest_stream_code_chunk=_code(9),
        ),
    )
    scheduler._requests["req"] = request

    finished = scheduler.update(
        SchedulerOutput(requests=[request], batch_data=None),
        ModelRunnerOutput(outputs={"req": RequestOutput("req", data=1)}),
    )

    assert finished == []
    assert freed == ["req"]
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._running_ids


def test_fish_scheduler_emit_finished_suppresses_aborted_result() -> None:
    adapted = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = {"req"}
    scheduler._requests = {}
    scheduler._submit_times = {"req": 1.0}

    def _result_adapter(data):
        adapted.append(data)
        return _stream_payload("req")

    scheduler._result_adapter = _result_adapter
    request = SchedulerRequest(
        request_id="req",
        status=SchedulerStatus.FINISHED,
        data=SimpleNamespace(req=SimpleNamespace(output_ids=[1])),
    )
    scheduler._requests["req"] = request

    scheduler.emit_finished([request])

    assert adapted == []
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_finish_preserves_abort_marker_for_emit_suppression() -> None:
    freed = []
    adapted = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = {"req"}
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = ["req"]
    scheduler._submit_times = {"req": 1.0}
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )
    scheduler._result_adapter = lambda data: adapted.append(data) or _stream_payload(
        "req"
    )
    request = SchedulerRequest(
        request_id="req",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(req=SimpleNamespace(output_ids=[1])),
    )
    scheduler._requests["req"] = request

    scheduler._finish_request(request)
    scheduler.emit_finished([request])

    assert freed == ["req"]
    assert adapted == []
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_abort_cleanup_frees_waiting_request_resources() -> None:
    freed = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque(["req"])
    scheduler._running_ids = []
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = set()
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )
    request = SchedulerRequest("req", data=SimpleNamespace())
    scheduler._requests["req"] = request

    scheduler.abort("req")

    assert freed == []
    assert request.status == SchedulerStatus.ABORTED
    assert "req" in scheduler._requests

    scheduler._cleanup_aborted_requests()

    assert freed == ["req"]
    assert request.status == SchedulerStatus.ABORTED
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._waiting
    assert "req" in scheduler._aborted_request_ids
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_abort_defers_inflight_resource_free_until_update() -> None:
    freed = []
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = ["req"]
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = {"req"}
    scheduler.resource_manager = SimpleNamespace(
        free=lambda request: freed.append(request.request_id)
    )
    request = SchedulerRequest(
        "req",
        status=SchedulerStatus.RUNNING,
        data=SimpleNamespace(),
    )
    scheduler._requests["req"] = request

    scheduler.abort("req")

    assert freed == []
    assert request.status == SchedulerStatus.ABORTED
    assert "req" in scheduler._requests

    scheduler._cleanup_aborted_requests()
    assert freed == []
    assert "req" in scheduler._requests

    finished = scheduler.update(
        SchedulerOutput(requests=[request], batch_data=None),
        ModelRunnerOutput(outputs={}),
    )

    assert finished == []
    assert freed == ["req"]
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._running_ids
    assert "req" not in scheduler._submit_times


def test_fish_scheduler_batch_exception_cleans_finished_request() -> None:
    scheduler = FishScheduler.__new__(FishScheduler)
    scheduler.outbox = queue.Queue()
    scheduler._aborted_request_ids = set()
    scheduler._requests = {}
    scheduler._waiting = deque()
    scheduler._running_ids = []
    scheduler._submit_times = {"req": 1.0}
    scheduler._inflight_request_ids = set()
    scheduler.resource_manager = SimpleNamespace(free=lambda request: None)

    request = SchedulerRequest(
        "req",
        status=SchedulerStatus.FINISHED,
        data=SimpleNamespace(),
    )
    scheduler._requests["req"] = request
    error = RuntimeError("adapter failed")

    scheduler._handle_batch_exception(
        SchedulerOutput(requests=[request], batch_data=None),
        error,
    )

    out = scheduler.outbox.get_nowait()
    assert out.request_id == "req"
    assert out.type == "error"
    assert out.data is error
    assert "req" not in scheduler._requests
    assert "req" not in scheduler._submit_times
    assert "req" not in scheduler._aborted_request_ids
