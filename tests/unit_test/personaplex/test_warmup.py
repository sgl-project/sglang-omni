# SPDX-License-Identifier: Apache-2.0
"""Startup warmup owns temporary allocations and must fail before readiness."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import DynamicGradMode

import sglang_omni.models.personaplex.warmup as warmup_module
from sglang_omni.models.personaplex.warmup import PersonaPlexSessionScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput, RequestOutput

WARMUP_BATCH_SIZES = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


class WarmupRequestPool:
    def __init__(self) -> None:
        self.owned_rows = {99}
        self.next_row = 1
        self.freed_rows = []

    def free(self, request) -> None:
        row = request.kv.req_pool_idx
        assert row != 99
        self.owned_rows.remove(row)
        self.freed_rows.append(row)
        request.kv.req_pool_idx = None


class WarmupRunner:
    def __init__(self, fail_step: int | None = None, device: str = "cpu") -> None:
        self.model_device = torch.device(device)
        self.completion_events = []
        self.model = SimpleNamespace(
            fusion_buffer=torch.zeros(3, 17), config=SimpleNamespace(vocab_size=32000)
        )
        self.fail_step = fail_step
        self.calls = []
        self.grad_modes = []

    def execute(self, output) -> ModelRunnerOutput:
        self.grad_modes.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        self.calls.append(
            [
                (
                    request.data.req.prefix_len,
                    request.data.req.extend_len,
                    request.data.talker_model_inputs["sampling"].seed,
                )
                for request in output.requests
            ]
        )
        torch.rand(8)
        if self.model_device.type == "cuda":
            torch.rand(8, device=self.model_device)
            event = torch.cuda.Event()
            event.record()
            self.completion_events.append(event)
        if len(self.calls) == self.fail_step:
            raise RuntimeError("injected forward failure")
        outputs = {
            request.request_id: RequestOutput(request_id=request.request_id, data=17)
            for request in output.requests
        }
        return ModelRunnerOutput(outputs=outputs)


@pytest.fixture
def warmup_backend(monkeypatch):
    pool = WarmupRequestPool()
    scheduler = PersonaPlexSessionScheduler.__new__(PersonaPlexSessionScheduler)
    scheduler.req_to_token_pool = pool
    scheduler.token_to_kv_pool_allocator = SimpleNamespace()
    scheduler.tree_cache = SimpleNamespace()
    scheduler.model_config = SimpleNamespace(context_len=8192)
    preparations = []
    fail_preparation = []

    def new_batch(**arguments):
        requests = arguments["reqs"]

        def prepare():
            preparations.append(requests)
            for request in requests:
                if not request.kv.holds_kv:
                    request.kv.req_pool_idx = pool.next_row
                    pool.owned_rows.add(pool.next_row)
                    pool.next_row += 1
                if fail_preparation:
                    raise RuntimeError("injected KV allocation failure")
                request.kv.kv_allocated_len = request.seqlen
                request.kv.kv_committed_len = request.seqlen

        return SimpleNamespace(prepare_for_extend=prepare)

    def release(request, tree_cache, *, checkpoint):
        assert tree_cache is scheduler.tree_cache
        assert checkpoint is False
        request.kv.mark_kv_released()
        pool.free(request)

    monkeypatch.setattr(
        "sglang_omni.models.personaplex.warmup.ScheduleBatch.init_new", new_batch
    )
    monkeypatch.setattr(
        "sglang_omni.models.personaplex.warmup.release_kv_cache", release
    )
    return scheduler, preparations, fail_preparation


def test_warmup_runs_prompt_and_single_row_extend_at_every_batch_size(warmup_backend):
    scheduler, preparations, _ = warmup_backend
    runner = WarmupRunner()
    scheduler.model_runner = runner
    original_grad_mode = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    with DynamicGradMode():
        serving_grad_mode = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    original_rng_state = torch.random.get_rng_state()
    scheduler.warm_up_serving_thread()
    assert set(runner.grad_modes) == {serving_grad_mode}
    assert (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
    ) == original_grad_mode
    assert [len(call) for call in runner.calls] == WARMUP_BATCH_SIZES
    assert [{seed for _, _, seed in call} for call in runner.calls] == [
        {None},
        {None},
        {0},
        {0},
    ] * 3
    for prompt_call, extend_call in zip(
        runner.calls[::2], runner.calls[1::2], strict=True
    ):
        assert all(prefix == 0 and rows > 1 for prefix, rows, _ in prompt_call)
        assert all(prefix > 0 and rows == 1 for prefix, rows, _ in extend_call)
    assert torch.equal(torch.random.get_rng_state(), original_rng_state)
    assert scheduler.req_to_token_pool.owned_rows == {99}
    assert len(scheduler.req_to_token_pool.freed_rows) == 12
    assert all(
        not request.kv.holds_kv for requests in preparations for request in requests
    )


@pytest.mark.parametrize("fail_step", range(1, 13))
def test_forward_failure_releases_only_warmup_requests_and_restores_rng(
    warmup_backend, fail_step
):
    scheduler, _, _ = warmup_backend
    runner = WarmupRunner(fail_step=fail_step)
    scheduler.model_runner = runner
    original_rng_state = torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        scheduler.warm_up_serving_thread()
    assert len(runner.calls) == fail_step
    assert scheduler.req_to_token_pool.owned_rows == {99}
    assert torch.equal(torch.random.get_rng_state(), original_rng_state)


def test_kv_allocation_failure_releases_request_slot_without_releasing_unowned_kv(
    warmup_backend,
):
    scheduler, _, fail_preparation = warmup_backend
    fail_preparation.append(True)
    scheduler.model_runner = WarmupRunner()
    with pytest.raises(RuntimeError, match="injected KV allocation failure"):
        scheduler.warm_up_serving_thread()
    assert scheduler.req_to_token_pool.owned_rows == {99}
    assert scheduler.req_to_token_pool.freed_rows == [1]


def test_request_build_failure_does_not_continue_warmup(warmup_backend, monkeypatch):
    scheduler, _, _ = warmup_backend
    original_builder = warmup_module.build_lm_request
    built_requests = []

    def build(payload, **arguments):
        if len(built_requests) == 3:
            raise RuntimeError("injected request initialization failure")
        request = original_builder(payload, **arguments)
        built_requests.append(request)
        return request

    monkeypatch.setattr(warmup_module, "build_lm_request", build)
    runner = WarmupRunner()
    scheduler.model_runner = runner
    with pytest.raises(RuntimeError, match="injected request initialization failure"):
        scheduler.warm_up_serving_thread()
    assert len(runner.calls) == 4
    assert all(not request.req.kv.holds_kv for request in built_requests)
    assert scheduler.req_to_token_pool.owned_rows == {99}


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fail_step", [None, 2, 4])
def test_cuda_work_finishes_before_kv_release_and_rng_restoration(
    warmup_backend, monkeypatch, fail_step
):
    scheduler, _, _ = warmup_backend
    runner = WarmupRunner(fail_step=fail_step, device="cuda")
    scheduler.model_runner = runner
    original_release = warmup_module.release_kv_cache
    original_synchronize = torch.accelerator.synchronize
    original_rng_state = torch.cuda.get_rng_state()
    synchronized_steps = []

    def synchronize(device):
        original_synchronize(device)
        synchronized_steps.append(len(runner.calls))

    def release(request, tree_cache, *, checkpoint):
        assert synchronized_steps and synchronized_steps[-1] == len(runner.calls)
        assert all(event.query() for event in runner.completion_events)
        original_release(request, tree_cache, checkpoint=checkpoint)

    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(warmup_module, "release_kv_cache", release)
    if fail_step is None:
        scheduler.warm_up_serving_thread()
    else:
        with pytest.raises(RuntimeError, match="injected forward failure"):
            scheduler.warm_up_serving_thread()
    assert torch.equal(torch.cuda.get_rng_state(), original_rng_state)
    assert scheduler.req_to_token_pool.owned_rows == {99}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "fail_step,request_capacity,token_capacity,expected_error",
    [
        (None, 4, 128, None),
        (8, 4, 128, "injected forward failure"),
        (None, 4, 2, "Out of memory"),
        (None, 3, 128, "alloc_req_slots runs out of memory"),
    ],
)
def test_real_batch_preparation_returns_request_and_token_capacity(
    device, fail_step, request_capacity, token_capacity, expected_error
):
    request_pool = ReqToTokenPool(request_capacity, 64, device, False)
    allocator = TokenToKVPoolAllocator(
        token_capacity, torch.float32, device, None, False
    )
    foreign_rows = request_pool.alloc_rows(1)
    foreign_tokens = allocator.alloc(1)
    request_pool.req_to_token[foreign_rows[0], 0] = foreign_tokens[0]
    with get_context().override_server_args(
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
        disable_radix_cache=True,
        chunked_prefill_size=None,
        enable_streaming_session=True,
        disaggregation_mode=None,
        speculative_algorithm=None,
        strip_thinking_cache=False,
    ):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=request_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            )
        )
        scheduler = PersonaPlexSessionScheduler.__new__(PersonaPlexSessionScheduler)
        scheduler.req_to_token_pool = request_pool
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.tree_cache = cache
        scheduler.model_config = SimpleNamespace(
            context_len=64, vocab_size=32000, is_encoder_decoder=False
        )
        runner = WarmupRunner(fail_step=fail_step, device=device)
        scheduler.model_runner = runner
        if expected_error is None:
            scheduler.warm_up_serving_thread()
            assert [len(call) for call in runner.calls] == WARMUP_BATCH_SIZES
        else:
            with pytest.raises(RuntimeError, match=expected_error):
                scheduler.warm_up_serving_thread()
        assert request_pool.available_size() == request_capacity - 1
        assert len(set(request_pool.free_slots)) == request_capacity - 1
        assert allocator.available_size() == token_capacity - 1
        assert torch.unique(allocator.free_pages).numel() == token_capacity - 1
        assert not torch.isin(foreign_tokens, allocator.free_pages).any()
        assert foreign_rows[0] not in request_pool.free_slots
        assert request_pool.req_to_token[foreign_rows[0], 0] == foreign_tokens[0]
        assert cache.session.slots == {}
