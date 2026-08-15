# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.flow_matching import (
    SenseNovaU1FlowMatchingRunner,
    _validated_flow_params_from_request,
)
from sglang_omni.models.sensenova_u1.interleave import (
    SenseNovaU1InterleaveRunner,
    _params_from_payload,
)
from sglang_omni.models.sensenova_u1.limits import (
    U1_MAX_TOTAL_TOKENS,
)
from sglang_omni.models.sensenova_u1.native_serving import (
    SenseNovaU1NativeServingExecutor,
    _NativeEagerDecodeGraph,
)


def _cache_only_executor() -> SenseNovaU1NativeServingExecutor:
    executor = object.__new__(SenseNovaU1NativeServingExecutor)
    executor._eager_text_prefill_cache = OrderedDict()
    executor._eager_text_decode_graphs = OrderedDict()
    executor._eager_text_prefill_cache_evictions = 0
    executor._eager_text_decode_graph_evictions = 0
    executor._eager_text_decode_graph_captures = 0
    executor.eager_prefix_cache_max_entries = 4
    executor.eager_decode_graph_cache_max_entries = 2
    executor.eager_decode_graph_max_captures = 4
    executor.eager_prefix_cache_max_tokens = 2048
    executor.eager_decode_graph_max_total_tokens = 1024
    return executor


def _prefix_cache_entry(
    length: int,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], bool]:
    caches = [
        (
            torch.zeros((length, 2), dtype=torch.bfloat16),
            torch.zeros((length, 2), dtype=torch.bfloat16),
        )
        for _ in range(2)
    ]
    return torch.zeros((1, 8), dtype=torch.bfloat16), caches, True


class _FakeCudaGraph:
    def __init__(self) -> None:
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True


def _decode_graph_runner(length: int) -> _NativeEagerDecodeGraph:
    initial_caches = [
        (
            torch.zeros((length, 2), dtype=torch.bfloat16),
            torch.zeros((length, 2), dtype=torch.bfloat16),
        )
    ]
    final_caches = [
        (
            torch.zeros((length + 1, 2), dtype=torch.bfloat16),
            torch.zeros((length + 1, 2), dtype=torch.bfloat16),
        )
    ]
    return _NativeEagerDecodeGraph(
        graphs=[_FakeCudaGraph()],  # type: ignore[list-item]
        graph_pool=object(),
        initial_caches=initial_caches,
        input_token=torch.zeros(1, dtype=torch.long),
        decode_indexes=[torch.zeros((3, 1), dtype=torch.long)],
        generated_tokens=[torch.zeros(1, dtype=torch.long)],
        last_logits=torch.zeros((1, 8), dtype=torch.bfloat16),
        final_caches=final_caches,
        cache_history=[final_caches],
        source_prefix_cache_key=f"prefix-{length}",
        prefix_len=length,
        verified_finite=True,
    )


def test_sensenova_u1_eager_caches_are_bounded_lru_and_release_references() -> None:
    executor = _cache_only_executor()
    prefix_cycle_bytes = []
    evicted_prefix_caches = None
    for cycle in range(3):
        for length in (8, 16, 24, 32, 40, 48):
            entry = _prefix_cache_entry(length)
            if cycle == 0 and length == 8:
                evicted_prefix_caches = entry[1]
            executor._put_eager_prefix_cache_entry(
                f"prefix-{cycle}-{length}",
                entry,
            )
            assert len(executor._eager_text_prefill_cache) <= 4
        prefix_cycle_bytes.append(
            executor.eager_runtime_cache_snapshot()[
                "prefix_cache_tensor_bytes"
            ]
        )

    assert prefix_cycle_bytes[0] == prefix_cycle_bytes[1] == prefix_cycle_bytes[2]
    assert evicted_prefix_caches == []
    assert executor._eager_text_prefill_cache_evictions == 14

    graph_cycle_bytes = []
    first_graph = None
    first_cuda_graph = None
    for cycle in range(3):
        for length in (8, 16, 24, 32):
            graph = _decode_graph_runner(length)
            if cycle == 0 and length == 8:
                first_graph = graph
                first_cuda_graph = graph.graphs[0]
            executor._put_eager_decode_graph(
                (cycle, length),
                graph,
            )
            assert len(executor._eager_text_decode_graphs) <= 2
        graph_cycle_bytes.append(
            executor.eager_runtime_cache_snapshot()[
                "decode_graph_tensor_bytes"
            ]
        )

    assert graph_cycle_bytes[0] == graph_cycle_bytes[1] == graph_cycle_bytes[2]
    assert first_graph is not None
    assert first_cuda_graph is not None
    assert first_cuda_graph.reset_called
    assert first_graph.graphs == []
    assert first_graph.initial_caches == []
    assert first_graph.input_token is None
    assert first_graph.graph_pool is None
    assert executor._eager_text_decode_graph_evictions == 10

    cleared = executor.clear_eager_runtime_caches()
    assert cleared["after"]["prefix_cache_entries"] == 0
    assert cleared["after"]["prefix_cache_tensor_bytes"] == 0
    assert cleared["after"]["decode_graph_entries"] == 0
    assert cleared["after"]["decode_graph_tensor_bytes"] == 0


def test_sensenova_u1_decode_graph_capture_attempts_have_lifetime_budget() -> None:
    executor = _cache_only_executor()

    assert [
        executor._try_reserve_eager_decode_graph_capture()
        for _ in range(5)
    ] == [True, True, True, True, False]
    assert (
        executor.eager_runtime_cache_snapshot()["decode_graph_captures"]
        == 4
    )

    executor.clear_eager_runtime_caches()

    assert not executor._try_reserve_eager_decode_graph_capture()


class _FakePool:
    def __init__(self, available: int) -> None:
        self.available = available

    def available_size(self) -> int:
        return self.available


class _FakeReq:
    def __init__(self, rid: str, token_count: int) -> None:
        self.rid = rid
        self.token_count = token_count
        self.req_pool_idx = None
        self.kv = None
        self.prefix_indices = []
        self.extra_key = None
        self.inflight_middle_chunks = 0

    def effective_kv_committed_len(self) -> int:
        return self.token_count


class _FakePrefillManager:
    def __init__(
        self,
        req_pool: _FakePool,
        token_pool: _FakePool,
        *,
        partial: bool,
    ) -> None:
        self.req_pool = req_pool
        self.token_pool = token_pool
        self.partial = partial
        self.waiting_queue: list[_FakeReq] = []
        self.chunked_req = None

    def add_one_request(self, req: _FakeReq) -> None:
        self.waiting_queue.append(req)

    def schedule_next_batch(
        self,
        running_batch: Any,
        num_allocatable_reqs: int,
    ) -> Any:
        del running_batch
        count = 1 if self.partial else num_allocatable_reqs
        selected = self.waiting_queue[:count]
        self.waiting_queue = self.waiting_queue[count:]
        for index, req in enumerate(selected):
            self.req_pool.available -= 1
            self.token_pool.available -= req.token_count
            req.req_pool_idx = index + 1
            req.kv = object()
        return SimpleNamespace(reqs=selected)


def _prefill_fault_executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault: str,
    partial: bool = False,
) -> tuple[
    SenseNovaU1NativeServingExecutor,
    list[_FakeReq],
    dict[str, int],
]:
    from sglang.srt.mem_cache import common as cache_common
    from sglang_omni.model_runner import base as model_runner_base
    from sglang_omni.models.sensenova_u1 import native_serving

    req_pool = _FakePool(8)
    token_pool = _FakePool(128)
    baseline = {
        "req_available": req_pool.available,
        "token_available": token_pool.available,
    }
    reqs = [_FakeReq("req-0", 7), _FakeReq("req-1", 11)]
    manager = _FakePrefillManager(
        req_pool,
        token_pool,
        partial=partial,
    )
    executor = object.__new__(SenseNovaU1NativeServingExecutor)
    executor.max_running_requests = 2
    executor.enable_radix_cache = True
    executor.prefill_cuda_graph_enabled = False
    executor.req_to_token_pool = req_pool
    executor.token_to_kv_pool_allocator = token_pool
    executor.tree_cache = SimpleNamespace(
        disable=False,
        protected_size_=0,
        evictable_size_=0,
        total_size=lambda: 0,
        evictable_size=lambda: 0,
        full_evictable_size=lambda: 0,
    )
    executor.prefill_manager = manager
    executor.decode_manager = SimpleNamespace(
        running_batch=SimpleNamespace(),
    )
    model = SimpleNamespace(last_forward_batch_prepare={"ok": True})
    model_runner = SimpleNamespace(
        model=model,
        attn_backend=SimpleNamespace(forward_metadata=None),
    )

    def forward_batch_generation(forward_batch: Any, *, batch: Any) -> Any:
        del forward_batch, batch
        if fault == "forward":
            raise RuntimeError("injected forward failure")
        logits = torch.zeros((2, 4))
        if fault == "logits":
            logits = torch.zeros((1, 4))
        return SimpleNamespace(
            can_run_cuda_graph=False,
            logits_output=SimpleNamespace(next_token_logits=logits),
        )

    executor.model_worker = SimpleNamespace(
        device="cpu",
        model_runner=model_runner,
        forward_batch_generation=forward_batch_generation,
    )
    prepared_by_id = {req.rid: req for req in reqs}

    def prepare(self: Any, **item: Any) -> dict[str, Any]:
        req = prepared_by_id[item["request_id"]]
        return {
            "request_id": item["request_id"],
            "input_ids": item["input_ids"],
            "indexes": item["indexes"],
            "image_token_tag": item["image_token_tag"],
            "image_gen_indicators": None,
            "input_embeds": None,
            "req": req,
            "cache_extra_key": None,
        }

    executor._prepare_prefill_request = MethodType(prepare, executor)

    def release_kv_cache(req: _FakeReq, tree_cache: Any, *, is_insert: bool) -> None:
        del tree_cache, is_insert
        req_pool.available += 1
        token_pool.available += req.token_count
        req.req_pool_idx = None
        req.kv = None

    monkeypatch.setattr(cache_common, "release_kv_cache", release_kv_cache)
    monkeypatch.setattr(
        model_runner_base,
        "resolve_deferred_prefill_inputs",
        lambda batch, device: None,
    )

    def init_new(
        batch: Any,
        model_runner_arg: Any,
        **kwargs: Any,
    ) -> Any:
        del model_runner_arg, kwargs
        batch_size = len(batch.reqs)
        return SimpleNamespace(
            forward_mode="EXTEND",
            batch_size=batch_size,
            input_ids=torch.arange(batch_size),
            positions=torch.arange(batch_size),
            mrope_positions=None,
            extend_seq_lens_cpu=[1] * batch_size,
            extend_prefix_lens_cpu=[0] * batch_size,
            rids=[req.rid for req in batch.reqs],
            cross_attention_custom_mask=None,
            model_specific_states={},
        )

    monkeypatch.setattr(
        native_serving,
        "ForwardBatch",
        SimpleNamespace(init_new=init_new),
    )
    if fault == "sidecar":

        def fail_sidecar(self: Any, **kwargs: Any) -> None:
            del self, kwargs
            raise RuntimeError("injected sidecar failure")

        executor._attach_prefill_sidecar = MethodType(fail_sidecar, executor)
    if fault == "metadata":

        def fail_metadata(self: Any, **kwargs: Any) -> None:
            del self, kwargs
            raise RuntimeError("injected metadata failure")

        executor._collect_prefill_metadata = MethodType(fail_metadata, executor)
    return executor, reqs, baseline


@pytest.mark.parametrize("fault", ["sidecar", "forward", "logits", "metadata"])
def test_sensenova_u1_prefill_faults_restore_scheduler_and_kv_baseline(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    executor, reqs, baseline = _prefill_fault_executor(
        monkeypatch,
        fault=fault,
    )
    requests = [
        {
            "request_id": req.rid,
            "input_ids": torch.arange(req.token_count),
            "indexes": torch.zeros((3, req.token_count), dtype=torch.long),
            "image_token_tag": torch.zeros(req.token_count, dtype=torch.bool),
        }
        for req in reqs
    ]

    with pytest.raises(RuntimeError, match="injected|logits batch size mismatch"):
        executor.run_prefill_batch(requests, cache_insert=True)

    assert executor.prefill_manager.waiting_queue == []
    assert executor.prefill_manager.chunked_req is None
    assert executor._pool_snapshot() == baseline
    assert executor._cache_snapshot()["total_size"] == 0
    assert all(req.req_pool_idx is None and req.kv is None for req in reqs)


def test_sensenova_u1_partial_prefill_failure_restores_all_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, reqs, baseline = _prefill_fault_executor(
        monkeypatch,
        fault="none",
        partial=True,
    )
    requests = [
        {
            "request_id": req.rid,
            "input_ids": torch.arange(req.token_count),
            "indexes": torch.zeros((3, req.token_count), dtype=torch.long),
            "image_token_tag": torch.zeros(req.token_count, dtype=torch.bool),
        }
        for req in reqs
    ]

    with pytest.raises(RuntimeError, match="did not schedule the full batch"):
        executor.run_prefill_batch(requests)

    assert executor.prefill_manager.waiting_queue == []
    assert executor._pool_snapshot() == baseline
    assert all(req.req_pool_idx is None and req.kv is None for req in reqs)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"width": 0}, "width must be positive"),
        ({"width": 33}, "width must be divisible"),
        ({"width": 2080}, "width exceeds"),
        ({"width": 2048, "height": 1024}, "pixel count exceeds"),
        ({"num_steps": 0}, "num_steps must be positive"),
        ({"num_steps": 65}, "num_steps exceeds"),
        ({"max_images": 0}, "max_images must be positive"),
        ({"max_images": 5}, "max_images exceeds"),
        ({"max_new_tokens": 0}, "max_new_tokens must be positive"),
        ({"max_new_tokens": 2049}, "max_new_tokens exceeds"),
        (
            {
                "width": 1024,
                "height": 1024,
                "max_images": 3,
                "max_new_tokens": 2048,
            },
            "token budget exceeds",
        ),
        ({"width": "wide"}, "width must be an integer"),
    ],
)
def test_sensenova_u1_interleave_rejects_malformed_or_oversized_payloads(
    params: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _params_from_payload("draw a lighthouse", params)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"height": -32}, "height must be positive"),
        ({"height": 48}, "height must be divisible"),
        ({"num_steps": "many"}, "num_steps must be an integer"),
        ({"num_steps": 65}, "num_steps exceeds"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"batch_size": 2}, "batch_size exceeds"),
        ({"images": "not-a-list"}, "images must be a list"),
    ],
)
def test_sensenova_u1_flow_rejects_malformed_or_oversized_payloads(
    params: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validated_flow_params_from_request(
            {"prompt": "draw a lighthouse"},
            params,
            max_total_tokens=U1_MAX_TOTAL_TOKENS,
        )


def test_sensenova_u1_rejects_excess_input_images() -> None:
    images = [Image.new("RGB", (32, 32)) for _ in range(5)]
    with pytest.raises(ValueError, match="input image count exceeds"):
        _params_from_payload(
            {"prompt": "compare", "images": images},
            {},
        )


def test_sensenova_u1_exact_prefix_budget_precedes_gpu_embedding_compose() -> None:
    runner = object.__new__(SenseNovaU1FlowMatchingRunner)
    runner.max_total_tokens = 32
    runner.tokenizer = lambda text, return_tensors: {
        "input_ids": torch.arange(40).view(1, -1)
    }
    runner._load_input_images = lambda images: (None, None)
    runner._replace_image_tokens = lambda query, grid_hw: query
    runner._get_thw_indexes = lambda input_ids, grid_hw: torch.zeros(
        (3, input_ids.numel()),
        dtype=torch.long,
    )
    runner._token_embeds = lambda input_ids: pytest.fail(
        "GPU embedding allocation must not run for an oversized prefix"
    )

    with pytest.raises(ValueError, match="token budget exceeds"):
        runner._build_prefix(
            prompt="oversized",
            images=[],
            system_message="system",
            assistant_append="<img>",
            reserved_image_tokens=1,
        )


def test_sensenova_u1_interleave_exact_budget_precedes_gpu_embedding_compose() -> None:
    runner = object.__new__(SenseNovaU1InterleaveRunner)
    runner.max_total_tokens = 32
    runner.tokenizer = lambda text, return_tensors: {
        "input_ids": torch.arange(40).view(1, -1)
    }
    runner.img_context_token_id = 7
    runner._load_interleave_prefix_images = lambda images, generated: (None, None)
    runner._replace_image_tokens = lambda query, grid_hw: query
    runner._get_thw_indexes = lambda input_ids, grid_hw: torch.zeros(
        (3, input_ids.numel()),
        dtype=torch.long,
    )
    runner._token_embeds = lambda input_ids: pytest.fail(
        "GPU embedding allocation must not run for an oversized prefix"
    )

    with pytest.raises(ValueError, match="token budget exceeds"):
        runner._build_condition_prefix(
            prompt="oversized",
            images=[],
            system_message="system",
            think_mode=True,
            reserved_text_tokens=1,
        )
