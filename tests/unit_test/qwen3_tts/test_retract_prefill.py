# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS re-prefill after KV-pressure retraction (#1555)."""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.penaltylib import BatchedRepetitionPenalizer
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

if current_platform.is_npu():
    from sgl_kernel_npu.mem_cache import allocator as allocator_kernels
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )

    from sglang_omni.models.qwen3_tts import engine_builder as qwen3_engine_builder
else:
    pass


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_npu(), reason="Requires an NPU")
@pytest.mark.parametrize("page_size", [16, 128])
def test_npu_allocator_preparation_preserves_pool_and_page_mapping(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup must not consume real KV pages or change partial-page allocation."""
    allocator = NPUPagedTokenToKVPoolAllocator(
        size=8192,
        page_size=page_size,
        dtype=torch.float16,
        device="npu:0",
        kvcache=None,
        need_sort=False,
    )
    free_pages_before = allocator.free_pages.clone()
    dependency_kernel = allocator_kernels.alloc_extend_kernel
    monkeypatch.setattr(allocator_kernels, "alloc_extend_kernel", dependency_kernel)
    dependency_function = dependency_kernel.fn
    dependency_annotations = dict(dependency_function.__annotations__)
    sampling_shapes: list[tuple[int, int]] = []
    original_sampler = qwen3_engine_builder.multinomial_with_seed

    def sample(
        logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        sampling_shapes.append(tuple(logprobs.shape))
        return original_sampler(logprobs, seed, positions)

    monkeypatch.setattr(qwen3_engine_builder, "multinomial_with_seed", sample)
    qwen3_engine_builder.prepare_npu_kernels(allocator, 3, 3072)
    prepared_kernel = allocator_kernels.alloc_extend_kernel
    qwen3_engine_builder.prepare_npu_kernels(allocator, 3, 3072)

    assert sampling_shapes == [(1, 3072), (2, 3072), (4, 3072)] * 2
    assert allocator_kernels.alloc_extend_kernel is prepared_kernel
    assert (
        prepared_kernel.arg_names.index("max_num_extend_tokens")
        not in prepared_kernel.constexprs
    )
    assert dependency_function.__annotations__ == dependency_annotations
    torch.testing.assert_close(allocator.free_pages, free_pages_before, rtol=0, atol=0)

    allocator.alloc(3 * page_size)
    free_pages = allocator.free_pages.cpu().tolist()
    prefixes_cpu = torch.tensor([0, 1, page_size], dtype=torch.int64)
    sequences_cpu = torch.tensor(
        [page_size + 3, page_size + 5, page_size + 2], dtype=torch.int64
    )
    last_positions = torch.tensor(
        [-1, 2 * page_size, 4 * page_size - 1], dtype=torch.int64, device="npu:0"
    )
    expected = [
        *range(free_pages[0] * page_size, (free_pages[0] + 1) * page_size),
        *range(free_pages[1] * page_size, free_pages[1] * page_size + 3),
        *range(2 * page_size + 1, 3 * page_size),
        *range(free_pages[2] * page_size, free_pages[2] * page_size + 5),
        *range(free_pages[3] * page_size, free_pages[3] * page_size + 2),
    ]
    actual = allocator.alloc_extend(
        prefixes_cpu.to("npu:0"),
        prefixes_cpu,
        sequences_cpu.to("npu:0"),
        sequences_cpu,
        last_positions,
        extend_num_tokens=int((sequences_cpu - prefixes_cpu).sum()),
    )

    assert actual.cpu().tolist() == expected
    assert allocator.free_pages.cpu().tolist() == free_pages[4:]


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(hidden))


def make_runner() -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = TinyModel()
    return runner


def make_sched_req(
    *,
    prompt: torch.Tensor,
    extend_len: int,
    prefix_len: int = 0,
    history: list[torch.Tensor] | None = None,
    pending_feedback: list[torch.Tensor] | None = None,
    pending_text: list[torch.Tensor] | None = None,
    rid: str = "req",
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            req=SimpleNamespace(
                rid=rid,
                prefix_len=prefix_len,
                extend_len=extend_len,
            ),
            prompt_input_embeds=prompt,
            prefill_input_embeds=prompt,
            decode_input_embeds=list(history or []),
            pending_feedback_queue=deque(pending_feedback or []),
            pending_text_queue=deque(pending_text or []),
            thinker_chunks_done=True,
            tts_pad_embed=torch.zeros(prompt.shape[-1], dtype=prompt.dtype),
        )
    )


def test_write_feedback_buffers_records_decode_input_history() -> None:
    embedding = torch.nn.Embedding(4, 2)
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.row_ids_cache = None
    runner.model = SimpleNamespace(
        decode_feedback_embedding=embedding,
        get_input_embeddings=lambda: embedding,
    )
    sched_req = SimpleNamespace(
        data=SimpleNamespace(
            pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
            pending_text_queue=deque([torch.tensor([20.0, 30.0])]),
            decode_input_embeds=[],
            thinker_chunks_done=True,
            tts_pad_embed=torch.zeros(2),
        )
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([99], dtype=torch.long))

    runner.write_feedback_buffers(forward_batch, [sched_req])

    assert len(sched_req.data.decode_input_embeds) == 1
    assert torch.equal(
        sched_req.data.decode_input_embeds[0],
        torch.tensor([21.0, 32.0]),
    )
    assert torch.equal(embedding.weight[0].detach(), torch.tensor([21.0, 32.0]))
    assert forward_batch.input_ids.tolist() == [0]
    assert list(sched_req.data.pending_feedback_queue) == []
    assert list(sched_req.data.pending_text_queue) == []


def test_write_feedback_buffers_batches_text_and_pad_rows() -> None:
    embedding = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.row_ids_cache = None
    runner.model = SimpleNamespace(
        decode_feedback_embedding=embedding,
        get_input_embeddings=lambda: embedding,
    )

    def data(feedback, text, pad):
        return SimpleNamespace(
            pending_feedback_queue=deque(feedback),
            pending_text_queue=deque(text),
            decode_input_embeds=[],
            thinker_chunks_done=True,
            tts_pad_embed=pad,
        )

    staged = SimpleNamespace(
        data=data(
            [torch.tensor([1.0, 2.0])], [torch.tensor([20.0, 30.0])], torch.zeros(2)
        )
    )
    padded = SimpleNamespace(
        data=data([torch.tensor([3.0, 4.0])], [], torch.tensor([0.5, 0.5]))
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([9, 9], dtype=torch.long))

    runner.write_feedback_buffers(forward_batch, [staged, padded])

    expected = torch.tensor([[21.0, 32.0], [3.5, 4.5]])
    assert torch.equal(embedding.weight[:2].detach(), expected)
    for row_idx, sched_req in enumerate((staged, padded)):
        assert len(sched_req.data.decode_input_embeds) == 1
        assert torch.equal(sched_req.data.decode_input_embeds[0], expected[row_idx])
    assert forward_batch.input_ids.tolist() == [0, 1]
    assert list(staged.data.pending_feedback_queue) == []
    assert list(staged.data.pending_text_queue) == []
    assert list(padded.data.pending_feedback_queue) == []


def test_write_feedback_buffers_rejects_a_row_without_queued_feedback() -> None:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.row_ids_cache = None
    runner.model = SimpleNamespace(decode_feedback_embedding=torch.nn.Embedding(4, 2))
    sched_req = SimpleNamespace(
        request_id="req",
        data=SimpleNamespace(
            pending_feedback_queue=deque(),
            pending_text_queue=deque([torch.tensor([40.0, 50.0])]),
            decode_input_embeds=[],
            thinker_chunks_done=True,
            tts_pad_embed=torch.zeros(2),
        ),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([3], dtype=torch.long))

    with pytest.raises(RuntimeError, match="no queued feedback"):
        runner.write_feedback_buffers(forward_batch, [sched_req])


def test_reprefill_after_retract_replays_prompt_plus_generated() -> None:
    # note (Richard Wang): 460 plus 134 is the #1555 594 vs 460 mismatch
    prompt_len, generated_len, hidden = 460, 134, 4
    prompt = torch.arange(prompt_len * hidden, dtype=torch.float32).reshape(
        prompt_len, hidden
    )
    history = [
        torch.full((hidden,), float(1000 + i), dtype=torch.float32)
        for i in range(generated_len - 1)
    ]
    leftover = torch.full((hidden,), 2000.0, dtype=torch.float32)
    extend_len = prompt_len + generated_len
    sched_req = make_sched_req(
        prompt=prompt,
        extend_len=extend_len,
        history=history,
        pending_feedback=[leftover],
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(extend_len, dtype=torch.long))

    out = make_runner().build_prefill_input_embeds(forward_batch, [sched_req])

    assert out.shape[0] == extend_len
    assert torch.equal(out[:prompt_len], prompt)
    assert torch.equal(out[prompt_len:-1], torch.stack(history))
    assert torch.equal(out[-1], leftover)
    assert list(sched_req.data.pending_feedback_queue) == []
    assert len(sched_req.data.decode_input_embeds) == generated_len


def test_reprefill_restores_retained_repetition_penalty_history() -> None:
    retained_req = SimpleNamespace(
        output_ids=[2, 5, 2],
        sampling_params=SimpleNamespace(repetition_penalty=1.05),
    )
    fresh_req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(repetition_penalty=1.05),
    )
    identity_req = SimpleNamespace(
        output_ids=[3],
        sampling_params=SimpleNamespace(repetition_penalty=1.0),
    )
    reqs = [retained_req, fresh_req, identity_req]

    class PenaltyOrchestrator:
        vocab_size = 8
        device = "cpu"

        def __init__(self) -> None:
            self.penalizers = {}

        def reqs(self):
            return reqs

    orchestrator = PenaltyOrchestrator()
    penalizer = BatchedRepetitionPenalizer(orchestrator)
    penalizer.prepare()
    orchestrator.penalizers[BatchedRepetitionPenalizer] = penalizer
    schedule_batch = SimpleNamespace(
        reqs=reqs,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        sampling_info=SimpleNamespace(penalizer_orchestrator=orchestrator),
    )

    scaling = penalizer.get_scaling_penalties()
    expected = torch.ones(3, 8)
    expected[0, [2, 5]] = 1.05

    class ExecutionBridge:
        def forward_context(self, batch, *, isolate_sampling):
            assert batch is schedule_batch
            assert isolate_sampling
            assert torch.equal(scaling, expected)
            return contextlib.nullcontext()

    runner = make_runner()
    runner.execution_bridge = ExecutionBridge()
    with runner.execution_context(schedule_batch, isolate_sampling=True):
        pass

    assert torch.equal(scaling, expected)

    logits = torch.tensor(
        [
            [1.0, 2.0, 10.0, 4.0, 5.0, -10.0, 7.0, 8.0],
            [1.0, 2.0, 10.0, 4.0, 5.0, -10.0, 7.0, 8.0],
            [1.0, 2.0, 10.0, 4.0, 5.0, -10.0, 7.0, 8.0],
        ]
    )
    original_logits = logits.clone()
    penalizer.apply(logits)

    assert logits[0, 2].item() == pytest.approx(10.0 / 1.05)
    assert logits[0, 5].item() == pytest.approx(-10.0 * 1.05)
    assert torch.equal(logits[1], original_logits[1])
    assert torch.equal(logits[2], original_logits[2])


def test_reprefill_replays_prompt_tail_and_generated_tail() -> None:
    prompt = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    history = [
        torch.tensor([100.0, 101.0]),
        torch.tensor([200.0, 201.0]),
        torch.tensor([300.0, 301.0]),
    ]
    sched_req = make_sched_req(
        prompt=prompt, prefix_len=8, extend_len=5, history=history
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    out = make_runner().build_prefill_input_embeds(forward_batch, [sched_req])

    expected = torch.cat([prompt[8:10], torch.stack(history)], dim=0)
    assert torch.equal(out, expected)


def test_reprefill_drains_leftover_feedback_when_history_is_short() -> None:
    prompt = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    history = [torch.tensor([10.0, 11.0])]
    sched_req = make_sched_req(
        prompt=prompt,
        extend_len=6,
        history=history,
        pending_feedback=[torch.tensor([3.0, 4.0])],
        pending_text=[torch.tensor([30.0, 40.0])],
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(6, dtype=torch.long))

    out = make_runner().build_prefill_input_embeds(forward_batch, [sched_req])

    expected = torch.cat(
        [
            prompt,
            torch.stack([torch.tensor([10.0, 11.0]), torch.tensor([33.0, 44.0])]),
        ],
        dim=0,
    )
    assert torch.equal(out, expected)
    assert len(sched_req.data.decode_input_embeds) == 2
    assert list(sched_req.data.pending_feedback_queue) == []
    assert list(sched_req.data.pending_text_queue) == []


def test_reprefill_without_generated_history_fails_loudly() -> None:
    prompt = torch.randn(460, 4)
    sched_req = make_sched_req(prompt=prompt, extend_len=594)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(594, dtype=torch.long))

    with pytest.raises(RuntimeError, match="missing feedback/text input embeds"):
        make_runner().build_prefill_input_embeds(forward_batch, [sched_req])


def test_decode_then_retract_reprefill_roundtrip() -> None:
    """N completed decodes leave G=N+1 tokens: history N plus one queued row."""
    hidden, prompt_len, n_decode = 2, 8, 3
    prompt = torch.arange(prompt_len * hidden, dtype=torch.float32).reshape(
        prompt_len, hidden
    )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decode_feedback_embedding = torch.nn.Embedding(8, hidden)
            self.embed = torch.nn.Embedding(8, hidden)

        def get_input_embeddings(self):
            return self.embed

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.row_ids_cache = None
    runner.model = Model()
    generated = n_decode + 1
    sched_req = make_sched_req(prompt=prompt, extend_len=prompt_len)
    decode_batch = SimpleNamespace(input_ids=torch.tensor([99], dtype=torch.long))
    for step in range(n_decode):
        sched_req.data.pending_feedback_queue.append(
            torch.full((hidden,), float(step + 1), dtype=torch.float32)
        )
        sched_req.data.pending_text_queue.append(torch.zeros(hidden))
        runner.write_feedback_buffers(decode_batch, [sched_req])

    leftover = torch.full((hidden,), float(generated), dtype=torch.float32)
    sched_req.data.pending_feedback_queue.append(leftover)
    sched_req.data.pending_text_queue.append(torch.zeros(hidden))
    assert len(sched_req.data.decode_input_embeds) == n_decode
    assert len(sched_req.data.pending_feedback_queue) == 1

    sched_req.data.req.prefix_len = 0
    sched_req.data.req.extend_len = prompt_len + generated
    prefill_batch = SimpleNamespace(
        input_ids=torch.zeros(prompt_len + generated, dtype=torch.long)
    )
    out = runner.build_prefill_input_embeds(prefill_batch, [sched_req])

    assert out.shape[0] == prompt_len + generated
    assert torch.equal(out[:prompt_len], prompt)
    assert torch.equal(
        out[prompt_len:-1],
        torch.stack([torch.full((hidden,), float(i + 1)) for i in range(n_decode)]),
    )
    assert torch.equal(out[-1], leftover)
    assert list(sched_req.data.pending_feedback_queue) == []
    assert len(sched_req.data.decode_input_embeds) == generated


def test_fresh_prefill_still_uses_prompt_only_buffer() -> None:
    prompt = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    sched_req = make_sched_req(prompt=prompt, prefix_len=1, extend_len=4)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(4, dtype=torch.long))

    out = make_runner().build_prefill_input_embeds(forward_batch, [sched_req])

    assert torch.equal(out, prompt[1:5])


@dataclass
class RadixChunkState:
    page_size: int
    allocator: TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    cache: RadixCache
    scheduler: OmniScheduler


@pytest.fixture(params=[1, 16])
def radix_chunk_state(request: pytest.FixtureRequest) -> Iterator[RadixChunkState]:
    page_size = request.param
    with get_context().override_server_args(page_size=page_size):
        kv_pool = MHATokenToKVPool(
            size=128,
            page_size=page_size,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        if page_size == 1:
            allocator = TokenToKVPoolAllocator(
                size=128,
                dtype=torch.float16,
                device="cpu",
                kvcache=kv_pool,
                need_sort=False,
            )
        else:
            allocator = PagedTokenToKVPoolAllocator(
                size=128,
                page_size=page_size,
                dtype=torch.float16,
                device="cpu",
                kvcache=kv_pool,
                need_sort=False,
            )
        req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=128, device="cpu", enable_memory_saver=False
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
            )
        )
        scheduler = object.__new__(OmniScheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = cache
        scheduler.req_to_token_pool = req_to_token_pool
        yield RadixChunkState(page_size, allocator, req_to_token_pool, cache, scheduler)


def prompt_only_req(state: RadixChunkState, prompt_len: int) -> Req:
    req = Req("chunk", "", list(range(prompt_len)), SamplingParams(max_new_tokens=40))
    req._omni_prompt_only_radix = True  # noqa: leading-underscore  # production name
    OmniScheduler.normalize_req_token_arrays(req)
    state.req_to_token_pool.alloc([req])
    req.init_next_round_input(state.cache)
    return req


def write_slots(state: RadixChunkState, req: Req, start: int, end: int) -> None:
    slots = state.allocator.alloc(
        -(-(end - start) // state.page_size) * state.page_size
    )
    state.req_to_token_pool.write(
        (req.kv.req_pool_idx, slice(start, end)), slots[: end - start]
    )


def compute_chunk(state: RadixChunkState, req: Req, start: int, end: int) -> None:
    req.prefix_len = start
    req.extend_end = end
    req.kv.kv_committed_len = req.kv.kv_allocated_len = end


def admit_next_chunk(state: RadixChunkState, req: Req) -> None:
    adder = PrefillAdder(
        page_size=state.page_size,
        tree_cache=state.cache,
        token_to_kv_pool_allocator=state.allocator,
        running_batch=ScheduleBatch(reqs=[], batch_is_full=False),
        new_token_ratio=1.0,
        rem_input_tokens=64,
        rem_chunk_tokens=32,
    )
    adder.add_chunked_req(req)


def assert_every_slot_returns_once(state: RadixChunkState, total: int) -> None:
    assert state.allocator.available_size() + state.cache.evictable_size() == total
    state.cache.evict(EvictParams(num_tokens=total))
    slots = state.allocator.alloc(total)
    assert slots.unique().numel() == total


@pytest.mark.parametrize("release_at_middle", [False, True])
def test_a_retracted_reprefill_advances_past_its_private_chunk(
    radix_chunk_state: RadixChunkState, release_at_middle: bool
) -> None:
    """A retracted re-prefill's next chunk starts after its private chunk, which stays out of the radix tree and frees once."""
    state = radix_chunk_state
    total = state.allocator.available_size()
    req = prompt_only_req(state, prompt_len=20)
    write_slots(state, req, 0, 16)
    compute_chunk(state, req, 0, 16)
    checkpoint_kv_cache(req, state.cache)
    req.output_ids.extend(range(30, 51))
    req.init_next_round_input()
    req.skip_radix_cache_insert = True
    write_slots(state, req, 16, 33 if release_at_middle else 41)
    compute_chunk(state, req, 16, 33)

    state.scheduler.stash_chunked_request(req)
    admit_next_chunk(state, req)

    assert (req.prefix_len, req.extend_end) == (33, 41)
    assert req.kv.cache_protected_len == 16
    if release_at_middle:
        pass
    else:
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 41
    release_kv_cache(req, state.cache, checkpoint=True)
    match = state.cache.match_prefix(
        MatchPrefixParams(key=RadixKey(req.origin_input_ids + req.output_ids))
    )
    assert match.device_prefix_len == 16
    assert_every_slot_returns_once(state, total)


def test_a_fresh_prompt_chunk_enters_the_radix_tree(
    radix_chunk_state: RadixChunkState,
) -> None:
    state = radix_chunk_state
    total = state.allocator.available_size()
    req = prompt_only_req(state, prompt_len=41)
    write_slots(state, req, 0, 32)
    compute_chunk(state, req, 0, 32)

    state.scheduler.stash_chunked_request(req)
    admit_next_chunk(state, req)

    assert (req.prefix_len, req.extend_end) == (32, 41)
    assert state.cache.total_size() == 32
    release_kv_cache(req, state.cache, checkpoint=True)
    assert_every_slot_returns_once(state, total)
