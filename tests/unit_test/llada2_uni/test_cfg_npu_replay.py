# SPDX-License-Identifier: Apache-2.0
"""CFG decode-graph replay and page-aligned padding on Ascend NPU."""

from __future__ import annotations

import threading
from array import array
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.llada2_uni.cfg_attention_ascend import (
    CompletedUpdateFuture,
    InlineUpdateExecutor,
    LLaDA2CFGAscendAttnBackend,
    install_cfg_npu_replay_hooks,
)
from sglang_omni.models.llada2_uni.request_builders import LLaDA2UniRequest
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


class RecordingExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class FakeBatch:
    def __init__(
        self,
        seq_lens: list[int],
        left_pad_lens: list[int] | None,
        needs_init: bool = False,
    ) -> None:
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.dllm_left_pad_lens_cpu = left_pad_lens
        self.needs_init = needs_init
        self.replay_batch: object = None
        self.untrimmed: list[int] | None = None

    def needs_forward_metadata_init(self) -> bool:
        return self.needs_init


class FakeGraphRunner:
    """Stands in for the NPU decode graph runner."""

    def __init__(self, attn_backend: LLaDA2CFGAscendAttnBackend) -> None:
        self.backend = SimpleNamespace(_update_executor=RecordingExecutor())
        self.attn_backend = attn_backend
        self.seen: list[list[int]] = []

    def execute(self, forward_batch, pp_proxy_tensors=None):
        self.seen.append(forward_batch.seq_lens.tolist())
        self.seen_batch = forward_batch
        self.seen_replay_batch = self.attn_backend.cfg_replay_batch
        self.seen_untrimmed = self.attn_backend.cfg_replay_untrimmed_seq_lens_cpu
        return "graph-out"

    def _replay_attn_backend(  # noqa: leading-underscore  # sglang private API
        self,
    ) -> LLaDA2CFGAscendAttnBackend:
        return self.attn_backend


def make_attn_backend() -> LLaDA2CFGAscendAttnBackend:
    backend = object.__new__(LLaDA2CFGAscendAttnBackend)
    backend.cfg_replay_batch = None
    backend.cfg_replay_untrimmed_seq_lens_cpu = None
    return backend


def test_inline_update_runs_before_replay_without_a_thread() -> None:
    events: list[str] = []
    executor = InlineUpdateExecutor()
    threads_before = threading.active_count()

    future = executor.submit(events.append, "update")
    events.append("replay")

    assert events == ["update", "replay"]
    assert isinstance(future, CompletedUpdateFuture)
    assert future.result() is None
    assert threading.active_count() == threads_before


def test_inline_update_propagates_update_failure() -> None:
    def fail() -> None:
        raise RuntimeError("graph update failed")

    with pytest.raises(RuntimeError, match="graph update failed"):
        InlineUpdateExecutor().submit(fail)


def test_replay_hook_trims_the_pad_and_restores_it() -> None:
    attn_backend = make_attn_backend()
    runner = FakeGraphRunner(attn_backend)
    old_executor = runner.backend._update_executor  # noqa: leading-underscore
    model_runner = SimpleNamespace(decode_cuda_graph_runner=runner)

    install_cfg_npu_replay_hooks(model_runner)

    assert runner.llada2_cfg_replay_hook is True
    assert isinstance(
        runner.backend._update_executor,  # noqa: leading-underscore
        InlineUpdateExecutor,
    )
    assert old_executor.shutdown_calls == [(False, True)]

    batch = FakeBatch(seq_lens=[10, 9], left_pad_lens=[4, 0])
    assert runner.execute(batch) == "graph-out"

    # The captured graph only sees the real tokens; the live batch is untouched.
    assert runner.seen == [[6, 9]]
    assert batch.seq_lens.tolist() == [10, 9]
    assert runner.seen_replay_batch is batch
    assert attn_backend.cfg_replay_batch is None


def test_replay_hook_snapshots_untrimmed_lengths_only_when_replanning() -> None:
    attn_backend = make_attn_backend()
    runner = FakeGraphRunner(attn_backend)
    install_cfg_npu_replay_hooks(SimpleNamespace(decode_cuda_graph_runner=runner))

    replan = FakeBatch(seq_lens=[10, 9], left_pad_lens=[4, 0], needs_init=True)
    runner.execute(replan)
    assert runner.seen == [[6, 9]]
    assert runner.seen_untrimmed == [10, 9]
    assert attn_backend.cfg_replay_untrimmed_seq_lens_cpu is None

    fast_path = FakeBatch(seq_lens=[10, 9], left_pad_lens=[4, 0])
    runner.execute(fast_path)
    assert runner.seen == [[6, 9], [6, 9]]
    assert runner.seen_untrimmed is None


def test_replay_hook_installs_once() -> None:
    attn_backend = make_attn_backend()
    runner = FakeGraphRunner(attn_backend)
    model_runner = SimpleNamespace(decode_cuda_graph_runner=runner)

    install_cfg_npu_replay_hooks(model_runner)
    install_cfg_npu_replay_hooks(model_runner)

    batch = FakeBatch(seq_lens=[10, 9], left_pad_lens=[4, 0])
    runner.execute(batch)

    # A second wrap would subtract the pad twice and hand the graph [2, 9].
    assert runner.seen == [[6, 9]]
    assert batch.seq_lens.tolist() == [10, 9]


def test_replay_hook_leaves_unpadded_batches_alone() -> None:
    attn_backend = make_attn_backend()
    runner = FakeGraphRunner(attn_backend)
    install_cfg_npu_replay_hooks(SimpleNamespace(decode_cuda_graph_runner=runner))

    no_pad = FakeBatch(seq_lens=[10, 9], left_pad_lens=[0, 0])
    runner.execute(no_pad)
    assert runner.seen == [[10, 9]]
    assert no_pad.seq_lens.tolist() == [10, 9]

    wrong_rows = FakeBatch(seq_lens=[10, 9], left_pad_lens=[4])
    runner.execute(wrong_rows)
    assert runner.seen == [[10, 9], [10, 9]]


def make_cond_request(
    total: int, block_size: int = 4, mask_id: int = 9
) -> LLaDA2UniRequest:
    config = DllmConfig(
        algorithm="LowConfidenceCFG",
        algorithm_config={"threshold": 0.2},
        block_size=block_size,
        mask_id=mask_id,
        max_running_requests=1,
        first_done_first_out_mode=False,
    )
    return LLaDA2UniRequest(
        rid="cond",
        origin_input_text="",
        origin_input_ids=array("q", [1] * total),
        sampling_params=SamplingParams(max_new_tokens=total, temperature=0.0),
        vocab_size=32,
        eos_token_ids=[0],
        dllm_config=config,
    )


def make_scheduler() -> SimpleNamespace:
    return SimpleNamespace(
        waiting_queue=[],
        cond_to_unconds={},
        uncond_to_cond={},
        uncond_rids=set(),
    )


def create_companion(
    monkeypatch: pytest.MonkeyPatch, total: int, left_pad: int, *, npu: bool
) -> tuple[SimpleNamespace, LLaDA2UniRequest]:
    monkeypatch.setattr(current_platform, "is_npu", lambda: npu, raising=False)
    scheduler = make_scheduler()
    cond = make_cond_request(total)
    ids = [9] * left_pad + [5] * (total - left_pad)
    DllmScheduler.create_uncond_companion(scheduler, cond, ids, left_pad, "_uncond", True)
    return scheduler, cond


def test_companion_pad_rounds_up_to_a_whole_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, _ = create_companion(monkeypatch, total=10, left_pad=6, npu=True)

    uncond = scheduler.waiting_queue[0]
    # Six real pad tokens become eight, so the pad ends on a page boundary.
    assert uncond.origin_input_ids.tolist() == [9] * 8 + [5] * 4
    assert uncond._dllm_left_pad_len == 8  # noqa: leading-underscore  # DLLM protocol
    assert uncond.rid == "cond_uncond"


def test_companion_pad_keeps_page_aligned_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, _ = create_companion(monkeypatch, total=10, left_pad=8, npu=True)

    uncond = scheduler.waiting_queue[0]
    assert uncond.origin_input_ids.tolist() == [9] * 8 + [5] * 2
    assert uncond._dllm_left_pad_len == 8  # noqa: leading-underscore  # DLLM protocol


def test_companion_pad_is_not_rounded_off_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, _ = create_companion(monkeypatch, total=10, left_pad=6, npu=False)

    uncond = scheduler.waiting_queue[0]
    assert uncond.origin_input_ids.tolist() == [9] * 6 + [5] * 4
    assert uncond._dllm_left_pad_len == 6  # noqa: leading-underscore  # DLLM protocol
