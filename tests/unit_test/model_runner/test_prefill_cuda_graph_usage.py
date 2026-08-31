# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

from sglang_omni.model_runner.model_worker import ModelWorker, _PrefillCudaGraphUsage


def _forward_batch(
    num_tokens: int,
    *,
    forward_mode: ForwardMode = ForwardMode.EXTEND,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        forward_mode=forward_mode,
    )


def test_prefill_cuda_graph_usage_instances_do_not_share_buckets() -> None:
    first = _PrefillCudaGraphUsage()
    second = _PrefillCudaGraphUsage()

    first.replay_buckets[16] += 1

    assert first.replay_buckets == {16: 1}
    assert second.replay_buckets == {}


def test_model_worker_reports_actual_prefill_graph_replays_by_bucket(
    monkeypatch,
) -> None:
    prefill_runner = object.__new__(PrefillCudaGraphRunner)
    prefill_runner.capture_num_tokens = [16, 32]
    prefill_runner.backend = SimpleNamespace()
    prefill_runner.buffer_registry = SimpleNamespace(
        has_slot=lambda name: name == "input_embeds"
    )
    outcomes = iter(
        [
            SimpleNamespace(
                logits_output="graph-16",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            SimpleNamespace(
                logits_output="eager",
                can_run_graph=False,
                expert_distribution_metrics=None,
            ),
            SimpleNamespace(
                logits_output="graph-32",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            # Decode graphs must not be reported as prefill replays.
            SimpleNamespace(
                logits_output="decode-graph",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            # TARGET_VERIFY is an extend-like mode but belongs to the decode
            # graph runner and must not consume a stale prefill bucket.
            SimpleNamespace(
                logits_output="target-verify-graph",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
        ]
    )

    def forward(*, forward_batch: object) -> object:
        del forward_batch
        return next(outcomes)

    runner = SimpleNamespace(
        forward=forward,
        prefill_cuda_graph_runner=prefill_runner,
    )
    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = None
    worker.model_runner = runner
    worker._prefill_cuda_graph_usage = _PrefillCudaGraphUsage()
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_model",
        lambda: SimpleNamespace(model_path="model", load_format="auto"),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )
    worker.server_args = SimpleNamespace(
        tp_size=1,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(
                backend="breakable",
                bs=[16, 32],
            )
        ),
    )
    worker.tp_rank = 0
    worker.model_arch_override = None
    ModelWorker.forward_batch_generation(worker, _forward_batch(5))
    ModelWorker.forward_batch_generation(worker, _forward_batch(40))
    ModelWorker.forward_batch_generation(worker, _forward_batch(31))
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(1, forward_mode=ForwardMode.DECODE),
    )
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(1, forward_mode=ForwardMode.TARGET_VERIFY),
    )
    ModelWorker.record_custom_prefill_eager(worker)

    stats = ModelWorker.model_info(worker)["prefill_cuda_graph"]

    assert stats["backend"] == "breakable"
    assert stats["capture_num_tokens"] == [16, 32]
    assert stats["runner"] == "PrefillCudaGraphRunner"
    assert stats["backend_runner"] == "SimpleNamespace"
    assert stats["input_embeds_slot"] is True
    assert stats["replay_count"] == 2
    assert stats["standard_eager_count"] == 1
    assert stats["custom_eager_count"] == 1
    assert stats["replay_buckets"] == {"16": 1, "32": 1}
    assert json.loads(json.dumps(stats)) == stats
