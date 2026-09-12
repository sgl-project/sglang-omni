# SPDX-License-Identifier: Apache-2.0
"""Retained output history survives the production forward sampling boundary."""

import contextlib
from types import SimpleNamespace

import torch
from sglang.srt.sampling.penaltylib import (
    BatchedFrequencyPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.base import ModelRunner


class _Batch(SimpleNamespace):
    # SimpleNamespace instances cannot be weakly referenced; the orchestrator holds a weakref to its batch.
    pass


def _sampling(batch):
    n = len(batch.reqs)
    return SamplingBatchInfo(
        temperatures=torch.ones(n, 1),
        top_ps=torch.ones(n),
        top_ks=torch.ones(n, dtype=torch.int32),
        min_ps=torch.zeros(n),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=8,
        device="cpu",
        penalizer_orchestrator=BatchedPenalizerOrchestrator(
            vocab_size=8,
            batch=batch,
            penalizers={
                BatchedRepetitionPenalizer,
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
            },
        ),
    )


def _batch(rows):
    batch = _Batch(reqs=[], device="cpu")
    batch.forward_mode = SimpleNamespace(is_extend=lambda: True)
    for history, rp, freq, presence in rows:
        batch.reqs.append(
            SimpleNamespace(
                output_ids=list(history),
                sampling_params=SamplingParams(
                    repetition_penalty=rp,
                    frequency_penalty=freq,
                    presence_penalty=presence,
                ),
            )
        )
    batch.sampling_info = _sampling(batch)
    return batch


def _forward(batch):
    def forward_context(batch, *, isolate_sampling):
        assert isolate_sampling
        return contextlib.nullcontext(batch.sampling_info.copy_for_forward())

    runner = object.__new__(ModelRunner)
    runner._execution_bridge = SimpleNamespace(forward_context=forward_context)
    with runner._execution_context(batch, isolate_sampling=True) as snapshot:
        return snapshot


def _logits(snapshot):
    logits = torch.tensor([[2.6, -2, 2.6, -2, 2.6, -2, 2.6, -2]]).repeat(
        len(snapshot), 1
    )
    with torch._dynamo.config.patch(disable=True):
        snapshot.apply_logits_bias(logits)
    return logits


def test_mixed_restore_and_forward_snapshot_isolation():
    batch = _batch(
        [
            ([2, 2, 5, -1, 8], 1.3, 0.25, -0.5),
            ([2, 5, 5], 2.0, 0.5, 0.25),
            ([2, 5], 1.0, 0.0, 0.0),
            ([], 1.1, -0.25, 0.5),
        ]
    )
    snapshot = _forward(batch)
    expected = torch.tensor([[2.6, -2, 2.6, -2, 2.6, -2, 2.6, -2]]).repeat(4, 1)
    expected[0, 2], expected[0, 5] = 2.0, -2.275
    expected[1, 2], expected[1, 5] = 0.925, -6.5
    # Additive first: (-2 - 0.25 + 0.5) * 1.3 = -2.275.
    torch.testing.assert_close(_logits(snapshot), expected)
    torch.testing.assert_close(_logits(_forward(batch)), expected)  # Idempotent.

    batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
        torch.tensor([3, 3, 3, 3])
    )
    changed = expected.clone()
    changed[:, 3] = torch.tensor([-2.275, -5.5, -2.0, -2.475])
    torch.testing.assert_close(_logits(batch.sampling_info.copy_for_forward()), changed)
    torch.testing.assert_close(_logits(snapshot), expected)


def test_chunked_reprefill_rebuild_then_one_committed_decode():
    batch = _batch([([2, 2, 5], 1.3, 0.25, -0.5)])
    for _ in range(3):
        previous = batch.sampling_info
        batch.sampling_info = _sampling(batch)
        assert (
            batch.sampling_info.penalizer_orchestrator
            is not previous.penalizer_orchestrator
        )
        torch.testing.assert_close(
            _logits(_forward(batch))[0, [2, 5]], torch.tensor([2.0, -2.275])
        )
    batch.reqs[0].output_ids.append(2)
    batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(torch.tensor([2]))
    batch.forward_mode.is_extend = lambda: False
    torch.testing.assert_close(
        _logits(_forward(batch))[0, [2, 5]], torch.tensor([2.35 / 1.3, -2.275])
    )


def test_sampling_batch_filter_merge_preserves_row_behavior():
    batch = _batch([([2, 2], 1.3, 0.25, -0.5), ([5], 2.0, 0.5, 0.25)])
    other = _batch([([3, 3, 3], 1.0, 0.0, 0.0), ([5, 5], 1.3, 0.25, -0.5)])
    _forward(batch)
    _forward(other)
    batch.reqs = [batch.reqs[1]]
    batch.sampling_info.filter_batch([1], torch.tensor([1]))
    batch.sampling_info.merge_batch(other.sampling_info)
    batch.reqs.extend(other.reqs)
    assert len(batch.sampling_info) == len(batch.reqs) == 3
    # Consume directly: another restore could conceal broken filter/merge state.
    expected = torch.tensor([[2.6, -2, 2.6, -2, 2.6, -2, 2.6, -2]]).repeat(3, 1)
    expected[0, 5], expected[2, 5] = -5.5, -2.6
    torch.testing.assert_close(
        _logits(batch.sampling_info.copy_for_forward()), expected
    )
