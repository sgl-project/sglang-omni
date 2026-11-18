# SPDX-License-Identifier: Apache-2.0
"""Speculative verify applies repetition_penalty exactly as plain decode does."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner import base as model_runner_base
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.speculative.repetition_penalty import (
    apply_verify_repetition_penalty,
    record_committed_tokens,
    verify_penalty_temporary_bytes,
)

# A small vocabulary, so outputs and drafts repeat often.
VOCAB_SIZE = 6
BLOCK_SIZE = 4
OUTPUT_LENGTH = 48


def plain_penalized_logits(
    logits: torch.Tensor, history: list[int], penalty: float
) -> torch.Tensor:
    """SGLang's plain decode penalty: every token already output is scaled once."""
    scales = torch.ones(logits.shape[-1])
    scales[history] = penalty
    return torch.where(logits < 0, logits * scales, logits / scales).to(logits.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_speculative_decode_commits_plain_decode_tokens_and_logits(
    dtype: torch.dtype,
) -> None:
    """Every committed verify row equals plain decode's penalized logits, so the outputs match."""
    penalties = [1.5, 0.6]
    request_slots = torch.tensor([3, 1])
    total_positions = 4 * OUTPUT_LENGTH + BLOCK_SIZE
    position_logits = torch.randn(
        len(penalties),
        total_positions,
        VOCAB_SIZE,
        generator=torch.Generator().manual_seed(0),
    ).to(dtype)
    plain_rows: list[list[torch.Tensor]] = []
    plain_outputs: list[list[int]] = []
    for request, penalty in enumerate(penalties):
        rows: list[torch.Tensor] = []
        outputs: list[int] = []
        for position in range(total_positions):
            rows.append(
                plain_penalized_logits(
                    position_logits[request, position], outputs, penalty
                )
            )
            outputs.append(int(rows[-1].argmax()))
        plain_rows.append(rows)
        plain_outputs.append(outputs)

    draft_random = random.Random(0)
    committed_token_presence = torch.zeros((5, VOCAB_SIZE + 1), dtype=torch.bool)
    # The prefill commits the first output token.
    committed_token_presence[
        request_slots, torch.tensor([outputs[0] for outputs in plain_outputs])
    ] = True
    speculative_outputs = [[outputs[0]] for outputs in plain_outputs]
    while min(len(outputs) for outputs in speculative_outputs) < OUTPUT_LENGTH:
        verify_input_ids = torch.tensor(
            [
                [outputs[-1]]
                + [
                    (
                        plain_outputs[request][len(outputs) + draft]
                        if draft_random.random() < 0.6
                        else draft_random.randrange(VOCAB_SIZE)
                    )
                    for draft in range(BLOCK_SIZE - 1)
                ]
                for request, outputs in enumerate(speculative_outputs)
            ]
        )
        verify_logits = torch.stack(
            [
                position_logits[request, len(outputs) : len(outputs) + BLOCK_SIZE]
                for request, outputs in enumerate(speculative_outputs)
            ]
        )
        apply_verify_repetition_penalty(
            verify_logits,
            verify_input_ids,
            committed_token_presence[request_slots, :-1],
            torch.tensor(penalties),
        )
        target_ids = verify_logits.argmax(dim=-1)
        committed_ids = torch.zeros((len(penalties), BLOCK_SIZE), dtype=torch.int64)
        committed_lengths = []
        for request, outputs in enumerate(speculative_outputs):
            accepted = 0
            while (
                accepted < BLOCK_SIZE - 1
                and verify_input_ids[request, accepted + 1]
                == target_ids[request, accepted]
            ):
                accepted += 1
            for row in range(accepted + 1):
                assert torch.equal(
                    verify_logits[request, row],
                    plain_rows[request][len(outputs) + row],
                ), (request, len(outputs), row)
            committed = verify_input_ids[request, 1 : accepted + 1].tolist() + [
                int(target_ids[request, accepted])
            ]
            committed_ids[request, : len(committed)] = torch.tensor(committed)
            committed_lengths.append(len(committed))
            outputs.extend(committed)
        record_committed_tokens(
            committed_token_presence,
            request_slots,
            committed_ids,
            torch.tensor(committed_lengths),
        )

    for speculative, plain in zip(speculative_outputs, plain_outputs):
        assert speculative[:OUTPUT_LENGTH] == plain[:OUTPUT_LENGTH]


def penalized_request(
    repetition_penalty: float, output_ids: list[int]
) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SimpleNamespace(repetition_penalty=repetition_penalty),
        output_ids=output_ids,
    )


@pytest.mark.parametrize("chunk_bytes", [64 << 20, 1])
def test_runner_history_follows_each_request_through_its_slot_and_committed_prefix(
    monkeypatch: pytest.MonkeyPatch, chunk_bytes: int
) -> None:
    """A slot's new request starts from its own retained output, only a block's committed prefix counts, and rows without a penalty stay as they are, whether adjacent penalized rows share a chunk or not."""
    monkeypatch.setattr(
        model_runner_base, "SPECULATIVE_PENALTY_CHUNK_BYTES", chunk_bytes
    )
    vocab_size = 8
    runner = object.__new__(ModelRunner)
    runner.speculative_repetition_penalty_owner = "verify_logits"
    runner.device = torch.device("cpu")
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            # SGLang hands out slots 1 to size and keeps row 0 for padded graph batches.
            req_to_token_pool=SimpleNamespace(size=3, req_to_token=torch.zeros((4, 1)))
        ),
        model_config=SimpleNamespace(vocab_size=vocab_size),
    )
    previous = penalized_request(2.0, [])
    runner.record_speculative_output(
        SimpleNamespace(reqs=[previous], req_pool_indices=torch.tensor([2])),
        SimpleNamespace(next_token_ids=torch.tensor([5])),
        is_prefill=True,
    )
    runner.record_speculative_output(
        SimpleNamespace(reqs=[previous], req_pool_indices=torch.tensor([2])),
        SimpleNamespace(
            next_token_ids=torch.tensor([6, 7, 1]), accept_lens=torch.tensor([3])
        ),
        is_prefill=False,
    )

    retracted = penalized_request(2.0, [3])
    unpenalized = penalized_request(1.0, [])
    fresh = penalized_request(0.5, [])
    batch = SimpleNamespace(
        reqs=[retracted, fresh, unpenalized], req_pool_indices=torch.tensor([2, 3, 1])
    )
    runner.record_speculative_output(
        batch, SimpleNamespace(next_token_ids=torch.tensor([4, 1, 6])), is_prefill=True
    )
    runner.record_speculative_output(
        batch,
        SimpleNamespace(
            next_token_ids=torch.tensor([1, 2, 7, 3, 0, 0, 5, 5, 5]),
            accept_lens=torch.tensor([2, 1, 3], dtype=torch.int32),
        ),
        is_prefill=False,
    )
    verify_input_ids = torch.tensor([[2, 7, 0], [3, 3, 5], [5, 6, 6]])
    original = torch.randn(3, 3, vocab_size, generator=torch.Generator().manual_seed(1))
    verify_logits = original.clone()
    runner.apply_speculative_repetition_penalty(
        verify_logits,
        SimpleNamespace(
            input_ids=verify_input_ids.reshape(-1),
            req_pool_indices=batch.req_pool_indices,
        ),
        [
            SchedulerRequest(request_id=str(index), data=SimpleNamespace(req=request))
            for index, request in enumerate(batch.reqs)
        ],
    )

    expected_histories = {
        0: ([3, 4, 1, 2], [3, 4, 1, 2, 7], [3, 4, 1, 2, 7, 0]),
        1: ([1, 3], [1, 3], [1, 3, 5]),
    }
    for request, histories in expected_histories.items():
        penalty = batch.reqs[request].sampling_params.repetition_penalty
        for row, history in enumerate(histories):
            assert torch.equal(
                verify_logits[request, row],
                plain_penalized_logits(original[request, row], history, penalty),
            ), (request, row)
    assert torch.equal(verify_logits[2], original[2])


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures CUDA allocations")
def test_verify_penalty_stays_within_its_temporary_byte_budget() -> None:
    """Chunk sizing trusts verify_penalty_temporary_bytes, so an extra full-size intermediate must break this bound."""
    request_count, block_size, vocab_size = 4, 8, 151_936
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    verify_logits = torch.randn(
        request_count, block_size, vocab_size, device=device, generator=generator
    )
    verify_input_ids = torch.randint(
        vocab_size, (request_count, block_size), device=device, generator=generator
    )
    committed_token_presence = (
        torch.rand(request_count, vocab_size, device=device, generator=generator) < 0.01
    )
    repetition_penalties = torch.full((request_count,), 1.2, device=device)
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    apply_verify_repetition_penalty(
        verify_logits, verify_input_ids, committed_token_presence, repetition_penalties
    )

    torch.cuda.synchronize()
    temporary_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    # note (wilsonzheng0327): the draft-prefix step adds block-by-block tensors, far below 1 MiB.
    assert temporary_bytes <= request_count * verify_penalty_temporary_bytes(
        block_size, vocab_size
    ) + (1 << 20), temporary_bytes
