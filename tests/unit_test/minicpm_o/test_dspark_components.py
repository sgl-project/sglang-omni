# SPDX-License-Identifier: Apache-2.0
"""Seeded proposer and verifier behavior across the model-forward boundary."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftForwardResult,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

from sglang_omni.models.minicpm_o import dspark_worker
from sglang_omni.models.minicpm_o.dspark_worker import (
    SeededDraftProposer,
    SeededVerifyExecutor,
)
from sglang_omni.sampling.speculative_noise import (
    ACCEPTANCE_NOISE_SEED_SALT,
    DRAFT_NOISE_SEED_SALT,
    RESIDUAL_NOISE_SEED_SALT,
)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA required",
                ),
            ],
        ),
    ]
)
def device(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "cpu":

        def fixed_draws(
            seeds: torch.Tensor,
            positions: torch.Tensor,
            *,
            salt: int,
            column_count: int,
        ) -> torch.Tensor:
            assert seeds.tolist() == [0, 42]
            assert positions.tolist() == [17, 29]
            return torch.full((seeds.numel(), column_count), 0.5)

        monkeypatch.setattr(dspark_worker, "seeded_uniform_noise", fixed_draws)
        monkeypatch.setattr(dspark_worker, "seeded_exponential_noise", fixed_draws)
    else:
        pass
    return request.param


def seeded_verifier(device: str, gamma: int) -> SeededVerifyExecutor:
    verifier = object.__new__(SeededVerifyExecutor)
    verifier.gamma = gamma
    verifier.verify_num_draft_tokens = gamma + 1
    verifier._tp_sync = SpecTpSync(
        SimpleNamespace(world_size=1)
    )  # noqa: leading-underscore  # upstream spelling
    verifier.step_state = (
        torch.tensor([0, 42], device=device),
        torch.tensor([17, 29], device=device),
    )
    return verifier


def test_mixed_greedy_ties_keep_original_logits_argmax(device: str) -> None:
    """Top-k ties must not change a greedy neighbor's accepted tokens or bonus."""
    verifier = seeded_verifier(device, gamma=2)
    candidates = torch.zeros((2, 3), dtype=torch.long, device=device)
    draft = DraftBlockResult(
        draft_tokens=candidates[:, 1:],
        corrected_logits=torch.full((2, 2, 4), 2.0, device=device),
        greedy_mask=torch.tensor([True, False], device=device),
        temperatures=torch.ones(2, device=device),
        greedy_draft=False,
    )
    sampling = SimpleNamespace(
        temperatures=torch.ones((2, 1), device=device),
        top_ks=torch.tensor([1, 2], device=device),
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        is_any_greedy=True,
    )
    draft_input = DFlashDraftInputV2.create_idle_input(device=device)
    draft_input.max_top_k = 2
    accepted = verifier.accept_and_finalize(
        folded_accept=False,
        bs=2,
        verify_ids_2d=candidates,
        target_logits=torch.full((6, 4), 2.0, device=device),
        draft_block=draft,
        sampling_info=sampling,
        draft_input=draft_input,
        layout=None,
        prefix_lens=torch.tensor([17, 29], device=device),
        draft_tokens=draft.draft_tokens,
    )
    assert accepted.correct_len[0].item() == 2
    assert accepted.bonus[0].item() == 0
    assert accepted.out_tokens[0].tolist() == [0, 0, 0]
    assert accepted.new_seq_lens[0].item() == 20


def test_seeded_proposal_and_verification_keep_committed_token_prefix(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gamma, vocab_size = 2, 17
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(2, gamma, vocab_size, generator=generator).to(device)
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=3).to(device)
    with torch.no_grad():
        head.markov_w1.weight.copy_(torch.randn(vocab_size, 3, generator=generator))
        head.markov_w2.weight.copy_(torch.randn(vocab_size, 3, generator=generator))
    sampling = SimpleNamespace(
        temperatures=torch.full((2, 1), 0.7, device=device),
        top_ks=torch.full((2,), vocab_size, device=device),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        is_any_greedy=False,
    )
    verifier = seeded_verifier(device, gamma)
    proposer = object.__new__(SeededDraftProposer)
    proposer.gamma = gamma
    proposer.sample_from_anchor = False
    proposer.step_state = verifier.step_state
    proposer.attach_draft_sampler(None)
    proposer.draft_model = SimpleNamespace(
        embed_tokens=torch.nn.Embedding(vocab_size, 3, device=device),
        markov_head=head,
        compute_base_logits=Mock(return_value=(logits, None)),
        confidence_head=None,
    )
    forward = DraftForwardResult(
        draft_block_ids=torch.tensor([[3, 0, 0], [5, 0, 0]], device=device),
        raw_hidden=torch.empty(4, 3, device=device),
        draft_hidden_3d=torch.empty(2, 2, 3, device=device),
        can_run_graph=True,
    )
    monkeypatch.setattr(proposer, "_run_forward", Mock(return_value=forward))
    monkeypatch.setattr(proposer, "_base_logits_context", nullcontext)
    exponential_draws = Mock(wraps=dspark_worker.seeded_exponential_noise)
    acceptance_draws = Mock(wraps=dspark_worker.seeded_uniform_noise)
    monkeypatch.setattr(dspark_worker, "seeded_exponential_noise", exponential_draws)
    monkeypatch.setattr(dspark_worker, "seeded_uniform_noise", acceptance_draws)
    proposal = proposer.propose(
        batch=None,
        draft_input=None,
        verify_window=None,
        bs=2,
        device=device,
        target_model=None,
        sampling_info=sampling,
    )
    repeated = proposer.propose(
        batch=None,
        draft_input=None,
        verify_window=None,
        bs=2,
        device=device,
        target_model=None,
        sampling_info=sampling,
    )
    assert not proposal.folded and proposal.verify_ids is None
    draft = proposal.draft_block
    torch.testing.assert_close(draft.draft_tokens, repeated.draft_block.draft_tokens)
    torch.testing.assert_close(
        draft.corrected_logits, repeated.draft_block.corrected_logits
    )
    target = torch.cat(
        [
            draft.corrected_logits,
            torch.full((2, 1, vocab_size), -torch.inf, device=device),
        ],
        dim=1,
    )
    target[:, -1, 3] = 0
    rejected_bonus = (draft.draft_tokens[1, 0] + 1) % vocab_size
    target[1, 0].fill_(-torch.inf)
    target[1, 0, rejected_bonus] = 0
    candidates = torch.cat([proposal.draft_block_ids[:, :1], draft.draft_tokens], dim=1)
    accepted = verifier.accept_and_finalize(
        folded_accept=False,
        bs=2,
        verify_ids_2d=candidates,
        target_logits=target.view(-1, vocab_size),
        draft_block=draft,
        sampling_info=sampling,
        draft_input=DFlashDraftInputV2.create_idle_input(device=device),
        layout=None,
        prefix_lens=torch.tensor([17, 29], device=device),
        draft_tokens=draft.draft_tokens,
    )
    assert [call.kwargs["salt"] for call in exponential_draws.call_args_list] == [
        DRAFT_NOISE_SEED_SALT,
        DRAFT_NOISE_SEED_SALT,
        RESIDUAL_NOISE_SEED_SALT,
    ]
    assert acceptance_draws.call_args.kwargs["salt"] == ACCEPTANCE_NOISE_SEED_SALT
    assert accepted.correct_len.tolist() == [2, 0]
    assert accepted.new_seq_lens.tolist() == [20, 30]
    assert accepted.bonus.tolist() == [3, rejected_bonus.item()]
    torch.testing.assert_close(accepted.out_tokens[0, :gamma], draft.draft_tokens[0])
    assert accepted.out_tokens[1, 0] == rejected_bonus
