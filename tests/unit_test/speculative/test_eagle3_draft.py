# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 drafts on targets whose runner composes the multimodal prefill embeddings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.speculative import target_worker
from sglang_omni.speculative.eagle3_draft import OmniLlamaForCausalLMEagle3
from sglang_omni.speculative.target_worker import SpeculativeTargetWorker


@pytest.mark.parametrize(
    ("speculative_algorithm", "is_published"),
    [("EAGLE3", True), ("DFLASH", False)],
)
def test_target_prefill_hands_an_eagle_draft_the_composed_embeds(
    monkeypatch, speculative_algorithm: str, is_published: bool
) -> None:
    monkeypatch.setattr(
        target_worker.ForwardBatch,
        "init_new",
        lambda batch, model_runner, **kwargs: SimpleNamespace(
            seq_lens=torch.tensor([3]), input_ids=torch.tensor([1, 2, 3])
        ),
    )
    composed_embeds = torch.ones(3, 2)

    class Runner:
        retain_prefill_input_embeds = False
        retained_prefill_input_embeds: torch.Tensor | None = None

        def prepare_and_forward(self, forward_batch, batch, requests, is_prefill):
            if self.retain_prefill_input_embeds:
                self.retained_prefill_input_embeds = composed_embeds
            else:
                pass
            return SimpleNamespace(
                logits_output=SimpleNamespace(mm_input_embeds=None),
                next_token_ids=torch.tensor([7]),
            )

    model_worker = SimpleNamespace(
        model_runner=None, model_config=None, device="cpu", random_seed=0
    )
    runner = Runner()
    runner.tp_worker = model_worker
    target = SpeculativeTargetWorker(
        model_worker, SpeculativeAlgorithm.from_string(speculative_algorithm)
    )
    target.attach_runner(runner)
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="req-1", omni_data=None)], is_prefill_only=False
    )

    result = target.forward_batch_generation(batch)

    assert result.logits_output.mm_input_embeds is (
        composed_embeds if is_published else None
    )
    assert runner.retained_prefill_input_embeds is None


@pytest.mark.parametrize("contains_mm_inputs", [False, True])
def test_draft_extend_embeds_each_request_last_token_the_parent_skips(
    monkeypatch, contains_mm_inputs: bool
) -> None:
    """The parent embeds a request's newly sampled last token only for requests
    it sees multimodal items on, and an Omni runner composes those items itself."""
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(
        LlamaForCausalLMEagle3,
        "forward",
        lambda self, input_ids, positions, forward_batch, *args: seen.append(
            forward_batch.mm_input_embeds.clone()
        )
        or "logits",
    )
    draft = OmniLlamaForCausalLMEagle3.__new__(OmniLlamaForCausalLMEagle3)
    torch.nn.Module.__init__(draft)
    draft.model = SimpleNamespace(
        embed_tokens=lambda token_ids: token_ids[:, None].float().expand(-1, 2) * 10
    )
    forward_batch = SimpleNamespace(
        mm_input_embeds=torch.zeros(5, 2),
        forward_mode=SimpleNamespace(
            is_extend=lambda: True, is_draft_extend_v2=lambda: False
        ),
        contains_mm_inputs=lambda: contains_mm_inputs,
        extend_start_loc=torch.tensor([0, 3]),
        extend_seq_lens=torch.tensor([3, 2]),
    )

    result = draft.forward(
        torch.tensor([1, 2, 3, 4, 5]), torch.arange(5), forward_batch
    )

    expected = torch.zeros(5, 2)
    if not contains_mm_inputs:
        expected[2] = 30.0
        expected[4] = 50.0
    else:
        pass
    assert result == "logits"
    assert torch.equal(seen[0], expected)
