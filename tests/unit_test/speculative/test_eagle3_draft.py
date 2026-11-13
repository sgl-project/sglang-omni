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
    (
        "speculative_algorithm",
        "has_target_embeddings",
        "has_retained_embeddings",
        "expected_source",
    ),
    [
        ("EAGLE3", False, True, "retained"),
        ("DFLASH", False, False, None),
        ("EAGLE3", True, False, "target"),
        ("NEXTN", True, False, "target"),
        ("EAGLE3", True, True, "retained"),
    ],
)
def test_target_prefill_preserves_and_publishes_multimodal_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    speculative_algorithm: str,
    has_target_embeddings: bool,
    has_retained_embeddings: bool,
    expected_source: str | None,
) -> None:
    """Target embeddings survive empty sidecars, and retained values are consumed once."""
    monkeypatch.setattr(
        target_worker.ForwardBatch,
        "init_new",
        lambda batch, model_runner, **kwargs: SimpleNamespace(
            seq_lens=torch.tensor([3]), input_ids=torch.tensor([1, 2, 3])
        ),
    )
    target_multimodal_embeddings = torch.ones(3, 2)
    retained_multimodal_embeddings = torch.full((3, 2), 2.0)
    second_request_multimodal_embeddings = torch.full((3, 2), 3.0)

    class Runner:
        retain_prefill_input_embeds: bool = False
        retained_prefill_input_embeds: torch.Tensor | None = None
        target_multimodal_embeddings: torch.Tensor | None
        next_retained_multimodal_embeddings: torch.Tensor | None
        tp_worker: SimpleNamespace

        def __init__(self) -> None:
            self.target_multimodal_embeddings = (
                target_multimodal_embeddings if has_target_embeddings else None
            )
            self.next_retained_multimodal_embeddings = (
                retained_multimodal_embeddings if has_retained_embeddings else None
            )

        def prepare_and_forward(
            self,
            forward_batch: SimpleNamespace,
            batch: SimpleNamespace,
            requests: list[SimpleNamespace],
            is_prefill: bool,
        ) -> SimpleNamespace:
            if self.retain_prefill_input_embeds:
                if self.next_retained_multimodal_embeddings is not None:
                    self.retained_prefill_input_embeds = (
                        self.next_retained_multimodal_embeddings
                    )
                    self.next_retained_multimodal_embeddings = None
                else:
                    pass
            else:
                pass
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    mm_input_embeds=self.target_multimodal_embeddings
                ),
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
    should_retain_prefill_embeddings = speculative_algorithm in ("EAGLE3", "NEXTN")
    assert target.publishes_prefill_input_embeds is should_retain_prefill_embeddings
    assert runner.retain_prefill_input_embeds is should_retain_prefill_embeddings
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="req-1", omni_data=None)], is_prefill_only=False
    )

    result = target.forward_batch_generation(batch)

    expected_embeddings = {
        "retained": retained_multimodal_embeddings,
        "target": target_multimodal_embeddings,
        None: None,
    }[expected_source]
    assert result.logits_output.mm_input_embeds is expected_embeddings
    if expected_embeddings is not None:
        assert result.logits_output.mm_input_embeds.shape == expected_embeddings.shape
        assert torch.equal(result.logits_output.mm_input_embeds, expected_embeddings)
    else:
        pass
    assert runner.retained_prefill_input_embeds is None

    runner.target_multimodal_embeddings = second_request_multimodal_embeddings
    batch.reqs[0].rid = "req-2"
    second_result = target.forward_batch_generation(batch)

    assert (
        second_result.logits_output.mm_input_embeds
        is second_request_multimodal_embeddings
    )
    assert torch.equal(
        second_result.logits_output.mm_input_embeds,
        second_request_multimodal_embeddings,
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
