# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


def probe_runner(monkeypatch, method: str):
    runner = object.__new__(ThinkerModelRunner)
    runner.text_model = SimpleNamespace(layers_to_capture=[0, 24])
    seen: list[list[int]] = []
    monkeypatch.setattr(
        ModelRunner,
        method,
        lambda self, _sched: seen.append(list(self.text_model.layers_to_capture)),
    )
    return runner, seen


def test_execute_keeps_layers_to_capture_stable(monkeypatch) -> None:
    runner, seen = probe_runner(monkeypatch, "execute")

    runner.execute(SimpleNamespace(requests=[object()]))

    assert seen == [[0, 24]]
    assert runner.text_model.layers_to_capture == [0, 24]


def test_execute_launch_keeps_layers_to_capture_stable(monkeypatch) -> None:
    runner, seen = probe_runner(monkeypatch, "execute_launch")

    runner.execute_launch(SimpleNamespace(requests=[object()]))

    assert seen == [[0, 24]]
    assert runner.text_model.layers_to_capture == [0, 24]


def test_audio_prefill_publishes_embeds_to_sglang_runner() -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    input_embeds = torch.ones(3, 4)
    runner.inject_multimodal_embeds = lambda *args: (input_embeds, None, None)
    forward_batch = SimpleNamespace(input_embeds=None)

    result = runner.custom_prefill_forward(
        forward_batch,
        SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True)),
        [],
    )

    assert result is None
    assert forward_batch.input_embeds is input_embeds


def test_visual_deepstack_prefill_keeps_model_specific_forward() -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    input_embeds = torch.ones(3, 4)
    deepstack_embeds = [torch.ones(1, 4)]
    visual_mask = torch.tensor([False, True, False])
    runner.inject_multimodal_embeds = lambda *args: (
        input_embeds,
        deepstack_embeds,
        visual_mask,
    )
    seen = []
    runner.forward_with_omni_embeds = lambda *args: seen.append(args) or "result"
    forward_batch = SimpleNamespace(input_embeds=None)

    result = runner.custom_prefill_forward(
        forward_batch,
        SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True)),
        [],
    )

    assert result == "result"
    assert forward_batch.input_embeds is None
    assert seen == [
        (forward_batch, input_embeds, deepstack_embeds, visual_mask),
    ]


@pytest.mark.parametrize("capture_auxiliary", [False, True])
def test_custom_omni_forward_publishes_sglang_forward_context(
    capture_auxiliary: bool,
) -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []
    final_hidden_states = torch.ones(3, 2)
    auxiliary_hidden_states = [torch.ones(3, 2), torch.full((3, 2), 2.0)]
    deepstack_embeddings = [torch.tensor([[3.0, 4.0]])]
    visual_mask = torch.tensor([False, True, False])
    mrope_positions = torch.arange(3).repeat(3, 1)

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        assert torch.equal(
            kwargs["input_deepstack_embeds"],
            torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]),
        )
        assert kwargs["positions"] is mrope_positions
        if capture_auxiliary:
            return final_hidden_states, auxiliary_hidden_states
        else:
            return final_hidden_states

    def logits_processor(input_ids, hidden_states, lm_head, forward_batch, auxiliary):
        seen.append(get_forward_context().attn_backend)
        assert input_ids is forward_batch.input_ids
        assert lm_head == "lm_head"
        assert hidden_states is final_hidden_states
        assert auxiliary is (auxiliary_hidden_states if capture_auxiliary else None)
        return "logits"

    runner.outer_model = SimpleNamespace(
        model=model,
        logits_processor=logits_processor,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.arange(3),
        mrope_positions=mrope_positions,
        input_ids=torch.tensor([1, 2, 3]),
    )

    assert not has_forward_context()
    result = runner.forward_with_omni_embeds(
        forward_batch,
        torch.ones(3, 2),
        deepstack_embeddings,
        visual_mask,
    )

    assert result.logits_output == "logits"
    assert not result.can_run_cuda_graph
    assert seen == [attn_backend, attn_backend]
    assert not has_forward_context()
