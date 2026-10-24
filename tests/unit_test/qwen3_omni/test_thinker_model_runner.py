# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

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


def test_custom_omni_forward_publishes_sglang_forward_context():
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        assert kwargs["input_deepstack_embeds"] is None
        return torch.ones(1, 2)

    def logits_processor(
        input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states=None
    ):
        seen.append(get_forward_context().attn_backend)
        assert input_ids is forward_batch.input_ids
        assert lm_head == "lm_head"
        return "logits"

    runner.outer_model = SimpleNamespace(
        model=model,
        logits_processor=logits_processor,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    assert not has_forward_context()
    result = runner.forward_with_omni_embeds(
        forward_batch,
        torch.ones(1, 2),
    )

    assert result.logits_output == "logits"


def test_omni_forward_hands_a_draft_its_captures_and_the_composed_embeds() -> None:
    """A draft reads the hidden layers the model captured and, for EAGLE3, the
    prefill embeddings the runner composed."""
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(init_forward_metadata=lambda batch: None)
        )
    )
    runner.retain_prefill_input_embeds = True
    hidden_states = torch.zeros(1, 2)
    auxiliary_hidden_states = [torch.ones(1, 2), torch.full((1, 2), 2.0)]
    seen: dict[str, torch.Tensor | list[torch.Tensor]] = {}

    def logits_processor(input_ids, hidden, lm_head, forward_batch, auxiliary):
        seen.update(hidden=hidden, auxiliary=auxiliary)
        return "logits"

    runner.outer_model = SimpleNamespace(
        model=lambda **kwargs: (hidden_states, auxiliary_hidden_states),
        logits_processor=logits_processor,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )
    input_embeds = torch.ones(1, 2)

    runner.forward_with_omni_embeds(forward_batch, input_embeds)

    assert seen["hidden"] is hidden_states
    assert seen["auxiliary"] is auxiliary_hidden_states
    assert runner.retained_prefill_input_embeds is input_embeds
