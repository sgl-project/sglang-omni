# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components import sglang_thinker
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)


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


@pytest.mark.parametrize("enabled", [False, True])
def test_audio_prefill_publishes_embeds_to_sglang_runner(enabled: bool) -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.outer_model = SimpleNamespace(enable_deepstack_prefill_graph=enabled)
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
    assert get_omni_prefill_inputs(forward_batch) is None


def test_visual_deepstack_prefill_keeps_model_specific_forward() -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.outer_model = SimpleNamespace(enable_deepstack_prefill_graph=False)
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

    def logits_processor(input_ids, hidden_states, lm_head, forward_batch):
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


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_visual_prefill_forwards_deepstack_inputs(mode: ForwardMode) -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.outer_model = SimpleNamespace(enable_deepstack_prefill_graph=True)
    input_embeds = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    layer_embeddings = [
        torch.arange(1, 5, dtype=torch.float32).reshape(2, 2) + 10 * index
        for index in range(3)
    ]
    visual_pos_masks = torch.tensor([True, False, True])
    runner.inject_multimodal_embeds = lambda forward, schedule: (
        input_embeds,
        layer_embeddings,
        visual_pos_masks,
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.arange(3),
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        rids=["one", "two"],
    )
    assert (
        runner.custom_prefill_forward(
            forward_batch, SimpleNamespace(forward_mode=mode), []
        )
        is None
    )
    assert forward_batch.input_embeds is None
    model_runner = SGLModelRunner.__new__(SGLModelRunner)
    model_runner.dtype = torch.float32
    model_runner.support_pp = False
    model_runner.is_generation = True
    kwargs = model_runner._extend_forward_kwargs(
        forward_batch, None
    )  # noqa: leading-underscore  # native runner contract
    assert kwargs["input_embeds"] is input_embeds
    torch.testing.assert_close(
        kwargs["input_deepstack_embeds"],
        torch.tensor(
            [[1, 2, 11, 12, 21, 22], [0, 0, 0, 0, 0, 0], [3, 4, 13, 14, 23, 24]],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )


def test_opt_in_rejects_native_without_deepstack_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sglang_thinker, "PrefillInputBuffers", SimpleNamespace(__dataclass_fields__={})
    )
    with pytest.raises(RuntimeError, match="native DeepStack replay slot"):
        Qwen3OmniThinkerForCausalLM(
            SimpleNamespace(enable_deepstack_prefill_graph=True)
        )


@pytest.mark.parametrize("enabled,expected", [(False, 0), (True, 2)])
def test_deepstack_capability_uses_decoder_layout(enabled: bool, expected: int) -> None:
    model = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(model)
    model.enable_deepstack_prefill_graph = enabled
    model.model = SimpleNamespace(deepstack_embed_to_decoder_layer=range(2))

    assert model.get_num_deepstack_embeddings() == expected
