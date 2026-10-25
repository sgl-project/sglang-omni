# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.scheduling.sglang_backend.prompt_hidden import PromptHiddenRequest
from tests.unit_test.fixtures.prompt_hidden import cuda_prompt_hidden_store


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
        input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states
    ):
        seen.append(get_forward_context().attn_backend)
        assert input_ids is forward_batch.input_ids
        assert lm_head == "lm_head"
        assert aux_hidden_states is None
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


def test_a_capturing_omni_forward_hands_the_captured_layer_to_the_logits():
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )
    captured = [torch.full((1, 2), 7.0)]
    runner.outer_model = SimpleNamespace(
        model=lambda **kwargs: (torch.ones(1, 2), captured),
        logits_processor=lambda input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states: aux_hidden_states,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    result = runner.forward_with_omni_embeds(forward_batch, torch.ones(1, 2))

    assert result.logits_output is captured


def multimodal_request(positions: list[int]) -> PromptHiddenRequest:
    request = object.__new__(PromptHiddenRequest)
    request.prompt_hidden_positions = torch.tensor(positions)
    return request


@pytest.mark.parametrize(
    ("prefix_length", "extend_length", "expected_mode"),
    [
        (0, 3, CaptureHiddenMode.NULL),
        (2, 2, CaptureHiddenMode.FULL),
        (4, 3, CaptureHiddenMode.NULL),
    ],
)
def test_only_a_prefill_chunk_computing_multimodal_rows_captures_hidden_states(
    prefix_length: int, extend_length: int, expected_mode: CaptureHiddenMode
) -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    schedule_batch = SimpleNamespace(
        reqs=[SimpleNamespace(), multimodal_request([3])],
        prefix_lens=[0, prefix_length],
        extend_lens=[4, extend_length],
    )

    assert (
        runner.requested_capture_hidden_mode_prefill(schedule_batch, [])
        == expected_mode
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host rows need CUDA")
def test_a_capturing_prefill_stores_each_multimodal_row_at_its_kv_slot() -> None:
    req_to_token = torch.zeros((1, 6), dtype=torch.int32, device="cuda")
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    schedule_batch = SimpleNamespace(
        reqs=[SimpleNamespace(), multimodal_request([3, 5])],
        prefix_lens=[0, 3],
        extend_lens=[2, 3],
    )
    hidden_states = torch.arange(10.0, device="cuda").view(5, 2)
    out_cache_loc = torch.tensor([10, 11, 12, 13, 14], device="cuda")

    with cuda_prompt_hidden_store(slot_count=16, req_to_token=req_to_token) as store:
        runner.prompt_hidden_store = store
        runner.post_prefill(
            SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden_states)),
            SimpleNamespace(
                capture_hidden_mode=CaptureHiddenMode.FULL,
                out_cache_loc=out_cache_loc,
                out_cache_loc_virtual=None,
            ),
            schedule_batch,
            [],
        )
        req_to_token[0, 3:6] = out_cache_loc[2:].to(torch.int32)

        assert torch.equal(
            store.read(0, torch.tensor([3, 5])), hidden_states[[2, 4]].cpu()
        )
