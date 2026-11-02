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
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload
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


def bare_thinker(
    *, accept_hidden_layer: int, num_hidden_layers: int
) -> Qwen3OmniThinkerForCausalLM:
    """A thinker of hidden size 2 with nothing set to capture."""
    thinker = object.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(thinker)
    thinker.root_config = SimpleNamespace(
        talker_config=SimpleNamespace(accept_hidden_layer=accept_hidden_layer)
    )
    thinker.config = SimpleNamespace(num_hidden_layers=num_hidden_layers, hidden_size=2)
    thinker.model = SimpleNamespace(layers_to_capture=[])
    return thinker


def capturing_thinker(
    *,
    accept_hidden_layer: int,
    draft_layer_ids: list[int] | None,
    is_draft_first: bool = True,
) -> Qwen3OmniThinkerForCausalLM:
    """An 8-layer thinker; bootstrap sets the draft's capture before the talker's."""
    thinker = bare_thinker(accept_hidden_layer=accept_hidden_layer, num_hidden_layers=8)
    if draft_layer_ids is None:
        thinker.set_talker_layers_to_capture()
    elif is_draft_first:
        thinker.set_dflash_layers_to_capture(draft_layer_ids)
        thinker.set_talker_layers_to_capture()
    else:
        thinker.set_talker_layers_to_capture()
        thinker.set_dflash_layers_to_capture(draft_layer_ids)
    return thinker


def layer_tagged_hidden_states(layer_ids: list[int], row_count: int) -> torch.Tensor:
    """A capture whose every element is 10 * its layer id + its row."""
    rows = torch.arange(row_count, dtype=torch.float32).view(-1, 1, 1)
    layers = torch.tensor(layer_ids, dtype=torch.float32).view(1, -1, 1)
    return (
        (layers * 10 + rows).expand(row_count, len(layer_ids), 2).reshape(row_count, -1)
    )


class HostPromptHiddenStore:
    """The store's contract on the host; the real store's rows are pinned, which needs CUDA."""

    def __init__(self, req_to_token: torch.Tensor) -> None:
        self.req_to_token = req_to_token
        self.staged: list[tuple[torch.Tensor, torch.Tensor]] = []

    def stage(self, slot_ids: torch.Tensor, hidden_states: torch.Tensor) -> None:
        self.staged.append((slot_ids, hidden_states))

    def prompt_slot_ids(
        self, request_pool_index: int, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.req_to_token[request_pool_index, positions]

    def read(self, slot_ids: torch.Tensor) -> torch.Tensor:
        rows_by_slot = {
            int(slot_id): row
            for staged_slot_ids, staged_rows in self.staged
            for slot_id, row in zip(staged_slot_ids, staged_rows)
        }
        return torch.stack([rows_by_slot[int(slot_id)] for slot_id in slot_ids])


@pytest.mark.parametrize("accept_hidden_layer", [3, 5])
def test_a_speculative_target_forward_splits_the_capture_between_talker_and_draft(
    accept_hidden_layer: int,
) -> None:
    """With speech output a draft read the talker's capture in place of its own layers,
    and the speculative prefill never stored the talker's prompt rows. The talker's
    layer may sit between the draft's (3) or be one of them (5)."""
    thinker = capturing_thinker(
        accept_hidden_layer=accept_hidden_layer, draft_layer_ids=[1, 4, 6]
    )
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.outer_model = thinker
    runner.prompt_hidden_store = HostPromptHiddenStore(torch.zeros(1, 8))
    captured_layer_ids = thinker.model.layers_to_capture
    request = multimodal_request([3, 5])
    # The prompt continues in a later chunk, so this chunk saves no prompt slots.
    request.origin_input_ids = list(range(8))
    schedule_batch = SimpleNamespace(
        reqs=[SimpleNamespace(), request],
        prefix_lens=[0, 3],
        extend_lens=[2, 3],
    )
    forward_batch = SimpleNamespace(
        capture_hidden_mode=CaptureHiddenMode.FULL,
        out_cache_loc=torch.tensor([10, 11, 12, 13, 14]),
        out_cache_loc_virtual=None,
    )
    prefill = SimpleNamespace(
        logits_output=SimpleNamespace(
            hidden_states=layer_tagged_hidden_states(captured_layer_ids, 5)
        )
    )
    verify = SimpleNamespace(
        logits_output=SimpleNamespace(
            hidden_states=layer_tagged_hidden_states(captured_layer_ids, 4)
        )
    )

    runner.post_speculative_target_forward(prefill, forward_batch, schedule_batch, [])
    runner.post_speculative_target_forward(verify, forward_batch, None, [])

    (staged,) = runner.prompt_hidden_store.staged
    staged_slot_ids, staged_hidden_states = staged
    assert staged_slot_ids.tolist() == [12, 14]
    assert torch.equal(
        staged_hidden_states,
        layer_tagged_hidden_states([accept_hidden_layer], 5)[[2, 4]],
    )
    assert torch.equal(
        prefill.logits_output.hidden_states, layer_tagged_hidden_states([2, 5, 7], 5)
    )
    assert torch.equal(
        verify.logits_output.hidden_states, layer_tagged_hidden_states([2, 5, 7], 4)
    )


@pytest.mark.parametrize("is_draft_first", [True, False])
def test_a_draft_refuses_a_talker_that_accepts_the_final_layer(
    is_draft_first: bool,
) -> None:
    """A draft's captures replace the final normalized output the talker would read;
    the talker's capture set before the draft's skipped the refusal."""
    with pytest.raises(ValueError, match="final layer"):
        capturing_thinker(
            accept_hidden_layer=8,
            draft_layer_ids=[1, 4, 6],
            is_draft_first=is_draft_first,
        )


@pytest.mark.parametrize(
    ("accept_hidden_layer", "captured_layer_ids"),
    [(24, [2, 24, 45]), (3, [2, 3, 24, 45])],
)
def test_an_eagle3_draft_naming_no_layers_reads_its_defaults_beside_the_talker(
    accept_hidden_layer: int, captured_layer_ids: list[int]
) -> None:
    """On a 48-layer thinker the default EAGLE3 layers include layer 24, the one the
    official talker accepts, so the two readers can share a column."""
    thinker = bare_thinker(
        accept_hidden_layer=accept_hidden_layer, num_hidden_layers=48
    )

    thinker.set_eagle3_layers_to_capture(None)
    thinker.set_talker_layers_to_capture()

    assert thinker.model.layers_to_capture == captured_layer_ids
    capture = layer_tagged_hidden_states(captured_layer_ids, 3)
    assert torch.equal(
        thinker.select_captured_hidden_states(capture, thinker.draft_capture_layer_ids),
        layer_tagged_hidden_states([2, 24, 45], 3),
    )
    assert torch.equal(
        thinker.select_captured_hidden_states(
            capture, thinker.talker_capture_layer_ids
        ),
        layer_tagged_hidden_states([accept_hidden_layer], 3),
    )


@pytest.mark.parametrize("chunk_lengths", [[6], [4, 2]])
def test_a_request_finishing_on_its_speculative_prefill_hands_the_talker_its_rows(
    chunk_lengths: list[int],
) -> None:
    """Under a draft, a request that finishes on its prefill streams its first talker
    chunk after the scheduler releases its KV, and the hand-off read the prompt rows
    through the request row it no longer held. The prompt's last chunk may hold no
    multimodal rows (4, 2) and still has to save the slots of the earlier chunk's."""
    req_to_token = torch.zeros((1, 6), dtype=torch.int32)
    thinker = capturing_thinker(accept_hidden_layer=3, draft_layer_ids=[1, 4, 6])
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.outer_model = thinker
    runner.prompt_hidden_store = HostPromptHiddenStore(req_to_token)
    request = multimodal_request([1, 3])
    request.origin_input_ids = list(range(6))
    request.inflight_middle_chunks = 0
    request.kv = SimpleNamespace(req_pool_idx=0)
    prefix_length = 0
    for chunk_length in chunk_lengths:
        out_cache_loc = torch.arange(10, 10 + chunk_length) + prefix_length
        req_to_token[0, prefix_length : prefix_length + chunk_length] = (
            out_cache_loc.to(torch.int32)
        )
        runner.post_speculative_target_forward(
            SimpleNamespace(
                logits_output=SimpleNamespace(
                    hidden_states=layer_tagged_hidden_states(
                        thinker.model.layers_to_capture, chunk_length
                    )
                )
            ),
            SimpleNamespace(
                capture_hidden_mode=CaptureHiddenMode.FULL,
                out_cache_loc=out_cache_loc,
                out_cache_loc_virtual=None,
            ),
            SimpleNamespace(
                reqs=[request],
                prefix_lens=[prefix_length],
                extend_lens=[chunk_length],
            ),
            [],
        )
        prefix_length += chunk_length
    req_to_token[0] = 0
    request.kv.req_pool_idx = None
    stage_payload = StagePayload(
        request_id="request-a",
        request=OmniRequest(
            inputs=[],
            params={"stream": False},
            metadata={"output_modalities": ["text", "audio"]},
        ),
        data={},
    )

    builder = make_thinker_stream_output_builder(
        speech_enabled=True, prompt_hidden_store=runner.prompt_hidden_store
    )
    (talker_message,) = builder(
        "request-a",
        SimpleNamespace(req=request, stage_payload=stage_payload),
        SimpleNamespace(data=11),
    )

    assert torch.equal(talker_message.data, layer_tagged_hidden_states([3], 6)[[1, 3]])


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
    runner.outer_model = capturing_thinker(accept_hidden_layer=3, draft_layer_ids=None)
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
            store.read(store.prompt_slot_ids(0, torch.tensor([3, 5]))),
            hidden_states[[2, 4]].cpu(),
        )


def test_omni_forward_retains_the_composed_embeds_for_an_eagle_draft() -> None:
    """An EAGLE3 draft extends on the prefill embeddings the runner composed."""
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(init_forward_metadata=lambda batch: None)
        )
    )
    runner.retain_prefill_input_embeds = True
    runner.outer_model = SimpleNamespace(
        model=lambda **kwargs: torch.zeros(1, 2),
        logits_processor=lambda input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states: "logits",
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )
    input_embeds = torch.ones(1, 2)

    runner.forward_with_omni_embeds(forward_batch, input_embeds)

    assert runner.retained_prefill_input_embeds is input_embeds
