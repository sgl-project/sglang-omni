# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni thinker requests under a speculative draft's verify."""

from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.qwen3_omni import request_builders
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)
from sglang_omni.scheduling.sglang_backend.prompt_hidden import PromptHiddenRequest
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload


def build_requests(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_speculative: bool,
    multimodal_inputs: list[SimpleNamespace | None],
) -> list[SimpleNamespace]:
    """Run the thinker's request builder over one request per multimodal input."""
    built = iter(multimodal_inputs)
    monkeypatch.setattr(
        request_builders,
        "build_sglang_thinker_request",
        lambda state, **kwargs: SimpleNamespace(
            req=SimpleNamespace(multimodal_inputs=next(built))
        ),
    )
    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=None,
        vocab_size=256,
        capture_prompt_hidden_states=False,
        is_speculative=is_speculative,
    )
    return [
        request_builder(make_qwen_payload(request_id=f"request-{index}")).req
        for index in range(len(multimodal_inputs))
    ]


def test_verify_positions_of_a_mixed_batch_stay_integer(monkeypatch) -> None:
    """compute_mrope_positions floors the positions but can return a float delta, and
    SGLang's verify stacks a multimodal request's delta with a text request's int64
    zero before adding it to the integer positions."""
    multimodal_request, text_request = build_requests(
        monkeypatch,
        is_speculative=True,
        multimodal_inputs=[
            SimpleNamespace(mrope_position_delta=torch.tensor([[-3.0]])),
            None,
        ],
    )
    delta = multimodal_request.multimodal_inputs.mrope_position_delta
    forward_batch = SimpleNamespace(seq_lens=torch.tensor([5, 9]))

    ForwardBatch.compute_spec_mrope_positions(
        forward_batch,
        SimpleNamespace(device="cpu"),
        SimpleNamespace(
            multimodal_inputs=[
                multimodal_request.multimodal_inputs,
                text_request.multimodal_inputs,
            ],
            spec_info=SimpleNamespace(positions=torch.tensor([5, 6, 7, 9, 10, 11])),
        ),
    )

    assert delta.dtype == torch.int64
    assert delta.tolist() == [[-3]]
    assert text_request.multimodal_inputs is None
    assert forward_batch.mrope_positions.dtype == torch.int64
    assert torch.equal(
        forward_batch.mrope_positions,
        torch.tensor([[2, 3, 4, 9, 10, 11]]).repeat(3, 1),
    )


def test_a_thinker_without_a_draft_keeps_the_delta_as_computed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (request,) = build_requests(
        monkeypatch,
        is_speculative=False,
        multimodal_inputs=[
            SimpleNamespace(mrope_position_delta=torch.tensor([[-3.0]])),
        ],
    )

    assert request.multimodal_inputs.mrope_position_delta.dtype == torch.float32


@pytest.mark.parametrize("capture_prompt_hidden_states", [False, True])
def test_retracted_multimodal_request_restores_inputs_and_mrope_positions(
    capture_prompt_hidden_states: bool,
) -> None:
    """A retracted thinker request must restore its cleared multimodal inputs, reset its
    chunk cursor, and rebuild generated M-RoPE positions from the prompt's delta."""
    vocab_size = 64
    hidden_size = 4
    image_token_id = 51
    video_token_id = 52
    audio_token_id = 53
    vision_start_token_id = 54
    audio_start_token_id = 55
    thinker_config = SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        audio_token_id=audio_token_id,
        vision_start_token_id=vision_start_token_id,
        audio_start_token_id=audio_start_token_id,
        position_id_per_seconds=25,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
    )
    runner = object.__new__(Qwen3OmniThinkerModelRunner)
    runner.tp_worker = SimpleNamespace(record_custom_prefill_eager=lambda: None)
    torch.manual_seed(0)
    runner.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    runner.image_token_id = image_token_id
    runner.video_token_id = video_token_id
    runner.audio_token_id = audio_token_id

    audio_embeds = torch.full((2, hidden_size), 99.0)
    audio_state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor(
                [7, audio_token_id, audio_token_id, 8], dtype=torch.long
            )
        },
        thinker_inputs={
            "model_inputs": {
                "audio_embeds": audio_embeds,
                "audio_feature_lengths": torch.tensor([13], dtype=torch.long),
            },
            "media_cache_keys": {"audio": "clip-0"},
        },
    )
    audio_request_data = request_builders.build_sglang_thinker_request(
        audio_state,
        params={"max_new_tokens": 8, "temperature": 0.0},
        tokenizer=None,
        vocab_size=vocab_size,
        request_id="request-audio",
        thinker_config=thinker_config,
        capture_prompt_hidden_states=capture_prompt_hidden_states,
    )
    audio_request = audio_request_data.req
    assert audio_request is not None
    assert (
        isinstance(audio_request, PromptHiddenRequest) is capture_prompt_hidden_states
    )
    audio_request.origin_input_ids = array("q", audio_request.origin_input_ids)
    audio_request.origin_input_ids_unpadded = audio_request.origin_input_ids

    audio_request.inflight_middle_chunks = 1
    first_chunk_forward = SimpleNamespace(
        input_ids=torch.tensor(audio_request.origin_input_ids[:2], dtype=torch.long),
        batch_size=1,
        extend_seq_lens_cpu=[2],
        extend_prefix_lens_cpu=[0],
        input_embeds=None,
        replace_embeds=None,
    )
    first_chunk_schedule = SimpleNamespace(
        reqs=[audio_request], forward_mode=ForwardMode.EXTEND, is_prefill_only=True
    )
    runner.before_prefill(first_chunk_forward, first_chunk_schedule, [audio_request])
    assert audio_request._omni_consumed == {
        "audio": 1
    }  # noqa: leading-underscore  # thinker runner contract

    audio_request.reset_for_retract()
    audio_request.init_next_round_input()
    assert (
        audio_request._omni_consumed
        is None  # noqa: leading-underscore  # thinker runner contract
    )

    full_prefill_forward = SimpleNamespace(
        input_ids=torch.tensor(audio_request.origin_input_ids, dtype=torch.long),
        batch_size=1,
        extend_seq_lens_cpu=[len(audio_request.origin_input_ids)],
        extend_prefix_lens_cpu=[0],
        input_embeds=None,
        replace_embeds=None,
    )
    full_prefill_schedule = SimpleNamespace(
        reqs=[audio_request], forward_mode=ForwardMode.EXTEND, is_prefill_only=False
    )
    runner.before_prefill(full_prefill_forward, full_prefill_schedule, [audio_request])
    initial_prefill_inputs = get_omni_prefill_inputs(full_prefill_forward)
    assert initial_prefill_inputs is not None
    assert audio_request.omni_model_inputs is None

    audio_request.output_ids.extend([11, 12])
    audio_request.reset_for_retract()
    audio_request.init_next_round_input()
    reprefill_token_ids = list(audio_request.full_untruncated_fill_ids)
    reprefill_forward = SimpleNamespace(
        input_ids=torch.tensor(reprefill_token_ids, dtype=torch.long),
        batch_size=1,
        extend_seq_lens_cpu=[len(reprefill_token_ids)],
        extend_prefix_lens_cpu=[0],
        input_embeds=None,
        replace_embeds=None,
    )
    reprefill_schedule = SimpleNamespace(
        reqs=[audio_request], forward_mode=ForwardMode.EXTEND, is_prefill_only=False
    )
    runner.before_prefill(reprefill_forward, reprefill_schedule, [audio_request])
    reprefill_inputs = get_omni_prefill_inputs(reprefill_forward)
    assert reprefill_inputs is not None
    assert torch.equal(
        reprefill_inputs.input_embeds[:4], initial_prefill_inputs.input_embeds
    )

    aiv_prompt_ids = torch.tensor(
        [7, vision_start_token_id, audio_start_token_id]
        + [video_token_id] * 25
        + [audio_token_id] * 2
        + [56, 57],
        dtype=torch.long,
    )
    aiv_state = Qwen3OmniPipelineState(
        prompt={"input_ids": aiv_prompt_ids},
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.ones(25, hidden_size),
                "audio_embeds": torch.ones(2, hidden_size),
                "video_grid_thw": torch.tensor([[1, 10, 10]], dtype=torch.long),
                "video_second_per_grid": torch.tensor([1.0], dtype=torch.float32),
                "audio_feature_lengths": torch.tensor([13], dtype=torch.long),
                "use_audio_in_video": True,
            },
        },
    )
    aiv_request = request_builders.build_sglang_thinker_request(
        aiv_state,
        params={"max_new_tokens": 8, "temperature": 0.0},
        tokenizer=None,
        vocab_size=vocab_size,
        request_id="request-aiv",
        thinker_config=thinker_config,
        capture_prompt_hidden_states=capture_prompt_hidden_states,
    ).req
    assert aiv_request is not None
    aiv_request.origin_input_ids = array("q", aiv_request.origin_input_ids)
    aiv_request.origin_input_ids_unpadded = aiv_request.origin_input_ids
    aiv_request.multimodal_inputs.mrope_position_delta = (
        aiv_request.multimodal_inputs.mrope_position_delta.long()
    )
    prompt_token_count = len(aiv_request.origin_input_ids)

    aiv_request.output_ids.extend([21, 22])
    aiv_request.reset_for_retract()
    aiv_request.init_next_round_input()
    assert aiv_request.multimodal_inputs.mrope_positions.shape == (
        3,
        prompt_token_count + 2,
    )

    aiv_request.output_ids.extend([23, 24])
    aiv_request.reset_for_retract()
    aiv_request.init_next_round_input()
    verify_forward = SimpleNamespace(seq_lens=torch.tensor([prompt_token_count]))
    ForwardBatch.compute_spec_mrope_positions(
        verify_forward,
        SimpleNamespace(device="cpu"),
        SimpleNamespace(
            multimodal_inputs=[aiv_request.multimodal_inputs],
            spec_info=SimpleNamespace(
                positions=torch.arange(prompt_token_count, prompt_token_count + 4)
            ),
        ),
    )
    assert aiv_request.multimodal_inputs.mrope_positions.shape == (
        3,
        prompt_token_count + 4,
    )
    assert torch.equal(
        aiv_request.multimodal_inputs.mrope_positions[:, prompt_token_count:].to(
            torch.int64
        ),
        verify_forward.mrope_positions,
    )
