# SPDX-License-Identifier: Apache-2.0
"""M-RoPE positions of Qwen3-Omni thinker requests, with and without a speculative draft."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.qwen3_omni import request_builders
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload, make_thinker_config


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
        speech_enabled=False,
        is_speculative=is_speculative,
    )
    return [
        request_builder(make_qwen_payload(request_id=f"request-{index}")).req
        for index in range(len(multimodal_inputs))
    ]


def extend_mrope_positions(
    request: request_builders.QwenThinkerRequest, token_count: int
) -> torch.Tensor:
    """The M-RoPE positions SGLang's extend builds for the request's first token_count tokens."""
    forward_batch = SimpleNamespace(seq_lens_cpu=torch.tensor([token_count]))
    ForwardBatch._compute_mrope_positions_extend(  # noqa: leading-underscore  # upstream name
        forward_batch,
        SimpleNamespace(device="cpu"),
        SimpleNamespace(
            multimodal_inputs=[request.multimodal_inputs],
            extend_lens=[token_count],
            prefix_lens=[0],
            reqs=[request],
        ),
    )
    return forward_batch.mrope_positions


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


@pytest.mark.parametrize("is_speculative", [False, True])
def test_a_retracted_audio_in_video_request_reprefills_at_its_original_positions(
    monkeypatch: pytest.MonkeyPatch, is_speculative: bool
) -> None:
    """Re-prefilling a retracted audio-in-video request put its generated tokens one position below where decode had put them."""
    monkeypatch.setattr(
        forward_batch_info,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        ),
    )
    video_token_id = 52
    audio_token_id = 53
    vision_start_token_id = 54
    audio_start_token_id = 55
    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=None,
        vocab_size=64,
        thinker_config=make_thinker_config(
            image_token_id=51,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
            vision_start_token_id=vision_start_token_id,
            audio_start_token_id=audio_start_token_id,
        ),
        capture_prompt_hidden_states=False,
        speech_enabled=False,
        is_speculative=is_speculative,
    )
    prompt_token_ids = (
        [7, vision_start_token_id, audio_start_token_id]
        + [video_token_id] * 25
        + [audio_token_id] * 2
        + [56, 57]
    )
    audio_in_video_state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor(prompt_token_ids)},
        thinker_inputs={
            "model_inputs": {
                "video_grid_thw": torch.tensor([[1, 10, 10]]),
                "video_second_per_grid": torch.tensor([1.0]),
                "audio_feature_lengths": torch.tensor([13]),
                "use_audio_in_video": True,
            }
        },
    )
    request_data = request_builder(make_qwen_payload(audio_in_video_state))
    request = request_data.req
    request.omni_data = request_data
    OmniScheduler.normalize_req_token_arrays(request)
    prompt_token_count = len(prompt_token_ids)
    generated_token_ids = [21, 22, 23, 24]
    prefill_positions = extend_mrope_positions(request, prompt_token_count)
    decode_sequence_lengths = (
        prompt_token_count + 1 + torch.arange(len(generated_token_ids))
    )
    decode_forward = SimpleNamespace(
        seq_lens_cpu=decode_sequence_lengths, seq_lens=decode_sequence_lengths
    )
    ForwardBatch._compute_mrope_positions_decode(  # noqa: leading-underscore  # upstream name
        decode_forward,
        SimpleNamespace(device="cpu"),
        SimpleNamespace(
            multimodal_inputs=[request.multimodal_inputs] * len(generated_token_ids)
        ),
    )

    request.output_ids.extend(generated_token_ids[:2])
    request.reset_for_retract()
    request.init_next_round_input()
    request.output_ids.extend(generated_token_ids[2:])
    request.reset_for_retract()
    request.init_next_round_input()

    assert torch.equal(
        extend_mrope_positions(request, len(request.full_untruncated_fill_ids)),
        torch.cat((prefill_positions, decode_forward.mrope_positions), dim=1),
    )
