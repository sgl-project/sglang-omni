# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.ming_tts.model_runner import (
    MingTTSModelRunner,
    MingTTSTPStepUpdate,
    _MingTTSRequestState,
)


def test_ming_tts_entry_tail_failure_is_published_before_reraise() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._tp_rank = 0
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4))
    )
    published = []

    def fail_tail(*_):
        raise RuntimeError("tail failed")

    runner._run_entry_tail_step = fail_tail
    runner._broadcast_tp_step_update = published.append
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(2, 1, 4))
    )

    with pytest.raises(RuntimeError, match="tail failed"):
        runner._collect_ming_tts_step(
            result,
            forward_batch=None,
            schedule_batch=SimpleNamespace(),
            requests=[SimpleNamespace(), SimpleNamespace()],
        )

    assert len(published) == 1
    assert published[0].tail_failed.tolist() == [1, 1]
    assert torch.count_nonzero(published[0].feedback_embeddings).item() == 0


def test_ming_tts_follower_rejects_tail_failure() -> None:
    runner = object.__new__(MingTTSModelRunner)
    update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )
    update.tail_failed.fill_(1)

    with pytest.raises(RuntimeError, match="acoustic tail failed"):
        runner._apply_follower_step_update(update, [SimpleNamespace()])


def _run_ming_tts_tail_step(
    *,
    stop_prob: float,
    generation_steps: int,
    max_new_tokens: int,
    is_streaming: bool,
) -> tuple[SimpleNamespace, _MingTTSRequestState, MingTTSTPStepUpdate]:
    runner = object.__new__(MingTTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4)),
        run_tail_step=lambda _inputs: SimpleNamespace(
            sampled=torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]),
            feedback_embeddings=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            stop_prob=torch.tensor([stop_prob]),
        ),
    )
    request_state = _MingTTSRequestState(latent_history=torch.zeros(1, 2, 3))
    runner._request_states = {"req-ming-tts": request_state}
    request = SimpleNamespace(
        request_id="req-ming-tts",
        data=SimpleNamespace(
            generation_steps=generation_steps,
            max_new_tokens=max_new_tokens,
            is_streaming=is_streaming,
            pending_stream_patch=None,
            generated_latents=None,
            stop_step=None,
            audio_patch_token_id=7,
            audio_eos_token_id=8,
            state=SimpleNamespace(cfg=2.0, sigma=0.25, temperature=0.0),
        ),
    )
    step_update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )

    runner._run_entry_tail_step(torch.ones(1, 1, 4), [request], step_update)

    return request.data, request_state, step_update


def test_ming_tts_streaming_length_limit_marks_terminal_patch() -> None:
    data, request_state, step_update = _run_ming_tts_tail_step(
        stop_prob=0.0,
        generation_steps=3,
        max_new_tokens=4,
        is_streaming=True,
    )

    assert data.pending_stream_patch is not None
    assert data.pending_stream_patch.is_last is True
    assert torch.equal(
        data.pending_stream_patch.latent,
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
    )
    assert data.stop_step is None
    assert request_state.generated_latents == []
    assert request_state.feedback_embeddings == []
    assert data.generated_latents is None
    assert step_update.next_token_ids.tolist() == [7]


def test_ming_tts_streaming_stop_head_marks_terminal_patch() -> None:
    data, request_state, step_update = _run_ming_tts_tail_step(
        stop_prob=0.9,
        generation_steps=4,
        max_new_tokens=256,
        is_streaming=True,
    )

    assert data.pending_stream_patch is not None
    assert data.pending_stream_patch.is_last is True
    assert data.stop_step == 4
    assert request_state.generated_latents == []
    assert data.generated_latents is None
    assert step_update.next_token_ids.tolist() == [8]


def test_ming_tts_streaming_mid_generation_patch_is_not_terminal() -> None:
    data, request_state, step_update = _run_ming_tts_tail_step(
        stop_prob=0.1,
        generation_steps=4,
        max_new_tokens=256,
        is_streaming=True,
    )

    assert data.pending_stream_patch is not None
    assert data.pending_stream_patch.is_last is False
    assert len(request_state.feedback_embeddings) == 1
    assert torch.equal(
        request_state.latent_history,
        torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]),
    )
    assert step_update.feedback_mask.tolist() == [1]
    assert step_update.feedback_embeddings.tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert step_update.next_token_ids.tolist() == [7]


def test_ming_tts_streaming_stop_head_is_gated_until_step_four() -> None:
    data, _request_state, step_update = _run_ming_tts_tail_step(
        stop_prob=0.9,
        generation_steps=3,
        max_new_tokens=256,
        is_streaming=True,
    )

    assert data.pending_stream_patch is not None
    assert data.pending_stream_patch.is_last is False
    assert step_update.next_token_ids.tolist() == [7]


def test_ming_tts_non_streaming_step_buffers_latents_without_stream_patch() -> None:
    data, request_state, _step_update = _run_ming_tts_tail_step(
        stop_prob=0.9,
        generation_steps=4,
        max_new_tokens=256,
        is_streaming=False,
    )

    assert data.pending_stream_patch is None
    assert len(request_state.generated_latents) == 1
    assert data.stop_step == 4
    assert data.generated_latents.dtype == torch.float32
    assert torch.equal(
        data.generated_latents,
        torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]),
    )


def test_ming_tts_text_prefill_stages_embeddings_without_mutating_admission_fields() -> (
    None
):
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner._tp_rank = 1
    runner._request_states = {}
    embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 11.0],
                [20.0, 21.0],
            ]
        )
    )
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=embedding.weight),
        get_input_embeddings=lambda: embedding,
    )
    request = SimpleNamespace(
        request_id="request-text",
        data=SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            state=SimpleNamespace(spk_emb=None, prompt_latent=None),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=2),
            ),
        ),
    )
    mm_inputs = [object()]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2]),
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=mm_inputs,
    )

    runner.before_prefill(forward_batch, SimpleNamespace(), [request])

    staged = get_omni_prefill_inputs(forward_batch)
    assert staged is not None
    assert torch.equal(
        staged.input_embeds,
        torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
    )
    assert forward_batch.input_embeds is None
    assert forward_batch.replace_embeds is None
    assert forward_batch.mm_inputs is mm_inputs


def test_ming_tts_reference_prefill_delegates_staged_embeddings_to_standard_worker() -> (
    None
):
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner._tp_rank = 0
    runner._request_states = {}
    standard_observations = []

    def standard_forward(forward_batch):
        staged = get_omni_prefill_inputs(forward_batch)
        standard_observations.append(
            (
                staged.input_embeds.clone(),
                forward_batch.input_embeds,
                forward_batch.replace_embeds,
                forward_batch.mm_inputs,
            )
        )
        return SimpleNamespace(
            logits_output="standard-logits",
            next_token_ids=None,
            can_run_cuda_graph=True,
        )

    runner.tp_worker = SimpleNamespace(forward_batch_generation=standard_forward)
    embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ]
        )
    )
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=embedding.weight),
        get_input_embeddings=lambda: embedding,
        spk_head=lambda _speaker: torch.tensor([[90.0, 91.0]]),
        linear_proj_audio=lambda _latent: torch.tensor(
            [[[70.0, 71.0]], [[80.0, 81.0]]]
        ),
        patch_size=1,
        latent_dim=1,
        history_patch_size=2,
    )
    request = SimpleNamespace(
        request_id="request-reference",
        data=SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
            state=SimpleNamespace(
                spk_emb=torch.tensor([[1.0, 2.0]]),
                prompt_latent=torch.tensor([[3.0], [4.0]]),
                spk_injection_positions=[1],
                prompt_latent_start_position=2,
                prompt_latent_token_count=2,
            ),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=5),
            ),
        ),
    )
    mm_inputs = [object()]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3, 4, 5]),
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=mm_inputs,
    )

    result = runner._prepare_and_forward(
        forward_batch,
        SimpleNamespace(is_prefill_only=True),
        [request],
        True,
    )

    assert result.logits_output == "standard-logits"
    assert result.can_run_cuda_graph is True
    assert len(standard_observations) == 1
    staged, input_embeds, replace_embeds, observed_mm_inputs = standard_observations[0]
    assert torch.equal(
        staged,
        torch.tensor(
            [
                [10.0, 11.0],
                [90.0, 91.0],
                [70.0, 71.0],
                [80.0, 81.0],
                [50.0, 51.0],
            ]
        ),
    )
    assert input_embeds is None
    assert replace_embeds is None
    assert observed_mm_inputs is mm_inputs
    assert get_omni_prefill_inputs(forward_batch) is None


def test_ming_tts_prefill_replays_prompt_and_generated_feedback() -> None:
    def fail_token_embedding(_input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("continuous rows must not use token embeddings")

    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty((1, 2), dtype=torch.float32)
        ),
        get_input_embeddings=lambda: fail_token_embedding,
    )
    runner._request_states = {
        "req-ming-tts": _MingTTSRequestState(
            prefill_input_embeds=torch.tensor(
                [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
            ),
            feedback_embeddings=[
                torch.tensor([40.0, 41.0]),
                torch.tensor([50.0, 51.0]),
            ],
        )
    }
    request = SimpleNamespace(
        request_id="req-ming-tts",
        data=SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3]),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=5),
            ),
        ),
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    actual = runner._build_prefill_input_embeds(forward_batch, [request])

    assert torch.equal(
        actual,
        torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ]
        ),
    )
