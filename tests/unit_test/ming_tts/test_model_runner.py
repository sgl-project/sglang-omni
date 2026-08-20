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

_PREFILL_PARITY_TOLERANCES = {
    torch.float32: {"max_abs_error": 1e-6, "mean_abs_error": 1e-7},
    torch.bfloat16: {"max_abs_error": 2e-2, "mean_abs_error": 5e-3},
}


def _tensor_error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    error = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }


def _cpu_prefill_graph_replay(*, bucket_size: int) -> SimpleNamespace:
    """Replace only the CUDA replay boundary with a fixed static input slot."""

    static_input_embeds = torch.full((bucket_size, 2), -999.0)

    def execute(forward_batch, *, input_embeds):
        num_tokens = len(forward_batch.input_ids)
        static_input_embeds[:num_tokens].copy_(input_embeds)
        return static_input_embeds[:num_tokens].clone()

    return SimpleNamespace(
        execute=execute,
        static_input_embeds=static_input_embeds,
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


def test_ming_tts_fixed_tail_outputs_preserve_patch_order_and_stop_step() -> None:
    tail_outputs = iter(
        [
            SimpleNamespace(
                sampled=torch.tensor([[[1.0, 1.5], [2.0, 2.5]]]),
                feedback_embeddings=torch.tensor([[10.0, 11.0, 12.0, 13.0]]),
                stop_prob=torch.tensor([0.1]),
            ),
            SimpleNamespace(
                sampled=torch.tensor([[[3.0, 3.5], [4.0, 4.5]]]),
                feedback_embeddings=torch.tensor([[20.0, 21.0, 22.0, 23.0]]),
                stop_prob=torch.tensor([0.9]),
            ),
        ]
    )
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4)),
        run_tail_step=lambda _inputs: next(tail_outputs),
    )
    request_state = _MingTTSRequestState(latent_history=torch.zeros(1, 2, 2))
    runner._request_states = {"ordered-tail": request_state}
    request = SimpleNamespace(
        request_id="ordered-tail",
        data=SimpleNamespace(
            generation_steps=4,
            max_new_tokens=256,
            is_streaming=False,
            pending_stream_patch=None,
            generated_latents=None,
            stop_step=None,
            audio_patch_token_id=7,
            audio_eos_token_id=8,
            state=SimpleNamespace(cfg=2.0, sigma=0.25, temperature=0.0),
        ),
    )

    first_update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )
    runner._run_entry_tail_step(torch.ones(1, 1, 4), [request], first_update)
    request.data.generation_steps = 5
    second_update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )
    runner._run_entry_tail_step(torch.ones(1, 1, 4), [request], second_update)

    assert first_update.next_token_ids.tolist() == [7]
    assert second_update.next_token_ids.tolist() == [8]
    assert request.data.stop_step == 5
    assert len(request_state.feedback_embeddings) == 1
    assert torch.equal(
        request_state.feedback_embeddings[0],
        torch.tensor([10.0, 11.0, 12.0, 13.0]),
    )
    assert torch.equal(
        request.data.generated_latents,
        torch.tensor(
            [
                [[1.0, 1.5], [2.0, 2.5]],
                [[3.0, 3.5], [4.0, 4.5]],
            ]
        ),
    )


def test_ming_tts_post_prefill_collects_the_first_acoustic_tail_step() -> None:
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner._tp_rank = 0
    runner._tp_size = 1
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4)),
        run_tail_step=lambda _inputs: SimpleNamespace(
            sampled=torch.tensor([[[1.0, 1.5], [2.0, 2.5]]]),
            feedback_embeddings=torch.tensor([[10.0, 11.0, 12.0, 13.0]]),
            stop_prob=torch.tensor([0.1]),
        ),
    )
    runner._request_states = {
        "prefill-tail": _MingTTSRequestState(latent_history=torch.zeros(1, 2, 2))
    }
    request = SimpleNamespace(
        request_id="prefill-tail",
        data=SimpleNamespace(
            generation_steps=0,
            max_new_tokens=256,
            is_streaming=True,
            pending_stream_patch=None,
            generated_latents=None,
            stop_step=None,
            audio_patch_token_id=7,
            audio_eos_token_id=8,
            state=SimpleNamespace(cfg=2.0, sigma=0.25, temperature=0.0),
        ),
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(1, 1, 4)),
        next_token_ids=None,
    )

    runner.post_prefill(
        result,
        SimpleNamespace(),
        SimpleNamespace(is_prefill_only=False),
        [request],
    )

    assert result.next_token_ids.tolist() == [7]
    assert request.data.pending_stream_patch.is_last is False
    assert torch.equal(
        request.data.pending_stream_patch.latent,
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
    )
    assert len(runner._request_states["prefill-tail"].feedback_embeddings) == 1


def test_ming_tts_tp_ranks_compose_the_same_logical_prefill_window() -> None:
    embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ]
        )
    )
    model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=embedding.weight),
        get_input_embeddings=lambda: embedding,
        spk_head=lambda _speaker: torch.tensor([[90.0, 91.0]]),
        linear_proj_audio=lambda _latent: torch.tensor([[[70.0, 71.0]]]),
        patch_size=1,
        latent_dim=1,
        history_patch_size=2,
    )
    request = SimpleNamespace(
        request_id="tp-request",
        data=SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3]),
            state=SimpleNamespace(
                spk_emb=torch.tensor([[1.0, 2.0]]),
                prompt_latent=torch.tensor([[3.0]]),
                spk_injection_positions=[1],
                prompt_latent_start_position=2,
                prompt_latent_token_count=1,
            ),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=3),
            ),
        ),
    )
    observed_windows = []
    request_states = []

    for tp_rank in (0, 1):
        runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
        runner._tp_rank = tp_rank
        runner.model = model
        runner._request_states = {}
        forward_batch = SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3]),
            input_embeds=None,
            replace_embeds=None,
        )

        runner.before_prefill(forward_batch, SimpleNamespace(), [request])

        staged = get_omni_prefill_inputs(forward_batch)
        assert staged is not None
        observed_windows.append(staged.input_embeds)
        request_states.append(runner._request_states["tp-request"])

    expected = torch.tensor([[10.0, 11.0], [90.0, 91.0], [70.0, 71.0]])
    assert torch.equal(observed_windows[0], expected)
    assert torch.equal(observed_windows[1], expected)
    assert request_states[0].latent_history is not None
    assert request_states[1].latent_history is None


def test_ming_tts_tp_follower_uses_synchronized_feedback_for_decode() -> None:
    _data, entry_state, step_update = _run_ming_tts_tail_step(
        stop_prob=0.1,
        generation_steps=4,
        max_new_tokens=256,
        is_streaming=True,
    )
    follower = MingTTSModelRunner.__new__(MingTTSModelRunner)
    follower._tp_rank = 1
    follower_state = _MingTTSRequestState(latent_history=None)
    follower._request_states = {"req-ming-tts": follower_state}
    staged_rows = []
    follower.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(2, 4)),
        stage_decode_feedback=lambda rows: (
            staged_rows.append(rows.clone()) or torch.tensor([1])
        ),
    )
    request = SimpleNamespace(request_id="req-ming-tts")

    follower._apply_follower_step_update(step_update, [request])
    forward_batch = SimpleNamespace(input_ids=torch.tensor([99]))
    follower.before_decode(
        forward_batch,
        SimpleNamespace(),
        [request],
    )

    assert len(entry_state.feedback_embeddings) == 1
    assert len(follower_state.feedback_embeddings) == 1
    assert torch.equal(
        follower_state.feedback_embeddings[0],
        entry_state.feedback_embeddings[0],
    )
    assert follower_state.latent_history is None
    assert len(staged_rows) == 1
    assert torch.equal(staged_rows[0], step_update.feedback_embeddings)
    assert forward_batch.input_ids.tolist() == [1]


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


def test_ming_tts_a_b_a_same_bucket_replay_uses_current_feedback_window() -> None:
    graph_runner = _cpu_prefill_graph_replay(bucket_size=4)
    observed_sidecars: list[torch.Tensor] = []

    def standard_forward(forward_batch):
        staged = get_omni_prefill_inputs(forward_batch)
        assert staged is not None
        observed_sidecars.append(staged.input_embeds.clone())
        hidden_states = graph_runner.execute(
            forward_batch,
            input_embeds=staged.input_embeds,
        )
        return SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden_states),
            next_token_ids=None,
            can_run_cuda_graph=True,
        )

    def fail_generated_token_embedding(_input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError(
            "Continuous Feedback Rows must not be reconstructed as token embeddings"
        )

    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.tp_worker = SimpleNamespace(forward_batch_generation=standard_forward)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty((1, 2), dtype=torch.float32)
        ),
        get_input_embeddings=lambda: fail_generated_token_embedding,
    )
    a_prompt = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    b_prompt = torch.tensor([[20.0, 21.0], [22.0, 23.0]])
    a_feedback = [torch.tensor([30.0, 31.0]), torch.tensor([40.0, 41.0])]
    runner._request_states = {
        "request-a": _MingTTSRequestState(prefill_input_embeds=a_prompt.clone()),
        "request-b": _MingTTSRequestState(prefill_input_embeds=b_prompt.clone()),
    }

    def request(request_id: str, *, extend_length: int) -> SimpleNamespace:
        return SimpleNamespace(
            request_id=request_id,
            data=SimpleNamespace(
                input_ids=torch.tensor([1, 2]),
                req=SimpleNamespace(
                    prefix_indices=torch.empty(0, dtype=torch.long),
                    extend_range=SimpleNamespace(length=extend_length),
                ),
            ),
        )

    def run(request_value: SimpleNamespace) -> tuple[torch.Tensor, SimpleNamespace]:
        num_tokens = int(request_value.data.req.extend_range.length)
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=[None],
        )
        result = runner._prepare_and_forward(
            forward_batch,
            SimpleNamespace(is_prefill_only=True),
            [request_value],
            True,
        )
        return result.logits_output.hidden_states, forward_batch

    first_a, first_batch = run(request("request-a", extend_length=2))
    b, b_batch = run(request("request-b", extend_length=2))
    runner._request_states["request-a"].feedback_embeddings.extend(a_feedback)
    current_a_window = torch.cat((a_prompt, torch.stack(a_feedback)), dim=0)
    resumed_a, resumed_batch = run(request("request-a", extend_length=4))

    assert torch.equal(first_a, a_prompt)
    assert torch.equal(b, b_prompt)
    assert torch.equal(resumed_a, current_a_window)
    expected_sidecars = [a_prompt, b_prompt, current_a_window]
    assert len(observed_sidecars) == len(expected_sidecars)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(observed_sidecars, expected_sidecars)
    )
    assert get_omni_prefill_inputs(first_batch) is None
    assert get_omni_prefill_inputs(b_batch) is None
    assert get_omni_prefill_inputs(resumed_batch) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ming_tts_controlled_rng_prefill_hidden_state_parity(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260819)
    prefill_window = torch.randn(4, 3, generator=generator).to(dtype=dtype)
    projection = torch.tensor(
        [[0.5, -0.25], [1.0, 0.75], [-0.5, 0.125]],
        dtype=dtype,
    )
    graph_replay = _cpu_prefill_graph_replay(bucket_size=4)
    graph_replay.static_input_embeds = torch.full((4, 3), -999.0, dtype=dtype)

    def execute(forward_batch, *, input_embeds):
        num_tokens = len(forward_batch.input_ids)
        graph_replay.static_input_embeds[:num_tokens].copy_(input_embeds)
        return graph_replay.static_input_embeds[:num_tokens].clone()

    graph_replay.execute = execute

    def run_candidate(*, use_graph: bool) -> torch.Tensor:
        def standard_forward(forward_batch):
            staged = get_omni_prefill_inputs(forward_batch)
            assert staged is not None
            backbone_inputs = (
                graph_replay.execute(
                    forward_batch,
                    input_embeds=staged.input_embeds,
                )
                if use_graph
                else staged.input_embeds
            )
            hidden_states = backbone_inputs @ projection
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=hidden_states),
                next_token_ids=None,
                can_run_cuda_graph=use_graph,
            )

        runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
        runner.tp_worker = SimpleNamespace(forward_batch_generation=standard_forward)
        runner.model = SimpleNamespace(
            _decode_input_embedding=SimpleNamespace(
                weight=torch.empty((1, 3), dtype=dtype)
            ),
            get_input_embeddings=lambda: lambda _input_ids: pytest.fail(
                "controlled prefill rows must be used"
            ),
        )
        runner._request_states = {
            "parity-request": _MingTTSRequestState(
                prefill_input_embeds=prefill_window.clone()
            )
        }
        request = SimpleNamespace(
            request_id="parity-request",
            data=SimpleNamespace(
                input_ids=torch.arange(4),
                req=SimpleNamespace(
                    prefix_indices=torch.empty(0, dtype=torch.long),
                    extend_range=SimpleNamespace(length=4),
                ),
            ),
        )
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.long),
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=[None],
        )
        result = runner._prepare_and_forward(
            forward_batch,
            SimpleNamespace(is_prefill_only=True),
            [request],
            True,
        )
        assert get_omni_prefill_inputs(forward_batch) is None
        return result.logits_output.hidden_states

    eager_hidden = run_candidate(use_graph=False)
    graph_hidden = run_candidate(use_graph=True)
    error_summary = _tensor_error_summary(graph_hidden, eager_hidden)
    tolerances = _PREFILL_PARITY_TOLERANCES[dtype]

    assert error_summary["max_abs_error"] <= tolerances["max_abs_error"], error_summary
    assert (
        error_summary["mean_abs_error"] <= tolerances["mean_abs_error"]
    ), error_summary


@pytest.mark.parametrize(
    "num_tokens",
    [
        pytest.param(257, id="above-maximum"),
        pytest.param(65, id="two-x-padding"),
    ],
)
def test_ming_tts_standard_eager_fallback_preserves_prefill_semantics(
    num_tokens: int,
) -> None:
    expected_window = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(
        num_tokens,
        2,
    )
    observed: list[torch.Tensor] = []

    def standard_eager_forward(forward_batch):
        staged = get_omni_prefill_inputs(forward_batch)
        assert staged is not None
        observed.append(staged.input_embeds.clone())
        return SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=staged.input_embeds.clone()),
            next_token_ids=None,
            can_run_cuda_graph=False,
        )

    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.tp_worker = SimpleNamespace(forward_batch_generation=standard_eager_forward)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty((1, 2), dtype=torch.float32)
        ),
        get_input_embeddings=lambda: lambda _input_ids: pytest.fail(
            "the materialized request window must be preserved"
        ),
    )
    runner._request_states = {
        "fallback-request": _MingTTSRequestState(
            prefill_input_embeds=expected_window.clone()
        )
    }
    request = SimpleNamespace(
        request_id="fallback-request",
        data=SimpleNamespace(
            input_ids=torch.arange(num_tokens),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=num_tokens),
            ),
        ),
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[None],
    )

    result = runner._prepare_and_forward(
        forward_batch,
        SimpleNamespace(is_prefill_only=True),
        [request],
        True,
    )

    assert result.can_run_cuda_graph is False
    assert torch.equal(result.logits_output.hidden_states, expected_window)
    assert len(observed) == 1
    assert torch.equal(observed[0], expected_window)
    assert get_omni_prefill_inputs(forward_batch) is None


def test_ming_tts_prefill_clears_sidecar_after_standard_forward_error() -> None:
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.tp_worker = SimpleNamespace(
        forward_batch_generation=lambda _forward_batch: (_ for _ in ()).throw(
            ValueError("standard forward failed")
        )
    )
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty((1, 2), dtype=torch.float32)
        ),
        get_input_embeddings=lambda: lambda _input_ids: pytest.fail(
            "stored rows must be used"
        ),
    )
    runner._request_states = {
        "error-request": _MingTTSRequestState(
            prefill_input_embeds=torch.tensor([[1.0, 2.0]])
        )
    }
    request = SimpleNamespace(
        request_id="error-request",
        data=SimpleNamespace(
            input_ids=torch.tensor([1]),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=1),
            ),
        ),
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1]),
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[None],
    )

    with pytest.raises(ValueError, match="standard forward failed"):
        runner._prepare_and_forward(
            forward_batch,
            SimpleNamespace(is_prefill_only=True),
            [request],
            True,
        )

    assert get_omni_prefill_inputs(forward_batch) is None


def test_ming_tts_reprefill_rejects_missing_continuous_feedback_rows() -> None:
    def fail_token_embedding(_input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("stored prompt rows must be used for this re-prefill")

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
            feedback_embeddings=[torch.tensor([40.0, 41.0])],
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
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(5, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Continuous Feedback Rows.*expected 2.*found 1",
    ):
        runner.before_prefill(forward_batch, SimpleNamespace(), [request])


def test_ming_tts_prefill_missing_request_state_is_an_explicit_error() -> None:
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty((1, 2), dtype=torch.float32)
        ),
        get_input_embeddings=lambda: torch.nn.Embedding(4, 2),
    )
    runner._request_states = {}
    request = SimpleNamespace(
        request_id="missing-state",
        data=SimpleNamespace(
            input_ids=torch.tensor([1]),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=1),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="missing request state.*missing-state"):
        runner._build_prefill_input_embeds(
            SimpleNamespace(input_ids=torch.tensor([1])),
            [request],
        )
