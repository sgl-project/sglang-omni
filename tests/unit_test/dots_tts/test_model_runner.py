# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner


def test_dots_abort_callback_clears_flow_state() -> None:
    flow_state = object()
    released = []
    request_data = SimpleNamespace(
        pending_feedback_queue=deque([object()]),
        flow_state=flow_state,
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=SimpleNamespace(release_request=released.append)
    )
    runner._request_data = {"req-1": request_data}

    runner.reset_request("req-1")
    runner.reset_request("req-1")

    assert runner._request_data == {}
    assert not request_data.pending_feedback_queue
    assert request_data.flow_state is None
    assert released == [flow_state]


def test_dots_finish_callback_releases_flow_state_once() -> None:
    flow_state = object()
    released = []
    request_data = SimpleNamespace(
        pending_feedback_queue=deque([object()]),
        flow_state=flow_state,
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=SimpleNamespace(release_request=released.append)
    )
    runner._request_data = {"req-1": request_data}

    runner.on_request_finished("req-1", request_data)
    runner.reset_request("req-1")

    assert request_data.flow_state is None
    assert released == [flow_state]


def test_dots_post_prefill_skips_prefill_only_batch() -> None:
    runner = object.__new__(DotsTTSModelRunner)

    runner.post_prefill(
        result=object(),
        forward_batch=object(),
        schedule_batch=SimpleNamespace(is_prefill_only=True),
        requests=[object()],
    )


def test_dots_prefill_batches_request_embeddings_in_scheduler_order() -> None:
    old_flow_state = object()
    suspended = []
    restored_rng = []

    class _Flow:
        def suspend_request(self, state):
            suspended.append(state)
            return torch.tensor([7], dtype=torch.uint8)

        def new_request(self, *, speaker_scale: float, rng, **_kwargs):
            restored_rng.append(rng)
            prompt = torch.full((1, 1, 4), speaker_scale)
            return object(), prompt

        def replay_feedback(self, _state, decoded_latent_patches):
            return torch.full((len(decoded_latent_patches), 4), 33.0)

    model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding.from_pretrained(
            torch.arange(40, dtype=torch.float32).reshape(10, 4)
        ),
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = model

    def _request(request_id: str, speaker_scale: float):
        retracted = request_id == "a"
        data = SimpleNamespace(
            flow_state=old_flow_state if retracted else None,
            generation_schedule=torch.tensor([[1, 2, 3]]),
            span_positions=torch.tensor([1, 2]),
            prompt_span_positions=torch.tensor([1]),
            prefill_end=3,
            pending_feedback_queue=deque(),
            decoded_latent_patches=([torch.full((1, 2, 2), 5.0)] if retracted else []),
            req=SimpleNamespace(
                prefix_indices=[],
                extend_range=SimpleNamespace(length=4 if retracted else 3),
                output_ids=[3] if retracted else [],
                is_retracted=retracted,
            ),
            state=SimpleNamespace(
                prompt_latents=torch.zeros(1, 4, 2),
                speaker_embedding=torch.zeros(1, 8),
                speaker_scale=speaker_scale,
                seed=42,
            ),
        )
        return SimpleNamespace(request_id=request_id, data=data)

    requests = [_request("a", 11.0), _request("b", 22.0)]
    runner._request_data = {"a": requests[0].data}
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3, 1, 2, 3]), input_embeds=None
    )

    runner.before_prefill(forward_batch, object(), requests)

    assert forward_batch.input_embeds.shape == (7, 4)
    torch.testing.assert_close(forward_batch.input_embeds[1], torch.full((4,), 11.0))
    torch.testing.assert_close(forward_batch.input_embeds[3], torch.full((4,), 33.0))
    torch.testing.assert_close(forward_batch.input_embeds[5], torch.full((4,), 22.0))
    assert suspended == [old_flow_state]
    torch.testing.assert_close(restored_rng[0], torch.tensor([7], dtype=torch.uint8))
    assert restored_rng[1] == 42


def test_dots_prefill_failure_releases_materialized_slots() -> None:
    flow_state = object()
    released = []

    class _Flow:
        calls = 0

        def new_request(self, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("materialize failed")
            return flow_state, None

        release_request = released.append

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding(10, 4),
    )
    runner._request_data = {}

    def _request(request_id: str):
        data = SimpleNamespace(
            flow_state=None,
            generation_schedule=torch.tensor([[1, 2, 3]]),
            span_positions=torch.tensor([1]),
            prompt_span_positions=None,
            prefill_end=3,
            pending_feedback_queue=deque(),
            decoded_latent_patches=[],
            req=SimpleNamespace(
                prefix_indices=[],
                extend_range=SimpleNamespace(length=3),
                output_ids=[],
                is_retracted=False,
            ),
            state=SimpleNamespace(
                prompt_latents=None,
                speaker_embedding=None,
                speaker_scale=1.0,
                seed=42,
            ),
        )
        return SimpleNamespace(request_id=request_id, data=data)

    requests = [_request("a"), _request("b")]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3, 1, 2, 3]), input_embeds=None
    )

    with pytest.raises(RuntimeError, match="materialize failed"):
        runner.before_prefill(forward_batch, object(), requests)

    assert released == [flow_state]
    assert runner._request_data == {}
    assert requests[0].data.flow_state is None


def _decode_request(request_id: str, fill: float):
    feedback = torch.full((1, 4), fill)
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(pending_feedback_queue=deque([feedback])),
    )


def test_dots_before_decode_writes_graph_feedback_buffer_in_batch_order(
    monkeypatch,
) -> None:
    buffer = torch.zeros(4, 4)
    stack_outputs = []
    stack = torch.stack

    def record_stack(values, *args, **kwargs):
        stack_outputs.append(kwargs.get("out"))
        return stack(values, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", record_stack)
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        graph_feedback_buffer=buffer,
        parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([0, 0]), input_embeds=None)

    runner.before_decode(
        forward_batch, object(), [_decode_request("a", 5.0), _decode_request("b", 6.0)]
    )

    torch.testing.assert_close(buffer[0], torch.full((4,), 5.0))
    torch.testing.assert_close(buffer[1], torch.full((4,), 6.0))
    torch.testing.assert_close(buffer[2], torch.zeros(4))
    assert len(stack_outputs) == 1
    assert stack_outputs[0].data_ptr() == buffer.data_ptr()
    assert forward_batch.input_embeds is None


def test_dots_before_decode_without_buffer_sets_forward_batch_embeds() -> None:
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        graph_feedback_buffer=None,
        parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([0]), input_embeds=None)

    runner.before_decode(forward_batch, object(), [_decode_request("a", 9.0)])

    torch.testing.assert_close(forward_batch.input_embeds, torch.full((1, 4), 9.0))


def test_dots_post_decode_resolve_applies_batched_eos_finish() -> None:
    finished_token = object()

    class _Flow:
        is_batched = True

        def decode_batch(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    feedback_embedding=torch.zeros(4),
                    latent_patch=torch.zeros(1, 2, 2),
                    finished=False,
                    emit=True,
                ),
                SimpleNamespace(
                    feedback_embedding=torch.ones(4),
                    latent_patch=torch.ones(1, 2, 2),
                    finished=False,
                    emit=True,
                ),
            ]

        def resolve_batched_eos(self):
            return [False, True]

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(flow=_Flow())

    def _request(request_id: str, control_token_id: int):
        return SimpleNamespace(
            request_id=request_id,
            data=SimpleNamespace(
                flow_state=object(),
                pending_feedback_queue=deque(),
                decoded_latent_patches=[],
                latent_patches=[],
                latest_latent_patch=None,
                control_token_id=control_token_id,
                state=SimpleNamespace(
                    num_steps=2,
                    ode_method="euler",
                    guidance_scale=1.0,
                    eos_threshold=0.5,
                ),
                req=SimpleNamespace(finished_reason=None),
            ),
        )

    requests = [_request("a", 7), _request("b", 9)]
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(2, 4)),
        next_token_ids=None,
    )

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "sglang_omni.models.dots_tts.model_runner.FINISH_MATCHED_TOKEN",
            lambda token_id: (finished_token, token_id),
        )
        runner.post_decode(result, object(), object(), requests)

    assert requests[0].data.req.finished_reason is None
    assert requests[1].data.req.finished_reason == (finished_token, 9)
    assert result.next_token_ids.tolist() == [7, 9]
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert len(requests[1].data.latent_patches) == 1


def test_dots_post_decode_resolve_uses_step_finished_for_single_request() -> None:
    finished_token = object()

    class _Flow:
        is_batched = False

        def decode_batch(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    feedback_embedding=torch.zeros(4),
                    latent_patch=torch.zeros(1, 2, 2),
                    finished=True,
                    emit=True,
                )
            ]

        def resolve_batched_eos(self):
            raise AssertionError("single-request path must not resolve batched EOS")

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(flow=_Flow())
    request = SimpleNamespace(
        request_id="a",
        data=SimpleNamespace(
            flow_state=object(),
            pending_feedback_queue=deque(),
            decoded_latent_patches=[],
            latent_patches=[],
            latest_latent_patch=None,
            control_token_id=3,
            state=SimpleNamespace(
                num_steps=2,
                ode_method="euler",
                guidance_scale=1.0,
                eos_threshold=0.5,
            ),
            req=SimpleNamespace(finished_reason=None),
        ),
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, 4)),
        next_token_ids=None,
    )

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "sglang_omni.models.dots_tts.model_runner.FINISH_MATCHED_TOKEN",
            lambda token_id: (finished_token, token_id),
        )
        runner.post_decode(result, object(), object(), [request])

    assert request.data.req.finished_reason == (finished_token, 3)
