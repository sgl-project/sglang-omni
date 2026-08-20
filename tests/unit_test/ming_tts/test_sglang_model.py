# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_ming_tts_fixed_tail_sampling_inputs_match_eager_and_graph() -> None:
    """Match fixed-input eager and graph tails through feedback and stop lifecycle."""
    from sglang_omni.models.ming_tts.model_runner import (
        MingTTSModelRunner,
        MingTTSTPStepUpdate,
        _MingTTSRequestState,
    )
    from sglang_omni.models.ming_tts.sglang_model import (
        MingTTSSGLangModel,
        MingTTSTailInputs,
        MingTTSTailOutputs,
    )

    fixed_noise = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    fixed_timesteps = torch.tensor([0.0, 0.5, 1.0])
    fixed_sde_random = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.5, 0.6], [0.7, 0.8]]],
        ]
    )
    inputs = MingTTSTailInputs(
        hidden_states=torch.tensor([[[1.0, 2.0, 3.0]]]),
        latent_history=torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
        cfg=torch.tensor([2.0]),
        sigma=torch.tensor([0.25]),
        temperature=torch.tensor([0.0]),
    )
    sampling_observations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fixed_sampling_inputs(*, batch_size: int, device: torch.device):
        assert batch_size == 1
        assert device.type == "cpu"
        return (
            fixed_noise.clone(),
            fixed_timesteps.clone(),
            fixed_sde_random.clone(),
        )

    def compute_tail_step(
        tail_inputs,
        *,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
    ) -> MingTTSTailOutputs:
        sampling_observations.append(
            (noise.clone(), timesteps.clone(), sde_random.clone())
        )
        sampled = tail_inputs.latent_history + noise.transpose(1, 2)
        return MingTTSTailOutputs(
            sampled=sampled,
            feedback_embeddings=sampled.reshape(1, -1),
            stop_prob=(tail_inputs.hidden_states[:, 0, 0] > 0.5).float(),
        )

    eager_model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4)),
        _tail_graphs=None,
        _make_tail_sampling_inputs=fixed_sampling_inputs,
        _compute_tail_step=compute_tail_step,
    )

    class FixedInputTailGraph:
        def replay(self, tail_inputs, *, noise, sde_random):
            return compute_tail_step(
                tail_inputs,
                noise=noise,
                timesteps=fixed_timesteps,
                sde_random=sde_random,
            )

    graph_model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4)),
        _tail_graphs=FixedInputTailGraph(),
        _make_tail_sampling_inputs=fixed_sampling_inputs,
    )
    eager_model.run_tail_step = lambda tail_inputs: MingTTSSGLangModel.run_tail_step(
        eager_model,
        tail_inputs,
    )
    graph_model.run_tail_step = lambda tail_inputs: MingTTSSGLangModel.run_tail_step(
        graph_model,
        tail_inputs,
    )

    eager = MingTTSSGLangModel.run_tail_step(eager_model, inputs)
    graph = MingTTSSGLangModel.run_tail_step(graph_model, inputs)

    assert torch.equal(graph.sampled, eager.sampled)
    assert torch.equal(graph.feedback_embeddings, eager.feedback_embeddings)
    assert torch.equal(graph.stop_prob, eager.stop_prob)

    def run_lifecycle(model) -> dict[str, object]:
        runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
        runner.model = model
        request_state = _MingTTSRequestState(latent_history=torch.zeros(1, 2, 2))
        runner._request_states = {"fixed-tail": request_state}
        request = SimpleNamespace(
            request_id="fixed-tail",
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
        step_updates = []
        for step, hidden_value in ((4, 0.0), (5, 1.0)):
            request.data.generation_steps = step
            update = MingTTSTPStepUpdate.empty_for_broadcast(
                batch_size=1,
                hidden_size=4,
                device=torch.device("cpu"),
                feedback_dtype=torch.float32,
            )
            runner._run_entry_tail_step(
                torch.full((1, 1, 4), hidden_value),
                [request],
                update,
            )
            step_updates.append(
                {
                    "next_token_ids": update.next_token_ids.clone(),
                    "feedback_mask": update.feedback_mask.clone(),
                }
            )
        return {
            "step_updates": step_updates,
            "generated_latents": request.data.generated_latents.clone(),
            "feedback_embeddings": [
                value.clone() for value in request_state.feedback_embeddings
            ],
            "stop_step": request.data.stop_step,
            "pending_stream_patch": request.data.pending_stream_patch,
        }

    eager_lifecycle = run_lifecycle(eager_model)
    graph_lifecycle = run_lifecycle(graph_model)

    assert eager_lifecycle["stop_step"] == graph_lifecycle["stop_step"] == 5
    assert (
        eager_lifecycle["pending_stream_patch"]
        is graph_lifecycle["pending_stream_patch"]
        is None
    )
    assert torch.equal(
        eager_lifecycle["generated_latents"],
        graph_lifecycle["generated_latents"],
    )
    assert len(eager_lifecycle["feedback_embeddings"]) == 1
    assert len(graph_lifecycle["feedback_embeddings"]) == 1
    assert torch.equal(
        eager_lifecycle["feedback_embeddings"][0],
        graph_lifecycle["feedback_embeddings"][0],
    )
    for eager_update, graph_update in zip(
        eager_lifecycle["step_updates"],
        graph_lifecycle["step_updates"],
    ):
        assert torch.equal(
            eager_update["next_token_ids"], graph_update["next_token_ids"]
        )
        assert torch.equal(eager_update["feedback_mask"], graph_update["feedback_mask"])
    assert len(sampling_observations) == 6
    for noise, timesteps, sde_random in sampling_observations:
        assert torch.equal(noise, fixed_noise)
        assert torch.equal(timesteps, fixed_timesteps)
        assert torch.equal(sde_random, fixed_sde_random)


def test_ming_tts_forward_accepts_sidecar_request_identity() -> None:
    from sglang_omni.models.ming_tts.sglang_model import MingTTSSGLangModel

    seen: dict[str, object] = {}

    def fake_model(**kwargs):
        seen.update(kwargs)
        return torch.ones((2, 3))

    wrapper = object.__new__(MingTTSSGLangModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = fake_model
    input_ids = torch.tensor([1, 2], dtype=torch.long)
    positions = torch.tensor([0, 1], dtype=torch.long)
    input_embeds = torch.ones((2, 3))
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda: False,
        ),
        mrope_positions=None,
    )

    result = wrapper.forward(
        input_ids=input_ids,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        omni_prefill_rids=("request-1",),
    )

    assert torch.equal(result.hidden_states, torch.ones((2, 3)))
    assert seen["input_embeds"] is input_embeds
    assert seen["positions"] is positions


def test_ming_sparse_moe_tp_collective_uses_forward_flags(monkeypatch) -> None:
    from sglang.srt.runtime_context import get_forward

    from sglang_omni.models.ming_tts import sglang_model

    helper_calls: list[tuple[bool, bool, bool]] = []

    def strict_0516_helper(*, is_tp_path: bool) -> bool:
        forward = get_forward()
        helper_calls.append(
            (
                is_tp_path,
                forward.fuse_mlp_allreduce,
                forward.mlp_reduce_scatter,
            )
        )
        return forward.fuse_mlp_allreduce or forward.mlp_reduce_scatter

    all_reduce_calls: list[torch.Tensor] = []

    def fake_all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        all_reduce_calls.append(hidden_states.clone())
        return hidden_states + 100

    monkeypatch.setattr(
        sglang_model,
        "should_skip_post_experts_all_reduce",
        strict_0516_helper,
    )
    monkeypatch.setattr(
        sglang_model,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )
    block = SimpleNamespace(
        tp_size=2,
        shared_experts=None,
        gate=lambda hidden_states: hidden_states,
        topk=lambda _hidden_states, _router_logits: object(),
        experts=lambda hidden_states, _topk_output: hidden_states + 1,
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    with get_forward().scoped(
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=False,
    ):
        ordinary = sglang_model.MingBailingMoeSparseMoeBlock.forward(
            block,
            hidden_states,
        )
    with get_forward().scoped(
        fuse_mlp_allreduce=True,
        mlp_reduce_scatter=False,
    ):
        fused = sglang_model.MingBailingMoeSparseMoeBlock.forward(
            block,
            hidden_states,
        )
    with get_forward().scoped(
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=True,
    ):
        reduce_scattered = sglang_model.MingBailingMoeSparseMoeBlock.forward(
            block,
            hidden_states,
        )

    assert torch.equal(ordinary, hidden_states + 101)
    assert torch.equal(fused, hidden_states + 1)
    assert torch.equal(reduce_scattered, hidden_states + 1)
    assert len(all_reduce_calls) == 1
    assert helper_calls == [
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ]


@pytest.mark.parametrize(
    ("fuse_mlp_allreduce", "mlp_reduce_scatter", "postprocess_calls"),
    [(True, False, 0), (False, True, 1)],
)
def test_ming_decoder_scopes_mlp_collective_flags(
    fuse_mlp_allreduce: bool,
    mlp_reduce_scatter: bool,
    postprocess_calls: int,
) -> None:
    from sglang.srt.runtime_context import get_forward

    from sglang_omni.models.ming_tts.sglang_model import MingBailingMoeDecoderLayer

    class FakeCommunicator:
        def __init__(self) -> None:
            self.postprocess_calls = 0

        def prepare_attn_and_capture_last_layer_outputs(
            self,
            hidden_states,
            residual,
            forward_batch,
        ):
            del forward_batch
            return hidden_states, residual

        def prepare_mlp(self, *, hidden_states, residual, forward_batch):
            del forward_batch
            return hidden_states, residual

        def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
            del forward_batch
            return fuse_mlp_allreduce

        def should_use_reduce_scatter(self, forward_batch):
            del forward_batch
            return mlp_reduce_scatter

        def postprocess_layer(self, hidden_states, residual, forward_batch):
            del forward_batch
            self.postprocess_calls += 1
            return hidden_states, residual

    seen_flags: list[tuple[bool, bool]] = []

    def mlp(hidden_states, forward_batch):
        del forward_batch
        forward = get_forward()
        seen_flags.append((forward.fuse_mlp_allreduce, forward.mlp_reduce_scatter))
        return hidden_states.clone()

    communicator = FakeCommunicator()
    layer = SimpleNamespace(
        layer_communicator=communicator,
        attention=lambda _positions, hidden_states, _forward_batch: hidden_states,
        mlp=mlp,
    )
    hidden_states = torch.ones((1, 2))

    with get_forward().scoped(
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=False,
    ):
        output, _ = MingBailingMoeDecoderLayer.forward(
            layer,
            positions=torch.tensor([0]),
            hidden_states=hidden_states,
            forward_batch=object(),
            residual=None,
        )
        assert get_forward().fuse_mlp_allreduce is False
        assert get_forward().mlp_reduce_scatter is False

    assert seen_flags == [(fuse_mlp_allreduce, mlp_reduce_scatter)]
    assert communicator.postprocess_calls == postprocess_calls
    assert getattr(output, "_sglang_needs_allreduce_fusion", False) is (
        fuse_mlp_allreduce
    )
