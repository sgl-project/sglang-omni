# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize(
    (
        "max_running_requests",
        "disable_cuda_graph",
        "graph_max_batch_size",
        "context_length",
        "expected_tail_capacity",
        "expected_aggregator_capacity",
    ),
    [
        (3, False, 8, 128, 8, 128),
        (5, True, 64, 2, 5, 5),
        (12, False, 8, 4, 12, 12),
    ],
)
def test_ming_tts_owns_tail_execution_geometry(
    monkeypatch: pytest.MonkeyPatch,
    max_running_requests: int,
    disable_cuda_graph: bool,
    graph_max_batch_size: int,
    context_length: int,
    expected_tail_capacity: int,
    expected_aggregator_capacity: int,
) -> None:
    from sglang_omni.models.ming_omni.talker.talker_module.execution import (
        TalkerExecutionConfig,
    )
    from sglang_omni.models.ming_tts import sglang_model

    stale_execution_config = TalkerExecutionConfig(
        attn_backend="stale",
        rope_kernel=Mock(),
    )
    config = SimpleNamespace(
        llm_config=SimpleNamespace(hidden_size=8, vocab_size=16),
        audio_tokenizer_config=SimpleNamespace(enc_kwargs={"latent_dim": 4}),
        aggregator_config={"execution_config": stale_execution_config},
        ditar_config={
            "patch_size": 2,
            "history_patch_size": 3,
            "execution_config": stale_execution_config,
        },
    )
    captured: dict[str, dict[str, object]] = {}

    class Backbone(torch.nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(
                16,
                8,
                dtype=torch.bfloat16,
            )

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.word_embeddings

    class CapturingAggregator(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["aggregator"] = kwargs

    class CapturingFlowLoss(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["dit"] = kwargs

    graph = SimpleNamespace(
        disable_cuda_graph=disable_cuda_graph,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=graph_max_batch_size)
        ),
    )
    kernel = Mock()
    provider = Mock(return_value=kernel)
    monkeypatch.setattr(
        sglang_model,
        "current_platform",
        SimpleNamespace(get_joint_rope_inplace_kernel=provider),
    )
    monkeypatch.setattr(sglang_model, "MingBailingMoeTextModel", Backbone)
    monkeypatch.setattr(sglang_model, "Aggregator", CapturingAggregator)
    monkeypatch.setattr(sglang_model, "FlowLoss", CapturingFlowLoss)
    monkeypatch.setattr(sglang_model, "get_exec", lambda: SimpleNamespace(graph=graph))
    monkeypatch.setattr(
        sglang_model,
        "get_schedule",
        lambda: SimpleNamespace(max_running_requests=max_running_requests),
    )
    monkeypatch.setattr(
        sglang_model,
        "get_model",
        lambda: SimpleNamespace(context_length=context_length),
    )

    model = sglang_model.MingTTSSGLangModel(config)

    aggregator_execution = captured["aggregator"]["execution_config"]
    dit_execution = captured["dit"]["execution_config"]
    assert aggregator_execution == TalkerExecutionConfig(
        attn_backend=sglang_model.MING_TTS_TAIL_ATTN_BACKEND,
        rope_kernel=kernel,
        rope_seq_len=3,
        rope_max_batch_size=expected_aggregator_capacity,
    )
    assert dit_execution == TalkerExecutionConfig(
        attn_backend=sglang_model.MING_TTS_TAIL_ATTN_BACKEND,
        rope_kernel=kernel,
        rope_seq_len=6,
        rope_max_batch_size=2 * expected_tail_capacity,
    )
    assert model._decode_input_embedding.num_embeddings == expected_tail_capacity
    assert config.aggregator_config["execution_config"] is stale_execution_config
    assert config.ditar_config["execution_config"] is stale_execution_config
    provider.assert_called_once_with()
    kernel.assert_not_called()


@pytest.mark.parametrize(
    "platform_name", ["CPUOmniPlatform", "ROCMOmniPlatform", "MUSAOmniPlatform"]
)
def test_ming_tts_rejects_missing_joint_rope_before_building_backbone(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
) -> None:
    from sglang_omni import platforms
    from sglang_omni.models.ming_tts import sglang_model

    backbone = Mock(side_effect=AssertionError("Backbone must not be built"))
    monkeypatch.setattr(sglang_model, "MingBailingMoeTextModel", backbone)
    monkeypatch.setattr(
        sglang_model, "current_platform", getattr(platforms, platform_name)()
    )

    with pytest.raises(
        RuntimeError, match=f"Ming-TTS requires.*{platform_name} does not provide one"
    ):
        sglang_model.MingTTSSGLangModel(SimpleNamespace())

    backbone.assert_not_called()


@pytest.mark.parametrize(
    ("weight_dtype", "autocast_enabled"),
    [
        (torch.bfloat16, True),
        (torch.float16, True),
        (torch.float32, False),
    ],
)
def test_ming_tts_tail_compute_owns_model_precision(
    weight_dtype: torch.dtype,
    autocast_enabled: bool,
) -> None:
    from sglang_omni.models.ming_tts.sglang_model import (
        MingTTSSGLangModel,
        MingTTSTailInputs,
    )

    observed: list[tuple[str, bool, torch.dtype]] = []

    class FlowLoss:
        def sample(self, **kwargs):
            observed.append(
                (
                    "flow",
                    torch.is_autocast_enabled("cpu"),
                    torch.get_autocast_dtype("cpu"),
                )
            )
            return kwargs["noise"]

    class Aggregator:
        def __call__(self, sampled):
            observed.append(
                (
                    "aggregator",
                    torch.is_autocast_enabled("cpu"),
                    torch.get_autocast_dtype("cpu"),
                )
            )
            return sampled

    owner = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty(1, dtype=weight_dtype)
        ),
        flowloss=FlowLoss(),
        linear_proj_audio=Aggregator(),
        stop_head=lambda hidden: torch.zeros(hidden.shape[0], 1, 2),
    )
    inputs = MingTTSTailInputs(
        hidden_states=torch.zeros(2, 1, 4),
        latent_history=torch.zeros(2, 1, 4),
        cfg=torch.ones(2),
        sigma=torch.zeros(2),
        temperature=torch.zeros(2),
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        MingTTSSGLangModel._compute_tail_step(
            owner,
            inputs,
            noise=torch.ones(2, 1, 4),
            timesteps=torch.arange(2),
            sde_random=torch.zeros(1, 2, 1, 4),
        )

    assert observed == [
        ("flow", autocast_enabled, weight_dtype),
        ("aggregator", autocast_enabled, weight_dtype),
    ]


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_ming_tts_reference_projection_disables_inherited_autocast(
    weight_dtype: torch.dtype,
) -> None:
    from sglang_omni.models.ming_tts.sglang_model import MingTTSSGLangModel

    observed: list[tuple[bool, torch.dtype]] = []

    class Projection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_embedder = torch.nn.Linear(3, 4).to(dtype=weight_dtype)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            observed.append((torch.is_autocast_enabled("cpu"), inputs.dtype))
            return self.x_embedder(inputs)

    owner = SimpleNamespace(
        linear_proj_audio=Projection(),
        patch_size=2,
        latent_dim=3,
        hidden_size=4,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = MingTTSSGLangModel.project_reference_latents(
            owner,
            torch.randn(2, 2, 3, dtype=torch.bfloat16),
        )

    assert observed == [(False, weight_dtype)]
    assert actual.shape == (4, 4)
    assert actual.dtype == weight_dtype


def test_ming_sparse_moe_tp_collective_uses_forward_flags(monkeypatch) -> None:
    from sglang.srt.runtime_context import get_forward

    from sglang_omni.models.ming_tts import sglang_model

    helper_calls: list[tuple[bool, bool, bool]] = []

    def strict_helper(*, is_tp_path: bool) -> bool:
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
        strict_helper,
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
