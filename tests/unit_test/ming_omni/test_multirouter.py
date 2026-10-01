from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from torch import nn

from sglang_omni.model_runner.ming_thinker_model_runner import MingThinkerModelRunner
from sglang_omni.models.ming_omni import thinker
from sglang_omni.models.ming_omni.configuration import BailingMoeV2Config


def test_config_preserves_legacy_default_and_checkpoint_router_type():
    assert BailingMoeV2Config().router_type == "topN"
    assert BailingMoeV2Config(router_type="MultiRouter").router_type == "MultiRouter"


class ReferenceExperts(nn.Module):
    def forward(self, hidden_states, topk_output):
        self.last_topk = topk_output
        return (
            (topk_output.topk_weights * (topk_output.topk_ids + 1))
            .sum(-1, keepdim=True)
            .expand_as(hidden_states)
        )


class TupleGate(nn.Linear):
    def forward(self, hidden_states):
        return super().forward(hidden_states), None


@pytest.mark.parametrize("multi_router", [False, True])
@pytest.mark.parametrize("with_modalities", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_multirouter_matches_independent_per_token_reference(
    multi_router, with_modalities, use_bias
):
    torch.manual_seed(91)
    block = object.__new__(thinker.BailingMoeV2SparseMoeBlock)
    nn.Module.__init__(block)
    block.multi_router = multi_router
    block.num_experts = 8
    block.num_experts_per_tok = 2
    block.n_group = 2
    block.topk_group = 1
    block.routed_scaling_factor = 2.5
    block.tp_size = 1
    block.shared_experts = None
    block.experts = ReferenceExperts()
    block.gate = TupleGate(4, 8, bias=False).bfloat16()
    block.image_gate = TupleGate(4, 8, bias=False).bfloat16()
    block.audio_gate = TupleGate(4, 8, bias=False).bfloat16()
    block.expert_bias = nn.Parameter(torch.arange(8).float() / 20)
    block.image_expert_bias = nn.Parameter(torch.arange(8).flip(0).float() / 7)
    block.audio_expert_bias = nn.Parameter(
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    )
    if not use_bias:
        block.expert_bias = None
        block.image_expert_bias = None
        block.audio_expert_bias = None
    hidden = torch.randn(6, 4).bfloat16()
    modalities = torch.tensor([0, 1, 2, 1, 0, 2]) if with_modalities else None
    text_before_multimodal = block(hidden, modality_ids=None)
    actual = block(hidden, modality_ids=modalities)
    expected = []
    expected_logits = []
    for index, row in enumerate(hidden):
        route = int(modalities[index]) if multi_router and modalities is not None else 0
        gate = (block.gate, block.image_gate, block.audio_gate)[route]
        bias = (block.expert_bias, block.image_expert_bias, block.audio_expert_bias)[
            route
        ]
        logits = (
            F.linear(row.float(), gate.weight.float())
            if multi_router
            else F.linear(row, gate.weight).float()
        )
        probabilities = logits.sigmoid()
        selection_scores = probabilities + bias if bias is not None else probabilities
        group = selection_scores.reshape(2, 4).topk(2, dim=-1).values.sum(-1).argmax()
        chosen = selection_scores[group * 4 : group * 4 + 4].topk(2).indices + group * 4
        weights = probabilities[chosen]
        weights = weights / weights.sum() * 2.5
        expected.append((weights * (chosen + 1)).sum().expand(4))
        expected_logits.append(logits)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        block.experts.last_topk.router_logits,
        torch.stack(expected_logits),
        rtol=1e-6,
        atol=1e-6,
    )
    if multi_router and not with_modalities:
        rounded_logits = F.linear(hidden, block.gate.weight).float()
        assert not torch.allclose(
            block.experts.last_topk.router_logits, rounded_logits, rtol=1e-5, atol=1e-5
        )
    if with_modalities:
        after_multimodal = block(hidden, modality_ids=None)
        torch.testing.assert_close(
            after_multimodal, text_before_multimodal, rtol=0, atol=0
        )


def test_prefill_routes_hashed_modalities_per_request_and_chunk():
    runner = object.__new__(MingThinkerModelRunner)
    runner.multi_router = True
    runner.image_token_id = 3
    runner.video_token_id = 4
    runner.audio_token_id = 5
    first = SimpleNamespace(
        omni_model_inputs={
            "image_embeds": torch.ones(3, 2),
            "audio_embeds": torch.ones(2, 2),
            "pad_values": {"image": 1003, "audio": 1005},
        }
    )
    second = SimpleNamespace(omni_model_inputs={"video_embeds": torch.ones(2, 2)})
    batch = SimpleNamespace(
        input_ids=torch.tensor([1003, 9, 1005, 4, 7]),
        extend_seq_lens_cpu=[3, 2],
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=None,
    )
    assert runner.prefill_modality_ids(
        batch, SimpleNamespace(reqs=[first, second])
    ).tolist() == [1, 0, 2, 1, 0]
    first.omni_model_inputs = None
    second.omni_model_inputs = None
    assert runner.prefill_modality_ids(
        batch, SimpleNamespace(reqs=[first, second])
    ).tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("cached_rows", [0, 2])
def test_real_request_prefix_chunks_keep_routes_aligned_with_injected_rows(cached_rows):
    runner = object.__new__(MingThinkerModelRunner)
    runner.multi_router = True
    runner.embed_tokens = nn.Embedding(20, 2)
    runner.image_token_id, runner.video_token_id, runner.audio_token_id = 3, 4, 5
    request = Req(
        rid="multirouter-prefix",
        origin_input_text="",
        origin_input_ids=[1, 1003, 1003, 1003, 1003, 2],
        sampling_params=SamplingParams(max_new_tokens=2),
        vocab_size=20,
    )
    images = torch.arange(8).reshape(4, 2).float()
    request.omni_model_inputs = {"image_embeds": images, "pad_values": {"image": 1003}}
    request._omni_consumed = None  # noqa: leading-underscore
    request.inflight_middle_chunks = 1
    calls = []

    def capture(forward_batch, input_embeds, modality_ids):
        calls.append((input_embeds.detach().clone(), modality_ids.clone()))
        return input_embeds

    runner.forward_with_omni_embeds = capture
    prefix = 1 + cached_rows
    for final, ids, start in (
        (False, [1003], prefix),
        (True, request.origin_input_ids[prefix + 1 :], prefix + 1),
    ):
        request.inflight_middle_chunks = 0 if final else 1
        batch = SimpleNamespace(
            input_ids=torch.tensor(ids),
            extend_seq_lens_cpu=[len(ids)],
            extend_prefix_lens_cpu=[start],
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=None,
        )
        schedule = SimpleNamespace(
            reqs=[request], forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        runner.custom_prefill_forward(batch, schedule, [])
    assert calls[0][1].tolist() == [1]
    torch.testing.assert_close(calls[0][0][0], images[cached_rows])
    assert calls[1][1].tolist() == [1] * (3 - cached_rows) + [0]
    torch.testing.assert_close(calls[1][0][:-1], images[cached_rows + 1 :])
    assert request.omni_model_inputs is None


@pytest.mark.parametrize("rank", [0, 1])
def test_scattered_modality_order_matches_attention_tp(monkeypatch, rank):
    layer = object.__new__(thinker.BailingMoeV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_scatter_modes = SimpleNamespace(mlp_mode=thinker.ScatterMode.SCATTERED)
    monkeypatch.setattr(
        thinker,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=2, attn_tp_rank=rank),
    )
    labels = torch.tensor([0, 1, 2, 1, 0, 2])
    assert torch.equal(
        layer.mlp_modality_ids(labels, SimpleNamespace()), labels.chunk(2)[rank]
    )


@pytest.mark.parametrize(
    "router_type,dp_size,rejected",
    [
        ("MultiRouter", 2, True),
        ("MultiRouter", 1, False),
        ("topN", 2, False),
    ],
)
def test_real_block_constructor_limits_multirouter_to_attention_tp(
    monkeypatch, router_type, dp_size, rejected
):
    config = BailingMoeV2Config(
        hidden_size=4,
        num_attention_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        num_shared_experts=0,
        router_type=router_type,
    )
    monkeypatch.setattr(
        thinker, "get_parallel", lambda: SimpleNamespace(attn_dp_size=dp_size)
    )
    monkeypatch.setattr(thinker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(thinker, "ReplicatedLinear", TupleGate)
    monkeypatch.setattr(
        thinker,
        "get_moe_impl_class",
        lambda quantization: lambda **kwargs: ReferenceExperts(),
    )
    if rejected:
        with pytest.raises(NotImplementedError, match="attention DP size 1"):
            thinker.BailingMoeV2SparseMoeBlock(config, layer_id=1)
    else:
        block = thinker.BailingMoeV2SparseMoeBlock(config, layer_id=1)
        assert block.multi_router == (router_type == "MultiRouter")


def test_legacy_text_model_does_not_forward_modality_labels():
    class Layer(nn.Module):
        def forward(self, hidden, batch, residual, modality_ids):
            assert modality_ids is None
            return hidden, residual

    class Norm(nn.Module):
        def forward(self, hidden, residual):
            return hidden, residual

    model = object.__new__(thinker.BailingMoeV2TextModel)
    nn.Module.__init__(model)
    model.config = BailingMoeV2Config(router_type="topN")
    model.layers = nn.ModuleList([Layer()])
    model.norm = Norm()
    hidden = torch.ones(3, 4)
    result = model(
        None,
        None,
        SimpleNamespace(),
        input_embeds=hidden,
        modality_ids=torch.tensor([1, 2, 0]),
    )
    torch.testing.assert_close(result, hidden)


@pytest.mark.parametrize(
    "mode,metadata,multimodal,multi_router,rejected",
    [
        (ForwardMode.EXTEND, SimpleNamespace(), True, True, True),
        (ForwardMode.EXTEND, None, True, True, False),
        (ForwardMode.DECODE, SimpleNamespace(), True, True, False),
        (ForwardMode.EXTEND, SimpleNamespace(), False, True, False),
        (ForwardMode.EXTEND, SimpleNamespace(), True, False, False),
    ],
)
def test_multimodal_cp_is_rejected_without_changing_plain_prefill_or_decode(
    mode, metadata, multimodal, multi_router, rejected
):
    runner = object.__new__(MingThinkerModelRunner)
    runner.multi_router = multi_router
    runner.image_token_id, runner.video_token_id, runner.audio_token_id = 3, 4, 5
    request = SimpleNamespace(
        omni_model_inputs={"image_embeds": torch.ones(1, 2)} if multimodal else {}
    )
    batch = SimpleNamespace(
        input_ids=torch.tensor([3, 7]),
        extend_seq_lens_cpu=[2],
        forward_mode=mode,
        attn_cp_metadata=metadata,
    )
    schedule = SimpleNamespace(reqs=[request])
    if rejected:
        with pytest.raises(
            NotImplementedError, match="context-parallel multimodal prefill"
        ):
            runner.prefill_modality_ids(batch, schedule)
    else:
        expected = [1, 0] if multimodal else [0, 0]
        assert runner.prefill_modality_ids(batch, schedule).tolist() == expected
