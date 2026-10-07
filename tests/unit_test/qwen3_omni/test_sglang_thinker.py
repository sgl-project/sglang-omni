# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3-Omni thinker's M-RoPE plumbing, padded-row routing and MoE
precompile."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_moe_kernel
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from sglang.srt.runtime_context import get_context

from sglang_omni.models.qwen3_omni.components import (
    sglang_thinker as sglang_thinker_module,
)
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    DecodeLiveRows,
    PaddedRowsTopK,
    Qwen3OmniThinkerForCausalLM,
    config_uses_mrope,
)
from sglang_omni.models.qwen3_omni.hf_config import (
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(rope_scaling={"mrope_section": [16, 24, 24]}), True),
        (SimpleNamespace(rope_parameters={"mrope_section": [16, 24, 24]}), True),
        (SimpleNamespace(rope_scaling={"rope_type": "default"}), False),
        (SimpleNamespace(), False),
    ],
)
def test_qwen_text_config_declares_mrope_only_for_mrope_sections(config, expected):
    assert config_uses_mrope(config) is expected


@pytest.mark.parametrize(
    ("rope_scaling", "expected_mrope"),
    [
        (
            {"rope_type": "default", "mrope_section": [16, 24, 24]},
            True,
        ),
        (None, False),
    ],
)
def test_real_qwen_config_drives_prefill_positions(
    monkeypatch: pytest.MonkeyPatch,
    rope_scaling: dict[str, object] | None,
    expected_mrope: bool,
):
    class LightweightTextModel(torch.nn.Module):
        def __init__(self, *, config, quant_config, prefix):
            super().__init__()
            del quant_config, prefix
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size,
                config.hidden_size,
            )
            self.layers = torch.nn.ModuleList()

    class LightweightLogitsProcessor(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    monkeypatch.setattr(
        sglang_thinker_module,
        "Qwen3MoeLLMModel",
        LightweightTextModel,
    )
    monkeypatch.setattr(
        sglang_thinker_module,
        "LogitsProcessor",
        LightweightLogitsProcessor,
    )

    text_config = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        rope_scaling=rope_scaling,
    )
    root_config = Qwen3OmniMoeThinkerConfig(text_config=text_config)
    wrapper = Qwen3OmniThinkerForCausalLM(root_config)

    assert wrapper.config is text_config
    assert wrapper.is_mrope_enabled is expected_mrope

    runner = object.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(model=wrapper)
    ordinary_positions = torch.arange(4, dtype=torch.long)
    mrope_positions = ordinary_positions.repeat(3, 1)
    forward_batch = SimpleNamespace(
        positions=ordinary_positions,
        mrope_positions=mrope_positions,
    )

    selected = runner._get_layer_model_positions(
        forward_batch
    )  # noqa: leading-underscore  # upstream name
    assert selected is (mrope_positions if expected_mrope else ordinary_positions)


def test_outer_thinker_forward_accepts_sidecar_request_identity():
    seen: dict[str, object] = {}

    def fake_model(**kwargs):
        seen.update(kwargs)
        return torch.zeros((2, 3))

    wrapper = object.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = fake_model
    wrapper.logits_processor = lambda *args: args[0]
    wrapper.lm_head = object()
    wrapper.fused_rope_gate = None
    wrapper.decode_live_rows = DecodeLiveRows()
    input_ids = torch.tensor([1, 2], dtype=torch.long)
    positions = torch.tensor([0, 1], dtype=torch.long)
    input_embeds = torch.ones((2, 3))
    forward_batch = SimpleNamespace(
        mrope_positions=None, forward_mode=ForwardMode.EXTEND
    )

    result = wrapper.forward(
        input_ids=input_ids,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        omni_prefill_rids=("request-1",),
    )

    assert result is input_ids
    assert seen["input_embeds"] is input_embeds
    assert seen["positions"] is positions


class FixedTopK(torch.nn.Module):
    def __init__(self, topk_output: StandardTopKOutput) -> None:
        super().__init__()
        self.topk_output = topk_output

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> StandardTopKOutput:
        return self.topk_output


def make_topk_output(row_count: int) -> StandardTopKOutput:
    return StandardTopKOutput(
        topk_weights=torch.rand(row_count, 2),
        topk_ids=torch.arange(row_count * 2, dtype=torch.int32).view(row_count, 2),
        router_logits=torch.rand(row_count, 2 * row_count),
    )


def test_padded_decode_rows_route_to_the_experts_of_row_zero() -> None:
    topk_output = make_topk_output(4)
    live_rows = DecodeLiveRows(is_live_row=torch.tensor([True, True, False, False]))

    routed = PaddedRowsTopK(FixedTopK(topk_output), live_rows)(
        torch.zeros(4, 8), topk_output.router_logits
    )

    expected_ids = torch.tensor([[0, 1], [2, 3], [0, 1], [0, 1]], dtype=torch.int32)
    assert torch.equal(routed.topk_ids, expected_ids)
    assert routed.topk_weights is topk_output.topk_weights


def test_topk_without_a_decode_mask_returns_the_routing_unchanged() -> None:
    topk_output = make_topk_output(3)

    routed = PaddedRowsTopK(FixedTopK(topk_output), DecodeLiveRows())(
        torch.zeros(3, 8), topk_output.router_logits
    )

    assert routed is topk_output


class RecordingTextModel(torch.nn.Module):
    def __init__(
        self, live_rows: DecodeLiveRows, error: RuntimeError | None = None
    ) -> None:
        super().__init__()
        self.live_rows = live_rows
        self.error = error
        self.seen_masks: list[torch.Tensor | None] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimpleNamespace,
        input_embeds: torch.Tensor | None,
        pp_proxy_tensors: None,
        input_deepstack_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        self.seen_masks.append(self.live_rows.is_live_row)
        if self.error is not None:
            raise self.error
        else:
            pass
        return torch.zeros((input_ids.shape[0], 3))


def make_bare_wrapper(error: RuntimeError | None = None) -> Qwen3OmniThinkerForCausalLM:
    wrapper = object.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.decode_live_rows = DecodeLiveRows()
    wrapper.model = RecordingTextModel(wrapper.decode_live_rows, error)
    wrapper.logits_processor = lambda *args: args[1]
    wrapper.lm_head = object()
    wrapper.fused_rope_gate = None
    return wrapper


def run_forward(
    wrapper: Qwen3OmniThinkerForCausalLM,
    forward_mode: ForwardMode,
    out_cache_loc: list[int],
) -> None:
    wrapper.forward(
        input_ids=torch.ones(len(out_cache_loc), dtype=torch.long),
        positions=torch.zeros(len(out_cache_loc), dtype=torch.long),
        forward_batch=SimpleNamespace(
            mrope_positions=None,
            forward_mode=forward_mode,
            out_cache_loc=torch.tensor(out_cache_loc),
        ),
    )


@pytest.mark.parametrize(
    ("forward_mode", "out_cache_loc", "expected_live_rows"),
    [
        (ForwardMode.DECODE, [7, 3, 0, 0], [True, True, False, False]),
        (ForwardMode.DECODE, [7], None),
        (ForwardMode.EXTEND, [7, 3, 0, 0], None),
    ],
)
def test_thinker_forward_marks_decode_rows_by_their_kv_slot(
    forward_mode: ForwardMode,
    out_cache_loc: list[int],
    expected_live_rows: list[bool] | None,
) -> None:
    wrapper = make_bare_wrapper()

    run_forward(wrapper, forward_mode, out_cache_loc)

    (seen_mask,) = wrapper.model.seen_masks
    if expected_live_rows is None:
        assert seen_mask is None
    else:
        assert seen_mask.tolist() == expected_live_rows
    assert wrapper.decode_live_rows.is_live_row is None


def test_decode_mask_ends_with_its_forward_so_a_direct_prefill_routes_as_before() -> (
    None
):
    wrapper = make_bare_wrapper()
    run_forward(wrapper, ForwardMode.DECODE, [7, 3])
    prefill_routing = make_topk_output(5)

    routed = PaddedRowsTopK(FixedTopK(prefill_routing), wrapper.decode_live_rows)(
        torch.zeros(5, 8), prefill_routing.router_logits
    )

    assert routed is prefill_routing


def test_decode_mask_is_cleared_when_the_model_call_fails() -> None:
    wrapper = make_bare_wrapper(RuntimeError("forward failed"))

    with pytest.raises(RuntimeError, match="forward failed"):
        run_forward(wrapper, ForwardMode.DECODE, [7, 3, 0, 0])

    assert wrapper.decode_live_rows.is_live_row is None


@pytest.mark.parametrize("quant_config", [None, Fp8Config()])
def test_thinker_routes_every_unquantized_moe_layer_through_the_shared_decode_mask(
    monkeypatch: pytest.MonkeyPatch, quant_config: Fp8Config | None
) -> None:
    def make_layer(mlp: torch.nn.Module) -> Qwen3MoeDecoderLayer:
        layer = object.__new__(Qwen3MoeDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.mlp = mlp
        return layer

    moe_block = object.__new__(Qwen3MoeSparseMoeBlock)
    torch.nn.Module.__init__(moe_block)
    moe_topk = torch.nn.Identity()
    moe_block.topk = moe_topk
    dense_mlp = torch.nn.Identity()

    class LayeredTextModel(torch.nn.Module):
        def __init__(self, *, config, quant_config, prefix):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size, config.hidden_size
            )
            self.layers = torch.nn.ModuleList(
                [make_layer(moe_block), make_layer(dense_mlp)]
            )

    monkeypatch.setattr(sglang_thinker_module, "Qwen3MoeLLMModel", LayeredTextModel)
    monkeypatch.setattr(
        sglang_thinker_module, "LogitsProcessor", lambda config: torch.nn.Identity()
    )
    text_config = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )

    wrapper = Qwen3OmniThinkerForCausalLM(
        Qwen3OmniMoeThinkerConfig(text_config=text_config), quant_config=quant_config
    )

    routed_topk = wrapper.model.layers[0].mlp.topk
    if quant_config is None:
        assert isinstance(routed_topk, PaddedRowsTopK)
        assert routed_topk.topk is moe_topk
        assert routed_topk.live_rows is wrapper.decode_live_rows
    else:
        assert routed_topk is moe_topk
        assert wrapper.decode_live_rows is None
    assert wrapper.model.layers[1].mlp is dense_mlp


def thinker_with_moe(
    runner_backend: MoeRunnerBackend,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    top_k: int,
    forward_normal: Callable[[torch.Tensor], torch.Tensor | None],
) -> Qwen3OmniThinkerForCausalLM:
    quant_method = object.__new__(UnquantizedFusedMoEMethod)
    quant_method.runner = SimpleNamespace(runner_backend=runner_backend)
    moe_block = object.__new__(Qwen3MoeSparseMoeBlock)
    torch.nn.Module.__init__(moe_block)
    moe_block.experts = SimpleNamespace(
        quant_method=quant_method, w13_weight=w13_weight, w2_weight=w2_weight
    )
    moe_block.forward_normal = forward_normal
    layer = object.__new__(Qwen3MoeDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.mlp = moe_block
    wrapper = object.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(layers=[layer])
    wrapper.config = SimpleNamespace(
        num_experts_per_tok=top_k, hidden_size=w13_weight.shape[2]
    )
    return wrapper


def test_precompile_runs_no_moe_pass_on_a_backend_other_than_triton() -> None:
    token_counts = []
    wrapper = thinker_with_moe(
        MoeRunnerBackend.FLASHINFER_CUTLASS,
        torch.zeros(128, 16, 8, dtype=torch.bfloat16),
        torch.zeros(128, 8, 8, dtype=torch.bfloat16),
        8,
        lambda hidden: token_counts.append(hidden.shape[0]),
    )

    wrapper.precompile_kernels_after_loading()

    assert token_counts == []


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused MoE kernel compiles on CUDA"
)
def test_no_token_count_up_to_the_ceiling_compiles_a_fused_moe_kernel_after_precompile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    num_experts, top_k, hidden_size, intermediate_size, ceiling = 128, 8, 128, 64, 600
    generator = torch.Generator(device=device).manual_seed(0)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            generator=generator,
        )
        / 16
    ).to(torch.bfloat16)
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            generator=generator,
        )
        / 16
    ).to(torch.bfloat16)
    router = torch.randn(hidden_size, num_experts, device=device, generator=generator)
    runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        inplace=False,
    )

    def forward_normal(hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = hidden_states.float() @ router
        topk_weights, topk_ids = torch.topk(
            torch.softmax(router_logits, dim=-1), top_k, dim=-1
        )
        return fused_experts(
            hidden_states,
            w13_weight,
            w2_weight,
            StandardTopKOutput(topk_weights, topk_ids.to(torch.int32), router_logits),
            runner_config,
        )

    monkeypatch.setattr(
        sglang_thinker_module, "max_prefill_buffer_tokens", lambda: ceiling
    )
    wrapper = thinker_with_moe(
        MoeRunnerBackend.TRITON, w13_weight, w2_weight, top_k, forward_normal
    )
    with get_context().override_server_args():
        wrapper.precompile_kernels_after_loading()
        compiled = fused_moe_kernel.device_caches[torch.cuda.current_device()][0]
        variants = len(compiled)
        for num_tokens in range(1, ceiling + 1):
            forward_normal(
                torch.randn(
                    num_tokens, hidden_size, device=device, generator=generator
                ).to(torch.bfloat16)
            )

    assert variants > 0
    assert len(compiled) == variants
