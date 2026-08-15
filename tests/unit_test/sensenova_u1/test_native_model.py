# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.sensenova_u1.config import (
    SenseNovaU1NativeServingPipelineConfig,
    SenseNovaU1NativePipelineConfig,
    Variants,
)


MODEL_PATH = Path("/mnt/afs/fanyijiat/models/SenseNova-U1-8B-MoT-Interleaved-bd39")


def test_sensenova_u1_native_pipeline_variant_is_registered() -> None:
    assert PIPELINE_CONFIG_REGISTRY.get_config("NEOChatModel").architecture == "NEOChatModel"
    assert Variants["native"] is SenseNovaU1NativePipelineConfig
    native = SenseNovaU1NativePipelineConfig(model_path="model")
    native_serving = SenseNovaU1NativeServingPipelineConfig(model_path="model")
    assert native.entry_stage == "u1_native"
    assert native_serving.entry_stage == "u1_native_serving"
    assert native.stages[0].factory.endswith(
        "create_sensenova_u1_native_executor"
    )
    assert native_serving.stages[0].factory.endswith(
        "create_sensenova_u1_native_serving_executor"
    )
    assert native_serving.stages[0].factory_args["max_concurrency"] > 1
    assert native_serving.stages[0].factory_args["max_running_requests"] >= (
        native_serving.stages[0].factory_args["max_concurrency"]
    )
    assert Variants["native_serving"] is SenseNovaU1NativeServingPipelineConfig


def test_sensenova_u1_native_expected_language_keys_match_checkpoint() -> None:
    from sglang_omni.models.sensenova_u1.sglang_model import (
        expected_language_weight_keys,
        load_u1_llm_config,
    )

    config = load_u1_llm_config(MODEL_PATH)
    expected = expected_language_weight_keys(config)
    index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    language_keys = {
        key[len("language_model.") :]
        for key in index
        if key.startswith("language_model.")
    }

    assert len(language_keys) == 1096
    assert language_keys == expected


def test_sensenova_u1_native_vision_loader_maps_checkpoint_keys() -> None:
    from sglang_omni.models.sensenova_u1.native_vision import (
        SenseNovaU1NativeVisionModel,
    )

    model = SenseNovaU1NativeVisionModel.from_model_path(
        MODEL_PATH,
        params_dtype="float32",
    )
    report = model.load_weights(MODEL_PATH)

    assert report.ok
    assert len(report.loaded_keys) == 4


def test_sensenova_u1_native_vision_refreshes_rope_cache_after_dtype_cast() -> None:
    from sglang_omni.models.sensenova_u1.native_vision import (
        SenseNovaU1NativeVisionModel,
    )

    model = SenseNovaU1NativeVisionModel.from_model_path(
        MODEL_PATH,
        params_dtype="bfloat16",
    )
    model.to(dtype=torch.bfloat16)
    assert model.cos_cached_x.dtype == torch.bfloat16

    model._ensure_fp32_rope_cache(torch.device("cpu"))

    assert model.cos_cached_x.dtype == torch.float32
    assert model.sin_cached_x.dtype == torch.float32
    assert model.cos_cached_y.dtype == torch.float32
    assert model.sin_cached_y.dtype == torch.float32
    assert model.cos_cached_x[0, 0].item() == pytest.approx(1.0)


def test_sensenova_u1_interleave_uses_verified_text_decode_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.sensenova_u1.interleave import (
        SenseNovaU1InterleaveRunner,
    )

    env_name = "SENSENOVA_U1_NATIVE_INTERLEAVE_EAGER_TEXT_DECODE"
    monkeypatch.delenv(env_name, raising=False)
    assert (
        SenseNovaU1InterleaveRunner._native_interleave_text_decode_mode()
        == "eager_text_decode"
    )

    monkeypatch.setenv(env_name, "0")
    assert (
        SenseNovaU1InterleaveRunner._native_interleave_text_decode_mode()
        == "sglang_cached_decode"
    )


def test_sensenova_u1_separate_qkv_defaults_to_single_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.sensenova_u1.sglang_model import (
        SenseNovaU1NativeAttention,
    )

    env_name = "SENSENOVA_U1_NATIVE_SEPARATE_QKV_PROJ"
    monkeypatch.delenv(env_name, raising=False)
    attention = object.__new__(SenseNovaU1NativeAttention)
    attention.layer_id = 0
    attention.tp_size = 1
    assert attention._use_separate_qkv_projection()

    attention.tp_size = 2
    assert not attention._use_separate_qkv_projection()

    monkeypatch.setenv(env_name, "1")
    assert attention._use_separate_qkv_projection()

    attention.tp_size = 1
    monkeypatch.setenv(env_name, "0")
    assert not attention._use_separate_qkv_projection()


def test_sensenova_u1_hf_runner_forces_requested_transformers_backend() -> None:
    from sglang_omni.models.sensenova_u1.hf_runner import (
        _force_official_llm_attn_implementation,
    )

    shared_config = SimpleNamespace(_attn_implementation="eager")
    language_model = torch.nn.Module()
    language_model.config = shared_config
    language_model.model = SimpleNamespace(config=shared_config)
    model = SimpleNamespace(
        config=SimpleNamespace(llm_config=shared_config),
        language_model=language_model,
    )

    _force_official_llm_attn_implementation(model, "sdpa")

    assert shared_config._attn_implementation == "sdpa"


def test_sensenova_u1_flow_prefix_cache_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.sensenova_u1.flow_matching import (
        SenseNovaU1FlowMatchingRunner,
    )

    env_name = "SENSENOVA_U1_NATIVE_FLOW_PREFIX_CACHE"
    monkeypatch.delenv(env_name, raising=False)
    assert SenseNovaU1FlowMatchingRunner._native_flow_prefix_cache_enabled()

    monkeypatch.setenv(env_name, "0")
    assert not SenseNovaU1FlowMatchingRunner._native_flow_prefix_cache_enabled()
    monkeypatch.setenv(env_name, "1")
    assert SenseNovaU1FlowMatchingRunner._native_flow_prefix_cache_enabled()

    graph_env_name = "SENSENOVA_U1_NATIVE_FLOW_PREFILL_CUDA_GRAPH"
    monkeypatch.delenv(graph_env_name, raising=False)
    assert SenseNovaU1FlowMatchingRunner._native_flow_prefill_cuda_graph_enabled()

    monkeypatch.setenv(graph_env_name, "false")
    assert not SenseNovaU1FlowMatchingRunner._native_flow_prefill_cuda_graph_enabled()

    eager_graph_env_name = "SENSENOVA_U1_NATIVE_EAGER_TEXT_FULL_LOOP_CUDA_GRAPH"
    monkeypatch.delenv(eager_graph_env_name, raising=False)
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_full_loop_cuda_graph_enabled()
    )
    monkeypatch.setenv(eager_graph_env_name, "off")
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_full_loop_cuda_graph_enabled()
    )

    eager_prefix_env_name = "SENSENOVA_U1_NATIVE_EAGER_TEXT_PREFIX_CACHE"
    monkeypatch.delenv(eager_prefix_env_name, raising=False)
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_prefix_cache_enabled()
    )
    monkeypatch.setenv(eager_prefix_env_name, "0")
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_prefix_cache_enabled()
    )

    graph_mode_env_name = "SENSENOVA_U1_NATIVE_EAGER_TEXT_GRAPH_MODE"
    monkeypatch.delenv(graph_mode_env_name, raising=False)
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_graph_mode()
        == "segmented"
    )
    monkeypatch.setenv(graph_mode_env_name, "monolithic")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_graph_mode()
        == "monolithic"
    )

    repeated_kv_env_name = "SENSENOVA_U1_NATIVE_EAGER_REPEATED_KV_CACHE"
    monkeypatch.delenv(repeated_kv_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_repeated_kv_cache_enabled()
    )
    monkeypatch.setenv(repeated_kv_env_name, "1")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_repeated_kv_cache_enabled()
    )

    static_kv_env_name = "SENSENOVA_U1_NATIVE_EAGER_STATIC_KV_CACHE"
    monkeypatch.delenv(static_kv_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_static_kv_cache_enabled()
    )
    monkeypatch.setenv(static_kv_env_name, "yes")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_static_kv_cache_enabled()
    )

    bf16_argmax_env_name = "SENSENOVA_U1_NATIVE_EAGER_BF16_ARGMAX"
    monkeypatch.delenv(bf16_argmax_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_bf16_argmax_enabled()
    )
    monkeypatch.setenv(bf16_argmax_env_name, "on")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_bf16_argmax_enabled()
    )

    lm_head_env_name = "SENSENOVA_U1_NATIVE_EAGER_LM_HEAD_LINEAR"
    monkeypatch.delenv(lm_head_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_lm_head_linear_enabled()
    )
    monkeypatch.setenv(lm_head_env_name, "true")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_lm_head_linear_enabled()
    )

    fast_result_env_name = "SENSENOVA_U1_NATIVE_EAGER_FAST_RESULT"
    monkeypatch.delenv(fast_result_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_fast_result_enabled()
    )
    monkeypatch.setenv(fast_result_env_name, "true")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_fast_result_enabled()
    )

    direct_embedding_env_name = "SENSENOVA_U1_NATIVE_EAGER_DIRECT_EMBEDDING"
    monkeypatch.delenv(direct_embedding_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_direct_embedding_enabled()
    )
    monkeypatch.setenv(direct_embedding_env_name, "1")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_direct_embedding_enabled()
    )

    compiled_add_rms_env_name = "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS"
    monkeypatch.delenv(compiled_add_rms_env_name, raising=False)
    assert not (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_compiled_add_rms_enabled()
    )
    monkeypatch.setenv(compiled_add_rms_env_name, "true")
    assert (
        __import__(
            "sglang_omni.models.sensenova_u1.native_serving",
            fromlist=["SenseNovaU1NativeServingExecutor"],
        )
        .SenseNovaU1NativeServingExecutor
        ._native_eager_compiled_add_rms_enabled()
    )


def test_sensenova_u1_flow_skips_cached_prefix_refresh() -> None:
    from sglang_omni.models.sensenova_u1.flow_matching import (
        NativeFlowPrefix,
        SenseNovaU1FlowMatchingRunner,
    )

    class FakeExecutor:
        @staticmethod
        def cached_prefix_length(
            *,
            input_ids: torch.Tensor,
            cache_extra_key: str | None,
        ) -> int:
            assert cache_extra_key == "image-key"
            return int(input_ids.numel())

        @staticmethod
        def run_prefill(**_: object) -> None:
            raise AssertionError("cached prefix must not run another forward")

    runner = object.__new__(SenseNovaU1FlowMatchingRunner)
    runner.executor = FakeExecutor()
    prefix = NativeFlowPrefix(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        indexes=torch.zeros((3, 3), dtype=torch.long),
        image_token_tag=torch.tensor([False, True, False]),
        input_embeds=torch.zeros((3, 4)),
        cache_extra_key="image-key",
        cache_insert_log={},
        cache_reuse_enabled=True,
    )

    runner._prime_prefix_cache(prefix)

    assert prefix.cache_insert_log == {
        "skipped": True,
        "reason": "static_prefix_already_cached",
        "prefix_tokens": 3,
        "image_token_count": 1,
        "cache_extra_key": "image-key",
        "cache_hit_tokens": 3,
    }


def test_sensenova_u1_cached_decode_sampler_suppresses_tokens() -> None:
    from sglang_omni.models.sensenova_u1.native_serving import (
        SenseNovaU1NativeServingExecutor,
    )

    batch_result = SimpleNamespace(
        next_token_ids=torch.tensor([2]),
        logits_output=SimpleNamespace(
            next_token_logits=torch.tensor([[0.0, 3.0, 5.0, 4.0]])
        ),
    )

    assert SenseNovaU1NativeServingExecutor._sample_next_token_ids(
        batch_result
    ) == [2]
    assert SenseNovaU1NativeServingExecutor._sample_next_token_ids(
        batch_result,
        suppress_token_ids=[2],
    ) == [3]


def test_sensenova_u1_rms_norm_matches_u1_hf_cast_order() -> None:
    from sglang_omni.models.sensenova_u1.sglang_model import _rms_norm
    from sglang_omni.vendor.sglang.layers import RMSNorm

    norm = RMSNorm(4, eps=1e-6)
    norm.weight.data = torch.tensor([1.0, 1.5, 0.75, -2.0], dtype=torch.bfloat16)
    x = torch.tensor(
        [[0.125, -1.5, 3.0, -8.0], [2.25, -0.5, 0.75, 1.0]],
        dtype=torch.bfloat16,
    )

    xf = x.float()
    expected = norm.weight * (
        xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    ).to(x.dtype)

    assert torch.equal(_rms_norm(norm, x), expected)


def test_sensenova_u1_native_tiny_forward_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.sensenova_u1.sglang_model import (
        SenseNovaU1NativeForCausalLM,
        assert_no_hf_modeling_imported,
        block_hf_modeling_imports,
    )

    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        max_position_embeddings=128,
        max_position_embeddings_hw=128,
        rope_theta=5000000.0,
        rope_theta_hw=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=False,
    )
    with block_hf_modeling_imports():
        model = SenseNovaU1NativeForCausalLM(
            config,
            params_dtype="float32",
            standalone=True,
        )
        assert model.is_mrope_enabled
        monkeypatch.setenv(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS",
            "1",
        )
        monkeypatch.setenv(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS_LAYERS",
            "30-41",
        )
        assert not model.model.layers[0]._use_compiled_eager_add_rms()
        monkeypatch.setenv(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS_LAYERS",
            "0",
        )
        assert model.model.layers[0]._use_compiled_eager_add_rms()
        monkeypatch.delenv(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS",
            raising=False,
        )
        monkeypatch.delenv(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS_LAYERS",
            raising=False,
        )
        torch.manual_seed(0)
        for param in model.parameters():
            param.data.normal_(mean=0.0, std=0.02)
        model.eval()
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        positions = torch.arange(input_ids.numel(), dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids, positions, None)
            generated_output = model(
                input_ids,
                positions,
                None,
                image_gen_indicators=torch.ones(
                    input_ids.numel(),
                    dtype=torch.bool,
                ),
            )
            indexes = torch.stack(
                [
                    positions,
                    torch.zeros_like(positions),
                    torch.zeros_like(positions),
                ]
            )
            base_hidden, base_caches = model.model.eager_text_prefill_with_cache(
                input_ids,
                positions,
                indexes=indexes,
            )
            repeated_hidden, repeated_caches = (
                model.model.eager_text_prefill_with_cache(
                    input_ids,
                    positions,
                    indexes=indexes,
                    repeat_kv_cache=True,
                )
            )
            next_id = torch.tensor([4], dtype=torch.long)
            next_indexes = torch.tensor([[3], [0], [0]], dtype=torch.long)
            base_decode, base_next_caches = (
                model.model.eager_text_decode_with_cache(
                    next_id,
                    next_indexes[0],
                    base_caches,
                    indexes=next_indexes,
                )
            )
            repeated_decode, repeated_next_caches = (
                model.model.eager_text_decode_with_cache(
                    next_id,
                    next_indexes[0],
                    repeated_caches,
                    indexes=next_indexes,
                    repeat_kv_cache=True,
                )
            )
            static_caches = [
                (
                    torch.empty(
                        (4, *cache_k.shape[1:]),
                        dtype=cache_k.dtype,
                    ),
                    torch.empty(
                        (4, *cache_v.shape[1:]),
                        dtype=cache_v.dtype,
                    ),
                )
                for cache_k, cache_v in repeated_caches
            ]
            for (static_k, static_v), (cache_k, cache_v) in zip(
                static_caches,
                repeated_caches,
            ):
                static_k[:3].copy_(cache_k)
                static_v[:3].copy_(cache_v)
            static_decode, static_next_caches = (
                model.model.eager_text_decode_with_static_cache(
                    next_id,
                    next_indexes[0],
                    static_caches,
                    cache_position=3,
                    indexes=next_indexes,
                    repeat_kv_cache=True,
                )
            )
            eager_matmul_logits = model.eager_text_logits(base_hidden)
            monkeypatch.setenv(
                "SENSENOVA_U1_NATIVE_EAGER_LM_HEAD_LINEAR",
                "1",
            )
            eager_linear_logits = model.eager_text_logits(base_hidden)
            eager_embedding = model.model._eager_embed(input_ids)
            direct_embedding = torch.nn.functional.embedding(
                input_ids,
                model.model.embed_tokens.weight[: config.vocab_size],
            )
    assert_no_hf_modeling_imported(context="tiny native forward")
    assert output.next_token_logits is not None
    assert tuple(output.next_token_logits.shape) == (1, 32)
    assert torch.isfinite(output.next_token_logits).all()
    assert generated_output.next_token_logits is not None
    assert torch.isfinite(generated_output.next_token_logits).all()
    assert torch.equal(base_hidden, repeated_hidden)
    assert torch.equal(base_decode, repeated_decode)
    assert torch.equal(repeated_decode, static_decode)
    assert torch.equal(eager_matmul_logits, eager_linear_logits)
    assert torch.equal(eager_embedding, direct_embedding)
    bf16_scores = torch.tensor(
        [[1.0, 2.0, 1.5, 2.0]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(
        torch.argmax(bf16_scores, dim=-1),
        torch.argmax(bf16_scores.float(), dim=-1),
    )
    attention = model.model.layers[0].self_attn
    assert torch.equal(
        repeated_caches[0][0],
        attention.repeat_eager_kv_cache(base_caches[0][0]),
    )
    assert torch.equal(
        repeated_caches[0][1],
        attention.repeat_eager_kv_cache(base_caches[0][1]),
    )
    assert torch.equal(
        repeated_next_caches[0][0],
        attention.repeat_eager_kv_cache(base_next_caches[0][0]),
    )
    assert torch.equal(
        repeated_next_caches[0][1],
        attention.repeat_eager_kv_cache(base_next_caches[0][1]),
    )
    assert torch.equal(
        static_next_caches[0][0][:4],
        repeated_next_caches[0][0],
    )
    assert torch.equal(
        static_next_caches[0][1][:4],
        repeated_next_caches[0][1],
    )


def test_sensenova_u1_backend_mask_layout_matches_dense_reference() -> None:
    from sglang_omni.models.sensenova_u1.hybrid_attention import (
        build_u1_hybrid_allowed_matrix,
        build_u1_hybrid_backend_mask,
    )

    indexes = torch.tensor(
        [
            [0, 1, 2, 2, 2, 3, 4],
            [0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 1, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    image_token_tag = torch.tensor([0, 0, 1, 1, 1, 0, 0], dtype=torch.bool)
    mask, indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_token_tag,
        [7],
        [0],
    )
    dense = build_u1_hybrid_allowed_matrix(indexes[0], image_token_tag)

    assert mask is not None
    assert indptr.tolist() == [0, 49]
    assert torch.equal(mask.bool().view(7, 7), dense)


def test_sensenova_u1_backend_mask_allows_image_span_across_kernel_blocks() -> None:
    from sglang_omni.models.sensenova_u1.hybrid_attention import (
        build_u1_hybrid_allowed_matrix,
        build_u1_hybrid_backend_mask,
    )

    prefix_len = 12
    image_len = 144
    suffix_len = 4
    total_len = prefix_len + image_len + suffix_len
    t_indexes = torch.arange(total_len, dtype=torch.long)
    t_indexes[prefix_len : prefix_len + image_len] = prefix_len
    indexes = torch.stack(
        [
            t_indexes,
            torch.zeros(total_len, dtype=torch.long),
            torch.zeros(total_len, dtype=torch.long),
        ]
    )
    image_token_tag = torch.zeros(total_len, dtype=torch.bool)
    image_token_tag[prefix_len : prefix_len + image_len] = True

    mask, indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_token_tag,
        [total_len],
        [0],
    )
    dense = build_u1_hybrid_allowed_matrix(indexes[0], image_token_tag)

    assert mask is not None
    assert indptr.tolist() == [0, total_len * total_len]
    assert torch.equal(mask.bool().view(total_len, total_len), dense)
    first_image_row = prefix_len
    last_image_col = prefix_len + image_len - 1
    assert mask.bool().view(total_len, total_len)[first_image_row, last_image_col]


def test_sensenova_u1_backend_mask_allows_cached_prefix_and_full_generated_image() -> None:
    from sglang_omni.models.sensenova_u1.hybrid_attention import (
        build_u1_hybrid_backend_mask,
    )

    prefix_len = 37
    image_len = 64
    indexes = torch.stack(
        [
            torch.full((image_len,), prefix_len, dtype=torch.long),
            torch.arange(image_len, dtype=torch.long) // 8,
            torch.arange(image_len, dtype=torch.long) % 8,
        ]
    )
    image_token_tag = torch.ones(image_len, dtype=torch.bool)

    mask, indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_token_tag,
        [image_len],
        [prefix_len],
    )

    assert mask is not None
    assert indptr.tolist() == [0, image_len * (prefix_len + image_len)]
    assert torch.all(mask.bool().view(image_len, prefix_len + image_len))


def test_sensenova_u1_attention_adapter_owns_custom_mask_policy() -> None:
    from sglang_omni.models.sensenova_u1.attention_backend import (
        _build_extend_wrapper,
        _inject_custom_mask_metadata,
    )

    captured = {}

    def original(
        *,
        custom_mask,
        is_causal,
        kv_indices,
        max_len_extend,
        skip_prefix=False,
        skip_extend=False,
        lse_extend=None,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
        sliding_window_size=-1,
        logit_cap=0.0,
        xai_temperature_len=-1,
        page_size=1,
    ):
        captured["is_causal"] = is_causal
        return "delegated"

    wrapped = _build_extend_wrapper(original)
    result = wrapped(
        custom_mask=torch.ones(1, dtype=torch.uint8),
        is_causal=True,
        kv_indices=torch.tensor([1], dtype=torch.long),
        max_len_extend=1,
    )
    assert result == "delegated"
    assert captured["is_causal"] is False

    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(custom_mask=None, mask_indptr=None),
        mask_indptr=torch.zeros(3, dtype=torch.int64),
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        extend_seq_lens=torch.tensor([3, 2], dtype=torch.int64),
        extend_prefix_lens=torch.tensor([4, 5], dtype=torch.int64),
        cross_attention_custom_mask=torch.ones(35, dtype=torch.uint8),
    )
    _inject_custom_mask_metadata(backend, forward_batch)
    assert backend.forward_metadata.custom_mask is (
        forward_batch.cross_attention_custom_mask
    )
    assert backend.forward_metadata.mask_indptr.tolist() == [0, 21, 35]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sensenova_u1_custom_mask_extend_attention_matches_dense_no_prefix() -> None:
    from sglang.kernels.ops.attention.extend_attention import (
        extend_attention_fwd as upstream_extend_attention_fwd,
    )

    from sglang_omni.models.sensenova_u1.attention_backend import (
        _build_extend_wrapper,
    )
    from sglang_omni.models.sensenova_u1.hybrid_attention import (
        build_u1_hybrid_allowed_matrix,
        build_u1_hybrid_backend_mask,
    )

    extend_attention_fwd = _build_extend_wrapper(
        upstream_extend_attention_fwd
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    prefix_len = 12
    image_len = 144
    suffix_len = 4
    total_len = prefix_len + image_len + suffix_len
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16
    scaling = head_dim**-0.5

    t_indexes = torch.arange(total_len, device=device, dtype=torch.long)
    t_indexes[prefix_len : prefix_len + image_len] = prefix_len
    indexes = torch.stack(
        [
            t_indexes,
            torch.zeros(total_len, device=device, dtype=torch.long),
            torch.zeros(total_len, device=device, dtype=torch.long),
        ]
    )
    image_token_tag = torch.zeros(total_len, device=device, dtype=torch.bool)
    image_token_tag[prefix_len : prefix_len + image_len] = True

    q = torch.randn(
        total_len,
        num_q_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        total_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        total_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_rep = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v_rep = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    allowed = build_u1_hybrid_allowed_matrix(indexes[0], image_token_tag)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_rep.float()) * scaling
    scores = scores.masked_fill(
        ~allowed.unsqueeze(0),
        torch.finfo(scores.dtype).min,
    )
    expected = torch.einsum(
        "hqk,khd->qhd",
        torch.softmax(scores, dim=-1).to(v.dtype),
        v_rep,
    )

    custom_mask, mask_indptr = build_u1_hybrid_backend_mask(
        indexes,
        image_token_tag,
        [total_len],
        [0],
    )
    assert custom_mask is not None
    actual = torch.empty_like(q)
    extend_attention_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        actual,
        torch.empty(1, num_kv_heads, head_dim, device=device, dtype=k.dtype),
        torch.empty(1, num_kv_heads, head_dim, device=device, dtype=v.dtype),
        torch.tensor([0, total_len], device=device, dtype=torch.int64),
        torch.tensor([0, 0], device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        custom_mask,
        True,
        mask_indptr,
        total_len,
        1.0,
        1.0,
        sm_scale=scaling,
    )

    assert torch.equal(actual, expected)


def test_sensenova_u1_prepare_forward_batch_installs_metadata() -> None:
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from sglang_omni.models.sensenova_u1.sglang_model import (
        SenseNovaU1NativeForCausalLM,
    )

    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        max_position_embeddings=128,
        max_position_embeddings_hw=128,
        rope_theta=5000000.0,
        rope_theta_hw=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=False,
    )
    model = SenseNovaU1NativeForCausalLM(
        config,
        params_dtype="float32",
        standalone=True,
    )
    indexes = torch.tensor(
        [
            [0, 1, 2, 2, 2, 3, 4],
            [0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 1, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    tag = torch.tensor([0, 0, 1, 1, 1, 0, 0], dtype=torch.bool)
    mm_inputs = MultimodalInputs(mm_items=[])
    mm_inputs.mrope_positions = indexes
    mm_inputs.u1_image_token_tag = tag
    fb = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        positions=torch.arange(7, dtype=torch.long),
        input_ids=torch.arange(7, dtype=torch.long),
        mrope_positions=indexes,
        extend_seq_lens_cpu=[7],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens=torch.tensor([7], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        mm_inputs=[mm_inputs],
        model_specific_states=None,
        cross_attention_custom_mask=None,
    )

    model.prepare_forward_batch(fb)  # type: ignore[arg-type]

    states = fb.model_specific_states["sensenova_u1"]
    assert torch.equal(states["indexes"], indexes)
    assert torch.equal(states["image_token_tag"], tag)
    assert fb.cross_attention_custom_mask is not None
    assert model.last_forward_batch_prepare["custom_mask_numel"] == 49


def test_sensenova_u1_native_stage_factory_fails_if_hf_modeling_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.sensenova_u1.stages import (
        create_sensenova_u1_flow_executor,
        create_sensenova_u1_interleave_executor,
        create_sensenova_u1_native_executor,
    )

    polluted_name = "sensenova_u1.models.neo_unify.modeling_neo_chat"
    monkeypatch.setitem(sys.modules, polluted_name, types.ModuleType(polluted_name))
    factories = [
        lambda: create_sensenova_u1_native_executor(
            str(MODEL_PATH),
            device="cpu",
            load_weights=False,
        ),
        lambda: create_sensenova_u1_flow_executor(
            str(MODEL_PATH),
            device="cpu",
        ),
        lambda: create_sensenova_u1_interleave_executor(
            str(MODEL_PATH),
            device="cpu",
        ),
    ]
    for factory in factories:
        with pytest.raises(
            RuntimeError,
            match="HF U1 modeling modules imported",
        ):
            factory()
