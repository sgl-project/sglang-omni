# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1 import neo_unify, stages
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    _copy_right_aligned_prefix_bnsd,
    _randn_with_seed,
    prepare_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    _flash_or_sdpa,
    _sdpa_attn_func,
    create_block_causal_mask,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    current_platform as model_platform,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    make_qwen3_rms_norm,
    npu_fia_available,
    npu_swiglu_available,
    position_ids_from_indexes,
)


def test_right_aligns_bnsd_prefix_for_npu_fia():
    source = torch.tensor(
        [
            [[[1], [2], [99], [99], [99]]],
            [[[3], [4], [5], [6], [7]]],
        ]
    )
    destination = torch.zeros(2, 1, 8, 1, dtype=source.dtype)

    _copy_right_aligned_prefix_bnsd(destination, source, [2, 5])

    assert destination[:, 0, :5, 0].tolist() == [
        [0, 0, 0, 1, 2],
        [3, 4, 5, 6, 7],
    ]
    assert destination[:, :, 5:].eq(0).all()


def _force_generator_fallback(monkeypatch, device_type):
    original_generator = torch.Generator

    def unsupported_device_generator(device="cpu"):
        if torch.device(device).type == device_type:
            raise RuntimeError(f"Generator is unsupported on {device_type}")
        return original_generator(device)

    monkeypatch.setattr(torch, "Generator", unsupported_device_generator)


def test_randn_fallback_preserves_cpu_rng(monkeypatch):
    _force_generator_fallback(monkeypatch, "cpu")
    rng_state = torch.get_rng_state().clone()

    first = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_randn_supports_per_sample_seeds():
    actual = _randn_with_seed(
        (2, 3, 4), device=torch.device("cpu"), dtype=torch.float32, seed=[7, 19]
    )
    expected = torch.cat(
        [
            _randn_with_seed(
                (1, 3, 4),
                device=torch.device("cpu"),
                dtype=torch.float32,
                seed=seed,
            )
            for seed in (7, 19)
        ]
    )

    assert torch.equal(actual, expected)


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, return_tensors):
        del return_tensors
        token_count = len(text.split()) + 1
        return {"input_ids": torch.arange(1, token_count + 1).unsqueeze(0)}


def test_builds_padded_batched_text_inputs():
    model = SimpleNamespace(device=torch.device("cpu"))

    input_ids, indexes, attention_mask, valid_mask, prefix_lengths = (
        NEOChatModel._build_t2i_text_inputs(
            model, _FakeTokenizer(), ["short", "a much longer prompt"]
        )
    )

    assert input_ids.shape == (2, 5)
    assert indexes.shape == (2, 3, 5)
    assert prefix_lengths.tolist() == [2, 5]
    assert valid_mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    mask = attention_mask["full_attention"]
    assert mask.shape == (2, 1, 5, 5)
    assert torch.isneginf(mask[0, :, :, 2:]).all()
    assert torch.isfinite(mask[0, :, :, :2]).any()


def test_position_indexes_support_batched_inputs():
    indexes = torch.tensor(
        [
            [[0, 1], [0, 0], [0, 0]],
            [[4, 4], [0, 1], [0, 0]],
        ]
    )

    assert torch.equal(position_ids_from_indexes(indexes, 0), indexes[:, 0])
    assert torch.equal(
        position_ids_from_indexes(indexes[0], 1), indexes[0, 1].unsqueeze(0)
    )


def test_singleton_text_matches_valid_batched_tokens():
    model = SimpleNamespace(device=torch.device("cpu"))
    tokenizer = _FakeTokenizer()
    batched = NEOChatModel._build_t2i_text_inputs(
        model, tokenizer, ["short", "a much longer prompt"]
    )
    for index, prompt in enumerate(["short", "a much longer prompt"]):
        single = NEOChatModel._build_t2i_text_inputs(model, tokenizer, prompt)
        length = single[0].shape[1]
        assert torch.equal(batched[0][index, :length], single[0][0])
        assert torch.equal(batched[1][index, :, :length], single[1])
        assert torch.equal(
            batched[2]["full_attention"][index, :, :length, :length],
            single[2]["full_attention"][0],
        )


def test_block_causal_mask_rejects_padded_keys():
    indexes = torch.tensor([[0, 1, 2], [0, 1, 2]])
    valid = torch.tensor([[True, True, False], [True, True, True]])

    mask = create_block_causal_mask(indexes, valid)

    assert mask.shape == (2, 1, 3, 3)
    assert torch.isneginf(mask[0, :, :, 2]).all()
    assert mask[1, 0, 2, 2] == 0


def test_builds_per_sample_image_indexes():
    indexes = NEOChatModel._build_t2i_image_indexes(
        SimpleNamespace(),
        token_h=2,
        token_w=2,
        text_len=torch.tensor([2, 5]),
        device=torch.device("cpu"),
    )

    assert indexes.shape == (2, 3, 4)
    assert indexes[:, 0].tolist() == [[2, 2, 2, 2], [5, 5, 5, 5]]
    assert indexes[:, 1].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert indexes[:, 2].tolist() == [[0, 1, 0, 1], [0, 1, 0, 1]]


def test_compacts_variable_length_kv_before_attention():
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    actual = _flash_or_sdpa(q, k, v, actual_seq_lengths_kv=[5, 8])

    expected_short = _sdpa_attn_func(
        q[:1],
        torch.cat((k[:1, :2], k[:1, 5:]), dim=1),
        torch.cat((v[:1, :2], v[:1, 5:]), dim=1),
    )
    expected_long = _sdpa_attn_func(q[1:], k[1:], v[1:])
    torch.testing.assert_close(actual, torch.cat((expected_short, expected_long)))


def test_sdpa_masks_padded_prefix_keys():
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 0.0]], [[0.0, 4.0]], [[100.0, 100.0]]]])
    key_mask = torch.tensor([[[[True, True, False]]]])

    actual = _sdpa_attn_func(q, k, v, attention_mask=key_mask)
    expected = _sdpa_attn_func(q, k[:, :2], v[:, :2])

    torch.testing.assert_close(actual, expected)


def test_randn_fallback_preserves_accelerator_rng(monkeypatch):
    device_type = model_platform.device_type
    if not device_type or device_type == "cpu":
        pytest.skip("No accelerator is available")

    device = torch.device(device_type, 0)
    device_module = torch.get_device_module(device)
    if not device_module.is_available():
        pytest.skip(f"{device_type} is not available")

    _force_generator_fallback(monkeypatch, device_type)
    cpu_rng_state = torch.get_rng_state().clone()
    device_rng_state = device_module.get_rng_state(device).clone()

    first = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    assert torch.equal(device_module.get_rng_state(device), device_rng_state)


@pytest.mark.parametrize(
    ("operator", "probe"),
    [
        ("npu_fused_infer_attention_score", npu_fia_available),
        ("npu_swiglu", npu_swiglu_available),
    ],
)
@pytest.mark.parametrize("available", [False, True])
def test_npu_operator_probes(monkeypatch, operator, probe, available):
    namespace = SimpleNamespace()
    if available:
        setattr(namespace, operator, object())
    monkeypatch.setattr(torch.ops, "npu", namespace, raising=False)

    assert probe() is available


@pytest.mark.parametrize(
    ("is_npu", "uses_native"),
    [(False, True), (True, False)],
)
def test_shared_rmsnorm_dispatch(monkeypatch, is_npu, uses_native):
    monkeypatch.setattr(model_platform, "is_npu", lambda: is_npu)

    norm = make_qwen3_rms_norm(64, eps=1e-6)

    assert isinstance(norm, RMSNorm)
    assert norm.cast_x_before_out_mul
    assert (norm._forward_method == norm.forward_native) is uses_native


@torch.no_grad()
def test_fused_dense_mlp_matches_original(monkeypatch):
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="silu",
    )
    with torch.random.fork_rng():
        torch.manual_seed(37)
        mlp = Qwen3MLP(config).eval()
        hidden_states = torch.randn(2, 5, config.hidden_size)
        expected = mlp(hidden_states)

    monkeypatch.setattr(mlp, "_use_npu_fused_mlp", lambda _x: True)
    monkeypatch.setattr(
        torch.ops,
        "npu",
        SimpleNamespace(
            npu_swiglu=lambda x, dim=-1: (
                F.silu(x.chunk(2, dim=dim)[0]) * x.chunk(2, dim=dim)[1]
            )
        ),
        raising=False,
    )
    actual = mlp(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert set(mlp.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }
    assert (
        mlp.gate_proj.weight.untyped_storage().data_ptr()
        == mlp.up_proj.weight.untyped_storage().data_ptr()
    )


def test_batched_gqa_matches_unpadded_singletons():
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 2:5] = False
    attention_mask = valid[:, None, None, :].expand(-1, -1, q.shape[1], -1)

    actual = _sdpa_attn_func(q, k, v, attention_mask=attention_mask)

    for index in range(2):
        expected = _sdpa_attn_func(
            q[index : index + 1],
            k[index : index + 1, valid[index]],
            v[index : index + 1, valid[index]],
        )
        torch.testing.assert_close(actual[index : index + 1], expected)


@torch.no_grad()
def test_prefix_and_denoise_attention_match_singletons():
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(23)
        attention = Qwen3Attention(config, layer_idx=0).eval()
    generator = torch.Generator().manual_seed(31)
    text = torch.randn(2, 5, 64, generator=generator)
    image = torch.randn(2, 3, 64, generator=generator)
    helper = SimpleNamespace(device=torch.device("cpu"))

    def run(prefix, lengths, image_states):
        batch_size, width, _ = prefix.shape
        positions = torch.arange(width).expand(batch_size, -1)
        indexes = torch.stack(
            [positions, torch.zeros_like(positions), torch.zeros_like(positions)],
            dim=1,
        )
        valid = positions < torch.tensor(lengths)[:, None]
        cache = DynamicCache(config=config)
        attention.forward_und(
            prefix, indexes, create_block_causal_mask(positions, valid), cache
        )
        prefix_keys = cache.layers[0].keys.clone()
        prepare_flash_kv_cache(
            cache,
            current_len=3,
            batch_size=batch_size,
            prefix_lengths=torch.tensor(lengths),
        )
        image_indexes = NEOChatModel._build_t2i_image_indexes(
            helper, 1, 3, torch.tensor(lengths), torch.device("cpu")
        )
        outputs = []
        for _ in range(2):
            image_states, _ = attention.forward_gen(
                image_states,
                image_indexes,
                None,
                cache,
                update_cache=False,
            )
            outputs.append(image_states)
        torch.testing.assert_close(cache.layers[0].keys, prefix_keys)
        return prefix_keys, outputs

    keys, batched = run(text, [2, 5], image)
    for index, length in enumerate([2, 5]):
        single_keys, single = run(
            text[index : index + 1, :length],
            [length],
            image[index : index + 1],
        )
        torch.testing.assert_close(keys[index : index + 1, :, :length], single_keys)
        for step in range(2):
            torch.testing.assert_close(
                batched[step][index : index + 1],
                single[step],
                atol=1e-5,
                rtol=1e-4,
            )


def test_generation_executor_uses_resolved_npu_device(monkeypatch):
    from transformers import AutoModel, AutoTokenizer

    from sglang_omni.models import weight_loader
    from sglang_omni.utils import device as device_utils

    calls = {}
    tokenizer = object()

    class FakeDevice:
        type = "npu"

        def __str__(self):
            return "npu:3"

    resolved_device = FakeDevice()

    class FakeModel:
        def eval(self):
            calls["eval"] = True
            return self

        def to(self, device):
            calls["device"] = device
            return self

    def load_model(model_path, **kwargs):
        calls["model_path"] = model_path
        calls["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        neo_unify, "register", lambda: calls.setdefault("registered", True)
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path: tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(weight_loader, "resolve_dtype", lambda dtype: torch.bfloat16)
    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, gpu_id: resolved_device,
    )
    monkeypatch.setattr(torch.ops, "npu", SimpleNamespace(), raising=False)

    scheduler = stages.create_generation_executor(
        "/model/sensenova",
        device=None,
        gpu_id=3,
        dtype="bfloat16",
        max_batch_size=2,
        max_batch_wait_ms=7,
        max_batch_cost=1234,
    )

    assert scheduler._fn is not None
    assert scheduler._batch_fn is not None
    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == 0.007
    assert scheduler._batch_key_fn is stages.image_generation_batch_key
    assert scheduler._request_cost_fn is stages.image_generation_request_cost
    assert scheduler._max_batch_cost == 1234
    assert calls == {
        "registered": True,
        "model_path": "/model/sensenova",
        "model_kwargs": {"torch_dtype": torch.bfloat16},
        "eval": True,
        "device": resolved_device,
    }
