# SPDX-License-Identifier: Apache-2.0
"""Numerical and lifecycle coverage for Fish's bounded Apple runner."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.engine_builder import FishS2ProEngineBuilder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3Config,
)
from sglang_omni.models.fishaudio_s2_pro.torch_mps import (
    FishS2ProTorchMpsRunner,
    S2ProTorchMpsTextModel,
    fast_attention,
)


def tiny_model():
    torch.manual_seed(19)
    config = FishQwen3Config(
        vocab_size=64,
        dim=32,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        head_dim=8,
        intermediate_size=48,
        max_seq_len=32,
        attention_qk_norm=True,
    )
    return S2ProTorchMpsTextModel(SimpleNamespace(text_config=config)).eval()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cached_decode_matches_full_prefill(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Metal")
    model = tiny_model().to(device)
    tokens = torch.tensor([1, 4, 9, 3, 8], device=device)
    model.forward_native(tokens[:3], request_id="a", prefill=True)
    model.forward_native(tokens[3:4], request_id="a")
    cached, _ = model.forward_native(tokens[4:], request_id="a")
    full, _ = model.forward_native(tokens, request_id="b", prefill=True)
    torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-5)
    assert set(model._request_caches) == {"b"}


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_fast_attention_masks_uninitialized_slots_and_matches_manual_gqa(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Metal")
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.audio_decoder import (
        flash_attn_kvcache_op,
    )

    torch.manual_seed(9)
    keys, values = torch.randn(1, 3, 2, 8).to(device), torch.randn(1, 3, 2, 8).to(
        device
    )
    q = torch.randn(1, 1, 4, 8).to(device)
    kc, vc = torch.full((1, 11, 2, 8), float("nan"), device=device), torch.full(
        (1, 11, 2, 8), float("nan"), device=device
    )
    attention = flash_attn_kvcache_op if device == "mps" else fast_attention
    for i in range(3):
        actual = attention(
            q, kc, vc, keys[:, i : i + 1], values[:, i : i + 1], cache_position=i
        )
    k = keys.repeat_interleave(2, dim=2).transpose(1, 2)
    v = values.repeat_interleave(2, dim=2).transpose(1, 2)
    probs = (q.transpose(1, 2) @ k.transpose(-2, -1) / 8**0.5).softmax(-1)
    expected = (probs @ v).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_weight_loader_requires_complete_matching_checkpoint():
    source, target = tiny_model(), tiny_model()
    weights = [
        ("text_model.model." + k.replace("embed_tokens.", "embeddings."), v)
        for k, v in source.named_parameters()
    ]
    target.load_weights(iter(weights))
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)
    with pytest.raises(ValueError, match="Missing Fish"):
        target.load_weights(iter(weights[:-1]))
    with pytest.raises(ValueError, match="Unexpected Fish"):
        target.load_weights(iter([("text_model.model.unexpected", torch.zeros(1))]))
    with pytest.raises(ValueError, match="shape mismatch"):
        target.load_weights(iter([(weights[0][0], torch.zeros(1))]))


def test_completion_and_abort_release_native_caches():
    model = tiny_model()
    runner = object.__new__(FishS2ProTorchMpsRunner)
    runner.model = model
    for finish in [
        lambda: runner.on_request_finished("r", None),
        lambda: runner.abort_request("r"),
    ]:
        model.forward_native(torch.tensor([1, 2]), request_id="r", prefill=True)
        finish()
        assert not model._request_caches
        finish()  # Cleanup is idempotent.
        with pytest.raises(RuntimeError, match="no cache"):
            model.forward_native(torch.tensor([3]), request_id="r")


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_running_requests", 2),
        ("disable_radix_cache", False),
        ("chunked_prefill_size", 128),
        ("quantization", "int8"),
        ("enable_torch_compile", True),
    ],
)
def test_apple_profile_rejects_unqualified_overrides(key, value):
    builder = FishS2ProEngineBuilder(max_new_tokens=32, ras_window=16)
    builder.device = "mps"
    overrides = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(overrides)
    overrides[key] = value
    with pytest.raises(ValueError, match="Fish Apple requires"):
        builder.adjust_overrides(overrides)


def test_native_mlx_switch_selects_native_architecture(monkeypatch):
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    from sglang_omni.models.fishaudio_s2_pro import engine_builder

    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    monkeypatch.setattr(engine_builder.current_platform, "is_mps", lambda: True)
    builder = FishS2ProEngineBuilder(max_new_tokens=32, ras_window=16)
    builder.pre_infra_setup("unused")
    assert builder.model_arch_override == "FishS2ProMlxModel"


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    "seed,step,expected", [(42, 17, 1), (0, 0, 2), (2**40 + 3, 19, 1), (42, 18, 2)]
)
def test_seeded_sampler_matches_independent_murmur3_vectors(
    device, seed, step, expected
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Metal")
    # Expected draws calculated with mmh3.hash(struct.pack('<QII', seed,
    # step, column), signed=False) and the documented Gumbel transform.
    model = tiny_model()
    probs = torch.tensor([[0.15, 0.25, 0.6]])
    seeds, steps = torch.tensor([seed]), torch.tensor([step])
    actual = model._sample_semantic_choice(
        probs.to(device), seeds.to(device), steps.to(device)
    )
    assert actual.item() == expected


@pytest.mark.parametrize(
    "device,expected", [("mps", "S2ProTorchMpsTextModel"), ("cuda:0", None)]
)
def test_apple_device_selects_the_native_architecture(monkeypatch, device, expected):
    from sglang_omni.models.fishaudio_s2_pro import bootstrap as fish_bootstrap

    monkeypatch.setattr(fish_bootstrap, "patch_fish_config_for_sglang", lambda: None)
    builder = FishS2ProEngineBuilder(max_new_tokens=32, ras_window=16)
    builder.device = device
    builder.pre_infra_setup("unused")
    assert builder.model_arch_override == expected


def test_apple_runner_supplies_the_scheduler_abort_callback(monkeypatch):
    import sglang_omni.models.fishaudio_s2_pro.torch_mps as torch_mps

    class StubRunner:
        def __init__(self, model_worker, output_proc):
            self.model_worker = model_worker
            self.output_proc = output_proc

        def abort_request(self, request_id):
            raise AssertionError("not called")

    monkeypatch.setattr(torch_mps, "FishS2ProTorchMpsRunner", StubRunner)
    builder = FishS2ProEngineBuilder(max_new_tokens=32, ras_window=16)
    builder.device = "mps"
    assert builder.make_abort_callback() is None

    runner = builder.make_model_runner("worker", "output")
    assert isinstance(runner, StubRunner)
    assert builder.make_abort_callback() == runner.abort_request
