# SPDX-License-Identifier: Apache-2.0
"""PR3 decoder-only tests; no SGLang runtime or model download required.

Set LLADA_PR3_REFERENCE_DIR to the original PR3 checkout to additionally run
numeric comparisons against its backbone and transport implementation.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from PIL import Image
from safetensors.torch import save_file

from sglang_omni.models.llada2_uni.components import image_decoder as decoder_module
from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageTransformer2DModelWrapper,
    _decoder_config,
    _semantic_checkpoint,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    LLaDA2ImageDecoder,
    _create_decoder_model_fn,
)
from sglang_omni.models.llada2_uni.components.sigvq import SigVQ
from sglang_omni.models.llada2_uni.components.transport import Sampler, create_transport


@pytest.fixture
def tiny_config():
    return _decoder_config(
        {
            "dim": 32,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 4,
            "cap_feat_dim": 16,
            "axes_dims": (2, 2, 4),
            "axes_lens": (128, 32, 32),
        }
    )


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_config):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = ZImageTransformer2DModel(**tiny_config).eval()
        torch.nn.init.normal_(model.x_pad_token, std=0.01)
        torch.nn.init.normal_(model.cap_pad_token, std=0.01)
    state = {
        key.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    return tmp_path, model, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_diffusers_checkpoint_forward(tiny_checkpoint, tiny_config, dtype):
    path, reference, _ = tiny_checkpoint
    wrapper = ZImageTransformer2DModelWrapper(path, tiny_config, "cpu", dtype)
    reference.to(dtype=dtype)
    x = [torch.randn(16, 1, 4, 6).to(dtype), torch.randn(16, 1, 6, 4).to(dtype)]
    cap = [torch.randn(7, 16).to(dtype), torch.randn(33, 16).to(dtype)]
    t = torch.tensor([0.125, 0.875])
    seen = []
    hook = wrapper.model.t_embedder.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].clone())
    )
    with torch.inference_mode():
        expected = reference(x=x, t=t, cap_feats=cap, return_dict=False)[0]
        actual = wrapper(x, t, cap, return_dict=False)[0]
        single = wrapper(x[:1], 0.125, cap[:1]).sample[0]
    hook.remove()
    torch.testing.assert_close(seen[0], t * tiny_config["t_scale"], rtol=0, atol=0)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == dtype and torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert single.shape == x[0].shape
    assert not wrapper.model.training
    assert all(
        not p.requires_grad and p.device.type == "cpu" for p in wrapper.parameters()
    )


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "duplicate"])
def test_checkpoint_errors_do_not_fall_back(tiny_checkpoint, tiny_config, failure):
    path, _, state = tiny_checkpoint
    if failure == "missing":
        state.pop("x_pad_token")
    elif failure == "unexpected":
        state["unknown.weight"] = torch.zeros(1)
    elif failure == "shape":
        state["x_pad_token"] = torch.zeros(1, 7)
    else:
        state["cap_embedder.0.weight"] = state["semantic_embedder.0.weight"].clone()
    save_file(state, str(path / "model.safetensors"))
    with pytest.raises((RuntimeError, ValueError)):
        ZImageTransformer2DModelWrapper(path, tiny_config, "cpu", torch.float32)


def test_checkpoint_mapping_is_prefix_only():
    weights = [("semantic_embedder.0.weight", 1), ("other.semantic_embedder.weight", 2)]
    assert dict(_semantic_checkpoint(weights)) == {
        "cap_embedder.0.weight": 1,
        "other.semantic_embedder.weight": 2,
    }


@pytest.mark.parametrize("invalid", ["batch", "latent", "patch", "caption", "timestep"])
def test_adapter_validates_inputs(tiny_checkpoint, tiny_config, invalid):
    path, _, _ = tiny_checkpoint
    model = ZImageTransformer2DModelWrapper(path, tiny_config, "cpu", torch.float32)
    x, cap, t = [torch.zeros(16, 1, 4, 4)], [torch.zeros(3, 16)], 0.5
    if invalid == "batch":
        cap = []
    elif invalid == "latent":
        x = [torch.zeros(1, 16, 1, 4, 4)]
    elif invalid == "patch":
        x = [torch.zeros(16, 1, 3, 4)]
    elif invalid == "caption":
        cap = [torch.zeros(3, 15)]
    else:
        t = torch.tensor([0.25, 0.75])
    with pytest.raises(ValueError):
        model(x, t, cap)


def test_config_metadata_is_not_forwarded(tiny_config):
    config = _decoder_config(dict(tiny_config, _class_name="old", siglip_feat_dim=123))
    assert config == tiny_config
    with pytest.raises(ValueError, match="Unsupported decoder config"):
        _decoder_config(dict(tiny_config, unsupported_architecture=True))


@pytest.mark.parametrize("backend", ["auto", "missing", None])
def test_only_explicit_supported_backend(tmp_path, backend):
    with pytest.raises(ValueError, match="Unsupported image decoder backend"):
        LLaDA2ImageDecoder(str(tmp_path), device="cpu", backend=backend)
    with pytest.raises(ValueError, match="Unsupported image decoder backend"):
        ZImageTransformer2DModelWrapper(
            tmp_path, {}, "cpu", torch.float32, backend=backend
        )


def test_constructor_contract_and_hf_resolution(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(
        decoder_module,
        "resolve_model_path",
        lambda path: requests.append(path) or tmp_path,
    )
    decoder = LLaDA2ImageDecoder("org/model", "cpu", torch.float32, "normal", 50, 2)
    assert requests == ["org/model"]
    assert decoder.model_path == str(tmp_path)
    assert decoder.backend == "diffusers"
    assert decoder._diff_model is decoder._sigvq is decoder._vae is None


def test_sigvq_math_and_checkpoint_names():
    model = SigVQ(vocab_size=8, inner_dim=4)
    state = model.state_dict()
    assert set(state) == {
        "prior_token_embedding.weight",
        "prior_projector.net.0.proj.weight",
        "prior_projector.net.0.proj.bias",
        "prior_projector.net.2.weight",
        "prior_projector.net.2.bias",
    }
    ids = torch.tensor([[0, 3, 3, 7]])
    embedded = F.embedding(ids, state["prior_token_embedding.weight"])
    hidden = F.silu(
        F.linear(
            embedded,
            state["prior_projector.net.0.proj.weight"],
            state["prior_projector.net.0.proj.bias"],
        )
    )
    expected = F.linear(
        hidden,
        state["prior_projector.net.2.weight"],
        state["prior_projector.net.2.bias"],
    )
    torch.testing.assert_close(model(ids), expected, rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.parameters())


def test_sigvq_lazy_load_strict_retry(tmp_path, monkeypatch):
    (tmp_path / "image_tokenizer").mkdir()
    path = tmp_path / "image_tokenizer/sigvq_embedding.pt"
    torch.save({}, path)
    requests = []

    def build(vocab_size, inner_dim):
        requests.append((vocab_size, inner_dim))
        return SigVQ(vocab_size=8, inner_dim=4)

    monkeypatch.setattr(decoder_module, "SigVQ", build)
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Missing key"):
        decoder._ensure_sigvq()
    assert decoder._sigvq is None
    reference = SigVQ(vocab_size=8, inner_dim=4)
    torch.save(reference.state_dict(), path)
    decoder._ensure_sigvq()
    decoder._ensure_sigvq()
    assert requests == [(16384, 4096), (16384, 4096)]
    ids = torch.tensor([[0, 7]])
    torch.testing.assert_close(decoder._sigvq(ids), reference(ids), rtol=0, atol=0)
    assert not decoder._sigvq.training


def test_vae_lazy_load_contract(tmp_path, monkeypatch):
    requests = []
    vae = torch.nn.Linear(2, 2)

    def load(path, torch_dtype):
        requests.append((path, torch_dtype))
        return vae.to(dtype=torch_dtype)

    monkeypatch.setattr(decoder_module.AutoencoderKL, "from_pretrained", load)
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu", dtype=torch.bfloat16)
    decoder._ensure_vae()
    decoder._ensure_vae()
    assert requests == [(str(tmp_path / "vae"), torch.bfloat16)]
    assert decoder._vae is vae and not vae.training
    assert vae.weight.device.type == "cpu" and vae.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("cfg_scale", [0.0, 1.0])
def test_cfg_batching_float32_norm_clamp_and_timestep(cfg_scale):
    calls = []
    positive = [torch.ones(4, 16), torch.ones(5, 16) * 2]
    negative = [torch.zeros_like(c) for c in positive]
    p = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1)
    ng = torch.tensor([-1.0, 2.0]).reshape(2, 1, 1, 1)

    def model(**kwargs):
        calls.append(kwargs)
        return ([p, p * 2, ng, ng * 2] if cfg_scale else [p, p * 2],)

    fn = _create_decoder_model_fn(
        model, positive, negative, cfg_scale, 2, 1, torch.bfloat16
    )
    x = torch.randn(2, 2, 1, 1, 1)
    actual = fn(x, torch.tensor(0.25))
    expected = p.float() + cfg_scale * (p.float() - ng.float())
    expected *= min(
        1.0, torch.linalg.vector_norm(p) / torch.linalg.vector_norm(expected)
    )
    torch.testing.assert_close(actual, torch.stack([expected, expected * 2]))
    assert actual.dtype == torch.float32
    call = calls[0]
    assert call["cap_feats"] == (positive + negative if cfg_scale else positive)
    assert len(call["x"]) == (4 if cfg_scale else 2)
    assert call["x"][0].dtype == torch.bfloat16
    torch.testing.assert_close(call["t"], torch.full((len(call["x"]),), 0.25))
    assert call["patch_size"] == 2 and call["f_patch_size"] == 1


@pytest.mark.parametrize(
    "positive,negative",
    [
        ([3.0, 4.0], [-1.0, 2.0]),
        ([3.0, 4.0], [4.0, 4.0]),
        ([3.0, 4.0], [6.0, 8.0]),
        ([0.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [1.0, 1.0]),
    ],
)
def test_cfg_tensor_clamp_is_exact_without_device_scalar_control_flow(
    positive, negative
):
    from torch.utils._python_dispatch import TorchDispatchMode

    p = torch.tensor(positive).reshape(2, 1, 1, 1)
    ng = torch.tensor(negative).reshape(2, 1, 1, 1)
    expected = p + (p - ng)
    original_norm = torch.linalg.vector_norm(p)
    guided_norm = torch.linalg.vector_norm(expected)
    if guided_norm > original_norm:
        expected *= original_norm / guided_norm

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("device scalar read in decoder CFG")
            return func(*args, **(kwargs or {}))

    fn = _create_decoder_model_fn(
        lambda **kwargs: ([p, ng],),
        [torch.ones(1, 1)],
        [torch.zeros(1, 1)],
        1.0,
        2,
        1,
        torch.float32,
    )
    with NoScalarRead():
        actual = fn(torch.zeros(1, 2, 1, 1, 1), torch.tensor(0.5))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


def test_normal_euler_grid_and_positive_velocity():
    times = []

    def model(x, t):
        times.append(t[0].clone())
        return torch.ones_like(x)

    init = torch.zeros(1, 2, 1, 2, 2)
    sample = Sampler(create_transport()).sample_ode(num_steps=5, time_shifting_factor=6)
    outputs = sample(init, model)
    raw_grid = torch.linspace(0, 1, 5)
    grid = raw_grid / (raw_grid + 6 - 6 * raw_grid)
    torch.testing.assert_close(torch.stack(times), grid[:-1])
    torch.testing.assert_close(outputs[-1], torch.ones_like(init))
    assert outputs.shape[0] == 5 and outputs.dtype == torch.float32


def test_normal_one_grid_point_preserves_pr3_no_step_behavior():
    init = torch.randn(1, 16, 1, 2, 2)
    sample = Sampler(create_transport()).sample_ode(num_steps=1)
    actual = sample(init, lambda x, t: pytest.fail("one grid point has no interval"))[
        -1
    ]
    torch.testing.assert_close(actual, init, rtol=0, atol=0)


def test_turbo_renoising_grid_and_seed():
    init = torch.zeros(1, 2, 1, 2, 2)

    def run(seed):
        times = []

        def model(x, t):
            times.append(t[0].clone())
            assert x.dtype == torch.float64
            return torch.full_like(x, 0.5)

        sample = Sampler(create_transport()).sample_ode(
            num_steps=4,
            stochast_ratio=1.0,
            time_shifting_factor=6,
            generator=torch.Generator().manual_seed(seed),
        )
        return sample(init, model)[-1], torch.stack(times)

    first, times = run(7)
    second, _ = run(7)
    third, _ = run(8)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, third)
    torch.testing.assert_close(
        times, torch.tensor([0, 0.25, 0.5, 0.75], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "kwargs", [{"path_type": "VP"}, {"path_type": "GVP"}, {"prediction": "unknown"}]
)
def test_transport_rejects_unimplemented_math(kwargs):
    with pytest.raises(ValueError, match="Linear/velocity"):
        create_transport(**kwargs)


@pytest.fixture
def decode_probe(monkeypatch, tmp_path):
    observed = SimpleNamespace(ids=None, latents=None, calls=[])
    decoder = LLaDA2ImageDecoder(
        str(tmp_path), device="cpu", dtype=torch.float32, num_steps=4
    )

    def sigvq(ids):
        observed.ids = ids.clone()
        return ids.float().unsqueeze(-1).expand(-1, -1, 16)

    def model(**kwargs):
        observed.calls.append(kwargs)
        return ([torch.ones_like(x) for x in kwargs["x"]],)

    def vae_decode(latents, return_dict):
        observed.latents = latents.clone()
        assert not return_dict
        px = torch.empty(1, 3, latents.shape[-2] * 8, latents.shape[-1] * 8)
        px[:, 0], px[:, 1], px[:, 2] = -2, 0, 2
        return (px,)

    decoder._sigvq = sigvq
    decoder._diff_model = model
    decoder._diff_config = {
        "all_patch_size": [2],
        "all_f_patch_size": [1],
        "cap_feat_dim": 16,
    }
    decoder._vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=2.0, shift_factor=3.0), decode=vae_decode
    )
    monkeypatch.setattr(decoder, "_ensure_diff_model", lambda mode: None)
    return decoder, observed


@pytest.mark.parametrize("mode", ["normal", "decoder-turbo"])
@pytest.mark.parametrize("multiplier", [1, 2])
def test_decode_upsampling_seed_vae_and_pixels(decode_probe, mode, multiplier):
    decoder, observed = decode_probe
    image = decoder.decode(
        [1, 2], 1, 2, decode_mode=mode, resolution_multiplier=multiplier, seed=9
    )
    assert image.size == (32 * multiplier, 16 * multiplier)
    assert image.mode == "RGB" and image.getpixel((0, 0)) == (0, 127, 255)
    torch.testing.assert_close(observed.ids, torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2]]))
    call = observed.calls[0]
    assert len(call["x"]) == (2 if mode == "normal" else 1)
    assert call["x"][0].shape == (16, 1, 2 * multiplier, 4 * multiplier)
    first_latents = observed.latents.clone()
    if mode == "normal":
        noise = torch.randn(
            (1, 16, 1, 2 * multiplier, 4 * multiplier),
            generator=torch.Generator().manual_seed(9),
        )
        torch.testing.assert_close(first_latents, (noise.squeeze(2) + 1) / 2 + 3)
    decoder.decode(
        [1, 2], 1, 2, decode_mode=mode, resolution_multiplier=multiplier, seed=9
    )
    torch.testing.assert_close(observed.latents, first_latents, rtol=0, atol=0)
    decoder.decode(
        [1, 2], 1, 2, decode_mode=mode, resolution_multiplier=multiplier, seed=10
    )
    assert not torch.equal(observed.latents, first_latents)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_decode_to_bytes(decode_probe, format):
    decoder, _ = decode_probe
    data = decoder.decode_to_bytes(
        [1], 1, 1, format=format, decode_mode="normal", num_steps=3, seed=2
    )
    image = Image.open(io.BytesIO(data))
    assert image.format == format and image.size == (32, 32)


def test_sp_follower_samples_without_sigvq_vae_or_image_encoding(
    decode_probe, monkeypatch
):
    from contextlib import nullcontext

    decoder, observed = decode_probe
    decoder.runtime = SimpleNamespace(
        is_leader=False,
        preparation=lambda phase: nullcontext(),
        request_seed=lambda metadata, seed: 19,
        broadcast_features=lambda features: features.fill_(1),
    )
    monkeypatch.setattr(
        decoder, "_ensure_sigvq", lambda: pytest.fail("follower loaded SigVQ")
    )
    monkeypatch.setattr(
        decoder, "_ensure_vae", lambda: pytest.fail("follower loaded VAE")
    )
    assert decoder.decode_to_bytes([1, 2], 1, 2) is None
    assert observed.ids is None and observed.latents is None
    assert observed.calls
    assert observed.calls[0]["cap_feats"][0].shape == (8, 16)
    first_input = observed.calls[0]["x"][0].clone()
    observed.calls.clear()
    assert decoder.decode([1, 2], 1, 2) is None
    torch.testing.assert_close(observed.calls[0]["x"][0], first_input, rtol=0, atol=0)


def test_decode_without_seed_uses_fresh_noise(decode_probe):
    decoder, observed = decode_probe
    decoder.decode([1], 1, 1)
    first = observed.latents.clone()
    decoder.decode([1], 1, 1)
    assert not torch.equal(observed.latents, first)


@pytest.mark.parametrize(
    "tokens,h,w,kwargs",
    [
        ([1], 0, 1, {}),
        ([1], 1, 2, {}),
        ([16384], 1, 1, {}),
        ([-1], 1, 1, {}),
        ([1.5], 1, 1, {}),
        ([1], 1, 1, {"num_steps": 0}),
        ([1], 1, 1, {"resolution_multiplier": 0}),
        ([1], 1, 1, {"decode_mode": "typo"}),
    ],
)
def test_invalid_decode_fails_before_loading(
    tmp_path, monkeypatch, tokens, h, w, kwargs
):
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    monkeypatch.setattr(
        decoder, "_ensure_sigvq", lambda: pytest.fail("invalid input loaded weights")
    )
    with pytest.raises(ValueError):
        decoder.decode(tokens, h, w, **kwargs)


def test_diffusion_modes_and_failed_load_are_not_cached(tmp_path, monkeypatch):
    for name in ("decoder", "decoder-turbo"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(
            json.dumps({"cap_feat_dim": 2560, "axes_lens": [1, 2, 3]})
        )
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        if Path(kwargs["decoder_dir"]).name == "decoder-turbo":
            raise RuntimeError("checkpoint load failed")
        return object()

    monkeypatch.setattr(decoder_module, "ZImageTransformer2DModelWrapper", load)
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    decoder._ensure_diff_model("normal")
    model = decoder._diff_model
    decoder._ensure_diff_model("normal")
    assert decoder._diff_model is model and len(calls) == 1
    assert calls[0]["cfg"] == {"cap_feat_dim": 4096, "axes_lens": [32768, 1024, 1024]}
    assert calls[0]["backend"] == "diffusers"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="checkpoint load failed"):
            decoder._ensure_diff_model("decoder-turbo")
        assert (
            decoder._diff_model
            is decoder._diff_config
            is decoder._diff_model_mode
            is None
        )
    assert len(calls) == 3
    decoder._ensure_diff_model("normal")
    assert decoder._diff_model_mode == "normal"


def _reference_module(name):
    root = os.environ.get("LLADA_PR3_REFERENCE_DIR")
    if not root:
        pytest.skip("set LLADA_PR3_REFERENCE_DIR for original PR3 numeric parity")
    path = Path(root) / "sglang_omni/models/llada2_uni/components" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pr3_reference_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_pr3_backbone_numeric_parity(tiny_checkpoint, tiny_config, dtype):
    original = _reference_module("decoder_model")
    path, _, state = tiny_checkpoint
    reference = original.ZImageTransformer2DModel(**tiny_config)
    reference.load_state_dict(state, strict=True)
    reference.to(dtype=dtype).eval()
    adapter = ZImageTransformer2DModelWrapper(path, tiny_config, "cpu", dtype)
    with torch.random.fork_rng():
        torch.manual_seed(11)
        x = [torch.randn(16, 1, 4, 6).to(dtype), torch.randn(16, 1, 6, 4).to(dtype)]
        cap = [torch.randn(7, 16).to(dtype), torch.randn(33, 16).to(dtype)]
    with torch.inference_mode():
        expected = reference(
            x=x, t=torch.tensor([0.25, 0.75]), cap_feats=cap, return_dict=False
        )[0]
        actual = adapter(x, torch.tensor([0.25, 0.75]), cap, return_dict=False)[0]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("stochastic", [0.0, 1.0])
def test_original_pr3_sampling_numeric_parity(stochastic):
    reference = _reference_module("transport")
    original = reference.Sampler(reference.create_transport())
    current = Sampler(create_transport())
    init = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4) / 16

    def sample(sampler):
        return sampler.sample_ode(
            num_steps=4,
            stochast_ratio=stochastic,
            time_shifting_factor=6,
            generator=torch.Generator().manual_seed(13),
        )(init.clone(), lambda x, t: x.float() * 0.1 + t.reshape(-1, 1, 1, 1, 1))[-1]

    torch.testing.assert_close(sample(current), sample(original), rtol=0, atol=0)
