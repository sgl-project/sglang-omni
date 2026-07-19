# SPDX-License-Identifier: Apache-2.0
"""ZImage advanced-path guard tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.models.ming_omni.diffusion.backend import ImageGenParams


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=["image"])


class FakeTextEncoder:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.calls = []

    def encode(self, text, *, tokenizer, device, max_length):
        self.calls.append((text, tokenizer, device, max_length))
        return (
            [self.torch.full((1, 3), 2.0)],
            [self.torch.full((1, 3), -2.0)],
        )


class FakeSemanticEncoder:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.calls = []
        self.unloaded = False

    def encode(self, prompt):
        self.calls.append(prompt)
        return (
            [self.torch.full((2, 3), 4.0)],
            [self.torch.zeros(2, 3)],
        )

    def unload(self):
        self.unloaded = True


class _FakeScheduler:
    calls: list[tuple] = []
    instances: list["_FakeScheduler"] = []

    def __init__(self) -> None:
        self.config: dict[str, object] = {}
        _FakeScheduler.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_path, *, subfolder):
        cls.calls.append((model_path, subfolder))
        return cls()


class _FakeVae:
    calls: list[tuple] = []
    instances: list["_FakeVae"] = []

    def __init__(self) -> None:
        _FakeVae.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_path, *, subfolder, torch_dtype):
        cls.calls.append((model_path, subfolder, torch_dtype))
        return cls()


class _FakeTransformer:
    calls: list[tuple] = []
    instances: list["_FakeTransformer"] = []

    def __init__(self) -> None:
        self.config = SimpleNamespace(cap_feat_dim=2560)
        _FakeTransformer.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_path, *, subfolder, torch_dtype):
        cls.calls.append((model_path, subfolder, torch_dtype))
        return cls()


class _FakeZImagePipeline:
    instances: list["_FakeZImagePipeline"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.to_calls = []
        _FakeZImagePipeline.instances.append(self)

    def to(self, device):
        self.to_calls.append(device)
        return self


def _install_fake_diffusers(monkeypatch):
    module = ModuleType("diffusers")
    module.FlowMatchEulerDiscreteScheduler = _FakeScheduler
    module.AutoencoderKL = _FakeVae
    module.ZImageTransformer2DModel = _FakeTransformer
    module.ZImagePipeline = _FakeZImagePipeline
    monkeypatch.setitem(sys.modules, "diffusers", module)
    _FakeScheduler.calls.clear()
    _FakeScheduler.instances.clear()
    _FakeVae.calls.clear()
    _FakeVae.instances.clear()
    _FakeTransformer.calls.clear()
    _FakeTransformer.instances.clear()
    _FakeZImagePipeline.instances.clear()


def _backend_with_fake_pipe():
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    backend._pipe = FakePipe()
    backend._device = "cpu"
    return backend


def test_extract_render_text_uses_last_quoted_span() -> None:
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import (
        _extract_render_text,
    )

    assert _extract_render_text('make a sign saying "SALE"') == "SALE"
    assert _extract_render_text('first "A" then "B"') == "B"
    assert _extract_render_text("no quoted text") == ""


def test_generate_requires_standalone_encoder_when_condition_embeds_missing() -> None:
    backend = _backend_with_fake_pipe()

    with pytest.raises(RuntimeError, match="standalone semantic encoder unavailable"):
        backend.generate("draw", ImageGenParams())


def test_generate_uses_explicit_standalone_encoder_when_loaded() -> None:
    torch = pytest.importorskip("torch")
    backend = _backend_with_fake_pipe()
    backend._semantic_encoder = FakeSemanticEncoder(torch)

    image = backend.generate("draw from reference", ImageGenParams())

    assert image == "image"
    assert backend._semantic_encoder.calls == ["draw from reference"]
    prompt_embeds = backend._pipe.calls[0]["prompt_embeds"]
    torch.testing.assert_close(prompt_embeds[0], torch.full((2, 3), 4.0))


def test_generate_concats_byt5_only_when_text_rendering_is_enabled() -> None:
    torch = pytest.importorskip("torch")
    backend = _backend_with_fake_pipe()
    backend._text_encoder = FakeTextEncoder(torch)
    backend._tokenizer = object()
    sem = torch.ones(2, 3)
    neg = torch.zeros(2, 3)

    backend.generate(
        'make a sign saying "SALE"',
        ImageGenParams(enable_text_rendering=False),
        condition_embeds=[sem],
        negative_condition_embeds=[neg],
    )
    backend.generate(
        'make a sign saying "SALE"',
        ImageGenParams(enable_text_rendering=True),
        condition_embeds=[sem],
        negative_condition_embeds=[neg],
    )

    first_prompt = backend._pipe.calls[0]["prompt_embeds"][0]
    second_prompt = backend._pipe.calls[1]["prompt_embeds"][0]
    torch.testing.assert_close(first_prompt, sem)
    assert second_prompt.shape == (3, 3)
    torch.testing.assert_close(second_prompt[-1], torch.full((3,), 2.0))
    assert backend._text_encoder.calls[0][0] == "SALE"


def test_generate_fails_when_text_rendering_requested_but_byt5_unloaded() -> None:
    torch = pytest.importorskip("torch")
    backend = _backend_with_fake_pipe()
    sem = torch.ones(2, 3)

    with pytest.raises(RuntimeError, match="ByT5 text rendering unavailable"):
        backend.generate(
            'make a sign saying "SALE"',
            ImageGenParams(enable_text_rendering=True),
            condition_embeds=[sem],
        )


@pytest.mark.parametrize(
    ("load_kwargs", "message"),
    [
        (
            {"load_semantic_encoder": True},
            "load_semantic_encoder=True requires ming_model_path",
        ),
        (
            {"load_byt5_text_encoder": True},
            "load_byt5_text_encoder=True requires ming_model_path",
        ),
    ],
)
def test_load_models_optional_assets_without_ming_model_path_raises(
    load_kwargs, message
) -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    with pytest.raises(ValueError, match=message):
        ZImageBackend().load_models(
            "/fake/dit",
            torch.device("cpu"),
            **load_kwargs,
        )


def test_load_models_byt5_missing_dir_raises(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    with pytest.raises(RuntimeError, match="ByT5 text rendering requested but"):
        ZImageBackend().load_models(
            "/fake/dit",
            torch.device("cpu"),
            ming_model_path=str(tmp_path),
            load_byt5_text_encoder=True,
        )


def test_load_models_success_assembles_zimage_pipeline_with_dit_components(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_diffusers(monkeypatch)
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    backend.load_models("/fake/dit", torch.device("cpu"), ming_model_path="/fake/ming")

    assert _FakeScheduler.calls == [("/fake/dit", "scheduler")]
    assert _FakeVae.calls == [("/fake/dit", "vae", torch.bfloat16)]
    assert _FakeTransformer.calls == [("/fake/dit", "transformer", torch.bfloat16)]
    scheduler = _FakeScheduler.instances[0]
    vae = _FakeVae.instances[0]
    transformer = _FakeTransformer.instances[0]
    pipe = _FakeZImagePipeline.instances[0]
    assert set(pipe.kwargs) == {
        "scheduler",
        "vae",
        "transformer",
        "text_encoder",
        "tokenizer",
    }
    assert pipe.kwargs["scheduler"] is scheduler
    assert pipe.kwargs["vae"] is vae
    assert pipe.kwargs["transformer"] is transformer
    assert pipe.kwargs["scheduler"].config["use_dynamic_shifting"] is True
    assert pipe.kwargs["text_encoder"] is None
    assert pipe.kwargs["tokenizer"] is None
    assert pipe.to_calls == [torch.device("cpu")]
    assert backend._pipe is pipe

    backend.unload()
    assert backend._pipe is None


def test_load_models_loads_optional_ming_assets_from_ming_model_path(
    monkeypatch, tmp_path
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_diffusers(monkeypatch)

    from sglang_omni.models.ming_omni.diffusion import semantic_encoder
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    class _FakeMingSemanticEncoder:
        instances: list["_FakeMingSemanticEncoder"] = []

        def __init__(self) -> None:
            self.load_calls = []
            self.unloaded = False
            _FakeMingSemanticEncoder.instances.append(self)

        def load(self, model_path, device):
            self.load_calls.append((model_path, device))

        def unload(self):
            self.unloaded = True

    text_encoder = SimpleNamespace()
    tokenizer = object()
    byt5_calls = []

    def fake_load_byt5_text_encoder(model_path, device, dtype):
        byt5_calls.append((model_path, device, dtype))
        return text_encoder, tokenizer

    byt5_module = ModuleType("sglang_omni.models.ming_omni.diffusion.byt5_encoder")
    byt5_module.load_byt5_text_encoder = fake_load_byt5_text_encoder
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.diffusion.byt5_encoder",
        byt5_module,
    )
    monkeypatch.setattr(
        semantic_encoder, "MingSemanticEncoder", _FakeMingSemanticEncoder
    )

    ming_root = tmp_path / "ming"
    (ming_root / "byt5").mkdir(parents=True)
    backend = ZImageBackend()
    backend.load_models(
        "/fake/dit",
        torch.device("cpu"),
        ming_model_path=str(ming_root),
        load_semantic_encoder=True,
        load_byt5_text_encoder=True,
    )

    semantic = _FakeMingSemanticEncoder.instances[0]
    assert semantic.load_calls == [(str(ming_root), torch.device("cpu"))]
    assert byt5_calls == [(str(ming_root), torch.device("cpu"), torch.bfloat16)]
    assert backend._semantic_encoder is semantic
    assert backend._text_encoder is text_encoder
    assert backend._tokenizer is tokenizer

    backend.unload()
    assert semantic.unloaded is True
    assert backend._semantic_encoder is None
    assert backend._text_encoder is None
    assert backend._tokenizer is None


def test_image_gen_params_production_defaults() -> None:
    """Lock the production-facing defaults a request inherits when fields are omitted."""
    params = ImageGenParams()
    assert params.width == 1024
    assert params.height == 1024
    assert params.num_inference_steps == 28
    # Z-Image-Turbo is distilled/low-CFG; SD's 7.0 default washes images out.
    assert params.guidance_scale == 2.0
    assert params.seed is None
    assert params.negative_prompt == ""
    assert params.semantic_source is None
    assert params.enable_text_rendering is False
