# SPDX-License-Identifier: Apache-2.0
"""Ming image-generation terminal executor tests."""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from types import ModuleType

import pytest

from sglang_omni.proto import OmniRequest, StagePayload


class StubBackend:
    def __init__(self, *, image=None, error: Exception | None = None):
        self.image = image
        self.error = error
        self.calls = []
        self.unloaded = False

    def load_models(self, *_args, **_kwargs):
        return None

    def generate(self, prompt, params, **kwargs):
        self.calls.append((prompt, params, kwargs))
        if self.error is not None:
            raise self.error
        return self.image

    def unload(self):
        self.unloaded = True


class StubConditioner:
    def __init__(self):
        self.project_calls = []
        self.load_calls = []
        self.unloaded = False

    def load(self, model_path, device):
        self.load_calls.append((model_path, str(device)))

    def project(self, query_hidden):
        self.project_calls.append(query_hidden.clone())
        return query_hidden + 100

    def unload(self):
        self.unloaded = True


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, output_ids, skip_special_tokens=True):
        self.calls.append((list(output_ids), skip_special_tokens))
        return "decoded prompt"


def _payload(
    *,
    request_id: str = "req-1",
    metadata: dict | None = None,
    inputs=None,
    data: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs if inputs is not None else "hello", metadata=metadata or {}
        ),
        data=data or {},
    )


def _png(width: int = 3, height: int = 2):
    pil = pytest.importorskip("PIL.Image")
    return pil.new("RGB", (width, height), color=(20, 40, 60))


def _decode_png_size(image_b64: str) -> tuple[int, int]:
    pil = pytest.importorskip("PIL.Image")
    raw = base64.b64decode(image_b64.encode("ascii"))
    with pil.open(io.BytesIO(raw)) as image:
        return image.size


def _install_fake_module(monkeypatch, module_name: str, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, module_name, module)
    parent_name, _, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child_name, module, raising=False)


def _image_record(result: StagePayload) -> dict:
    assert result.data["modality"] == "image"
    assert result.data["finish_reason"] == "stop"
    assert len(result.data["images"]) == 1
    return result.data["images"][0]


def test_start_calls_load_models_through_to_thread(monkeypatch) -> None:
    from sglang_omni.models.ming_omni.components import image_gen_executor as module

    executor = module.MingImageGenExecutor(model_path="/fake/model")
    calls = []

    def fake_load_models():
        calls.append("load")

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(("to_thread", fn))
        return fn(*args, **kwargs)

    monkeypatch.setattr(executor, "_load_models", fake_load_models)
    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

    asyncio.run(executor.start())

    assert calls == [("to_thread", fake_load_models), "load"]


def test_start_and_stop_loads_and_unloads_conditioner(monkeypatch) -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    common_module = ModuleType("sglang_omni.models.ming_omni.components.common")
    common_module.load_ming_tokenizer = lambda _model_path: FakeTokenizer()
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.components.common",
        common_module,
    )

    backend = StubBackend()
    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        dit_type="zimage",
        dit_model_path="/fake/dit",
        device="cpu",
        backend=backend,
        conditioner=conditioner,
    )

    asyncio.run(executor.start())
    asyncio.run(executor.stop())

    assert conditioner.load_calls == [("/fake/model", "cpu")]
    assert conditioner.unloaded is True


def test_create_backend_lazy_imports_zimage_and_rejects_unknown(
    monkeypatch,
) -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        _create_backend,
    )

    class FakeZImageBackend:
        pass

    zimage_module = ModuleType("sglang_omni.models.ming_omni.diffusion.zimage_backend")
    zimage_module.ZImageBackend = FakeZImageBackend
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.diffusion.zimage_backend",
        zimage_module,
    )

    assert isinstance(_create_backend("zimage"), FakeZImageBackend)
    with pytest.raises(ValueError, match="Unknown dit_type: 'sd3'.*zimage"):
        _create_backend("sd3")
    with pytest.raises(ValueError, match="Unknown dit_type: 'other'.*zimage"):
        _create_backend("other")


@pytest.mark.parametrize(
    ("executor_kwargs", "expected_load_kwargs"),
    [
        ({}, {"ming_model_path": "/fake/model"}),
        (
            {
                "enable_standalone_semantic_encoder": True,
                "enable_byt5_text_rendering": True,
            },
            {
                "ming_model_path": "/fake/model",
                "load_semantic_encoder": True,
                "load_byt5_text_encoder": True,
            },
        ),
    ],
)
def test_load_models_uses_fake_backend_tokenizer_and_zimage_flags(
    monkeypatch, executor_kwargs, expected_load_kwargs
) -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    loaded_backends = []

    class FakeZImageBackend:
        def __init__(self):
            self.load_calls = []
            loaded_backends.append(self)

        def load_models(self, *args, **kwargs):
            self.load_calls.append((args, kwargs))

    zimage_module = ModuleType("sglang_omni.models.ming_omni.diffusion.zimage_backend")
    zimage_module.ZImageBackend = FakeZImageBackend
    common_module = ModuleType("sglang_omni.models.ming_omni.components.common")
    tokenizer = FakeTokenizer()
    common_module.load_ming_tokenizer = lambda model_path: (model_path, tokenizer)
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.diffusion.zimage_backend",
        zimage_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.components.common",
        common_module,
    )

    executor = MingImageGenExecutor(
        model_path="/fake/model",
        dit_type="zimage",
        dit_model_path="/fake/dit",
        device="cpu",
        **executor_kwargs,
    )

    executor._load_models()

    assert executor._backend is loaded_backends[0]
    assert loaded_backends[0].load_calls[0][0][0] == "/fake/dit"
    assert str(loaded_backends[0].load_calls[0][0][1]) == "cpu"
    assert loaded_backends[0].load_calls[0][1] == expected_load_kwargs
    assert executor._thinker_tokenizer == ("/fake/model", tokenizer)


def test_abort_prevents_generation_and_get_result_skips_aborted_result() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png())
    executor = MingImageGenExecutor(model_path="/fake/model", backend=backend)
    payload = _payload(
        request_id="aborted-before-add",
        metadata={
            "output_modalities": ["image"],
            "image_generation": {"semantic_source": "thinker"},
        },
        data={"generated_text": "draw"},
    )
    later_payload = _payload(request_id="later")

    async def run_request():
        await executor.abort(payload.request_id)
        await executor.add_request(payload)
        assert executor._results.empty()
        await executor._results.put(executor.build_empty_image_result(payload))
        await executor._results.put(executor.build_empty_image_result(later_payload))
        return await executor.get_result()

    result = asyncio.run(run_request())

    assert backend.calls == []
    assert result.request_id == "later"


def test_stop_unloads_and_clears_backend() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend()
    executor = MingImageGenExecutor(model_path="/fake/model", backend=backend)

    asyncio.run(executor.stop())

    assert backend.unloaded is True
    assert executor._backend is None


def test_create_image_gen_executor_passes_explicit_advanced_flags(
    monkeypatch,
) -> None:
    from sglang_omni.models.ming_omni import stages

    constructed = []
    conditioners = []

    class FakeImageGenExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            constructed.append(self)

        def should_generate_image(self, _payload):
            return False

        def build_empty_image_result(self, payload):
            return StagePayload(request_id=payload.request_id, request=payload.request)

    class FakeSemanticConditioner:
        def __init__(self):
            conditioners.append(self)

    class FakeSimpleScheduler:
        def __init__(self, fn):
            self.fn = fn

    image_executor_module = ModuleType(
        "sglang_omni.models.ming_omni.components.image_gen_executor"
    )
    image_executor_module.MingImageGenExecutor = FakeImageGenExecutor
    semantic_conditioner_module = ModuleType(
        "sglang_omni.models.ming_omni.diffusion.semantic_conditioner"
    )
    semantic_conditioner_module.SemanticConditioner = FakeSemanticConditioner
    weight_loader_module = ModuleType("sglang_omni.models.weight_loader")
    weight_loader_module.resolve_model_path = lambda model_path: (
        f"/resolved/{model_path}"
    )
    scheduler_module = ModuleType("sglang_omni.scheduling.simple_scheduler")
    scheduler_module.SimpleScheduler = FakeSimpleScheduler
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.components.image_gen_executor",
        image_executor_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.diffusion.semantic_conditioner",
        semantic_conditioner_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.weight_loader",
        weight_loader_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.scheduling.simple_scheduler",
        scheduler_module,
    )

    stages.create_image_gen_executor(
        "repo",
        dit_type="zimage",
        dit_model_path="/dit",
        device="cpu",
        enable_standalone_semantic_encoder=True,
        enable_byt5_text_rendering=True,
    )

    assert constructed[0].kwargs == {
        "model_path": "/resolved/repo",
        "dit_type": "zimage",
        "dit_model_path": "/dit",
        "device": "cpu",
        "conditioner": conditioners[0],
        "enable_standalone_semantic_encoder": True,
        "enable_byt5_text_rendering": True,
    }


def test_create_image_gen_executor_resolves_model_and_lazy_starts_for_images(
    monkeypatch,
) -> None:
    from sglang_omni.models.ming_omni import stages

    constructed = []

    class FakeImageGenExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = 0
            self.requests = []
            constructed.append(self)

        def should_generate_image(self, payload):
            return payload.request.metadata.get("output_modalities") == ["image"]

        def build_empty_image_result(self, payload):
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={
                    "modality": "image",
                    "images": [],
                    "skipped": True,
                    "finish_reason": "stop",
                },
            )

        async def start(self):
            self.started += 1

        async def add_request(self, payload):
            self.requests.append(payload.request_id)

        async def get_result(self):
            return StagePayload(
                request_id=self.requests[-1],
                request=OmniRequest(inputs=""),
                data={"modality": "image", "images": [], "finish_reason": "stop"},
            )

    class FakeSimpleScheduler:
        def __init__(self, fn):
            self.fn = fn

    conditioners = []

    class FakeSemanticConditioner:
        def __init__(self):
            conditioners.append(self)

    image_executor_module = ModuleType(
        "sglang_omni.models.ming_omni.components.image_gen_executor"
    )
    image_executor_module.MingImageGenExecutor = FakeImageGenExecutor
    semantic_conditioner_module = ModuleType(
        "sglang_omni.models.ming_omni.diffusion.semantic_conditioner"
    )
    semantic_conditioner_module.SemanticConditioner = FakeSemanticConditioner
    weight_loader_module = ModuleType("sglang_omni.models.weight_loader")
    weight_loader_module.resolve_model_path = lambda model_path: (
        f"/resolved/{model_path}"
    )
    scheduler_module = ModuleType("sglang_omni.scheduling.simple_scheduler")
    scheduler_module.SimpleScheduler = FakeSimpleScheduler
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.components.image_gen_executor",
        image_executor_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.ming_omni.diffusion.semantic_conditioner",
        semantic_conditioner_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.weight_loader",
        weight_loader_module,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.scheduling.simple_scheduler",
        scheduler_module,
    )

    scheduler = stages.create_image_gen_executor(
        "repo",
        dit_type="zimage",
        dit_model_path="/dit",
        device="cpu",
    )
    executor = constructed[0]
    assert isinstance(scheduler, FakeSimpleScheduler)
    assert executor.kwargs == {
        "model_path": "/resolved/repo",
        "dit_type": "zimage",
        "dit_model_path": "/dit",
        "device": "cpu",
        "conditioner": conditioners[0],
    }

    text_payload = _payload(
        request_id="text",
        metadata={"output_modalities": ["text"]},
    )
    image_payload_1 = _payload(
        request_id="image-1",
        metadata={"output_modalities": ["image"]},
    )
    image_payload_2 = _payload(
        request_id="image-2",
        metadata={"output_modalities": ["image"]},
    )

    async def run_requests():
        skipped = await scheduler.fn(text_payload)
        first = await scheduler.fn(image_payload_1)
        second = await scheduler.fn(image_payload_2)
        return skipped, first, second

    skipped, first, second = asyncio.run(run_requests())

    assert skipped.data == {
        "modality": "image",
        "images": [],
        "skipped": True,
        "finish_reason": "stop",
    }
    assert executor.started == 1
    assert executor.requests == ["image-1", "image-2"]
    assert first.request_id == "image-1"
    assert second.request_id == "image-2"


def test_import_and_instantiate_with_stub_backend_and_conditioner() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png())
    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=conditioner,
    )

    assert executor._backend is backend
    assert executor._conditioner is conditioner


def test_should_generate_image_and_empty_result_skip_text_only_without_generation(
    monkeypatch,
) -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(
        model_path="/fake/model", backend=StubBackend(image=_png())
    )
    payload = _payload(
        request_id="text-only",
        metadata={"output_modalities": ["text"]},
        data={"thinker_out": {"output_ids": [1, 2, 3]}},
    )
    monkeypatch.setattr(
        executor,
        "_generate_with_condition_embeds",
        lambda *_args, **_kwargs: pytest.fail("text-only request should not generate"),
    )

    async def run_request():
        assert executor.should_generate_image(payload) is False
        empty = executor.build_empty_image_result(payload)
        await executor.add_request(payload)
        return empty, await executor.get_result()

    empty, result = asyncio.run(run_request())

    assert empty.data == result.data
    assert result.data == {
        "modality": "image",
        "images": [],
        "skipped": True,
        "finish_reason": "stop",
    }


def test_try_condition_from_hidden_states_slices_2d_and_string_dict() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(image=_png()),
        conditioner=conditioner,
    )
    lower = torch.zeros(4, 3)
    higher = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    data = {
        "thinker_out": {
            "extra_model_outputs": {"hidden_states": {"2": lower, "10": higher}},
        },
        "mm_inputs": {"image_gen": {"gen_mask": [0, 1, 0, 1]}},
    }

    condition_embeds, negative_embeds = executor._try_condition_from_hidden_states(data)

    expected_query = higher[[1, 3]]
    torch.testing.assert_close(
        conditioner.project_calls[0], expected_query.unsqueeze(0)
    )
    assert len(condition_embeds) == 1
    assert len(negative_embeds) == 1
    torch.testing.assert_close(condition_embeds[0], expected_query + 100)
    torch.testing.assert_close(negative_embeds[0], torch.zeros_like(expected_query))


def test_try_condition_from_hidden_states_integer_dict_selects_highest_key() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(image=_png()),
        conditioner=conditioner,
    )
    lower = torch.zeros(3, 2)
    higher = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    data = {
        "thinker_out": {
            "extra_model_outputs": {"hidden_states": {1: lower, 7: higher}},
        },
        "mm_inputs": {"image_gen": {"gen_mask": [1, 0, 1]}},
    }

    condition_embeds, _negative_embeds = executor._try_condition_from_hidden_states(
        data
    )

    expected_query = higher[[0, 2]]
    torch.testing.assert_close(
        conditioner.project_calls[0], expected_query.unsqueeze(0)
    )
    torch.testing.assert_close(condition_embeds[0], expected_query + 100)


def test_try_condition_from_hidden_states_invalid_shapes_return_none() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(image=_png()),
        conditioner=conditioner,
    )
    mismatched_mask = {
        "thinker_out": {
            "extra_model_outputs": {
                "hidden_states": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            },
        },
        "mm_inputs": {"image_gen": {"gen_mask": [1, 0]}},
    }
    unsupported_hidden_rank = {
        "thinker_out": {
            "extra_model_outputs": {
                "hidden_states": torch.zeros(1, 2, 3, 4),
            },
        },
        "mm_inputs": {"image_gen": {"gen_mask": [1, 0]}},
    }
    nested_mask = {
        "thinker_out": {
            "extra_model_outputs": {
                "hidden_states": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            },
        },
        "mm_inputs": {"image_gen": {"gen_mask": [[1], [0], [1]]}},
    }

    assert executor._try_condition_from_hidden_states(mismatched_mask) == (None, None)
    assert executor._try_condition_from_hidden_states(unsupported_hidden_rank) == (
        None,
        None,
    )
    assert executor._try_condition_from_hidden_states(nested_mask) == (None, None)
    assert conditioner.project_calls == []


def test_add_request_hidden_state_path_returns_png_and_passes_condition_embeds() -> (
    None
):
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png(width=5, height=4))
    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=conditioner,
    )
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    payload = _payload(
        metadata={
            "output_modalities": ["image"],
            "image_generation": {"semantic_source": "thinker"},
        },
        data={
            "thinker_out": {
                "extra_model_outputs": {"hidden_states": hidden},
                "output_ids": [],
            },
            "mm_inputs": {"image_gen": {"gen_mask": [1, 0, 1, 0, 1]}},
            "prompt": {"prompt_text": "draw text"},
        },
    )

    async def run_request():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run_request())

    image = _image_record(result)
    assert image["format"] == "png"
    assert image["width"] == 5
    assert image["height"] == 4
    assert _decode_png_size(image["b64_json"]) == (5, 4)
    prompt, _params, kwargs = backend.calls[0]
    assert prompt == "draw text"
    assert kwargs["condition_embeds"] is not None
    assert kwargs["negative_condition_embeds"] is not None
    torch.testing.assert_close(kwargs["condition_embeds"][0], hidden[[0, 2, 4]] + 100)
    torch.testing.assert_close(
        kwargs["negative_condition_embeds"][0],
        torch.zeros_like(hidden[[0, 2, 4]]),
    )


def test_add_request_hidden_state_generation_error_returns_image_error_payload() -> (
    None
):
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(error=RuntimeError("hidden diffusion failed")),
        conditioner=StubConditioner(),
    )
    payload = _payload(
        metadata={
            "output_modalities": ["image"],
            "image_generation": {"semantic_source": "thinker"},
        },
        data={
            "thinker_out": {
                "extra_model_outputs": {
                    "hidden_states": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                },
            },
            "mm_inputs": {"image_gen": {"gen_mask": [1, 0, 1]}},
        },
    )

    async def run_request():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run_request())

    assert result.data == {
        "modality": "image",
        "images": [],
        "status": "failed",
        "stage": "image_gen",
        "reason": "image_generation_failed",
        "error": "hidden diffusion failed",
        "finish_reason": "error",
    }


def test_add_request_standalone_source_requires_explicit_enable() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png())
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=StubConditioner(),
    )
    payload = _payload(
        metadata={
            "output_modalities": ["image"],
            "image_generation": {"semantic_source": "standalone"},
        },
        data={"prompt": {"prompt_text": "draw"}},
    )

    async def run_request():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run_request())

    assert backend.calls == []
    assert result.data["reason"] == "semantic_conditioning_unavailable"
    assert "standalone semantic encoder" in result.data["error"]


def test_add_request_standalone_source_uses_backend_without_condition_embeds() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png(width=9, height=8))
    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=conditioner,
        enable_standalone_semantic_encoder=True,
    )
    payload = _payload(
        metadata={
            "output_modalities": ["image"],
            "image_generation": {"semantic_source": "standalone"},
        },
        data={"prompt": {"prompt_text": "draw with reference encoder"}},
    )

    async def run_request():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run_request())

    prompt, params, kwargs = backend.calls[0]
    assert prompt == "draw with reference encoder"
    assert params.semantic_source == "standalone"
    assert kwargs == {}
    assert conditioner.project_calls == []
    image = _image_record(result)
    assert image["width"] == 9
    assert image["height"] == 8


def test_add_request_invalid_semantic_source_returns_image_error() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png())
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=StubConditioner(),
    )
    payload = _payload(
        metadata={
            "output_modalities": ["image"],
            "image_generation": {"semantic_source": "random"},
        },
        data={"prompt": {"prompt_text": "draw"}},
    )

    async def run_request():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run_request())

    assert backend.calls == []
    assert result.data["reason"] == "invalid_image_generation_params"
    assert "semantic_source" in result.data["error"]


def test_add_request_missing_semantic_source_defaults_to_thinker() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png())
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=StubConditioner(),
    )
    payload = _payload(
        metadata={"output_modalities": ["image"], "image_generation": {"width": 8}},
        data={"prompt": {"prompt_text": "draw"}},
    )

    async def run_request():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run_request())

    # Missing semantic_source defaults to "thinker", so the request is NOT
    # rejected as invalid; it reaches the thinker path and fails only because
    # no thinker hidden states / gen_mask were captured.
    assert backend.calls == []
    assert result.data["reason"] == "semantic_conditioning_unavailable"


def test_extract_params_prefers_raw_then_mm_then_request_helper() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    executor = MingImageGenExecutor(model_path="/fake/model", backend=StubBackend())
    request = OmniRequest(
        inputs={"image_generation": {"size": "900x901"}},
        metadata={"image_generation": {"size": "800x801"}},
    )

    raw_params = executor._extract_params(
        {
            "raw_inputs": {
                "image_generation": {
                    "size": "512x768",
                    "num_inference_steps": 12,
                    "guidance_scale": 4.5,
                    "seed": 9,
                    "negative_prompt": "blur",
                    "semantic_source": "standalone",
                    "enable_text_rendering": True,
                }
            },
            "mm_inputs": {"image_gen": {"image_gen_params": {"size": "640x640"}}},
        },
        request,
    )
    mm_params = executor._extract_params(
        {"mm_inputs": {"image_gen": {"image_gen_params": {"size": "640x672"}}}},
        request,
    )
    request_params = executor._extract_params({}, request)

    assert (raw_params.width, raw_params.height) == (512, 768)
    assert raw_params.num_inference_steps == 12
    assert raw_params.guidance_scale == 4.5
    assert raw_params.seed == 9
    assert raw_params.negative_prompt == "blur"
    assert raw_params.semantic_source == "standalone"
    assert raw_params.enable_text_rendering is True
    assert (mm_params.width, mm_params.height) == (640, 672)
    assert mm_params.semantic_source == "thinker"
    assert (request_params.width, request_params.height) == (800, 801)
    assert request_params.semantic_source == "thinker"
    raw_without_semantic_source = executor._extract_params(
        {"raw_inputs": {"image_generation": {"size": "64x64"}}}, None
    )
    bogus = executor._extract_params(
        {"raw_inputs": {"image_generation": {"semantic_source": "bogus"}}}, None
    )
    assert raw_without_semantic_source.semantic_source == "thinker"
    assert bogus.semantic_source == "bogus"


def test_try_condition_tail_aligns_mask_when_hidden_states_are_suffix() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(image=_png()),
        conditioner=conditioner,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    data = {
        "thinker_out": {"extra_model_outputs": {"hidden_states": hidden}},
        "mm_inputs": {"image_gen": {"gen_mask": [0, 0, 0, 1, 0, 1]}},
    }

    condition_embeds, negative_embeds = executor._try_condition_from_hidden_states(data)

    expected_query = hidden[[1, 3]]
    torch.testing.assert_close(
        conditioner.project_calls[0], expected_query.unsqueeze(0)
    )
    torch.testing.assert_close(condition_embeds[0], expected_query + 100)
    torch.testing.assert_close(negative_embeds[0], torch.zeros_like(expected_query))


def test_try_condition_drops_conditioning_when_tail_misses_query_positions(
    caplog,
) -> None:
    torch = pytest.importorskip("torch")
    import logging

    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(image=_png()),
        conditioner=conditioner,
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    data = {
        "thinker_out": {"extra_model_outputs": {"hidden_states": hidden}},
        "mm_inputs": {"image_gen": {"gen_mask": [0, 1, 0, 1, 0, 0]}},
    }

    with caplog.at_level(logging.WARNING):
        result = executor._try_condition_from_hidden_states(data)

    assert result == (None, None)
    assert conditioner.project_calls == []
    assert any("query positions" in record.message for record in caplog.records)


def test_try_condition_accepts_3d_batched_hidden_states() -> None:
    import torch

    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(model_path="/fake", conditioner=conditioner)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    data = {
        "thinker_out": {"extra_model_outputs": {"hidden_states": hidden}},
        "mm_inputs": {"image_gen": {"gen_mask": [1, 0, 1, 0]}},
    }
    cond, neg = executor._try_condition_from_hidden_states(data)

    expected_query = hidden[:, [0, 2], :]
    torch.testing.assert_close(conditioner.project_calls[0], expected_query)
    torch.testing.assert_close(cond[0], expected_query[0] + 100)
    torch.testing.assert_close(neg[0], torch.zeros_like(expected_query[0]))


def test_conditioner_set_but_no_hidden_states_fails_loudly() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    backend = StubBackend(image=_png())
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=backend,
        conditioner=StubConditioner(),
    )
    payload = _payload(
        metadata={
            "output_modalities": ["image"],
            "image_generation": {
                "width": 8,
                "height": 8,
                "semantic_source": "thinker",
            },
        },
        data={"thinker_out": {"output_ids": [1, 2, 3]}},
    )

    async def run():
        await executor.add_request(payload)
        return await executor.get_result()

    result = asyncio.run(run())
    assert result.data["images"] == []
    assert result.data["status"] == "failed"
    assert result.data["stage"] == "image_gen"
    assert result.data["reason"] == "semantic_conditioning_unavailable"
    assert "semantic conditioning unavailable" in result.data["error"]
    assert backend.calls == []


def test_try_condition_returns_none_when_no_hidden_states_captured() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(
        model_path="/fake/model",
        backend=StubBackend(image=_png()),
        conditioner=conditioner,
    )
    hidden = torch.zeros(0, 3, dtype=torch.float32)
    data = {
        "thinker_out": {"extra_model_outputs": {"hidden_states": hidden}},
        "mm_inputs": {"image_gen": {"gen_mask": [0, 1, 0, 1]}},
    }

    result = executor._try_condition_from_hidden_states(data)

    assert result == (None, None)
    assert conditioner.project_calls == []


def test_try_condition_reads_via_typed_state() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        MingImageGenExecutor,
    )

    conditioner = StubConditioner()
    executor = MingImageGenExecutor(model_path="/fake", conditioner=conditioner)
    hs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    data = {
        "thinker_out": {"extra_model_outputs": {"hidden_states": hs}},
        "mm_inputs": {"image_gen": {"gen_mask": [0, 1, 0, 1]}},
    }
    cond, neg = executor._try_condition_from_hidden_states(data)
    assert cond is not None and neg is not None
    assert cond[0].shape == (2, 2)
    expected_query = hs[[1, 3]]
    torch.testing.assert_close(cond[0], expected_query + 100)
    torch.testing.assert_close(neg[0], torch.zeros(2, 2))
