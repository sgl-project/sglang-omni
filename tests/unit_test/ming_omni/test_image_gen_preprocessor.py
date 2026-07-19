# SPDX-License-Identifier: Apache-2.0
"""Ming image-generation preprocessor tests."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_omni.io import MingOmniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload

MING_PREPROCESSOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "ming_omni"
    / "components"
    / "preprocessor.py"
)


class FakeTokenizer:
    unk_token_id = -1
    _ids = {
        "<audio>": 10,
        "</audio>": 11,
        "<audioPatch>": 12,
        "<image>": 20,
        "</image>": 21,
        "<imagePatch>": 22,
    }

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self._ids and "Patch" not in text:
            return [self._ids[text]]
        return [1000 + (ord(ch) % 127) for ch in str(text)]


def _import_preprocessor(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.preprocessing",
        ModuleType("sglang_omni.preprocessing"),
    )

    audio_module = ModuleType("sglang_omni.preprocessing.audio")
    audio_module.compute_audio_cache_key = lambda _audios: "audio-cache"
    audio_module.load_audio_path = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "sglang_omni.preprocessing.audio", audio_module)

    image_module = ModuleType("sglang_omni.preprocessing.image")
    image_module.compute_image_cache_key = lambda _images: "image-cache"
    image_module.ensure_image_list_async = lambda _images: []
    monkeypatch.setitem(sys.modules, "sglang_omni.preprocessing.image", image_module)

    video_module = ModuleType("sglang_omni.preprocessing.video")
    video_module.compute_video_cache_key = lambda _videos: "video-cache"
    video_module.ensure_video_list_async = lambda _videos: []
    monkeypatch.setitem(sys.modules, "sglang_omni.preprocessing.video", video_module)

    module_name = "_ming_image_gen_preprocessor_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MING_PREPROCESSOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _install_common_deps(monkeypatch, module, *, query_tokens=None):
    monkeypatch.setattr(module, "load_ming_tokenizer", lambda _path: FakeTokenizer())
    monkeypatch.setattr(
        module,
        "load_ming_config",
        lambda _path: SimpleNamespace(
            audio_config=SimpleNamespace(ds_kernel_size=1, ds_stride=1),
            vision_config=SimpleNamespace(
                patch_size=14,
                temporal_patch_size=2,
                spatial_merge_size=2,
            ),
            llm_config=SimpleNamespace(image_patch_token=22),
        ),
    )
    if query_tokens is not None:
        from sglang_omni.models.ming_omni.diffusion.query_info import ImageGenQueryInfo

        monkeypatch.setattr(
            module,
            "load_image_gen_query_info",
            lambda _path: ImageGenQueryInfo(
                query_tokens=query_tokens.to(dtype=torch.bfloat16),
                image_patch_token_id=22,
            ),
        )


def _run_preprocessor(processor, payload):
    return asyncio.run(processor(payload))


def test_image_gen_preprocessor_appends_query_tokens_and_aligns_gen_mask(
    monkeypatch,
) -> None:
    module = _import_preprocessor(monkeypatch)

    _install_common_deps(
        monkeypatch,
        module,
        query_tokens=torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )
    processor = module.MingPreprocessor("fake-model", enable_image_gen=True)
    payload = StagePayload(
        request_id="req-img",
        request=OmniRequest(
            inputs={"messages": [{"role": "user", "content": "draw a cat"}]},
            metadata={
                "output_modalities": ["image"],
                "image_generation": {"size": "512x512"},
            },
        ),
        data={},
    )

    result = _run_preprocessor(processor, payload)
    state = MingOmniPipelineState.from_dict(result.data)
    image_gen = state.mm_inputs["image_gen"]
    input_ids = state.prompt["input_ids"].flatten().tolist()
    gen_mask = image_gen["gen_mask"]

    assert input_ids.count(22) == 3
    assert gen_mask.count(1) == 3
    assert len(gen_mask) == len(input_ids)
    assert image_gen["query_tokens"][2][3] == pytest.approx(11.0)
    assert image_gen["prefill_only"] is True
    assert image_gen["image_patch_token_id"] == 22
    assert image_gen["image_gen_params"] == {"size": "512x512"}
    assert len(gen_mask) == len(input_ids)
    assert len(gen_mask) == state.prompt["input_ids"].shape[-1]

    one_positions = [i for i, v in enumerate(gen_mask) if v == 1]
    assert one_positions == list(range(one_positions[0], one_positions[-1] + 1))
    assert len(one_positions) == 3

    patch_id = image_gen["image_patch_token_id"]
    assert all(input_ids[i] == patch_id for i in one_positions)
    start_idx = one_positions[0] - 1
    end_idx = one_positions[-1] + 1
    assert gen_mask[start_idx] == 0  # <image> start marker excluded
    assert gen_mask[end_idx] == 0  # </image> end marker excluded
    assert input_ids[start_idx] == 20
    assert input_ids[end_idx] == 21


def test_image_gen_preprocessor_disabled_by_default(monkeypatch) -> None:
    module = _import_preprocessor(monkeypatch)

    _install_common_deps(
        monkeypatch,
        module,
        query_tokens=torch.ones((2, 4), dtype=torch.float32),
    )
    processor = module.MingPreprocessor("fake-model")
    payload = StagePayload(
        request_id="req-text",
        request=OmniRequest(
            inputs={"messages": [{"role": "user", "content": "draw a cat"}]},
            metadata={
                "output_modalities": ["image"],
                "image_generation": {"size": "512x512"},
            },
        ),
        data={},
    )

    result = _run_preprocessor(processor, payload)
    state = MingOmniPipelineState.from_dict(result.data)

    assert "image_gen" not in state.mm_inputs


def test_image_gen_preprocessor_rejects_input_images_before_loading(
    monkeypatch,
) -> None:
    module = _import_preprocessor(monkeypatch)

    _install_common_deps(
        monkeypatch,
        module,
        query_tokens=torch.ones((2, 4), dtype=torch.float32),
    )

    def fail_load_images(_images):
        raise AssertionError("input images should be rejected before loading")

    monkeypatch.setattr(module, "ensure_image_list_async", fail_load_images)
    processor = module.MingPreprocessor("fake-model", enable_image_gen=True)
    payload = StagePayload(
        request_id="req-mixed",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "edit this"}],
                "images": ["file:///tmp/in.png"],
            },
            metadata={
                "output_modalities": ["image"],
                "image_generation": {"size": "512x512"},
            },
        ),
        data={},
    )

    with pytest.raises(ValueError, match="input images"):
        _run_preprocessor(processor, payload)
