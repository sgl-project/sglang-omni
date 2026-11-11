# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest


def test_processor_class_keeps_hub_repo_id(monkeypatch, tmp_path) -> None:
    from transformers import dynamic_module_utils
    from transformers.utils import hub

    from sglang_omni.models.moss_tts.hf_loading import load_moss_processor_class

    processor_config = tmp_path / "processor_config.json"
    processor_config.write_text(
        json.dumps(
            {"auto_map": {"AutoProcessor": "processing_moss_tts.MossProcessor"}}
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeProcessor:
        attributes = ["feature_extractor", "tokenizer"]

    def fake_cached_file(checkpoint, filename):
        calls["cached_file"] = (checkpoint, filename)
        return str(processor_config)

    def fake_get_class(class_ref, checkpoint):
        calls["get_class"] = (class_ref, checkpoint)
        return FakeProcessor

    monkeypatch.setattr(hub, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        dynamic_module_utils, "get_class_from_dynamic_module", fake_get_class
    )

    processor_cls = load_moss_processor_class("org/moss-checkpoint")

    assert calls == {
        "cached_file": ("org/moss-checkpoint", "processor_config.json"),
        "get_class": (
            "processing_moss_tts.MossProcessor",
            "org/moss-checkpoint",
        ),
    }
    assert processor_cls is FakeProcessor
    assert processor_cls.attributes == ["tokenizer"]


def test_processor_compat_restores_nested_patches_after_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.configuration_utils as configuration_utils
    from transformers import PreTrainedModel, processing_utils

    from sglang_omni.models.moss_tts.hf_loading import (
        moss_transformers_processor_compat,
    )

    sentinel = object()
    tokenizer_sentinel = object()
    auto_mapping = {"existing": sentinel}
    monkeypatch.setattr(
        processing_utils, "AUTO_TO_BASE_CLASS_MAPPING", auto_mapping, raising=False
    )
    monkeypatch.delattr(
        processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING", raising=False
    )
    monkeypatch.delattr(configuration_utils, "PreTrainedConfig", raising=False)
    monkeypatch.setattr(
        processing_utils,
        "PreTrainedAudioTokenizerBase",
        tokenizer_sentinel,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="outer"):
        with moss_transformers_processor_compat():
            with pytest.raises(RuntimeError, match="inner"):
                with moss_transformers_processor_compat():
                    raise RuntimeError("inner")
            assert (
                vars(configuration_utils)["PreTrainedConfig"]
                is configuration_utils.PretrainedConfig
            )
            assert vars(processing_utils)["MODALITY_TO_BASE_CLASS_MAPPING"] is (
                auto_mapping
            )
            assert auto_mapping["AutoModel"] == "PreTrainedModel"
            assert (
                vars(processing_utils)["PreTrainedAudioTokenizerBase"]
                is PreTrainedModel
            )
            raise RuntimeError("outer")

    assert "PreTrainedConfig" not in vars(configuration_utils)
    assert "MODALITY_TO_BASE_CLASS_MAPPING" not in vars(processing_utils)
    assert vars(processing_utils)["AUTO_TO_BASE_CLASS_MAPPING"] is auto_mapping
    assert auto_mapping == {"existing": sentinel}
    assert vars(processing_utils)["PreTrainedAudioTokenizerBase"] is tokenizer_sentinel
