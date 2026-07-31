# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json


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
