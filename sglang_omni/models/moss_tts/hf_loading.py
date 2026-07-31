# SPDX-License-Identifier: Apache-2.0
"""Shared Hugging Face loading helpers for MOSS TTS models."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def moss_transformers_processor_compat() -> Iterator[None]:
    """Scope Transformers API-drift patches to MOSS processor/code loading."""
    import transformers.configuration_utils as configuration_utils
    from transformers import PreTrainedModel, processing_utils

    missing = object()
    undo: list[tuple[str, Any, str, Any]] = []

    def patch_attr(obj: Any, name: str, value: Any) -> None:
        undo.append(("attr", obj, name, getattr(obj, name, missing)))
        setattr(obj, name, value)

    def patch_item(mapping: dict, key: str, value: Any) -> None:
        undo.append(("item", mapping, key, mapping.get(key, missing)))
        mapping[key] = value

    try:
        if not hasattr(configuration_utils, "PreTrainedConfig"):
            patch_attr(
                configuration_utils,
                "PreTrainedConfig",
                configuration_utils.PretrainedConfig,
            )
        auto_mapping = getattr(processing_utils, "AUTO_TO_BASE_CLASS_MAPPING", None)
        if isinstance(auto_mapping, dict):
            if "AutoModel" not in auto_mapping:
                patch_item(auto_mapping, "AutoModel", "PreTrainedModel")
            if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
                patch_attr(
                    processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING", auto_mapping
                )
        if hasattr(processing_utils, "PreTrainedAudioTokenizerBase"):
            patch_attr(
                processing_utils, "PreTrainedAudioTokenizerBase", PreTrainedModel
            )
        yield
    finally:
        for kind, obj, key, old in reversed(undo):
            if kind == "attr":
                if old is missing:
                    if hasattr(obj, key):
                        delattr(obj, key)
                else:
                    setattr(obj, key, old)
            elif old is missing:
                obj.pop(key, None)
            else:
                obj[key] = old


def load_moss_processor_class(checkpoint: str) -> type:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.utils.hub import cached_file

    processor_config_path = cached_file(checkpoint, "processor_config.json")
    if processor_config_path is None:
        raise RuntimeError("MOSS-TTS checkpoint lacks processor_config.json")
    with open(processor_config_path, encoding="utf-8") as f:
        processor_config = json.load(f)

    class_ref = (processor_config.get("auto_map") or {}).get("AutoProcessor")
    if not class_ref:
        raise RuntimeError("MOSS-TTS processor_config.json lacks AutoProcessor map")

    # Keep Hub repo IDs as repo IDs. Turning one into snapshot_download's
    # symlink-based cache directory makes recent Transformers releases treat
    # the blob targets as a local source tree. Relative imports are then
    # resolved beside ``blobs/<hash>`` instead of beside the snapshot module.
    processor_cls = get_class_from_dynamic_module(class_ref, checkpoint)
    if list(getattr(processor_cls, "attributes", [])) == [
        "feature_extractor",
        "tokenizer",
    ]:
        processor_cls.attributes = ["tokenizer"]
    return processor_cls
