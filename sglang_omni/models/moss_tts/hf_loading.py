# SPDX-License-Identifier: Apache-2.0
"""Shared Hugging Face loading helpers for MOSS TTS models."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from numbers import Integral, Real
from typing import Any

from sglang.srt.utils.hf_transformers import (
    CONTEXT_LENGTH_KEYS,
    get_config,
    get_context_length,
    get_hf_text_config,
)

MOSS_TTS_DEFAULT_CONTEXT_LENGTH = 8192


def validate_context_length_metadata(text_config: Any) -> bool:
    context_value = None
    for key in CONTEXT_LENGTH_KEYS:
        value = getattr(text_config, key, None)
        if value is None:
            continue
        else:
            pass
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"MOSS-TTS context metadata {key} must be an integer, got {value!r}"
            )
        else:
            pass
        context_value = int(value)
        if context_value <= 0:
            raise ValueError(
                f"MOSS-TTS context metadata {key} must be a positive integer, "
                f"got {value!r}"
            )
        else:
            pass
        break
    if context_value is None:
        return False
    else:
        pass
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if not rope_scaling:
        return True
    else:
        pass
    if not isinstance(rope_scaling, Mapping):
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling must be a mapping, "
            f"got {rope_scaling!r}"
        )
    else:
        pass
    if (
        "original_max_position_embeddings" in rope_scaling
        or rope_scaling.get("rope_type") == "llama3"
    ):
        return True
    else:
        pass
    factor = rope_scaling.get("factor", 1)
    if isinstance(factor, bool) or not isinstance(factor, Real):
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling.factor must be a finite "
            f"number, got {factor!r}"
        )
    else:
        pass
    try:
        factor_is_finite = math.isfinite(float(factor))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling.factor must be a finite "
            f"number, got {factor!r}"
        ) from exc
    if not factor_is_finite:
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling.factor must be a finite "
            f"number, got {factor!r}"
        )
    else:
        pass
    try:
        scaled_value = factor * context_value
        scaled_integer = int(scaled_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling.factor produces an "
            f"invalid context length: {factor!r} * {context_value!r}"
        ) from exc
    if scaled_value != scaled_integer:
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling.factor must produce an "
            f"integer context length, got {factor!r} * {context_value!r}"
        )
    else:
        pass
    if scaled_integer <= 0:
        raise ValueError(
            "MOSS-TTS context metadata rope_scaling.factor must produce a "
            f"positive context length, got {factor!r} * {context_value!r}"
        )
    else:
        pass
    return True


def resolve_moss_tts_context_length(
    checkpoint_dir: str,
    *,
    server_args_overrides: Mapping[str, Any] | None = None,
) -> int:
    """Resolve MOSS-TTS text context from the runtime model settings."""
    overrides = server_args_overrides or {}
    raw_model_override_args = overrides.get("json_model_override_args", "{}")
    try:
        model_override_args = json.loads(raw_model_override_args)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "json_model_override_args must be a valid JSON object string"
        ) from exc
    if not isinstance(model_override_args, Mapping):
        raise ValueError("json_model_override_args must decode to a JSON object")
    else:
        pass

    config_kwargs: dict[str, Any] = {
        "trust_remote_code": overrides.get("trust_remote_code", True),
        "model_config_parser": overrides.get("model_config_parser", "auto"),
        "model_override_args": dict(model_override_args),
    }
    config_file = overrides.get("decrypted_config_file")
    if config_file and config_file.strip():
        config_kwargs["_configuration_file"] = config_file.strip()
    else:
        pass
    config = copy.deepcopy(get_config(checkpoint_dir, **config_kwargs))
    text_config = get_hf_text_config(config)
    if not validate_context_length_metadata(text_config):
        return MOSS_TTS_DEFAULT_CONTEXT_LENGTH
    else:
        pass
    try:
        return int(get_context_length(text_config))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("MOSS-TTS context metadata is invalid") from exc


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
        else:
            pass
        auto_mapping = getattr(processing_utils, "AUTO_TO_BASE_CLASS_MAPPING", None)
        if isinstance(auto_mapping, dict):
            if "AutoModel" not in auto_mapping:
                patch_item(auto_mapping, "AutoModel", "PreTrainedModel")
            else:
                pass
            if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
                patch_attr(
                    processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING", auto_mapping
                )
            else:
                pass
        else:
            pass
        if hasattr(processing_utils, "PreTrainedAudioTokenizerBase"):
            patch_attr(
                processing_utils, "PreTrainedAudioTokenizerBase", PreTrainedModel
            )
        else:
            pass
        yield
    finally:
        for kind, obj, key, old in reversed(undo):
            if kind == "attr":
                if old is missing:
                    if hasattr(obj, key):
                        delattr(obj, key)
                    else:
                        pass
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
    else:
        pass
    with open(processor_config_path, encoding="utf-8") as f:
        processor_config = json.load(f)

    class_ref = (processor_config.get("auto_map") or {}).get("AutoProcessor")
    if not class_ref:
        raise RuntimeError("MOSS-TTS processor_config.json lacks AutoProcessor map")
    else:
        pass

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
    else:
        pass
    return processor_cls
