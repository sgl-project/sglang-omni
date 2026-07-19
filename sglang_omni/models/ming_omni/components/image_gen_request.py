# SPDX-License-Identifier: Apache-2.0
"""Shared Ming image-generation request helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sglang_omni.proto import StagePayload

IMAGE_OUTPUT_MODALITY = "image"
IMAGE_GENERATION_KEY = "image_generation"
OUTPUT_MODALITIES_KEY = "output_modalities"


def _metadata(payload: StagePayload) -> dict[str, Any]:
    request = getattr(payload, "request", None)
    metadata = getattr(request, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def output_modalities(payload: StagePayload) -> tuple[str, ...]:
    metadata = _metadata(payload)
    raw = metadata.get(OUTPUT_MODALITIES_KEY)
    if raw is None:
        return ("text",)
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple, set)):
        return tuple(str(item) for item in raw)
    return ("text",)


def should_generate_image(payload: StagePayload) -> bool:
    return IMAGE_OUTPUT_MODALITY in output_modalities(payload)


def extract_image_generation_params(payload: StagePayload) -> dict[str, Any]:
    metadata = _metadata(payload)
    params = metadata.get(IMAGE_GENERATION_KEY)
    if isinstance(params, Mapping):
        return dict(params)

    request = getattr(payload, "request", None)
    inputs = getattr(request, "inputs", None)
    if isinstance(inputs, Mapping):
        input_params = inputs.get(IMAGE_GENERATION_KEY)
        if isinstance(input_params, Mapping):
            return dict(input_params)
    return {}
