# SPDX-License-Identifier: Apache-2.0
"""Helpers to build engine request data from StagePayloads."""

from __future__ import annotations

from typing import Any

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.proto import StagePayload


def _extract_input_ids(payload: StagePayload) -> tuple[Any, dict[str, Any]]:
    params = payload.request.params
    data = payload.data
    if isinstance(data, dict):
        input_ids = data.get("input_ids")
        if input_ids is None:
            input_ids = data.get("raw_inputs", data)
    else:
        input_ids = data

    return input_ids, params


def build_encoder_request(payload: StagePayload) -> EncoderRequestData:
    """Build EncoderRequestData from StagePayload."""
    input_ids, _ = _extract_input_ids(payload)
    return EncoderRequestData(input_ids=input_ids)


def build_ar_request(payload: StagePayload) -> ARRequestData:
    """Build ARRequestData from StagePayload."""
    input_ids, params = _extract_input_ids(payload)
    return ARRequestData(
        input_ids=input_ids,
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )
