# SPDX-License-Identifier: Apache-2.0
"""Helpers to build engine request data from pipeline payloads."""

from __future__ import annotations

from typing import Any

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.proto import StagePayload


def build_encoder_request(request_id: str, data: Any) -> EncoderRequestData:
    """Build EncoderRequestData from StagePayload or raw input ids."""
    del request_id
    input_ids = data.data if isinstance(data, StagePayload) else data
    return EncoderRequestData(input_ids=input_ids)


def build_ar_request(request_id: str, data: Any) -> ARRequestData:
    """Build ARRequestData from StagePayload or raw input ids."""
    del request_id
    params: dict[str, Any] = {}
    input_ids = data
    if isinstance(data, StagePayload):
        input_ids = data.data
        params = data.request.params

    return ARRequestData(
        input_ids=input_ids,
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )
