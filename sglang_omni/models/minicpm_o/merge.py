# SPDX-License-Identifier: Apache-2.0
"""Join preprocessing and encoder branches into thinker inputs."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.proto import StagePayload


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Merge fan-in payloads into a single thinker-ready payload.

    The preprocessing payload carries prompt/mm metadata; each encoder payload
    carries its ``encoder_outs`` entry. The merged state moves encoder outputs
    into ``thinker_inputs["model_inputs"]`` and drops the raw encoder fields so
    embeddings do not cross the wire twice. A text-only request merges to the
    canonical empty shape ``{"model_inputs": {}}``.
    """
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = MiniCPMOPipelineState.from_dict(base.data)

    model_inputs: dict[str, Any] = {}
    for payload in payloads.values():
        branch = MiniCPMOPipelineState.from_dict(payload.data)
        for encoder_out in branch.encoder_outs.values():
            if isinstance(encoder_out, dict):
                model_inputs.update(encoder_out)

    state.thinker_inputs = {"model_inputs": model_inputs}
    state.encoder_inputs = {}
    state.encoder_outs = {}
    return StagePayload(
        request_id=base.request_id,
        request=base.request,
        data=state.to_dict(),
    )
