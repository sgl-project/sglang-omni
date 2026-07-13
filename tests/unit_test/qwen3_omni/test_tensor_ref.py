# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni TensorRef configuration and relay-merge contracts."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.comm.data_ref import (
    BackendRef,
    DataKind,
    DataLayout,
    DataRef,
    TransportKind,
)
from sglang_omni.comm.tensor_ref import TensorRef
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.models.qwen3_omni.merge import merge_for_thinker
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload, make_qwen_state

_TENSOR_REF_BYTES = 3 * 1024 * 1024


def _visual_deepstack_ref(modality: str) -> TensorRef:
    return TensorRef(
        request_id="req-1",
        producer_stage="image_encoder",
        consumer_stage="thinker",
        path=("encoder_outs.image_encoder." f"deepstack_visual_embeds_{modality}[0]"),
        nbytes=_TENSOR_REF_BYTES,
        data_ref=DataRef(
            version=1,
            kind=DataKind.TENSOR_REF,
            object_id=f"tensor-ref-{modality}",
            transport=TransportKind.SHM,
            layout=DataLayout.RAW_TENSOR,
            buffer=BackendRef(
                transport=TransportKind.SHM,
                info={"transfer_info": {"size": _TENSOR_REF_BYTES}},
                length=_TENSOR_REF_BYTES,
            ),
            shape=(_TENSOR_REF_BYTES // 4,),
            dtype="torch.float32",
            offset=0,
        ),
    )


def test_qwen_config_declares_visual_tensor_ref_edge() -> None:
    config = Qwen3OmniPipelineConfig(model_path="model")
    image_encoder = next(
        stage for stage in config.stages if stage.name == "image_encoder"
    )
    edge = image_encoder.tensor_ref_edges["mm_aggregate"]

    assert edge.consumer_stage == "thinker"
    assert edge.threshold_mb == 2.0
    assert edge.paths == (
        "encoder_outs.image_encoder.deepstack_visual_embeds_image",
        "encoder_outs.image_encoder.deepstack_visual_embeds_video",
    )


@pytest.mark.parametrize(
    "deepstack_output_keys",
    [
        {"image": "deepstack_visual_embeds"},
        {"video": "deepstack_visual_embeds"},
        {
            "image": "image_deepstack_visual_embeds",
            "video": "video_deepstack_visual_embeds",
        },
    ],
    ids=("image-only", "video-only", "image-and-video"),
)
def test_qwen_mm_aggregate_forwards_configured_deepstack_tensor_refs(
    deepstack_output_keys: dict[str, str],
) -> None:
    refs = {
        modality: _visual_deepstack_ref(modality) for modality in deepstack_output_keys
    }
    encoder_out: dict[str, object] = {}
    for modality, ref in refs.items():
        encoder_out[f"{modality}_embeds"] = torch.ones((1, 1))
        encoder_out[f"deepstack_visual_embeds_{modality}"] = [ref.to_dict()]

    preprocessing = make_qwen_payload(
        make_qwen_state(mm_inputs={"image": {}}), request_id="req-1"
    )
    image_encoder = make_qwen_payload(
        Qwen3OmniPipelineState(encoder_outs={"image_encoder": encoder_out}),
        request_id="req-1",
    )

    merged = merge_for_thinker(
        {"preprocessing": preprocessing, "image_encoder": image_encoder}
    )
    state = Qwen3OmniPipelineState.from_dict(merged.data)
    model_inputs = state.thinker_inputs["model_inputs"]

    assert {key for key in model_inputs if "deepstack" in key} == set(
        deepstack_output_keys.values()
    )
    for modality, output_key in deepstack_output_keys.items():
        assert model_inputs[output_key] == [refs[modality].to_dict()]
