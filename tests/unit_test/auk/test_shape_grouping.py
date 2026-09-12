# SPDX-License-Identifier: Apache-2.0
"""AuK opt-in shape-aware DiT grouping."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.auk.config import AuKPipelineConfig
from sglang_omni.models.auk.flow_matching import AuKSampleItem
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.stages import (
    _partition_sample_items_by_target,
    _sample_batch,
    create_auk_engine_executor,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _item(
    target_frames: int, reference_frames: int = 20, text_frames: int = 10
) -> AuKSampleItem:
    return AuKSampleItem(
        conditioning=torch.zeros(text_frames, 4),
        text_mask=torch.ones(text_frames, dtype=torch.bool),
        target_frames=target_frames,
        ref_latent=torch.zeros(reference_frames, 4),
        ref_length=reference_frames,
    )


def test_shape_grouping_is_disabled_by_default():
    items = [_item(frames) for frames in (100, 110, 400)]

    groups = _partition_sample_items_by_target(items, None)

    assert [[index for index, _ in group] for group in groups] == [[0, 1, 2]]
    engine = next(
        stage
        for stage in AuKPipelineConfig.model_fields["stages"].default
        if stage.name == "auk_engine"
    )
    assert engine.factory.min_batch_work_savings is None


def test_shape_grouping_is_bounded_to_two_groups_and_preserves_indices():
    items = [_item(frames) for frames in (400, 100, 110)]

    groups = _partition_sample_items_by_target(items, 0.25)

    assert [[index for index, _ in group] for group in groups] == [[1, 2], [0]]


def test_shape_grouping_keeps_similar_shapes_together():
    items = [_item(frames) for frames in (100, 110, 120)]

    groups = _partition_sample_items_by_target(items, 0.25)

    assert [[index for index, _ in group] for group in groups] == [[0, 1, 2]]


def test_shape_grouping_rejects_invalid_savings_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        _partition_sample_items_by_target([_item(100), _item(400)], 1.1)


def test_shape_grouping_scores_reference_and_text_padding():
    items = [
        _item(100, reference_frames=400, text_frames=10),
        _item(110, reference_frames=10, text_frames=400),
        _item(400, reference_frames=400, text_frames=400),
    ]

    groups = _partition_sample_items_by_target(items, 0.2)

    assert [[index for index, _ in group] for group in groups] == [[0, 1, 2]]


def test_shape_grouping_rejects_invalid_threshold_before_loading_model():
    with pytest.raises(ValueError, match="between 0 and 1"):
        create_auk_engine_executor("unused", min_batch_work_savings=1.1)


def test_sample_batch_restores_request_order_after_split():
    payloads = [
        StagePayload(
            request_id=str(index),
            request=OmniRequest(inputs="hello"),
            data=AuKState(
                gen_frames=frames,
                conditioning=torch.zeros(10, 4),
                text_mask=torch.ones(10, dtype=torch.bool),
                ref_latent=torch.zeros(20, 4),
                ref_length=20,
            ).to_dict(),
        )
        for index, frames in enumerate((400, 100, 110))
    ]
    flow = Mock()
    flow.sample_batch.side_effect = lambda items, **_: [
        torch.full((item.target_frames, 4), item.target_frames) for item in items
    ]

    sampled = _sample_batch(
        payloads,
        flow,
        torch.device("cpu"),
        "float32",
        500,
        {},
        min_batch_work_savings=0.2,
    )

    assert [
        [item.target_frames for item in call.args[0]]
        for call in flow.sample_batch.call_args_list
    ] == [[100, 110], [400]]
    states = [AuKState.from_dict(payload.data) for payload in sampled]
    assert [state.latent.shape[0] for state in states] == [400, 100, 110]
    assert [state.latent[0, 0].item() for state in states] == [400, 100, 110]
