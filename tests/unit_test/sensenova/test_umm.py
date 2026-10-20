# SPDX-License-Identifier: Apache-2.0
"""SenseNova phase, output, and Images API contracts on the real controller."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image
from pydantic import JsonValue

from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import UMMGenerateChunk
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.sensenova_u1.cancellation import (
    SENSENOVA_DEADLINE_METADATA_KEY,
    GenerationScope,
    SenseNovaCancelledError,
    generation_checkpoint,
)
from sglang_omni.models.sensenova_u1.image_processing import prepare_reference_images
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling
from sglang_omni.models.sensenova_u1.umm import SenseNovaUMMAdapter
from sglang_omni.models.sensenova_u1.umm_config import SenseNovaU1UMMPipelineConfig
from sglang_omni.pipeline.umm import (
    UMMController,
    UMMRequestError,
    create_umm_controller,
)
from sglang_omni.proto.request import OmniRequest, StagePayload

pytest_plugins = ["tests.unit_test.fixtures.sensenova_umm"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("image_count", [1, 3])
def test_outputs_survive_controller_completion(
    controller: UMMController,
    encoded_images: list[str],
    streaming: bool,
    image_count: int,
) -> None:
    request = OmniRequest("a red mug", params={"n": image_count, "stream": streaming})
    phase = controller.transition(StagePayload("request", request, None))
    assert phase is not None
    assert phase.request.params["stream"] is False
    phase.data = {"modality": "image", "images_b64": encoded_images[:image_count]}
    result = controller.transition(phase)
    assert result is not None
    completion = Client.default_result_builder("request", result.data)
    assert isinstance(completion, UMMGenerateChunk)
    assert completion.segments is not None
    assert [segment.segment_index for segment in completion.segments] == list(
        range(image_count)
    )
    assert [segment.kind for segment in completion.segments] == ["image"] * image_count
    if streaming:
        segments = [controller.outbox.get_nowait().data for _ in range(image_count)]
        assert all(segment.data is None for segment in completion.segments)
        assert [
            InlineImage.model_validate(segment["data"]).encoded_image()
            for segment in segments
        ] == encoded_images[:image_count]
    else:
        assert [
            InlineImage.model_validate(segment.data).encoded_image()
            for segment in completion.segments
        ] == encoded_images[:image_count]
    assert not controller.contexts
    assert not controller.adapter.executions


def test_bad_result_releases_adapter_state(
    controller: UMMController, encoded_image: str
) -> None:
    phase = controller.transition(
        StagePayload("request", OmniRequest("prompt", {"n": 2}), None)
    )
    assert phase is not None
    phase.data = {"modality": "image", "image_b64": encoded_image}
    with pytest.raises(ValueError, match="output count"):
        controller.transition(phase)
    assert not controller.contexts
    assert not controller.adapter.executions


@pytest.mark.parametrize("parameters", [{"think_mode": True}, {"n": 11}])
def test_invalid_request_never_dispatches(
    controller: UMMController, parameters: dict[str, int | bool]
) -> None:
    with pytest.raises(UMMRequestError):
        controller.transition(
            StagePayload("request", OmniRequest("prompt", parameters), None)
        )
    assert not controller.contexts
    assert not controller.adapter.executions


def test_umm_config_retains_tp_constraints() -> None:
    configuration = SenseNovaU1UMMPipelineConfig(model_path="unused")
    assert configuration.resolved_entry_stage == "orchestrator"
    assert configuration.stage_named("generate").next == "orchestrator"
    assert configuration.stage_named("output").terminal
    configuration.stage_named("generate").tp_size = 3
    configuration.stage_named("generate").gpu = [0, 1, 2]
    with pytest.raises(ValueError, match="tp_size"):
        SenseNovaU1UMMPipelineConfig.model_validate(configuration.model_dump())


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"width": 65536, "height": 65536}, "max_output_pixels"),
        ({"num_inference_steps": 101}, "max_steps"),
    ],
)
def test_admission_limits_reject_before_generation(
    controller: UMMController, parameters: dict[str, int], message: str
) -> None:
    with pytest.raises(UMMRequestError, match=message):
        controller.transition(
            StagePayload("request", OmniRequest("prompt", parameters), None)
        )
    assert not controller.contexts
    assert controller.outbox.empty()


def test_output_reservation_rejects_before_generation(
    controller: UMMController,
) -> None:
    controller.limits = replace(controller.limits, max_retained_bytes=1024)
    with pytest.raises(UMMRequestError, match="max_retained_bytes"):
        controller.transition(StagePayload("request", OmniRequest("prompt"), None))
    assert not controller.contexts
    assert controller.outbox.empty()


def test_batch_cancellation_preserves_live_members_and_clears_scope() -> None:
    cancelled_requests = {"first"}
    payloads = [
        StagePayload(request_id, OmniRequest("prompt"), None)
        for request_id in ("first", "second")
    ]
    with GenerationScope(
        payloads, cancelled_requests.__contains__, torch.device("cpu"), None
    ):
        generation_checkpoint()
        cancelled_requests.add("second")
        with pytest.raises(SenseNovaCancelledError):
            generation_checkpoint()
    generation_checkpoint()
    with GenerationScope(
        [StagePayload("next", OmniRequest("prompt"), None)],
        cancelled_requests.__contains__,
        torch.device("cpu"),
        None,
    ):
        generation_checkpoint()


def test_expired_phase_stops_at_checkpoint() -> None:
    payload = StagePayload(
        "expired",
        OmniRequest("prompt", metadata={SENSENOVA_DEADLINE_METADATA_KEY: 0.0}),
        None,
    )
    with GenerationScope(
        [payload], lambda request_id: False, torch.device("cpu"), None
    ):
        with pytest.raises(SenseNovaCancelledError):
            generation_checkpoint()


def test_abort_releases_sensenova_state(controller: UMMController) -> None:
    controller.transition(StagePayload("request", OmniRequest("prompt"), None))
    controller.abort("request")
    assert not controller.adapter.executions


@pytest.mark.parametrize("reference", ["not base64", "", []])
def test_invalid_edit_reference_never_reaches_generation(
    controller: UMMController, reference: str | list[str]
) -> None:
    request = OmniRequest(
        {"task": "image_edit", "prompt": "edit", "image_b64": reference}
    )
    with pytest.raises(UMMRequestError):
        controller.transition(StagePayload("request", request, None))
    assert not controller.contexts
    assert controller.outbox.empty()


def test_reference_pixels_and_input_bytes_are_admission_bounds(
    controller: UMMController, encoded_image: str
) -> None:
    adapter = controller.adapter
    assert isinstance(adapter, SenseNovaUMMAdapter)
    adapter.limits = adapter.limits.model_copy(update={"max_reference_pixels": 1})
    request = OmniRequest(
        {"task": "image_edit", "prompt": "edit", "image_b64": encoded_image}
    )
    with pytest.raises(UMMRequestError, match="max_reference_pixels"):
        controller.transition(StagePayload("pixels", request, None))
    adapter.limits = adapter.limits.model_copy(update={"max_input_bytes": 1})
    with pytest.raises(UMMRequestError, match="max_input_bytes"):
        controller.transition(StagePayload("bytes", OmniRequest("prompt"), None))
    assert not adapter.executions


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "url"])
def test_inline_media_rejects_tampered_content(
    encoded_image: str, field: Literal["sha256", "size_bytes", "url"]
) -> None:
    media = InlineImage.from_png(encoded_image)
    changes = {
        "sha256": "0" * 64,
        "size_bytes": media.size_bytes + 1,
        "url": "https://example.invalid/image.png",
    }
    media = media.model_copy(update={field: changes[field]})
    with pytest.raises(ValueError):
        media.encoded_image()


def test_loaded_configuration_constructs_controller_and_dispatches() -> None:
    config_path = (
        Path(__file__).resolve().parents[3] / "examples/configs/sensenova_u1_umm.yaml"
    )
    configuration = ConfigManager.from_file(str(config_path)).config
    assert isinstance(configuration, SenseNovaU1UMMPipelineConfig)
    stage = configuration.stage_named("orchestrator")
    arguments = {
        **resolve_stage_factory_kwargs(stage, configuration),
        **resolve_stage_typed_kwargs(stage),
    }
    controller = create_umm_controller("unused", **arguments)
    phase = controller.transition(StagePayload("request", OmniRequest("prompt"), None))
    assert phase is not None
    assert phase.request.metadata[SENSENOVA_DEADLINE_METADATA_KEY] > 0
    controller.abort("request")
    assert not controller.adapter.executions


@pytest.mark.parametrize("violation", ["outputs", "cancellation"])
def test_config_rejects_insufficient_bounds_and_disabled_cancellation(
    violation: Literal["outputs", "cancellation"]
) -> None:
    configuration = SenseNovaU1UMMPipelineConfig(model_path="unused")
    if violation == "outputs":
        configuration.stage_named("orchestrator").factory.max_outputs = 1
    else:
        configuration.stage_named("generate").factory.cooperative_cancellation = False
    with pytest.raises(ValueError):
        SenseNovaU1UMMPipelineConfig.model_validate(configuration.model_dump())


@pytest.mark.parametrize(
    "parameters, reference_count, pixel_limit",
    [
        ({}, 1, 512 * 512),
        ({"input_max_pixels": 512 * 512}, 1, 512 * 512),
        ({}, 3, 2048 * 2048),
    ],
)
def test_resize_budget_matches_preprocessing(
    controller: UMMController,
    encoded_image: str,
    parameters: dict[str, JsonValue],
    reference_count: int,
    pixel_limit: int,
) -> None:
    """Small references must not bypass the resized pixel budget."""
    adapter = controller.adapter
    assert isinstance(adapter, SenseNovaUMMAdapter)
    adapter.limits = adapter.limits.model_copy(
        update={"max_reference_pixels": pixel_limit}
    )
    references = [encoded_image] * reference_count
    request = OmniRequest(
        {"task": "image_edit", "prompt": "edit", "image_b64": references}, parameters
    )
    images = prepare_reference_images(
        references, SenseNovaU1ImageEditSampling.from_params(parameters)
    )
    assert sum(image.width * image.height for image in images) > pixel_limit
    with pytest.raises(UMMRequestError, match="max_reference_pixels"):
        controller.transition(StagePayload("request", request, None))
    assert not adapter.executions


def test_disabled_resize_preserves_small_reference_admission(
    controller: UMMController,
    encoded_image: str,
) -> None:
    adapter = controller.adapter
    assert isinstance(adapter, SenseNovaUMMAdapter)
    adapter.limits = adapter.limits.model_copy(update={"max_reference_pixels": 64 * 32})
    parameters = {"do_resize": False, "input_max_pixels": 2048 * 2048}
    request = OmniRequest(
        {"task": "image_edit", "prompt": "edit", "image_b64": encoded_image}, parameters
    )
    images = prepare_reference_images(
        encoded_image, SenseNovaU1ImageEditSampling.from_params(parameters)
    )
    assert images[0].size == (64, 32)
    assert controller.transition(StagePayload("request", request, None)) is not None
    controller.abort("request")


def test_decoder_pixel_limit_is_a_request_error(
    controller: UMMController,
    encoded_image: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Images over the decoder pixel limit must be rejected as invalid requests."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    request = OmniRequest(
        {"task": "image_edit", "prompt": "edit", "image_b64": encoded_image}
    )
    with pytest.raises(UMMRequestError, match="pixel limit"):
        controller.transition(StagePayload("request", request, None))
    assert not controller.contexts
    assert controller.outbox.empty()


@pytest.mark.parametrize(
    "overrides, error_message",
    [
        ({"num_inference_steps": 101}, "max_steps"),
        ({"width": 8192, "height": 8192}, "max_output_pixels"),
    ],
)
def test_stage_overrides_cannot_bypass_admission(
    controller: UMMController,
    overrides: dict[str, JsonValue],
    error_message: str,
) -> None:
    """Stage overrides bypassed the limits applied to top-level options."""
    request = OmniRequest("prompt", {"stage_params": {"generate": overrides}})
    with pytest.raises(UMMRequestError, match=error_message):
        controller.transition(StagePayload("request", request, None))
    assert not controller.adapter.executions
    assert controller.outbox.empty()


def test_stage_multi_output_count_matches_admitted_outputs(
    controller: UMMController, encoded_images: list[str]
) -> None:
    """An overridden output count was rejected after successful generation."""
    request = OmniRequest("prompt", {"stage_params": {"generate": {"n": 2}}})
    phase = controller.transition(StagePayload("request", request, None))
    assert phase is not None
    phase.data = {"modality": "image", "images_b64": encoded_images[:2]}
    result = controller.transition(phase)
    assert result is not None
    assert len(result.data["segments"]) == 2
    assert not controller.adapter.executions


@pytest.mark.parametrize("limit", ["pixels", "retained_bytes"])
def test_edit_resolved_dimensions_are_bounded_before_dispatch(
    controller: UMMController,
    encoded_image: str,
    limit: Literal["pixels", "retained_bytes"],
) -> None:
    """Aspect-ratio rounding exceeded the reserved pixel or byte budget."""
    if limit == "pixels":
        controller.adapter.limits = controller.adapter.limits.model_copy(
            update={"max_output_pixels": 96 * 64}
        )
        error_message = "max_output_pixels"
    else:
        controller.limits = replace(controller.limits, max_retained_bytes=126000)
        error_message = "max_retained_bytes"
    request = OmniRequest(
        {"task": "image_edit", "prompt": "edit", "image_b64": encoded_image},
        {"width": 96, "height": 64, "do_resize": False},
    )
    with pytest.raises(UMMRequestError, match=error_message):
        controller.transition(StagePayload("request", request, None))
    assert not controller.adapter.executions
    assert controller.outbox.empty()
