# SPDX-License-Identifier: Apache-2.0
"""Image generation request gating helpers."""

from __future__ import annotations

from sglang_omni.proto import OmniRequest, StagePayload


def _payload(*, metadata=None, inputs=None) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=inputs or "hello", metadata=metadata or {}),
        data={},
    )


def test_should_generate_image_requires_image_output_modality() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_request import (
        should_generate_image,
    )

    assert should_generate_image(_payload(metadata={"output_modalities": ["image"]}))
    assert should_generate_image(
        _payload(metadata={"output_modalities": ["text", "image"]})
    )
    assert not should_generate_image(_payload(metadata={"output_modalities": ["text"]}))
    assert not should_generate_image(_payload(metadata={}))


def test_extract_image_generation_params_prefers_metadata() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_request import (
        extract_image_generation_params,
    )

    payload = _payload(
        metadata={"image_generation": {"size": "512x512", "seed": 7}},
        inputs={"image_generation": {"size": "1024x1024"}},
    )

    assert extract_image_generation_params(payload) == {
        "size": "512x512",
        "seed": 7,
    }


def test_extract_image_generation_params_falls_back_to_inputs() -> None:
    from sglang_omni.models.ming_omni.components.image_gen_request import (
        extract_image_generation_params,
    )

    payload = _payload(inputs={"image_generation": {"size": "768x768"}})

    assert extract_image_generation_params(payload) == {"size": "768x768"}


def test_extract_image_generation_params_passes_semantic_source_through() -> None:
    # The request helper is a pure pass-through: it must NOT inject a
    # semantic_source default (that belongs to the executor's _extract_params).
    # A request that omits semantic_source stays omitted here; an explicit
    # value is preserved verbatim.
    from sglang_omni.models.ming_omni.components.image_gen_request import (
        extract_image_generation_params,
    )

    omitted = _payload(metadata={"image_generation": {"size": "512x512"}})
    assert "semantic_source" not in extract_image_generation_params(omitted)

    explicit = _payload(
        metadata={"image_generation": {"semantic_source": "standalone"}}
    )
    assert extract_image_generation_params(explicit)["semantic_source"] == "standalone"
