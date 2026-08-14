# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sglang_omni.client.client import Client, _extract_inputs
from sglang_omni.client.types import (
    CompletionImage,
    GenerateChunk,
    GenerateRequest,
    Message,
)
from sglang_omni.serve import protocol


def _load_openai_adapter_functions(*names: str) -> dict[str, object]:
    source_path = Path(__file__).parents[3] / "sglang_omni" / "serve" / "openai_api.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    requested = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    assert requested.keys() == set(names)
    namespace = {
        "ChatCompletionRequest": object,
        "HTTPException": HTTPException,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=list(requested.values()), type_ignores=[])
            ),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace


def test_image_generation_request_reuses_generic_image_contract() -> None:
    request = protocol.ChatCompletionRequest(
        model="inclusionAI/LLaDA2.0-Uni",
        messages=[{"role": "user", "content": "paint a red fox"}],
        modalities=["image"],
        image_generation={"width": 1024, "height": 1024, "seed": 7},
    )

    assert request.image_generation == {
        "width": 1024,
        "height": 1024,
        "seed": 7,
    }


def test_image_generation_requires_image_output_modality() -> None:
    functions = _load_openai_adapter_functions(
        "_requested_modalities", "_validate_chat_image_generation_request"
    )
    validate = functions["_validate_chat_image_generation_request"]

    with pytest.raises(HTTPException, match="requires modalities"):
        validate(SimpleNamespace(modalities=None, image_generation={}))

    validate(SimpleNamespace(modalities=["image"], image_generation={}))


def test_image_generation_config_is_forwarded_with_message_inputs() -> None:
    inputs = _extract_inputs(
        GenerateRequest(
            messages=[Message(role="user", content="paint a red fox")],
            metadata={"image_generation": {"width": 1024, "height": 768}},
        )
    )

    assert inputs == {
        "messages": [{"role": "user", "content": "paint a red fox"}],
        "image_generation": {"width": 1024, "height": 768},
    }


def test_image_modality_without_params_is_preserved_as_request_intent() -> None:
    request = Client._build_omni_request(
        GenerateRequest(
            messages=[Message(role="user", content="paint a red fox")],
            output_modalities=["image"],
        )
    )

    assert request.metadata["output_modalities"] == ["image"]
    assert "image_generation" not in request.metadata


def test_streaming_rejection_uses_image_modality_not_optional_params() -> None:
    functions = _load_openai_adapter_functions(
        "_requested_modalities", "_reject_streaming_image_generation"
    )
    reject_streaming = functions["_reject_streaming_image_generation"]

    with pytest.raises(HTTPException, match="streaming image generation"):
        reject_streaming(
            SimpleNamespace(
                modalities=["image"],
                image_generation=None,
                stream=True,
            )
        )

    reject_streaming(
        SimpleNamespace(
            modalities=["text"],
            image_generation=None,
            stream=True,
        )
    )


def test_image_response_uses_images_array_without_private_singular_field() -> None:
    image = protocol.ChatCompletionImage(
        id="image-request-0",
        data="aW1hZ2U=",
        format="png",
        width=1024,
        height=1024,
    )
    message = {"role": "assistant", "images": [image.model_dump()]}

    assert message == {
        "role": "assistant",
        "images": [
            {
                "id": "image-request-0",
                "data": "aW1hZ2U=",
                "format": "png",
                "width": 1024,
                "height": 1024,
            }
        ],
    }
    assert "image" not in message


def test_client_result_builder_collects_generic_images_array() -> None:
    chunk = Client._default_result_builder(
        "request-0",
        {
            "modality": "image",
            "images": [
                {
                    "id": "image-request-0",
                    "data": "aW1hZ2U=",
                    "format": "png",
                    "width": 1024,
                    "height": 768,
                }
            ],
        },
    )

    assert chunk.images == [
        CompletionImage(
            id="image-request-0",
            data="aW1hZ2U=",
            format="png",
            width=1024,
            height=768,
        )
    ]
    assert "image" not in chunk.to_dict()


def test_completion_preserves_image_results() -> None:
    class _ImageClient(Client):
        async def generate(self, request, request_id=None):
            del request
            yield GenerateChunk(
                request_id=request_id or "request-0",
                images=[
                    CompletionImage(
                        id="image-request-0",
                        data="aW1hZ2U=",
                        format="png",
                        width=1024,
                        height=768,
                    )
                ],
                modality="image",
                finish_reason="stop",
            )

    result = asyncio.run(
        _ImageClient(SimpleNamespace()).completion(
            SimpleNamespace(), request_id="request-0"
        )
    )

    assert [image.id for image in result.images] == ["image-request-0"]


def test_openai_adapter_preserves_image_generation_metadata_source() -> None:
    source_path = Path(__file__).parents[3] / "sglang_omni" / "serve" / "openai_api.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_chat_generate_request"
    )
    source = ast.get_source_segment(source_path.read_text(encoding="utf-8"), function)

    assert 'metadata["image_generation"] = dict(req.image_generation)' in source
    assert "output_modalities = _requested_modalities(req)" in source
