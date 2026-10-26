# SPDX-License-Identifier: Apache-2.0
import base64
import io
from unittest.mock import Mock, create_autospec

import pytest
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit
from sglang_omni.models.sensenova_u1.generation import generate_image, generate_images
from sglang_omni.models.sensenova_u1.stages import (
    image_generation_batch_key,
    image_generation_request_cost,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    return create_autospec(
        PreTrainedTokenizerBase, instance=True, return_value={"input_ids": [1, 2, 3]}
    )


pytest_plugins = ["tests.unit_test.fixtures.sensenova_generation"]


def payload(n: int = 1) -> StagePayload:
    return StagePayload(
        request_id=str(n),
        request=OmniRequest(
            inputs="a cat",
            params={"width": 32, "height": 32, "seed": 7, "n": n},
        ),
        data=None,
    )


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_native_outputs_and_seed_order(
    t2i_generate_mock: Mock,
    n: int,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> None:
    model = Mock()
    t2i_generate_mock.return_value = torch.stack(
        [torch.full((3, 32, 32), -1 + i / 8) for i in range(n)]
    )
    request = payload(n)
    generate_image(request, model, tokenizer, cache_dit)
    t2i_generate_mock.assert_called_once()
    assert t2i_generate_mock.call_args.args[0] is model
    assert t2i_generate_mock.call_args.args[1] is tokenizer
    assert t2i_generate_mock.call_args.args[2] == "a cat"
    assert t2i_generate_mock.call_args.kwargs["sampling"].n == n
    assert t2i_generate_mock.call_args.kwargs["sampling"].seed == 7
    outputs = [request.data["image_b64"]] if n == 1 else request.data["images_b64"]
    assert len(outputs) == n
    for index, encoded in enumerate(outputs):
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert image.size == (32, 32)
        assert image.getpixel((0, 0)) == (int(index / 8 * 127.5),) * 3
    assert image_generation_request_cost(request) == n * 32 * 32 * 50 * 2
    if n > 1:
        assert image_generation_batch_key(request) == (
            "text_to_image_multi_output",
            str(n),
        )


def test_compatible_batch_reaches_the_model_in_one_call(
    t2i_generate_mock: Mock,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> None:
    model = Mock()
    t2i_generate_mock.return_value = torch.zeros(2, 3, 32, 32)
    requests = [payload(1), payload(1)]
    assert generate_images(requests, model, tokenizer, cache_dit) == requests
    t2i_generate_mock.assert_called_once()
    assert t2i_generate_mock.call_args.kwargs["seeds"] == [7, 7]


@pytest.mark.parametrize("output", [None, torch.zeros(1, 3, 32, 32)])
def test_invalid_multi_output_result_is_rejected(
    t2i_generate_mock: Mock,
    output: torch.Tensor | None,
    tokenizer: PreTrainedTokenizerBase,
    cache_dit: SenseNovaCacheDit,
) -> None:
    model = Mock()
    t2i_generate_mock.return_value = output
    with pytest.raises(ValueError, match="invalid multi-output"):
        generate_image(payload(2), model, tokenizer, cache_dit)
