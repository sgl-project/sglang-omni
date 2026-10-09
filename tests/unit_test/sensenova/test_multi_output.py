# SPDX-License-Identifier: Apache-2.0
import base64
import io
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.stages import (
    generate_image,
    generate_images,
    image_generation_batch_key,
    image_generation_request_cost,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


def payload(n=1):
    return StagePayload(
        request_id=str(n),
        request=OmniRequest(
            inputs="a cat",
            params={"width": 32, "height": 32, "seed": 7, "n": n},
        ),
        data=None,
    )


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_native_outputs_and_seed_order(n):
    model = Mock()
    model.t2i_generate.return_value = torch.stack(
        [torch.full((3, 32, 32), -1 + i / 8) for i in range(n)]
    )
    request = payload(n)
    generate_image(request, model, None)
    model.t2i_generate.assert_called_once()
    assert model.t2i_generate.call_args.args[1] == "a cat"
    assert model.t2i_generate.call_args.kwargs["batch_size"] == n
    assert model.t2i_generate.call_args.kwargs["seed"] == (
        7 if n == 1 else list(range(7, 7 + n))
    )
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


def test_multi_output_requests_execute_separately():
    model = Mock()
    model.t2i_generate.side_effect = [
        torch.zeros(2, 3, 32, 32),
        torch.zeros(1, 3, 32, 32),
    ]
    requests = [payload(2), payload(1)]
    assert generate_images(requests, model, None) == requests
    assert [
        call.kwargs["batch_size"] for call in model.t2i_generate.call_args_list
    ] == [2, 1]


@pytest.mark.parametrize("output", [None, torch.zeros(1, 3, 32, 32)])
def test_invalid_multi_output_result_is_rejected(output):
    model = Mock()
    model.t2i_generate.return_value = output
    with pytest.raises(ValueError, match="invalid multi-output"):
        generate_image(payload(2), model, None)
