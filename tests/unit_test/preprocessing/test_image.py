# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

import pytest
from PIL import Image

from sglang_omni.preprocessing.image import (
    ImageMediaIO,
    ensure_image_list_async,
    load_image_path,
)
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["bytes", "file", "path", "data_url"])
@pytest.mark.parametrize("orientation", [None, 2, 6])
async def test_image_loading_applies_exif_orientation(tmp_path, source, orientation):
    image = Image.new("RGB", (12, 8), "red")
    image.paste("blue", (0, 0, 6, 4))
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    data = buffer.getvalue()
    expected = Image.open(BytesIO(data)).convert("RGB")
    if orientation == 2:
        expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        expected = expected.transpose(Image.Transpose.ROTATE_270)

    if source == "bytes":
        actual = ImageMediaIO().load_bytes(data)
    elif source == "data_url":
        url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        actual = (
            await ensure_image_list_async(
                [url], media_connector=MultiModalResourceConnector()
            )
        )[0]
    else:
        path = tmp_path / "image.jpg"
        path.write_bytes(data)
        actual = (
            ImageMediaIO().load_file(path)
            if source == "file"
            else load_image_path(path)
        )

    assert actual.mode == "RGB"
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    assert actual.getexif().get(274) is None
