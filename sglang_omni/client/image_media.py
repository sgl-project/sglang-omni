# SPDX-License-Identifier: Apache-2.0
"""Inline image validation and producer usage at UMM boundaries."""

import base64
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from sglang_omni.client.types import ImageUsageDict


class InlineImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["image"] = "image"
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str

    @classmethod
    def from_png(cls, encoded_image: str) -> "InlineImage":
        image_bytes = base64.b64decode(encoded_image, validate=True)
        return cls(
            mime_type="image/png",
            size_bytes=len(image_bytes),
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            url=f"data:image/png;base64,{encoded_image}",
        )

    def encoded_image(self) -> str:
        prefix = f"data:{self.mime_type};base64,"
        if not self.url.startswith(prefix):
            raise ValueError(
                "UMM image must be an inline data URL matching its MIME type"
            )
        else:
            encoded_image = self.url[len(prefix) :]
        image_bytes = base64.b64decode(encoded_image, validate=True)
        if (
            len(image_bytes) != self.size_bytes
            or hashlib.sha256(image_bytes).hexdigest() != self.sha256
        ):
            raise ValueError("UMM image size or SHA-256 does not match its content")
        else:
            return encoded_image


def parse_image_usage(value: object) -> ImageUsageDict | None:
    if value is None:
        return None
    else:
        usage = TypeAdapter(ImageUsageDict).validate_python(value, strict=True)
    counts = [usage.get("input_tokens"), usage.get("total_tokens")]
    details = usage.get("input_tokens_details")
    if details is not None:
        counts.extend((details["text_tokens"], details["image_tokens"]))
    else:
        pass
    if any(count is not None and count < 0 for count in counts):
        raise ValueError("Image usage token counts must be non-negative")
    else:
        return usage
